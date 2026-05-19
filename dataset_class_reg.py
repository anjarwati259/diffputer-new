import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import os
import json

DATA_DIR = 'datasets'

# ===========================================================================
#  Supervised Learnable Embedding Model dengan Regression Head
#
#  [MODIFIKASI dari dataset_class.py]
#  Perubahan utama:
#    - Classification head (n_classes output + CrossEntropy)
#      → Regression head (1 output + Sigmoid + MSELoss)
#    - Normalisasi label: min-max → sigmoid → (0, 1)
#    - Loss: alpha * MSE(reg_output, label_sigmoid) + beta * CE(recon, idx)
#
#  Yang TIDAK BERUBAH:
#    - Arsitektur encoder (nn.Embedding + MLP + LayerNorm)
#    - Decoder (Linear per kolom → logits rekonstruksi)
#    - Reconstruction loss (tetap CrossEntropy)
#    - encode_with_embedding, decode_cat_from_embedding
#    - load_dataset, get_eval, mean_std
#    - Pipeline diffusion di main_class.py tidak perlu diubah sama sekali
# ===========================================================================

def compute_embedding_size(n_categories: int) -> int:
    """
    Hitung ukuran embedding optimal berdasarkan jumlah kategori.
    Rumus: min(600, round(1.6 * n_categories^0.56))
    Referensi: Guo & Berkhahn (2016)
    [TIDAK BERUBAH]
    """
    return min(600, round(1.6 * n_categories ** 0.56))


def sigmoid_normalize_labels(labels_raw: np.ndarray,
                              label_min: float = None,
                              label_max: float = None):
    """
    Normalisasi label ke range (0, 1) menggunakan min-max → sigmoid.

    Alur:
        label_raw → min-max → [0, 1] → sigmoid → (0, 1)

    Parameter
    ---------
    labels_raw  : np.ndarray [N]  — nilai label float asli
    label_min   : float — min dari train labels (untuk transform test)
    label_max   : float — max dari train labels (untuk transform test)

    Return
    ------
    labels_norm : np.ndarray [N]  float32  — label ternormalisasi (0, 1)
    label_min   : float
    label_max   : float
    """
    labels_raw = labels_raw.astype(np.float32)

    if label_min is None:
        label_min = float(labels_raw.min())
    if label_max is None:
        label_max = float(labels_raw.max())

    label_range = label_max - label_min + 1e-8

    # Step 1: min-max → [0, 1]
    labels_minmax = (labels_raw - label_min) / label_range

    # Step 2: sigmoid → (0, 1)
    labels_norm = (1.0 / (1.0 + np.exp(-labels_minmax))).astype(np.float32)

    return labels_norm, label_min, label_max


def inverse_sigmoid_labels(labels_norm: np.ndarray,
                            label_min: float,
                            label_max: float) -> np.ndarray:
    """
    Inverse normalisasi label dari (0, 1) kembali ke skala asli.

    Alur (kebalikan sigmoid_normalize_labels):
        labels_norm → inverse sigmoid (logit) → inverse min-max → skala asli

    Dipakai HANYA jika ingin melaporkan performa prediksi label
    dalam skala asli (bukan untuk training loss maupun evaluasi imputasi).

    Parameter
    ---------
    labels_norm : np.ndarray [N]  float32  — output sigmoid regressor (0, 1)
    label_min   : float
    label_max   : float

    Return
    ------
    labels_asli : np.ndarray [N]  float32  — nilai label di skala asli
    """
    labels_norm  = np.clip(labels_norm, 1e-7, 1.0 - 1e-7)  # hindari log(0)
    label_range  = label_max - label_min + 1e-8

    # Step 1 inverse: logit (inverse sigmoid)
    logit        = np.log(labels_norm / (1.0 - labels_norm))

    # Step 2 inverse: inverse min-max
    labels_asli  = logit * label_range + label_min

    return labels_asli.astype(np.float32)


class SupervisedLearnableEmbeddingModel(nn.Module):
    """
    Model Supervised Learnable Embedding dengan Regression Head.

    [MODIFIKASI dari dataset_class.py]
    Perubahan:
      - self.classifier (CrossEntropy, n_classes output)
        → self.regressor (MSE, 1 output + Sigmoid)

    Alur (tidak berubah kecuali bagian supervised head):
      cat_idx [batch, n_cat_cols]
        → nn.Embedding per kolom → concat → [batch, total_emb_dim]
        → (opsional) Linear → SiLU → Linear   (use_mlp=True)
        → LayerNorm
        → z [batch, total_emb_dim]
        → Regressor → Sigmoid → [batch]        ← BERUBAH (dari classifier)
        → (+ noise σ=noise_std saat training)
        → Linear Decoder per kolom → logits rekonstruksi

    Parameter
    ---------
    cat_dims   : list[int]   jumlah kategori unik per kolom
    emb_sizes  : list[int]   dimensi embedding per kolom
    dropout    : float       dropout pada regressor head
    hidden_dim : int         dimensi hidden layer untuk regressor
    use_mlp    : bool        aktifkan 1 hidden layer setelah concat
    mlp_ratio  : float       hidden_dim_mlp = int(total_emb_dim * mlp_ratio)
    noise_std  : float       std Gaussian noise sebelum decoding saat training
    """

    def __init__(self, cat_dims: list, emb_sizes: list,
                 dropout: float = 0.1, hidden_dim: int = 256,
                 use_mlp: bool = True, mlp_ratio: float = 1.5,
                 noise_std: float = 0.1):
        super().__init__()

        # Satu nn.Embedding per kolom kategorikal [TIDAK BERUBAH]
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=n_cat, embedding_dim=emb_dim)
            for n_cat, emb_dim in zip(cat_dims, emb_sizes)
        ])

        self.total_emb_dim = sum(emb_sizes)
        self.n_cols        = len(cat_dims)
        self.cat_dims      = cat_dims
        self.emb_sizes     = emb_sizes
        self.noise_std     = noise_std
        self.use_mlp       = use_mlp

        # Optional MLP setelah concat [TIDAK BERUBAH]
        if use_mlp:
            hidden_dim_mlp = max(self.total_emb_dim, int(self.total_emb_dim * mlp_ratio))
            self.mlp = nn.Sequential(
                nn.Linear(self.total_emb_dim, hidden_dim_mlp),
                nn.SiLU(),
                nn.Linear(hidden_dim_mlp, self.total_emb_dim),
            )
        else:
            self.mlp = None

        # LayerNorm [TIDAK BERUBAH]
        self.layer_norm = nn.LayerNorm(self.total_emb_dim)
        self.out_dim    = self.total_emb_dim

        # ── [BERUBAH] Regression Head — menggantikan Classifier Head ────────
        # Output: 1 nilai kontinu + Sigmoid → range (0, 1)
        # Cocok dengan label yang dinormalisasi via sigmoid
        # Loss: MSELoss (bukan CrossEntropyLoss)
        self.dropout   = nn.Dropout(dropout)
        self.regressor = nn.Sequential(
            nn.Linear(self.total_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()          # output → (0, 1), konsisten dengan label sigmoid
        )

        # Linear Decoder per kolom [TIDAK BERUBAH]
        self.decoders = nn.ModuleList([
            nn.Linear(emb_size, n_cat)
            for n_cat, emb_size in zip(cat_dims, emb_sizes)
        ])

    def encode(self, x_cat: torch.Tensor) -> torch.Tensor:
        """
        Encode integer index kategorikal → vektor embedding dense + LayerNorm.
        [TIDAK BERUBAH]
        """
        embedded = [
            self.embeddings[i](x_cat[:, i])
            for i in range(self.n_cols)
        ]
        z = torch.cat(embedded, dim=1)

        if self.mlp is not None:
            z = self.mlp(z)

        z = self.layer_norm(z)
        return z

    def regress(self, z: torch.Tensor) -> torch.Tensor:
        """
        Regression head: embedding → prediksi label (0, 1).

        [BERUBAH — menggantikan classify()]
        z      : [batch, total_emb_dim]
        return : [batch]  — nilai float (0, 1) setelah sigmoid
        """
        return self.regressor(z).squeeze(1)   # [batch, 1] → [batch]

    def decode(self, z: torch.Tensor) -> list:
        """
        Linear Decoder: embedding → logit tiap kolom kategorikal.
        [TIDAK BERUBAH]
        """
        per_col = torch.split(z, self.emb_sizes, dim=1)
        return [self.decoders[i](per_col[i]) for i in range(self.n_cols)]

    def forward(self, x_cat: torch.Tensor, add_noise: bool = False):
        """
        Full forward: encode → regress + (noise) decode.

        [BERUBAH] return reg_output (float) bukan class_logits

        return : (z, reg_output, recon_logits)
            z            [batch, total_emb_dim]
            reg_output   [batch]               — prediksi label (0, 1)
            recon_logits list[n_cat_cols] of [batch, vocab_i]
        """
        z          = self.encode(x_cat)
        reg_output = self.regress(z)           # [batch] float (0, 1)

        # Noise sebelum decoding saat training [TIDAK BERUBAH]
        if add_noise and self.training and self.noise_std > 0:
            z_noisy = z + torch.randn_like(z) * self.noise_std
        else:
            z_noisy = z

        recon_logits = self.decode(z_noisy)
        return z, reg_output, recon_logits


def train_supervised_embedding_model(cat_idx_array: np.ndarray,
                                     labels: np.ndarray,
                                     cat_dims: list,
                                     emb_sizes: list,
                                     device: str,
                                     n_epochs: int = 1000,
                                     batch_size: int = 1024,
                                     lr: float = 1e-3,
                                     dropout: float = 0.1,
                                     hidden_dim: int = 256,
                                     use_mlp: bool = True,
                                     mlp_ratio: float = 1.5,
                                     noise_std: float = 0.01,
                                     patience: int = 40
                                     ) -> SupervisedLearnableEmbeddingModel:
    """
    Latih SupervisedLearnableEmbeddingModel dengan Regression Head.

    [MODIFIKASI dari dataset_class.py]
    Perubahan:
      - Parameter n_classes dihapus (tidak relevan untuk regression)
      - Loss: MSELoss menggantikan CrossEntropyLoss untuk supervised signal
      - Label dtype: float32 (bukan long/int)
      - Nama variabel: reg_output, reg_loss (bukan class_logits, class_loss)

    Loss total:
      loss = alpha * MSE(reg_output, label_sigmoid)   ← BERUBAH
           + beta  * CE(recon_logits, cat_idx)        ← TIDAK BERUBAH

    Parameter
    ---------
    labels : np.ndarray [N] float32 — label yang sudah dinormalisasi sigmoid (0,1)
    """
    model = SupervisedLearnableEmbeddingModel(
        cat_dims,
        emb_sizes,
        dropout   = dropout,
        hidden_dim = hidden_dim,
        use_mlp   = use_mlp,
        mlp_ratio = mlp_ratio,
        noise_std = noise_std,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # ── [BERUBAH] Loss functions ─────────────────────────────────────────
    mse_loss = nn.MSELoss()          # untuk regression (supervised signal)
    ce_loss  = nn.CrossEntropyLoss() # untuk reconstruction (tetap CE)

    # ── [BERUBAH] Label dtype: float32 (bukan long) ──────────────────────
    cat_tensor   = torch.tensor(cat_idx_array, dtype=torch.long,    device=device)
    label_tensor = torch.tensor(labels,        dtype=torch.float32, device=device)

    dataset  = torch.utils.data.TensorDataset(cat_tensor, label_tensor)
    cpu_gen  = torch.Generator(device='cpu')
    loader   = torch.utils.data.DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = True,
        num_workers = 0,
        pin_memory  = False,
        generator   = cpu_gen,
    )

    best_loss        = float('inf')
    patience_counter = 0
    best_model_state = None

    # Loss weights [alpha sama, beta sama — tidak berubah]
    alpha = 0.7   # regression loss weight
    beta  = 1.0   # reconstruction loss weight

    model.train()
    for epoch in range(n_epochs):
        total_loss      = 0.0
        total_reg_loss  = 0.0
        total_recon_loss = 0.0
        n_batches       = 0

        for batch_cat, batch_labels in loader:
            optimizer.zero_grad()

            z, reg_output, recon_logits = model(batch_cat, add_noise=True)

            # ── [BERUBAH] Regression loss (MSE) ──────────────────────────
            # reg_output : [batch] float (0, 1) — output sigmoid regressor
            # batch_labels: [batch] float (0, 1) — label ternormalisasi sigmoid
            reg_loss = mse_loss(reg_output, batch_labels)

            # ── [TIDAK BERUBAH] Reconstruction loss (CrossEntropy) ───────
            recon_loss = sum(
                ce_loss(recon_logits[i], batch_cat[:, i])
                for i in range(model.n_cols)
            ) / model.n_cols

            # Combined loss [TIDAK BERUBAH strukturnya]
            loss = alpha * reg_loss + beta * recon_loss

            loss.backward()
            optimizer.step()

            total_loss       += loss.item()
            total_reg_loss   += reg_loss.item()
            total_recon_loss += recon_loss.item()
            n_batches        += 1

        avg_loss       = total_loss       / n_batches
        avg_reg_loss   = total_reg_loss   / n_batches
        avg_recon_loss = total_recon_loss / n_batches

        if (epoch + 1) % 10 == 0:
            print(f'[Embedding] Epoch {epoch+1}/{n_epochs} - '
                  f'Loss: {avg_loss:.4f} (Reg: {avg_reg_loss:.4f}, '
                  f'Recon: {avg_recon_loss:.4f})')

        if avg_loss < best_loss:
            best_loss        = avg_loss
            patience_counter = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f'[Embedding] Early stopping triggered at epoch {epoch+1}')
            print(f'[Embedding] Best loss: {best_loss:.4f}')
            break

    if best_model_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
        print(f'[Embedding] Loaded best model from epoch {epoch + 1 - patience_counter}')

    model.eval()

    # Monitor distribusi embedding [TIDAK BERUBAH]
    with torch.no_grad():
        sample_cat = cat_tensor[:min(2048, len(cat_tensor))]
        z_sample   = model.encode(sample_cat)
        print(f'[Embedding] Distribusi embedding (N={z_sample.shape[0]}):')
        print(f'  mean={z_sample.mean().item():.4f}  '
              f'std={z_sample.std().item():.4f}  '
              f'norm_mean={z_sample.norm(dim=1).mean().item():.4f}')

        # Monitor prediksi label regression (skala sigmoid)
        reg_sample = model.regress(z_sample)
        print(f'[Embedding] Prediksi label regression (skala sigmoid):')
        print(f'  min={reg_sample.min().item():.4f}  '
              f'max={reg_sample.max().item():.4f}  '
              f'mean={reg_sample.mean().item():.4f}')

    # Freeze seluruh parameter embedding [TIDAK BERUBAH]
    for param in model.parameters():
        param.requires_grad_(False)
    print('[Embedding] Seluruh parameter embedding di-freeze untuk training diffusion.')

    return model


def encode_with_embedding(model: SupervisedLearnableEmbeddingModel,
                          cat_idx_array: np.ndarray,
                          device: str,
                          batch_size: int = 4096) -> np.ndarray:
    """
    Encode seluruh data kategorikal → embedding numpy array.
    [TIDAK BERUBAH]
    """
    model.eval()
    cat_tensor = torch.tensor(cat_idx_array, dtype=torch.long, device=device)
    dataset    = torch.utils.data.TensorDataset(cat_tensor)
    loader     = torch.utils.data.DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = 0,
        pin_memory  = False,
    )

    all_z = []
    with torch.no_grad():
        for (batch,) in loader:
            z, _, _ = model(batch, add_noise=False)
            all_z.append(z.cpu().numpy())

    return np.concatenate(all_z, axis=0).astype(np.float32)


def decode_cat_from_embedding(model: SupervisedLearnableEmbeddingModel,
                              emb_array: np.ndarray,
                              device: str,
                              batch_size: int = 4096) -> np.ndarray:
    """
    Decode embedding → prediksi kelas kategorikal (argmax logits).
    [TIDAK BERUBAH]
    """
    model.eval()
    emb_tensor = torch.tensor(emb_array, dtype=torch.float32, device=device)
    dataset    = torch.utils.data.TensorDataset(emb_tensor)
    loader     = torch.utils.data.DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = 0,
        pin_memory  = False,
    )

    all_pred = []
    with torch.no_grad():
        for (batch,) in loader:
            recon_logits = model.decode(batch)
            pred_idx = torch.stack([
                torch.argmax(logits, dim=1)
                for logits in recon_logits
            ], dim=1)
            all_pred.append(pred_idx.cpu().numpy())

    return np.concatenate(all_pred, axis=0).astype(np.int64)


# ===========================================================================
#  Load Dataset
# ===========================================================================

def load_dataset(dataname, idx=0, mask_type='MCAR', ratio='30'):
    """
    Load dataset dengan regression embedding learning.

    [MODIFIKASI dari dataset_class.py]
    Perubahan:
      - Label dinormalisasi dengan sigmoid (bukan LabelEncoder integer)
      - train_supervised_embedding_model dipanggil tanpa n_classes
      - label_min, label_max disimpan untuk keperluan inverse jika dibutuhkan

    Return sama persis dengan dataset_class.py untuk kompatibilitas main_class.py:
      train_X, test_X, ori_train_mask, ori_test_mask,
      train_num, test_num, train_cat_idx, test_cat_idx,
      extend_train_mask, extend_test_mask,
      cat_bin_num (None), emb_model, emb_sizes
    """
    ratio = str(ratio)

    data_dir  = f'datasets/{dataname}'
    info_path = f'datasets/Info/{dataname}.json'

    with open(info_path, 'r') as f:
        info = json.load(f)

    num_col_idx    = info['num_col_idx']
    cat_col_idx    = info['cat_col_idx']
    target_col_idx = info['target_col_idx']

    data_path       = f'{data_dir}/data.csv'
    train_path      = f'{data_dir}/train.csv'
    test_path       = f'{data_dir}/test.csv'
    train_mask_path = f'{data_dir}/masks/rate{ratio}/{mask_type}/train_mask_{idx}.npy'
    test_mask_path  = f'{data_dir}/masks/rate{ratio}/{mask_type}/test_mask_{idx}.npy'

    data_df  = pd.read_csv(data_path)
    train_df = pd.read_csv(train_path)
    test_df  = pd.read_csv(test_path)

    train_mask = np.load(train_mask_path)
    test_mask  = np.load(test_mask_path)

    cols = train_df.columns

    # Fitur numerik [TIDAK BERUBAH]
    data_num  = data_df[cols[num_col_idx]].values.astype(np.float32)
    train_num = train_df[cols[num_col_idx]].values.astype(np.float32)
    test_num  = test_df[cols[num_col_idx]].values.astype(np.float32)

    # ── [BERUBAH] Normalisasi label pakai sigmoid ─────────────────────────
    train_y = train_df[cols[target_col_idx]]
    test_y  = test_df[cols[target_col_idx]]

    train_labels_raw = train_y.values.ravel().astype(np.float32)
    test_labels_raw  = test_y.values.ravel().astype(np.float32)

    # Sigmoid normalisasi — fit min/max dari train saja
    train_labels, label_min, label_max = sigmoid_normalize_labels(train_labels_raw)
    test_labels,  _,         _         = sigmoid_normalize_labels(
        test_labels_raw, label_min=label_min, label_max=label_max
    )

    print(f'[Dataset] Label normalisasi sigmoid:')
    print(f'  label_min={label_min:.4f}, label_max={label_max:.4f}')
    print(f'  train_labels range: [{train_labels.min():.4f}, {train_labels.max():.4f}]')

    # Kasus: hanya fitur numerik [TIDAK BERUBAH]
    if len(cat_col_idx) == 0:
        train_X = train_num
        test_X  = test_num

        extend_train_mask = train_mask[:, num_col_idx]
        extend_test_mask  = test_mask[:, num_col_idx]

        return (train_X, test_X,
                train_mask, test_mask,
                train_num, test_num,
                None, None,
                extend_train_mask, extend_test_mask,
                None, None, None)

    # Kasus: ada fitur kategorikal [TIDAK BERUBAH kecuali pemanggilan train_emb]
    cat_columns = cols[cat_col_idx]

    data_cat  = data_df[cat_columns].astype(str)
    train_cat = train_df[cat_columns].astype(str)
    test_cat  = test_df[cat_columns].astype(str)

    encoders           = {}
    cat_dims           = []
    train_cat_idx_list = []
    test_cat_idx_list  = []

    for col in cat_columns:
        le = LabelEncoder()
        le.fit(data_cat[col])
        encoders[col] = le
        cat_dims.append(len(le.classes_))

        train_cat_idx_list.append(
            le.transform(train_cat[col]).astype(np.int64)
        )
        test_cat_idx_list.append(
            le.transform(test_cat[col]).astype(np.int64)
        )

    train_cat_idx = np.stack(train_cat_idx_list, axis=1)
    test_cat_idx  = np.stack(test_cat_idx_list,  axis=1)

    emb_sizes = [compute_embedding_size(n) for n in cat_dims]

    print(f'[Embedding] cat_dims={cat_dims}, emb_sizes={emb_sizes}, '
          f'total_emb_dim={sum(emb_sizes)}')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ── [BERUBAH] Training embedding dengan regression head ───────────────
    # - Tidak ada n_classes parameter
    # - labels sekarang float32 sigmoid (0, 1)
    print('[Embedding] Melatih SupervisedLearnableEmbeddingModel '
          '(regression MSE + reconstruction CE loss) ...')
    emb_model = train_supervised_embedding_model(
        cat_idx_array = train_cat_idx,
        labels        = train_labels,   # float32 sigmoid (0, 1)
        cat_dims      = cat_dims,
        emb_sizes     = emb_sizes,
        device        = device,
        n_epochs      = 1000,
        batch_size    = 1024,
        lr            = 1e-3,
        dropout       = 0.1,
        hidden_dim    = 256,
        use_mlp       = True,
        mlp_ratio     = 1.5,
        noise_std     = 0.01,
        patience      = 40,
    )
    print('[Embedding] Training selesai. Parameter di-freeze untuk diffusion.')

    # Encode [TIDAK BERUBAH]
    train_cat_emb = encode_with_embedding(emb_model, train_cat_idx, device)
    test_cat_emb  = encode_with_embedding(emb_model, test_cat_idx,  device)

    # Gabungkan numerik + embedding [TIDAK BERUBAH]
    train_X = np.concatenate([train_num, train_cat_emb], axis=1)
    test_X  = np.concatenate([test_num,  test_cat_emb],  axis=1)

    # Extended mask [TIDAK BERUBAH]
    train_num_mask = train_mask[:, num_col_idx]
    train_cat_mask = train_mask[:, cat_col_idx]
    test_num_mask  = test_mask[:, num_col_idx]
    test_cat_mask  = test_mask[:, cat_col_idx]

    emb_sizes_arr = np.array(emb_sizes, dtype=int)

    def extend_mask_emb(mask: np.ndarray, sizes: np.ndarray) -> np.ndarray:
        N      = mask.shape[0]
        cum    = np.concatenate(([0], sizes.cumsum()))
        result = np.zeros((N, sizes.sum()), dtype=bool)
        for j in range(len(sizes)):
            col_mask = mask[:, j][:, np.newaxis]
            result[:, cum[j]:cum[j + 1]] = np.tile(col_mask, sizes[j])
        return result

    ext_train_cat_mask = extend_mask_emb(train_cat_mask, emb_sizes_arr)
    ext_test_cat_mask  = extend_mask_emb(test_cat_mask,  emb_sizes_arr)

    extend_train_mask = np.concatenate([train_num_mask, ext_train_cat_mask], axis=1)
    extend_test_mask  = np.concatenate([test_num_mask,  ext_test_cat_mask],  axis=1)

    return (train_X, test_X,
            train_mask, test_mask,
            train_num, test_num,
            train_cat_idx, test_cat_idx,
            extend_train_mask, extend_test_mask,
            None,       # cat_bin_num (legacy)
            emb_model,
            emb_sizes)


# ===========================================================================
#  Utilities
# ===========================================================================

def mean_std(data, mask):
    """[TIDAK BERUBAH]"""
    mask      = (~mask).astype(np.float32)
    mask_sum  = mask.sum(0)
    mask_sum[mask_sum == 0] = 1
    mean      = (data * mask).sum(0) / mask_sum
    var       = ((data - mean) ** 2 * mask).sum(0) / mask_sum
    std       = np.sqrt(var)
    return mean, std


# ===========================================================================
#  Evaluasi
# ===========================================================================

def get_eval(dataname, X_recon, X_true, truth_cat_idx,
             num_num, emb_model, emb_sizes, mask,
             device='cpu', oos=False):
    """
    Hitung MAE, RMSE (numerik) dan Accuracy (kategorikal).
    [TIDAK BERUBAH] — evaluasi imputasi tidak menyentuh label regression sama sekali.

    Label regression (target_col_idx) tidak dievaluasi di sini.
    Jika ingin evaluasi prediksi label, gunakan inverse_sigmoid_labels()
    secara terpisah di luar fungsi ini.
    """
    info_path = f'datasets/Info/{dataname}.json'
    with open(info_path, 'r') as f:
        info = json.load(f)

    num_col_idx = info['num_col_idx']
    cat_col_idx = info['cat_col_idx']

    num_mask = mask[:, num_col_idx].astype(bool)
    cat_mask = mask[:, cat_col_idx].astype(bool) if len(cat_col_idx) > 0 else None

    num_pred = X_recon[:, :num_num]
    num_true = X_true[:, :num_num]

    cat_emb_pred = X_recon[:, num_num:]

    # Special case: news dataset [TIDAK BERUBAH]
    if dataname == 'news' and oos:
        drop = 6265
        num_mask     = np.delete(num_mask,     drop, axis=0)
        num_pred     = np.delete(num_pred,     drop, axis=0)
        num_true     = np.delete(num_true,     drop, axis=0)
        if cat_mask is not None:
            cat_mask = np.delete(cat_mask,     drop, axis=0)
        if truth_cat_idx is not None:
            truth_cat_idx = np.delete(truth_cat_idx, drop, axis=0)
        cat_emb_pred = np.delete(cat_emb_pred, drop, axis=0)

    # Numerik: MAE & RMSE [TIDAK BERUBAH]
    div  = num_pred[num_mask] - num_true[num_mask]
    mae  = np.abs(div).mean()
    rmse = np.sqrt((div ** 2).mean())

    # Kategorikal: Akurasi via Linear Decoder [TIDAK BERUBAH]
    acc = np.nan
    if (truth_cat_idx is not None
            and len(cat_col_idx) > 0
            and emb_model is not None
            and emb_sizes is not None):

        pred_cat_idx = decode_cat_from_embedding(emb_model, cat_emb_pred, device)

        correct_total = 0
        total_missing = 0

        for j in range(len(cat_col_idx)):
            rows_miss = cat_mask[:, j]
            if rows_miss.sum() == 0:
                continue

            pred_j = pred_cat_idx[:, j]
            true_j = truth_cat_idx[:, j].astype(int)

            correct = (pred_j[rows_miss] == true_j[rows_miss]).sum()
            correct_total += int(correct)
            total_missing += int(rows_miss.sum())

        if total_missing > 0:
            acc = correct_total / total_missing

    return mae, rmse, acc