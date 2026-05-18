import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
import os
import json
import time
import pickle

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

DATA_DIR = 'datasets'

# ===========================================================================
#  MRmD Discretizer (Max-Relevance-Min-Divergence)
#  [TIDAK BERUBAH] — identik dengan dataset_mrmd.py
#  Berdasarkan: Wang et al., Pattern Recognition 149 (2024) 110236
# ===========================================================================

def _mutual_information(a_discrete: np.ndarray, c: np.ndarray) -> float:
    n = len(a_discrete)
    bins_a = np.unique(a_discrete)
    bins_c = np.unique(c)
    mi = 0.0
    for a_val in bins_a:
        mask_a = (a_discrete == a_val)
        p_a = mask_a.sum() / n
        for c_val in bins_c:
            p_ac = ((mask_a) & (c == c_val)).sum() / n
            p_c  = (c == c_val).sum() / n
            if p_ac > 0 and p_a > 0 and p_c > 0:
                mi += p_ac * np.log(p_ac / (p_a * p_c))
    return max(mi, 0.0)


def _js_divergence(p_t: np.ndarray, p_v: np.ndarray) -> float:
    eps    = 1e-10
    p_t    = np.clip(p_t, eps, 1.0)
    p_v    = np.clip(p_v, eps, 1.0)
    p_star = 0.5 * (p_t + p_v)
    kl_t   = np.sum(p_t * np.log(p_t / p_star))
    kl_v   = np.sum(p_v * np.log(p_v / p_star))
    return float(np.clip(0.5 * (kl_t + kl_v), 0.0, 1.0))


def _get_distributions(a_train: np.ndarray, a_val: np.ndarray):
    all_bins = np.union1d(np.unique(a_train), np.unique(a_val))
    p_t = np.array([(a_train == b).sum() for b in all_bins], dtype=float)
    p_v = np.array([(a_val   == b).sum() for b in all_bins], dtype=float)
    if p_t.sum() > 0: p_t /= p_t.sum()
    if p_v.sum() > 0: p_v /= p_v.sum()
    return p_t, p_v


def _make_bins(cut_points: np.ndarray, x_min: float, x_max: float) -> np.ndarray:
    lo = x_min - 1e-10
    hi = x_max + 1e-10
    if len(cut_points) == 0:
        return np.array([lo, hi])
    return np.concatenate([[lo], np.sort(cut_points), [hi]])


def _discretize_mrmd(x: np.ndarray, cut_points: np.ndarray,
                     x_min: float, x_max: float) -> np.ndarray:
    if len(cut_points) == 0:
        return np.zeros(len(x), dtype=int)
    bins = _make_bins(cut_points, x_min, x_max)
    return (np.digitize(x, bins[1:-1])).astype(int)


class MRmDDiscretizer(BaseEstimator, TransformerMixin):
    """
    MRmD (Max-Relevance-Min-Divergence) Discretizer.
    [TIDAK BERUBAH] — identik dengan dataset_mrmd.py
    """

    def __init__(self, val_size: float = 0.125, N_D: int = 50,
                 random_state=None, verbose: bool = False):
        self.val_size     = val_size
        self.N_D          = N_D
        self.random_state = random_state
        self.verbose      = verbose

    def fit(self, X, y):
        if hasattr(X, 'columns'):
            self.feature_names_in_ = np.array(X.columns)
            X = np.array(X, dtype=float)
        else:
            X = np.array(X, dtype=float)

        y = np.array(y)
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        rng       = np.random.RandomState(self.random_state)
        val_n     = max(1, int(n_samples * self.val_size))
        val_idx   = rng.choice(n_samples, size=val_n, replace=False)
        train_idx = np.setdiff1d(np.arange(n_samples), val_idx)

        X_tr, y_tr = X[train_idx], y[train_idx]
        X_vl       = X[val_idx]

        if self.verbose:
            print(f'[MRmD] n_train={len(train_idx)}, n_val={len(val_idx)}, '
                  f'n_features={n_features}')

        self.cut_points_ = []
        self.x_min_      = []
        self.x_max_      = []
        self.n_bins_     = []

        for j in range(n_features):
            x_tr_j = X_tr[:, j]
            x_vl_j = X_vl[:, j]

            x_min = float(x_tr_j.min())
            x_max = float(x_tr_j.max())
            self.x_min_.append(x_min)
            self.x_max_.append(x_max)

            unique_tr = np.unique(x_tr_j)
            if len(unique_tr) <= 1:
                self.cut_points_.append(np.array([]))
                self.n_bins_.append(1)
                if self.verbose:
                    print(f'  [MRmD] Col {j}: konstan, skip.')
                continue

            cp = self._fit_one_attribute(x_tr_j, x_vl_j, y_tr,
                                         unique_tr, x_min, x_max, j)
            self.cut_points_.append(cp)
            n_bins = len(cp) + 1
            self.n_bins_.append(n_bins)
            print(f'  [MRmD] Col {j}: {len(cp)} cut points → {n_bins} bins')

        return self

    def _fit_one_attribute(self, x_tr, x_vl, c_tr,
                            unique_all, x_min, x_max, j_idx):
        D_star_j = np.array([])
        S_j      = unique_all.copy()
        psi_max  = -np.inf

        while len(S_j) > 0:
            best_psi = -np.inf
            best_dk  = None

            for dk in S_j:
                D_k_j     = np.append(D_star_j, dk)
                a_tr_disc = _discretize_mrmd(x_tr, D_k_j, x_min, x_max)
                a_vl_disc = _discretize_mrmd(x_vl, D_k_j, x_min, x_max)

                n_cuts   = len(D_star_j) + 1
                lam      = np.exp(-n_cuts / self.N_D)
                mi_val   = _mutual_information(a_tr_disc, c_tr)
                p_t, p_v = _get_distributions(a_tr_disc, a_vl_disc)
                jsd_val  = _js_divergence(p_t, p_v)
                psi_k    = lam * mi_val - jsd_val

                if psi_k > best_psi:
                    best_psi = psi_k
                    best_dk  = dk

            if best_dk is None or best_psi <= psi_max:
                break

            psi_max  = best_psi
            D_star_j = np.append(D_star_j, best_dk)
            S_j      = S_j[S_j != best_dk]

        result = np.sort(D_star_j)
        if self.verbose:
            print(f'  Fitur [{j_idx}]: {len(result)} cut points '
                  f'→ {np.round(result, 4).tolist()}')
        return result

    def transform(self, X) -> np.ndarray:
        check_is_fitted(self, 'cut_points_')
        if hasattr(X, 'values'):
            X = X.values
        X = np.array(X, dtype=float)

        out = np.empty(X.shape, dtype=np.int64)
        for j, cp in enumerate(self.cut_points_):
            out[:, j] = _discretize_mrmd(
                X[:, j], cp, self.x_min_[j], self.x_max_[j]
            ).astype(np.int64)

        return out

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y).transform(X)

    def get_n_bins(self) -> np.ndarray:
        check_is_fitted(self, 'cut_points_')
        return np.array(self.n_bins_)

    def get_bin_midpoints(self, X_norm: np.ndarray,
                          X_norm_binned: np.ndarray) -> list:
        """
        Hitung nilai tengah (midpoint) setiap bin dalam skala normalisasi.
        [TIDAK BERUBAH]
        """
        n_cols    = X_norm.shape[1]
        midpoints = []

        for col in range(n_cols):
            n_bins = self.n_bins_[col]
            mids   = np.zeros(n_bins, dtype=np.float32)

            for b in range(n_bins):
                mask = X_norm_binned[:, col] == b
                if mask.sum() > 0:
                    mids[b] = float(X_norm[mask, col].mean())
                else:
                    mids[b] = float(b) / max(n_bins - 1, 1)

            midpoints.append(mids)

        return midpoints

    def summary(self):
        check_is_fitted(self, 'cut_points_')
        print('=' * 60)
        print(f"{'MRmD Discretizer — Summary':^60}")
        print('=' * 60)
        print(f"  {'Fitur':<20} {'# Bins':>7}   Cut Points")
        print('  ' + '-' * 56)
        for j, cp in enumerate(self.cut_points_):
            name   = str(self.feature_names_in_[j])[:20] if hasattr(self, 'feature_names_in_') else f'fitur_{j}'
            cp_str = np.round(cp, 4).tolist() if len(cp) > 0 else '[ ]'
            print(f'  {name:<20} {len(cp)+1:>7}   {cp_str}')
        print('=' * 60)
        print(f'  Total cut points: {sum(len(c) for c in self.cut_points_)}')
        print(f'  N_D={self.N_D}, val_size={self.val_size}')
        print('=' * 60)


# ===========================================================================
#  One-Hot Encoding + Decode helpers
#  [BARU] Menggantikan nn.Embedding — representasi langsung sebagai float vector
#
#  Alur encode:
#    bin/label index (int) → one-hot float [0,0,...,1,...,0]
#
#  Alur decode numerik (identik logikanya dengan decode_num_from_embedding):
#    logits rekonstruksi → softmax → weighted sum midpoints → nilai kontinu
#
#  Alur decode kategorikal:
#    logits rekonstruksi → argmax → predicted class index
# ===========================================================================

def onehot_encode(idx_array: np.ndarray, vocab_sizes: list) -> np.ndarray:
    """
    Encode integer index array → one-hot float array (concat semua kolom).

    idx_array   : [N, n_cols]  — integer index per kolom (bin atau label)
    vocab_sizes : list[int]    — jumlah kategori/bin per kolom

    Return : [N, sum(vocab_sizes)]  float32
    """
    parts = []
    for col, v in enumerate(vocab_sizes):
        idx = idx_array[:, col].astype(np.int64)
        oh  = np.zeros((len(idx), v), dtype=np.float32)
        oh[np.arange(len(idx)), idx] = 1.0
        parts.append(oh)
    return np.concatenate(parts, axis=1)


def decode_num_from_onehot(X_recon: np.ndarray,
                           bin_midpoints: list,
                           num_vocab_sizes: list,
                           num_offsets: list) -> np.ndarray:
    """
    Decode bagian numerik dari rekonstruksi one-hot → nilai kontinu (skala norm).

    Logika identik dengan decode_num_from_embedding di dataset_mrmd.py,
    tapi tanpa embedding model — langsung dari slice X_recon.

    Metode: softmax weighted sum midpoints
        logits = X_recon[:, start:end]          # [N, n_bins]
        probs  = softmax(logits)                 # [N, n_bins]
        pred   = probs @ midpoints               # [N]  — nilai kontinu

    Parameter
    ---------
    X_recon        : [N, total_onehot_dim]
    bin_midpoints  : list[n_num_cols] of np.ndarray — midpoint per bin, skala norm
    num_vocab_sizes: list[n_num_cols] int — jumlah bin per kolom numerik
    num_offsets    : list[n_num_cols] int — posisi awal tiap kolom di X_recon

    Return : [N, n_num_cols]  float32
    """
    N     = X_recon.shape[0]
    n_num = len(num_vocab_sizes)
    preds = np.zeros((N, n_num), dtype=np.float32)

    for col in range(n_num):
        start  = num_offsets[col]
        end    = start + num_vocab_sizes[col]
        logits = X_recon[:, start:end]                           # [N, n_bins]

        # Numerically stable softmax
        logits_shifted = logits - logits.max(axis=1, keepdims=True)
        exp_l          = np.exp(logits_shifted)
        probs          = exp_l / exp_l.sum(axis=1, keepdims=True)  # [N, n_bins]

        # Weighted sum midpoints → nilai kontinu
        mids          = bin_midpoints[col]                        # [n_bins]
        preds[:, col] = (probs * mids[np.newaxis, :]).sum(axis=1)

    return preds


def decode_cat_from_onehot(X_recon: np.ndarray,
                           cat_vocab_sizes: list,
                           cat_offsets: list) -> np.ndarray:
    """
    Decode bagian kategorikal dari rekonstruksi one-hot → predicted class index.

    Parameter
    ---------
    X_recon        : [N, total_onehot_dim]
    cat_vocab_sizes: list[n_cat_cols] int — jumlah kelas per kolom
    cat_offsets    : list[n_cat_cols] int — posisi awal tiap kolom di X_recon

    Return : [N, n_cat_cols]  int64
    """
    N     = X_recon.shape[0]
    n_cat = len(cat_vocab_sizes)
    preds = np.zeros((N, n_cat), dtype=np.int64)

    for col in range(n_cat):
        start         = cat_offsets[col]
        end           = start + cat_vocab_sizes[col]
        logits        = X_recon[:, start:end]                    # [N, n_classes]
        preds[:, col] = np.argmax(logits, axis=1)

    return preds


# ===========================================================================
#  Load Dataset
#  [MODIFIKASI dari dataset_mrmd.py] — tanpa embedding model sama sekali
#
#  Alur:
#    Numerik  : float → normalisasi → MRmD → bin index → one-hot float
#    Kategorikal: string → LabelEncoder → class index → one-hot float
#    Concat   : [num_onehot | cat_onehot] → train_X / test_X → diffusion
# ===========================================================================

def load_dataset(dataname, idx=0, mask_type='MCAR', ratio='30'):
    """
    Load dataset dengan MRmD discretization + one-hot encoding (tanpa embedding).

    Return
    ------
    train_X           : [N_train, total_onehot_dim]  float32
    test_X            : [N_test,  total_onehot_dim]  float32
    ori_train_mask    : mask asli [N_train, total_cols]
    ori_test_mask     : mask asli [N_test,  total_cols]
    train_num         : [N_train, n_num_cols]  float32 ternormalisasi (untuk eval)
    test_num          : [N_test,  n_num_cols]  float32 ternormalisasi
    train_all_idx     : [N_train, n_num_cols + n_cat_cols]  int64
    test_all_idx      : [N_test,  n_num_cols + n_cat_cols]  int64
    extend_train_mask : [N_train, total_onehot_dim]  bool
    extend_test_mask  : [N_test,  total_onehot_dim]  bool
    num_vocab_sizes   : list[n_num_cols] int — jumlah bin per kolom numerik
    cat_vocab_sizes   : list[n_cat_cols] int — jumlah kelas per kolom kategorikal
    num_offsets       : list[n_num_cols] int — posisi awal kolom num di X
    cat_offsets       : list[n_cat_cols] int — posisi awal kolom cat di X
    bin_midpoints     : list[n_num_cols] of np.ndarray (skala norm)
    n_num_cols        : int
    t_mrmd            : float (detik)
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

    cols       = train_df.columns
    n_num_cols = len(num_col_idx)

    # ── Fitur numerik (nilai float asli) ─────────────────────────────────
    if n_num_cols > 0:
        train_num_raw = train_df[cols[num_col_idx]].values.astype(np.float32)
        test_num_raw  = test_df[cols[num_col_idx]].values.astype(np.float32)
    else:
        train_num_raw = np.zeros((len(train_df), 0), dtype=np.float32)
        test_num_raw  = np.zeros((len(test_df),  0), dtype=np.float32)

    # ── Labels untuk MRmD (supervised discretization) ────────────────────
    train_y = train_df[cols[target_col_idx]]
    test_y  = test_df[cols[target_col_idx]]

    label_encoder = LabelEncoder()
    all_labels    = pd.concat([train_y, test_y]).values.ravel()
    label_encoder.fit(all_labels.astype(str))
    train_labels  = label_encoder.transform(train_y.values.ravel().astype(str))
    n_classes     = len(label_encoder.classes_)

    print(f'[Dataset] Detected {n_classes} classes (untuk MRmD supervised discretization)')

    # ── Normalisasi numerik ───────────────────────────────────────────────
    if n_num_cols > 0:
        num_mask_train = train_mask[:, num_col_idx].astype(bool)
        mask_obs       = (~num_mask_train).astype(np.float32)
        mask_sum       = mask_obs.sum(0)
        mask_sum[mask_sum == 0] = 1.0

        num_mean = (train_num_raw * mask_obs).sum(0) / mask_sum
        num_var  = ((train_num_raw - num_mean) ** 2 * mask_obs).sum(0) / mask_sum
        num_std  = np.sqrt(num_var)
        num_std[num_std == 0] = 1.0

        train_num_norm = (train_num_raw - num_mean) / num_std
        test_num_norm  = (test_num_raw  - num_mean) / num_std

        # Disimpan untuk dikembalikan → dipakai get_eval sebagai ground truth float
        train_num = train_num_norm.astype(np.float32)
        test_num  = test_num_norm.astype(np.float32)

        # ── MRmD Discretization (dengan cache) ──────────────────────────
        mrmd_cache_path = f'cache/{dataname}/mrmd.pkl'
        os.makedirs(f'cache/{dataname}', exist_ok=True)

        if os.path.exists(mrmd_cache_path):
            print(f'[MRmD] Cache ditemukan di {mrmd_cache_path}, skip fitting.')
            with open(mrmd_cache_path, 'rb') as f:
                mrmd = pickle.load(f)
            t_mrmd = 0.0
            print(f'[MRmD] n_bins per kolom: {mrmd.n_bins_}')
        else:
            print(f'[MRmD] Cache belum ada. Menjalankan MRmD discretization '
                  f'pada {n_num_cols} kolom numerik ...')
            t_mrmd_start = time.time()
            mrmd = MRmDDiscretizer(val_size=0.125, N_D=50, random_state=42, verbose=False)
            mrmd.fit(train_num_raw, train_labels)
            t_mrmd = time.time() - t_mrmd_start

            with open(mrmd_cache_path, 'wb') as f:
                pickle.dump(mrmd, f)
            print(f'[MRmD] Cache disimpan ke {mrmd_cache_path}')
            print(f'[MRmD] Waktu komputasi diskritisasi: {t_mrmd:.4f}s')

        # Transform: float → bin index (int)
        train_num_bin = mrmd.transform(train_num_raw)   # [N_train, n_num_cols] int64
        test_num_bin  = mrmd.transform(test_num_raw)    # [N_test,  n_num_cols] int64

        # Midpoints dalam skala normalisasi (untuk decode prediksi → nilai kontinu)
        bin_midpoints   = mrmd.get_bin_midpoints(train_num_norm, train_num_bin)
        num_vocab_sizes = list(mrmd.n_bins_)

        print(f'[MRmD] n_bins per kolom: {mrmd.n_bins_}')
        print(f'[MRmD] Total bins: {sum(mrmd.n_bins_)}')

    else:
        train_num       = np.zeros((len(train_df), 0), dtype=np.float32)
        test_num        = np.zeros((len(test_df),  0), dtype=np.float32)
        train_num_bin   = np.zeros((len(train_df), 0), dtype=np.int64)
        test_num_bin    = np.zeros((len(test_df),  0), dtype=np.int64)
        bin_midpoints   = []
        num_vocab_sizes = []
        mrmd            = None
        t_mrmd          = 0.0

    # ── Encoding kolom kategorikal ────────────────────────────────────────
    cat_vocab_sizes    = []
    train_cat_idx_list = []
    test_cat_idx_list  = []

    if len(cat_col_idx) > 0:
        cat_columns = cols[cat_col_idx]
        data_cat    = data_df[cat_columns].astype(str)
        train_cat   = train_df[cat_columns].astype(str)
        test_cat    = test_df[cat_columns].astype(str)

        for col in cat_columns:
            le = LabelEncoder()
            le.fit(data_cat[col])             # fit pada seluruh data agar konsisten
            cat_vocab_sizes.append(len(le.classes_))
            train_cat_idx_list.append(le.transform(train_cat[col]).astype(np.int64))
            test_cat_idx_list.append(le.transform(test_cat[col]).astype(np.int64))

        train_cat_idx = np.stack(train_cat_idx_list, axis=1)   # [N, n_cat_cols]
        test_cat_idx  = np.stack(test_cat_idx_list,  axis=1)
    else:
        train_cat_idx = np.zeros((len(train_df), 0), dtype=np.int64)
        test_cat_idx  = np.zeros((len(test_df),  0), dtype=np.int64)

    # ── Gabungkan index: [num_bin | cat_idx] → one array ─────────────────
    if n_num_cols > 0 and len(cat_col_idx) > 0:
        train_all_idx = np.concatenate([train_num_bin, train_cat_idx], axis=1)
        test_all_idx  = np.concatenate([test_num_bin,  test_cat_idx],  axis=1)
    elif n_num_cols > 0:
        train_all_idx = train_num_bin
        test_all_idx  = test_num_bin
    else:
        train_all_idx = train_cat_idx
        test_all_idx  = test_cat_idx

    # ── Hitung offsets tiap kolom di ruang one-hot ────────────────────────
    # Urutan: [num_col_0, ..., num_col_k, cat_col_0, ..., cat_col_m]
    all_vocab_sizes = num_vocab_sizes + cat_vocab_sizes
    offsets         = [0]
    for v in all_vocab_sizes[:-1]:
        offsets.append(offsets[-1] + v)

    num_offsets     = offsets[:n_num_cols]
    cat_offsets     = offsets[n_num_cols:]
    total_onehot_dim = sum(all_vocab_sizes)

    print(f'[OneHot] num_vocab_sizes={num_vocab_sizes}')
    print(f'[OneHot] cat_vocab_sizes={cat_vocab_sizes}')
    print(f'[OneHot] total_onehot_dim={total_onehot_dim}')

    # ── One-Hot Encoding → train_X / test_X ──────────────────────────────
    train_X = onehot_encode(train_all_idx, all_vocab_sizes)   # [N_train, total_onehot_dim]
    test_X  = onehot_encode(test_all_idx,  all_vocab_sizes)   # [N_test,  total_onehot_dim]

    print(f'[OneHot] train_X shape: {train_X.shape}')
    print(f'[OneHot] test_X  shape: {test_X.shape}')

    # ── Extended mask: [N, n_cols] → [N, total_onehot_dim] ───────────────
    # Kolom ke-j yang missing → semua vocab_sizes[j] dimensi one-hot = True
    train_num_mask = train_mask[:, num_col_idx].astype(bool) if n_num_cols > 0      else np.zeros((len(train_df), 0), dtype=bool)
    train_cat_mask = train_mask[:, cat_col_idx].astype(bool) if len(cat_col_idx) > 0 else np.zeros((len(train_df), 0), dtype=bool)
    test_num_mask  = test_mask[:, num_col_idx].astype(bool)  if n_num_cols > 0      else np.zeros((len(test_df),  0), dtype=bool)
    test_cat_mask  = test_mask[:, cat_col_idx].astype(bool)  if len(cat_col_idx) > 0 else np.zeros((len(test_df),  0), dtype=bool)

    if n_num_cols > 0 and len(cat_col_idx) > 0:
        train_all_mask = np.concatenate([train_num_mask, train_cat_mask], axis=1)
        test_all_mask  = np.concatenate([test_num_mask,  test_cat_mask],  axis=1)
    elif n_num_cols > 0:
        train_all_mask = train_num_mask
        test_all_mask  = test_num_mask
    else:
        train_all_mask = train_cat_mask
        test_all_mask  = test_cat_mask

    all_vocab_arr = np.array(all_vocab_sizes, dtype=int)

    def extend_mask_onehot(mask: np.ndarray, sizes: np.ndarray) -> np.ndarray:
        """Perluas mask [N, n_cols] → [N, total_onehot_dim]."""
        N      = mask.shape[0]
        cum    = np.concatenate(([0], sizes.cumsum()))
        result = np.zeros((N, sizes.sum()), dtype=bool)
        for j in range(len(sizes)):
            col_mask = mask[:, j][:, np.newaxis]
            result[:, cum[j]:cum[j + 1]] = np.tile(col_mask, sizes[j])
        return result

    extend_train_mask = extend_mask_onehot(train_all_mask, all_vocab_arr)
    extend_test_mask  = extend_mask_onehot(test_all_mask,  all_vocab_arr)

    return (train_X, test_X,
            train_mask, test_mask,
            train_num, test_num,
            train_all_idx, test_all_idx,
            extend_train_mask, extend_test_mask,
            num_vocab_sizes,    # list[n_num_cols] — jumlah bin per kolom numerik
            cat_vocab_sizes,    # list[n_cat_cols]  — jumlah kelas per kolom kategorikal
            num_offsets,        # list[n_num_cols]  — posisi awal kolom num di X
            cat_offsets,        # list[n_cat_cols]  — posisi awal kolom cat di X
            bin_midpoints,      # list[n_num_cols] of np.ndarray (skala norm)
            n_num_cols,
            t_mrmd)


# ===========================================================================
#  Utilities
# ===========================================================================

def mean_std(data, mask):
    """
    Hitung mean & std hanya dari observed entries (mask=False → observed).
    [TIDAK BERUBAH]
    """
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

def get_eval(dataname, X_recon, X_true, truth_all_idx,
             mask, device='cpu', oos=False,
             num_vocab_sizes=None, cat_vocab_sizes=None,
             num_offsets=None, cat_offsets=None,
             bin_midpoints=None, n_num_cols=0,
             num_true_norm=None):
    """
    Hitung MAE, RMSE (numerik) dan Accuracy (kategorikal).

    [MODIFIKASI dari dataset_mrmd.py — tanpa embedding model]

    Decode langsung dari rekonstruksi one-hot (X_recon):
      - Numerik  : decode_num_from_onehot → softmax weighted sum → nilai kontinu
      - Kategorikal: decode_cat_from_onehot → argmax → predicted class index

    Parameter
    ---------
    X_recon / X_true : [N, total_onehot_dim]
    truth_all_idx    : [N, n_num_cols + n_cat_cols]  int
    num_vocab_sizes  : list[n_num_cols]  — jumlah bin per kolom numerik
    cat_vocab_sizes  : list[n_cat_cols]  — jumlah kelas per kolom kategorikal
    num_offsets      : list[n_num_cols]  — posisi awal kolom num di X_recon
    cat_offsets      : list[n_cat_cols]  — posisi awal kolom cat di X_recon
    bin_midpoints    : list[n_num_cols] of np.ndarray (skala norm)
    n_num_cols       : int
    num_true_norm    : [N, n_num_cols] float — ground truth nilai float ternormalisasi
    """
    info_path = f'datasets/Info/{dataname}.json'
    with open(info_path, 'r') as f:
        info = json.load(f)

    num_col_idx = info['num_col_idx']
    cat_col_idx = info['cat_col_idx']

    num_mask = mask[:, num_col_idx].astype(bool) if len(num_col_idx) > 0 else None
    cat_mask = mask[:, cat_col_idx].astype(bool) if len(cat_col_idx) > 0 else None

    # ── Special case: news dataset ────────────────────────────────────────
    if dataname == 'news' and oos:
        drop = 6265
        if num_mask is not None:
            num_mask = np.delete(num_mask, drop, axis=0)
        if cat_mask is not None:
            cat_mask = np.delete(cat_mask, drop, axis=0)
        if truth_all_idx is not None:
            truth_all_idx = np.delete(truth_all_idx, drop, axis=0)
        if num_true_norm is not None:
            num_true_norm = np.delete(num_true_norm, drop, axis=0)
        X_recon = np.delete(X_recon, drop, axis=0)
        X_true  = np.delete(X_true,  drop, axis=0)

    # ── Numerik: MAE & RMSE ───────────────────────────────────────────────
    mae  = np.nan
    rmse = np.nan

    if (n_num_cols > 0
            and num_mask is not None
            and bin_midpoints is not None
            and num_vocab_sizes is not None
            and num_offsets is not None):

        # Decode: one-hot logits → softmax weighted sum midpoints → nilai kontinu
        num_pred_norm = decode_num_from_onehot(
            X_recon, bin_midpoints, num_vocab_sizes, num_offsets
        )  # [N, n_num_cols]

        # Ground truth: nilai float asli ternormalisasi
        if num_true_norm is not None:
            gt_norm = num_true_norm
        else:
            # Fallback: lookup midpoint dari true bin index
            N = X_recon.shape[0]
            gt_norm = np.zeros((N, n_num_cols), dtype=np.float32)
            for col in range(n_num_cols):
                mids     = bin_midpoints[col]
                true_bin = truth_all_idx[:, col].astype(int)
                true_bin = np.clip(true_bin, 0, len(mids) - 1)
                gt_norm[:, col] = mids[true_bin]

        diff = num_pred_norm[num_mask] - gt_norm[num_mask]
        mae  = float(np.abs(diff).mean())
        rmse = float(np.sqrt((diff ** 2).mean()))

    # ── Kategorikal: Akurasi ──────────────────────────────────────────────
    acc = np.nan

    if (truth_all_idx is not None
            and len(cat_col_idx) > 0
            and cat_vocab_sizes is not None
            and cat_offsets is not None
            and cat_mask is not None):

        # Decode: one-hot logits → argmax → predicted class index
        pred_cat_idx = decode_cat_from_onehot(
            X_recon, cat_vocab_sizes, cat_offsets
        )  # [N, n_cat_cols]

        n_cat_cols    = len(cat_col_idx)
        correct_total = 0
        total_missing = 0

        for j in range(n_cat_cols):
            rows_miss = cat_mask[:, j]
            if rows_miss.sum() == 0:
                continue

            col_offset = n_num_cols + j        # offset di truth_all_idx
            pred_j     = pred_cat_idx[:, j]
            true_j     = truth_all_idx[:, col_offset].astype(int)

            correct = (pred_j[rows_miss] == true_j[rows_miss]).sum()
            correct_total += int(correct)
            total_missing += int(rows_miss.sum())

        if total_missing > 0:
            acc = correct_total / total_missing

    return mae, rmse, acc