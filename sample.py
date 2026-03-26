# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained DiT.
Optional: Track block-wise cross-timestep cosine similarity during sampling.
Output: cross_sim_4d [T, L, T, L] and cross_sim_2d [T*L, T*L]
"""
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
from models import DiT_models
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


def compute_cross_timestep_similarity(block_tokens_per_timestep, tracked_steps):
    """
    Compute cross-timestep block cosine similarity.

    From the old setting [T, L, L] (within-timestep only) to the new setting [T, L, T, L]
    which captures similarity between any pair (timestep_k, block_a) and (timestep_k', block_b).

    Args:
        block_tokens_per_timestep: dict {timestep -> list of L tensors (B, N, D)}
        tracked_steps: ordered list of tracked timesteps

    Returns:
        cross_sim_4d: numpy array [T, L, T, L]
        cross_sim_2d: numpy array [T*L, T*L]  (flattened, idx(k,a) = k*L + a)
        tracked: list of timesteps actually present in block_tokens_per_timestep (ordered)
    """
    # Keep only timesteps that were actually collected, in order
    tracked = [t for t in tracked_steps if t in block_tokens_per_timestep]
    T = len(tracked)

    # Stack into H: [T, L, B, N, D]
    H = torch.stack(
        [torch.stack(block_tokens_per_timestep[t], dim=0) for t in tracked],
        dim=0
    )   # [T, L, B, N, D]
    T, L, B, N, D = H.shape
    BN = B * N

    # Normalize along hidden dimension; free H immediately after
    h_norm = H / (torch.linalg.norm(H, dim=-1, keepdim=True) + 1e-8)
    del H
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Reshape to [T, L, B*N*D] for memory-efficient pairwise matmul.
    # No new allocation: view shares storage with h_norm.
    h_flat = h_norm.view(T, L, BN * D)   # [T, L, BN*D]

    # Compute cross_sim_2d [(T*L) x (T*L)] iteratively over timestep pairs.
    # For pair (k1, k2):
    #   sim[k1, l1, k2, l2] = (1/BN) * sum_{b,n,d} h[k1,l1,b,n,d] * h[k2,l2,b,n,d]
    #                        = ([L, BN*D] @ [BN*D, L])[l1, l2] / BN
    # Each matmul: [L, BN*D] @ [BN*D, L] -> [L, L]  (no large intermediate)
    cross_sim_2d = torch.zeros(T * L, T * L, dtype=torch.float32, device='cpu')
    for k1 in tqdm(range(T), desc="Cross-sim (rows)", leave=False):
        x1 = h_flat[k1].reshape(L, BN * D)   # view [L, BN*D]
        for k2 in range(T):
            x2 = h_flat[k2].reshape(L, BN * D)
            blk = (x1 @ x2.T).to(torch.float32) / BN   # [L, L]
            cross_sim_2d[k1*L:(k1+1)*L, k2*L:(k2+1)*L] = blk.cpu()

    del h_flat, h_norm
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    cross_sim_4d = cross_sim_2d.numpy().reshape(T, L, T, L)
    return cross_sim_4d, cross_sim_2d.numpy(), tracked


class ModelWrapper:
    """
    Wrapper that collects per-timestep block tokens during sampling.
    After p_sample_loop completes, call compute_cross_timestep_similarity
    on block_tokens_per_timestep.
    """
    def __init__(self, model, model_kwargs, track_timesteps):
        self.model = model
        self.model_kwargs = model_kwargs
        self.track_timesteps = set(track_timesteps)
        self.block_tokens_per_timestep = {}   # {t_val -> list of L tensors (B, N, D)}

    def __call__(self, x, t):
        t_val = t[0].item() if isinstance(t, torch.Tensor) else t

        if t_val in self.track_timesteps:
            out, block_tokens = self.model.forward_with_cfg(
                x, t, self.model_kwargs['y'], self.model_kwargs['cfg_scale'],
                return_block_tokens=True
            )
            # Detach to avoid holding computation graph in memory
            self.block_tokens_per_timestep[t_val] = [bt.detach() for bt in block_tokens]
            print(f"  [Timestep {t_val}] Collected block tokens (L={len(block_tokens)})")
            return out
        else:
            return self.model.forward_with_cfg(
                x, t, self.model_kwargs['y'], self.model_kwargs['cfg_scale']
            )


def sample_with_similarity_tracking(model, diffusion, z, model_kwargs, device, track_timesteps):
    """
    Custom sampling loop with cross-timestep block cosine similarity tracking.

    Args:
        model: DiT model
        diffusion: Diffusion object
        z: (B, C, H, W) initial noise
        model_kwargs: dict with 'y' (labels) and 'cfg_scale'
        device: torch device
        track_timesteps: list of timestep indices to track

    Returns:
        samples: (B, C, H, W) generated samples
        cross_sim_4d: numpy array [T, L, T, L]
        cross_sim_2d: numpy array [T*L, T*L]
        tracked: list of timesteps in order
    """
    wrapper = ModelWrapper(model, model_kwargs, track_timesteps)

    samples = diffusion.p_sample_loop(
        wrapper,
        z.shape,
        noise=z,
        clip_denoised=False,
        model_kwargs={},   # kwargs already embedded in wrapper
        device=device,
        progress=True
    )

    print(f"\nComputing cross-timestep similarity for {len(wrapper.block_tokens_per_timestep)} timesteps...")
    cross_sim_4d, cross_sim_2d, tracked = compute_cross_timestep_similarity(
        wrapper.block_tokens_per_timestep, track_timesteps
    )
    print(f"  cross_sim_4d shape: {cross_sim_4d.shape}  (T={len(tracked)}, L={cross_sim_4d.shape[1]})")
    print(f"  cross_sim_2d shape: {cross_sim_2d.shape}")

    return samples, cross_sim_4d, cross_sim_2d, tracked


def visualize_cross_timestep_similarity(cross_sim_4d, cross_sim_2d, tracked_timesteps, save_prefix=None):
    """
    Produce two figures:
      Figure 1 – Grid T×T of L×L heatmaps.
      Figure 2 – Large flattened heatmap [T*L, T*L].

    Args:
        cross_sim_4d: numpy array [T, L, T, L]
        cross_sim_2d: numpy array [T*L, T*L]
        tracked_timesteps: list of tracked timesteps (length T)
        save_prefix: if given, save figures as <save_prefix>_grid.png and <save_prefix>_flat.png
    """
    T, L, _, _ = cross_sim_4d.shape

    # ── Figure 1: Grid T×T of L×L sub-heatmaps ────────────────────────────
    cell_size = max(1.5, 80 / max(T, L))   # adaptive cell size
    fig, axes = plt.subplots(T, T, figsize=(cell_size * T, cell_size * T))
    if T == 1:
        axes = np.array([[axes]])
    elif axes.ndim == 1:
        axes = axes[np.newaxis, :]

    for i in range(T):
        for j in range(T):
            ax = axes[i, j]
            sub = cross_sim_4d[i, :, j, :]   # [L, L]
            sns.heatmap(sub, ax=ax, cmap='RdYlGn', vmin=-1, vmax=1,
                        xticklabels=False, yticklabels=False, cbar=False)
            if i == 0:
                ax.set_title(f't={tracked_timesteps[j]}', fontsize=7)
            if j == 0:
                ax.set_ylabel(f't={tracked_timesteps[i]}', fontsize=7)

    fig.suptitle(
        f'Cross-Timestep Block Similarity Grid  (T={T}, L={L})\n'
        f'Row = source timestep, Col = target timestep, Cell = L×L block similarity',
        fontsize=10
    )
    plt.tight_layout()
    if save_prefix:
        fig.savefig(f'{save_prefix}_grid.png', dpi=100, bbox_inches='tight')
        print(f"  Saved {save_prefix}_grid.png")
    plt.show()

    # ── Figure 2: Large flat heatmap [T*L, T*L] ───────────────────────────
    fig2, ax2 = plt.subplots(figsize=(12, 10))
    sns.heatmap(cross_sim_2d, ax=ax2, cmap='RdYlGn', vmin=-1, vmax=1,
                xticklabels=False, yticklabels=False,
                cbar_kws={'label': 'Cosine Similarity'})
    # Draw white lines to mark timestep boundaries
    for k in range(1, T):
        ax2.axhline(y=k * L, color='white', linewidth=1.0)
        ax2.axvline(x=k * L, color='white', linewidth=1.0)
    # Tick labels at timestep centres
    centres = [k * L + L // 2 for k in range(T)]
    ax2.set_xticks(centres)
    ax2.set_xticklabels([f't={tracked_timesteps[k]}' for k in range(T)], fontsize=8)
    ax2.set_yticks(centres)
    ax2.set_yticklabels([f't={tracked_timesteps[k]}' for k in range(T)], fontsize=8)
    ax2.set_title(
        f'Cross-Timestep Block Similarity  [{T*L} × {T*L}]  '
        f'(each block = {L}×{L} within-pair similarity)',
        fontsize=11
    )
    plt.tight_layout()
    if save_prefix:
        fig2.savefig(f'{save_prefix}_flat.png', dpi=100, bbox_inches='tight')
        print(f"  Saved {save_prefix}_flat.png")
    plt.show()


##############################################################################
#  PCA Analysis  (Token-wise, Image-wise, Timestep-trajectory)
#  Follows: dit_pca_measurement_spec.txt
##############################################################################

def _run_pca(X):
    """
    PCA on centered matrix X [M, F].
    Preprocessing: center by feature (no standardization).
    Backend: SVD (never explicitly forms covariance matrix).

    Returns dict with all metrics specified in section 2.4 of the spec:
        S         : singular values
        lambdas   : eigenvalues  = S^2 / (M-1)
        evr       : explained variance ratio  r_i
        cum_evr   : cumulative explained variance  c_k
        PR        : participation ratio  = (sum lambda)^2 / sum(lambda^2)
        top1      : r_1   (top1_ratio)
        top5      : c_5   (top5_cum)
        top10     : c_10  (top10_cum)
        n80/90/95 : components needed to reach 80/90/95% variance
        components: right singular vectors Vt  [K, F]  (principal axes)
        mean      : feature mean used for centering
    """
    M = X.shape[0]
    if M < 2:
        return None
    mean = X.mean(axis=0, keepdims=True)
    xc = X - mean
    _, S, Vt = np.linalg.svd(xc, full_matrices=False)
    lambdas = S ** 2 / max(M - 1, 1)
    total   = lambdas.sum() + 1e-12
    evr     = lambdas / total
    cum_evr = np.cumsum(evr)
    PR      = float(total ** 2 / (lambdas ** 2).sum())
    K       = len(evr)
    return dict(
        S=S, lambdas=lambdas,
        evr=evr, cum_evr=cum_evr,
        PR=PR,
        top1 =float(evr[0]),
        top5 =float(cum_evr[min(4,  K-1)]),
        top10=float(cum_evr[min(9,  K-1)]),
        n80=int(np.searchsorted(cum_evr, 0.80)) + 1,
        n90=int(np.searchsorted(cum_evr, 0.90)) + 1,
        n95=int(np.searchsorted(cum_evr, 0.95)) + 1,
        components=Vt,          # [K, F]  principal axes
        mean=mean.squeeze(0),   # [F]
    )


def compute_token_pca(H):
    """
    PCA type A – Token-wise (spec section 3).
    H: numpy [B, N, D]
    Reshape to [B*N, D] then run PCA in channel space D.
    Returns PCA metrics dict.
    """
    B, N, D = H.shape
    X = H.reshape(B * N, D).astype(np.float32)
    result = _run_pca(X)
    if result is not None:
        # optional: keep 2D projection onto PC1-PC2 for scatter (spec 3.6.C)
        n_pc = min(2, len(result['evr']))
        result['proj2d'] = (X - result['mean']) @ result['components'][:n_pc].T  # [B*N, 2]
        result['proj2d_shape'] = (B, N)   # original shape before flatten
    return result


def compute_image_pca(H):
    """
    PCA type B – Image-wise (spec section 4).
    H: numpy [B, N, D]
    Pool tokens by mean -> [B, D], then run PCA.
    Returns PCA metrics dict + proj_img [B, 2].
    """
    Z = H.mean(axis=1).astype(np.float32)   # [B, D]
    result = _run_pca(Z)
    if result is not None:
        n_pc = min(2, len(result['evr']))
        result['proj_img'] = (Z - result['mean']) @ result['components'][:n_pc].T  # [B, 2]
    return result


def compute_trajectory_pca(H_over_t, pooling='mean', shared_basis=True):
    """
    PCA type C – Timestep-trajectory (spec section 5).
    H_over_t : list of T numpy [B, N, D]
    pooling  : 'mean' (mean over tokens, spec default)
    shared_basis : True = fit one PCA on all (B*T) pooled vectors (spec recommended default)

    Returns:
        pca_result  : PCA metrics on [B*T, D]
        traj_proj   : numpy [B, T, 3]  – projected trajectory (first 3 PCs)
        motion_stats: dict with per-image motion statistics (spec section 5.6)
    """
    T = len(H_over_t)
    B = H_over_t[0].shape[0]

    # Pool tokens -> [B, D] per timestep  (spec 5.3)
    z_list = [H.mean(axis=1).astype(np.float32) for H in H_over_t]   # T x [B, D]

    # Shared basis: fit PCA on Z_all [B*T, D]  (spec 5.5)
    z_all = np.concatenate(z_list, axis=0)   # [B*T, D]
    pca_result = _run_pca(z_all)
    if pca_result is None:
        return None, None, None

    # Project each timestep's Z using the shared mean and top-3 PCs  (spec 5.5)
    n_pc  = min(3, len(pca_result['evr']))
    vt_top = pca_result['components'][:n_pc]   # [n_pc, D]
    mean_vec = pca_result['mean']

    proj_list = [(z - mean_vec) @ vt_top.T for z in z_list]   # T x [B, n_pc]
    traj_proj = np.stack(proj_list, axis=1)                    # [B, T, n_pc]

    # Motion statistics per image  (spec 5.6)
    steps      = np.diff(traj_proj, axis=1)                        # [B, T-1, n_pc]
    step_dists = np.linalg.norm(steps, axis=-1)                    # [B, T-1]
    path_length    = step_dists.sum(axis=1)                        # [B]
    avg_step       = step_dists.mean(axis=1)                       # [B]
    start_end_dist = np.linalg.norm(
        traj_proj[:, -1, :] - traj_proj[:, 0, :], axis=-1         # [B]
    )
    # Curvature proxy: mean turning angle between consecutive segments
    if T > 2:
        v1 = steps[:, :-1, :]                                      # [B, T-2, n_pc]
        v2 = steps[:, 1:,  :]
        cos_angles = (
            (v1 * v2).sum(axis=-1)
            / (np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1) + 1e-8)
        )
        curvature = np.arccos(np.clip(cos_angles, -1, 1)).mean(axis=1)  # [B] radians
    else:
        curvature = np.zeros(B)

    motion_stats = dict(
        path_length=path_length,        # [B] total Euclidean path in PCA space
        start_end_dist=start_end_dist,  # [B] direct distance first→last timestep
        avg_step=avg_step,              # [B] mean step size
        curvature=curvature,            # [B] mean turning angle (radians)
    )

    return pca_result, traj_proj, motion_stats


def run_pca_analysis(block_tokens_per_timestep, tracked_steps):
    """
    Run all three PCA types on collected block tokens.

    Returns dict:
        tracked : list of tracked timesteps (ordered)
        L, T    : num blocks, num tracked timesteps
        token   : list[L] of list[T] of metric dicts  (type A)
        image   : list[L] of list[T] of metric dicts  (type B)
        traj    : list[L] of (pca_result, traj_proj[B,T,3], motion_stats)  (type C)
    """
    tracked = [t for t in tracked_steps if t in block_tokens_per_timestep]
    T = len(tracked)
    if T == 0:
        return None

    block_np = {ki: [bt.cpu().numpy() for bt in block_tokens_per_timestep[t]]
                for ki, t in enumerate(tracked)}
    L = len(block_np[0])

    token_metrics = [[None] * T for _ in range(L)]
    image_metrics = [[None] * T for _ in range(L)]

    print("Running token-wise and image-wise PCA (type A + B)...")
    for li in tqdm(range(L), desc="Blocks (token+image)"):
        for ki in range(T):
            H = block_np[ki][li]
            token_metrics[li][ki] = compute_token_pca(H)
            image_metrics[li][ki] = compute_image_pca(H)

    traj_results = []
    print("Running timestep-trajectory PCA (type C)...")
    for li in tqdm(range(L), desc="Blocks (trajectory)"):
        H_over_t = [block_np[ki][li] for ki in range(T)]
        traj_results.append(compute_trajectory_pca(H_over_t))

    return dict(tracked=tracked, L=L, T=T,
                token=token_metrics, image=image_metrics, traj=traj_results)


def _ext(metrics_2d, key, L, T):
    """Extract a [L, T] metric array from a list-of-lists of dicts."""
    arr = np.zeros((L, T))
    for li in range(L):
        for ki in range(T):
            m = metrics_2d[li][ki]
            if m:
                arr[li, ki] = m[key]
    return arr


def save_pca_metrics(pca_results):
    """
    Save PCA metrics to .npz files.
    Artifact names follow spec section 8:
        token_pca_metrics.npz
        image_pca_metrics.npz
        trajectory_pca_metrics.npz
    """
    L, T    = pca_results['L'], pca_results['T']
    tracked = np.array(pca_results['tracked'])

    # Type A: token-wise  →  token_pca_metrics.npz
    np.savez('token_pca_metrics.npz',
             tracked=tracked,
             top1 =_ext(pca_results['token'], 'top1',  L, T),
             top5 =_ext(pca_results['token'], 'top5',  L, T),
             top10=_ext(pca_results['token'], 'top10', L, T),
             PR   =_ext(pca_results['token'], 'PR',    L, T),
             n80  =_ext(pca_results['token'], 'n80',   L, T),
             n90  =_ext(pca_results['token'], 'n90',   L, T),
             n95  =_ext(pca_results['token'], 'n95',   L, T))

    # Type B: image-wise  →  image_pca_metrics.npz
    np.savez('image_pca_metrics.npz',
             tracked=tracked,
             top1 =_ext(pca_results['image'], 'top1',  L, T),
             top5 =_ext(pca_results['image'], 'top5',  L, T),
             top10=_ext(pca_results['image'], 'top10', L, T),
             PR   =_ext(pca_results['image'], 'PR',    L, T),
             n80  =_ext(pca_results['image'], 'n80',   L, T),
             n90  =_ext(pca_results['image'], 'n90',   L, T),
             n95  =_ext(pca_results['image'], 'n95',   L, T))

    # Type C: trajectory  →  trajectory_pca_metrics.npz
    traj_PR    = np.array([r[0]['PR']   if r[0] else 0.0 for r in pca_results['traj']])
    traj_top1  = np.array([r[0]['top1'] if r[0] else 0.0 for r in pca_results['traj']])
    path_lens   = np.array([r[2]['path_length'].mean()    if r[2] else 0.0
                            for r in pca_results['traj']])
    se_dists    = np.array([r[2]['start_end_dist'].mean() if r[2] else 0.0
                            for r in pca_results['traj']])
    avg_steps   = np.array([r[2]['avg_step'].mean()       if r[2] else 0.0
                            for r in pca_results['traj']])
    np.savez('trajectory_pca_metrics.npz',
             tracked=tracked,
             PR=traj_PR, top1=traj_top1,
             path_length_mean=path_lens,
             start_end_dist_mean=se_dists,
             avg_step_mean=avg_steps)

    print("Saved: token_pca_metrics.npz, image_pca_metrics.npz, trajectory_pca_metrics.npz")


def _heatmap_fig(mat, title, xlabel, ylabel, xlabels, ylabels, fname):
    """Helper: one heatmap figure saved to fname."""
    fig, ax = plt.subplots(figsize=(max(6, len(xlabels) * 0.8), max(5, len(ylabels) * 0.25)))
    sns.heatmap(mat, ax=ax, cmap='viridis', annot=False,
                xticklabels=xlabels, yticklabels=ylabels)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    fig.savefig(fname, dpi=100, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    print(f"  Saved {fname}")


def visualize_pca_results(pca_results):
    """
    Produce all minimum figures required by spec section 7.5:

    Spec-named output files:
        token_pca_heatmap_top1.png   – token top1 variance [L × T]
        token_pca_heatmap_pr.png     – token PR [L × T]
        image_pca_heatmap_top1.png   – image top1 variance [L × T]
        image_pca_heatmap_pr.png     – image PR [L × T]
        trajectory_block_{l}.png     – per-block trajectory in PC1-PC2 (selected blocks)
        pca_scree.png                – scree plots for selected (l, t) pairs

    Additional figures (spec sections 3.6, 4.6, 5.7):
        pca_motion_heatmap.png       – avg path length / start-end dist per block (spec 5.7.C)
        pca_token_scatter.png        – 2D token scatter PC1-PC2 for selected (l,t) (spec 3.6.C)
        image_pca_scatter.png        – 2D image scatter PC1-PC2 for selected (l,t) (spec 4.6.B)
    """
    tracked = pca_results['tracked']
    L, T    = pca_results['L'], pca_results['T']
    xlabels = [f't={t}' for t in tracked]
    blk_step = max(1, L // 10)
    ylabels  = [f'L{li}' if li % blk_step == 0 else '' for li in range(L)]

    # ─────────────────────────────────────────────────────────────────────────
    # 1. token_pca_heatmap_top1.png  (spec fig 1)
    # ─────────────────────────────────────────────────────────────────────────
    _heatmap_fig(_ext(pca_results['token'], 'top1', L, T),
                 'Token-wise PCA: top-1 explained variance ratio',
                 'Timestep', 'Block', xlabels, ylabels,
                 'token_pca_heatmap_top1.png')

    # ─────────────────────────────────────────────────────────────────────────
    # 2. token_pca_heatmap_pr.png  (spec fig 2)
    # ─────────────────────────────────────────────────────────────────────────
    _heatmap_fig(_ext(pca_results['token'], 'PR', L, T),
                 'Token-wise PCA: Participation Ratio',
                 'Timestep', 'Block', xlabels, ylabels,
                 'token_pca_heatmap_pr.png')

    # ─────────────────────────────────────────────────────────────────────────
    # 3. image_pca_heatmap_top1.png  (spec fig 3)
    # ─────────────────────────────────────────────────────────────────────────
    _heatmap_fig(_ext(pca_results['image'], 'top1', L, T),
                 'Image-wise PCA: top-1 explained variance ratio',
                 'Timestep', 'Block', xlabels, ylabels,
                 'image_pca_heatmap_top1.png')

    # ─────────────────────────────────────────────────────────────────────────
    # 4. image_pca_heatmap_pr.png  (spec fig 4)
    # ─────────────────────────────────────────────────────────────────────────
    _heatmap_fig(_ext(pca_results['image'], 'PR', L, T),
                 'Image-wise PCA: Participation Ratio',
                 'Timestep', 'Block', xlabels, ylabels,
                 'image_pca_heatmap_pr.png')

    # ─────────────────────────────────────────────────────────────────────────
    # 5. trajectory_block_{l}.png  (spec fig 5) – one figure per selected block
    # ─────────────────────────────────────────────────────────────────────────
    sel_blocks = sorted({0, L // 4, L // 2, 3 * L // 4, L - 1})
    cmap_ts = plt.cm.plasma
    for li in sel_blocks:
        traj_m, traj_p, _ = pca_results['traj'][li]
        if traj_p is None:
            continue
        B_vis = traj_p.shape[0]
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Left: individual trajectories (spec 5.7.A)
        ax = axes[0]
        for b in range(B_vis):
            xs, ys = traj_p[b, :, 0], traj_p[b, :, 1]
            for ki in range(T - 1):
                col = cmap_ts(ki / max(T - 1, 1))
                ax.plot(xs[ki:ki+2], ys[ki:ki+2], '-o',
                        color=col, markersize=5, linewidth=1.5, alpha=0.7)
        sm = plt.cm.ScalarMappable(cmap=cmap_ts,
                                   norm=plt.Normalize(vmin=tracked[0], vmax=tracked[-1]))
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label='Timestep', shrink=0.8)
        ax.set_title(f'Block {li}: individual trajectories', fontsize=11)
        ax.set_xlabel('PC1'); ax.set_ylabel('PC2')

        # Right: average trajectory (spec 5.7.B)
        ax2 = axes[1]
        avg_traj = traj_p.mean(axis=0)   # [T, n_pc]
        for ki in range(T - 1):
            col = cmap_ts(ki / max(T - 1, 1))
            ax2.plot(avg_traj[ki:ki+2, 0], avg_traj[ki:ki+2, 1], '-o',
                     color=col, markersize=7, linewidth=2)
        # Label timesteps on average trajectory
        for ki, t in enumerate(tracked):
            ax2.annotate(f't={t}', (avg_traj[ki, 0], avg_traj[ki, 1]),
                         fontsize=7, ha='center', va='bottom')
        ax2.set_title(f'Block {li}: average trajectory', fontsize=11)
        ax2.set_xlabel('PC1'); ax2.set_ylabel('PC2')

        evr_str = f"PR={traj_m['PR']:.1f}  top1={traj_m['top1']:.2f}"
        fig.suptitle(f'Trajectory PCA – Block {li}  ({evr_str})', fontsize=12)
        plt.tight_layout()
        fname = f'trajectory_block_{li}.png'
        fig.savefig(fname, dpi=100, bbox_inches='tight')
        plt.show()
        plt.close(fig)
        print(f"  Saved {fname}")

    # ─────────────────────────────────────────────────────────────────────────
    # 6. pca_scree.png  (spec fig 6) – selected (block, timestep) pairs
    # ─────────────────────────────────────────────────────────────────────────
    sel_pairs = [(0, 0), (L // 4, T // 4), (L // 2, T // 2),
                 (3 * L // 4, 3 * T // 4), (L - 1, T - 1)]
    sel_pairs = list(dict.fromkeys(sel_pairs))  # deduplicate preserving order
    fig, axes = plt.subplots(1, len(sel_pairs), figsize=(5 * len(sel_pairs), 4))
    if len(sel_pairs) == 1:
        axes = [axes]
    for ax, (li, ki) in zip(axes, sel_pairs):
        m = pca_results['token'][li][ki]
        if m is None:
            continue
        n_show = min(30, len(m['evr']))
        ax.bar(range(1, n_show + 1), m['evr'][:n_show] * 100, alpha=0.75, color='steelblue')
        ax.plot(range(1, n_show + 1), m['cum_evr'][:n_show] * 100,
                'r-o', markersize=3, linewidth=1.2, label='Cumulative')
        ax.axhline(80, color='orange', linestyle='--', linewidth=0.8, label='80%')
        ax.axhline(95, color='red',    linestyle='--', linewidth=0.8, label='95%')
        ax.set_xlabel('Component')
        ax.set_ylabel('Explained Var (%)')
        ax.set_title(f'L{li}, t={tracked[ki]}\nPR={m["PR"]:.1f}  top1={m["top1"]*100:.1f}%',
                     fontsize=9)
        ax.legend(fontsize=7)
    fig.suptitle('Scree Plots – Token-wise PCA (selected block × timestep)', fontsize=11)
    plt.tight_layout()
    fig.savefig('pca_scree.png', dpi=100, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    print("  Saved pca_scree.png")

    # ─────────────────────────────────────────────────────────────────────────
    # 7. pca_motion_heatmap.png  (spec 5.7.C) – motion statistics per block
    # ─────────────────────────────────────────────────────────────────────────
    path_lens = np.array([r[2]['path_length'].mean()    if r[2] else 0.0
                          for r in pca_results['traj']])
    se_dists  = np.array([r[2]['start_end_dist'].mean() if r[2] else 0.0
                          for r in pca_results['traj']])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    bx = np.arange(L)
    ax1.bar(bx, path_lens, alpha=0.8, color='steelblue')
    ax1.set_xlabel('Block'); ax1.set_ylabel('Mean Path Length (PCA units)')
    ax1.set_title('Avg Total Path Length per Block', fontsize=11)
    ax1.set_xticks(range(0, L, blk_step))
    ax2.bar(bx, se_dists, alpha=0.8, color='tomato')
    ax2.set_xlabel('Block'); ax2.set_ylabel('Mean Start-End Distance (PCA units)')
    ax2.set_title('Avg Start→End Distance per Block', fontsize=11)
    ax2.set_xticks(range(0, L, blk_step))
    fig.suptitle('Trajectory Motion Statistics per Block', fontsize=12)
    plt.tight_layout()
    fig.savefig('pca_motion_heatmap.png', dpi=100, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    print("  Saved pca_motion_heatmap.png")

    # ─────────────────────────────────────────────────────────────────────────
    # 8. pca_token_scatter.png  (spec 3.6.C) – 2D token scatter for selected (l,t)
    # ─────────────────────────────────────────────────────────────────────────
    sel_lt = [(0, 0), (L // 2, T // 2), (L - 1, T - 1)]
    fig, axes = plt.subplots(1, len(sel_lt), figsize=(5 * len(sel_lt), 5))
    if len(sel_lt) == 1:
        axes = [axes]
    for ax, (li, ki) in zip(axes, sel_lt):
        m = pca_results['token'][li][ki]
        if m is None or 'proj2d' not in m:
            continue
        p2 = m['proj2d']             # [B*N, 2]
        B_orig, N_orig = m['proj2d_shape']
        # Color by spatial (patch) index to reveal spatial structure
        patch_idx = np.tile(np.arange(N_orig), B_orig)
        sc = ax.scatter(p2[:, 0], p2[:, 1], c=patch_idx, cmap='hsv',
                        s=3, alpha=0.4)
        plt.colorbar(sc, ax=ax, label='Patch index', shrink=0.8)
        ax.set_title(f'Token PCA L{li}, t={tracked[ki]}\nPC1 vs PC2', fontsize=10)
        ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
    fig.suptitle('Token-wise PCA: 2D scatter (color = patch spatial index)', fontsize=11)
    plt.tight_layout()
    fig.savefig('pca_token_scatter.png', dpi=100, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    print("  Saved pca_token_scatter.png")

    # ─────────────────────────────────────────────────────────────────────────
    # 9. image_pca_scatter.png  (spec 4.6.B) – 2D image scatter for selected (l,t)
    # ─────────────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, len(sel_lt), figsize=(5 * len(sel_lt), 5))
    if len(sel_lt) == 1:
        axes = [axes]
    for ax, (li, ki) in zip(axes, sel_lt):
        m = pca_results['image'][li][ki]
        if m is None or 'proj_img' not in m:
            continue
        p2 = m['proj_img']           # [B, 2]
        ax.scatter(p2[:, 0], p2[:, 1], s=80, alpha=0.9, zorder=3)
        for b in range(p2.shape[0]):
            ax.annotate(str(b), (p2[b, 0], p2[b, 1]), fontsize=8, ha='center')
        ax.set_title(f'Image PCA L{li}, t={tracked[ki]}\nPC1 vs PC2', fontsize=10)
        ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
    fig.suptitle('Image-wise PCA: 2D scatter (each point = one image)', fontsize=11)
    plt.tight_layout()
    fig.savefig('image_pca_scatter.png', dpi=100, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    print("  Saved image_pca_scatter.png")


##############################################################################
#  Layer-wise Cross-Noise Patch-wise Cosine Similarity
#  Spec: Layer-wise Cross-Noise Patch-wise Cosine
##############################################################################

def compute_layer_cross_similarity(cross_sim_4d):
    """
    Derive layer-major cross-noise similarity from existing cross_sim_4d [T, L, T, L].

    The underlying similarity values are identical to cross_sim_4d — this is a
    pure axis-reorder + reshape.

    Indexing:
        cross_sim_4d    [T, L, T, L]  – timestep-major (C[t, l, t', l'])
        layer_cross_4d  [L, T, L, T]  – layer-major    (S[l, t, l', t'])

    Layer-major flatten (spec): row/col index = l * T + t
        → block (l, l') of the [L*T, L*T] matrix is T×T
        → diagonal block (l, l) shows how layer l evolves across noise levels

    Returns:
        layer_cross_4d : numpy [L, T, L, T]
        layer_cross_2d : numpy [L*T, L*T]   layer-major flattened
    """
    # cross_sim_4d axes: (t, l, t', l')  →  transpose to (l, t, l', t')
    layer_cross_4d = cross_sim_4d.transpose(1, 0, 3, 2)          # [L, T, L, T]
    L, T = layer_cross_4d.shape[0], layer_cross_4d.shape[1]
    layer_cross_2d = layer_cross_4d.reshape(L * T, L * T)        # [L*T, L*T]
    return layer_cross_4d, layer_cross_2d


def visualize_layer_cross_similarity(layer_cross_4d, layer_cross_2d, tracked,
                                     save_prefix='layer_cross'):
    """
    Produce visualizations for layer-wise cross-noise patch cosine similarity.

    Outputs:
        {save_prefix}_heatmap_full.png      – [L*T, L*T] heatmap, layer-major blocks
        {save_prefix}_diagonal_blocks.png   – diagonal T×T blocks: noise stability per layer
    """
    L, T = layer_cross_4d.shape[0], layer_cross_4d.shape[1]
    t_labels = [str(t) for t in tracked]

    # ── 1. Full [L*T, L*T] heatmap ───────────────────────────────────────────
    vmin = float(np.percentile(layer_cross_2d, 2))
    vmax = float(np.percentile(layer_cross_2d, 98))
    fig_sz = max(10, min(20, L * T // 10))
    fig, ax = plt.subplots(figsize=(fig_sz, fig_sz))
    im = ax.imshow(layer_cross_2d, cmap='viridis', vmin=vmin, vmax=vmax,
                   aspect='auto', interpolation='nearest')
    plt.colorbar(im, ax=ax, shrink=0.6, label='Cosine similarity')

    # Draw separator lines every T rows/cols to highlight L×L block structure
    for i in range(1, L):
        ax.axhline(i * T - 0.5, color='white', linewidth=0.4, alpha=0.6)
        ax.axvline(i * T - 0.5, color='white', linewidth=0.4, alpha=0.6)

    # Axis ticks: one per block (layer index)
    tick_pos = [l * T + T // 2 for l in range(L)]
    step = max(1, L // 14)
    shown = [l for l in range(L) if l % step == 0]
    ax.set_xticks([tick_pos[l] for l in shown])
    ax.set_xticklabels([f'L{l}' for l in shown], fontsize=7, rotation=45)
    ax.set_yticks([tick_pos[l] for l in shown])
    ax.set_yticklabels([f'L{l}' for l in shown], fontsize=7)
    ax.set_title(f'Layer-wise Cross-Noise Patch Cosine Similarity  [{L*T}×{L*T}]\n'
                 f'(layer-major: each {T}×{T} block = one layer pair across noise levels)',
                 fontsize=10)
    plt.tight_layout()
    fname1 = f'{save_prefix}_heatmap_full.png'
    fig.savefig(fname1, dpi=120, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    print(f"  Saved {fname1}")

    # ── 2. Diagonal T×T blocks: noise stability per selected layer ────────────
    sel = sorted({0, L // 4, L // 2, 3 * L // 4, L - 1})
    n_sel = len(sel)
    fig, axes = plt.subplots(1, n_sel, figsize=(3.5 * n_sel, 3.5), squeeze=False)
    for col, li in enumerate(sel):
        # S[l, :, l, :] in R^{T × T}: similarity of layer li with itself across noise pairs
        block = layer_cross_4d[li, :, li, :]   # [T, T]
        ax = axes[0, col]
        sns.heatmap(block, ax=ax, cmap='viridis',
                    vmin=float(block.min()), vmax=1.0,
                    xticklabels=t_labels, yticklabels=t_labels, annot=(T <= 12),
                    fmt='.2f' if T <= 12 else '')
        ax.set_title(f'Layer {li}\n(noise × noise)', fontsize=9)
        ax.set_xlabel('Noise level t\'', fontsize=8)
        if col == 0:
            ax.set_ylabel('Noise level t', fontsize=8)
        ax.tick_params(axis='x', labelsize=7, rotation=45)
        ax.tick_params(axis='y', labelsize=7)
    fig.suptitle('Diagonal blocks S[l,:,l,:] – noise-level stability per layer\n'
                 '(high off-diagonal = layer is stable across noise levels)', fontsize=10)
    plt.tight_layout()
    fname2 = f'{save_prefix}_diagonal_blocks.png'
    fig.savefig(fname2, dpi=120, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    print(f"  Saved {fname2}")


##############################################################################
#  REPA-style PCA feature map visualization
#  Spec: repa_pca_visualization_only_spec.txt
##############################################################################

def pca_feature_map(tokens, gh, gw, eps=1e-8, robust=True):
    """
    Per-panel PCA feature map – Mode A (spec recommended default).

    For one (image, timestep, layer):
      1. center tokens: Xc = tokens - mean(tokens, axis=0)
      2. SVD → top-3 principal directions
      3. project:  Y = Xc @ V3        [N, 3]
      4. reshape:  Y_grid [gh, gw, 3]
      5. normalize each channel to [0, 1] via 1/99 percentile clip (robust)

    tokens : numpy [N, D]  – spatial patch tokens for one image
    Returns: numpy [gh, gw, 3]  float32 in [0, 1] – RGB feature map panel
    """
    Xc = tokens - tokens.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    V3 = Vt[:3].T                       # [D, 3]
    Y  = (Xc @ V3).reshape(gh, gw, 3)  # [gh, gw, 3]
    for c in range(3):
        yc = Y[:, :, c]
        lo = np.percentile(yc, 1)  if robust else yc.min()
        hi = np.percentile(yc, 99) if robust else yc.max()
        Y[:, :, c] = (yc - lo) / (hi - lo + eps)
    return np.clip(Y, 0, 1).astype(np.float32)


def visualize_repa_pca(
    models_data,
    tracked_steps,
    image_b=0,
    sel_layers=None,
    upsample_factor=8,
    out_prefix='repa_pca',
):
    """
    REPA-style PCA feature map grid (spec: repa_pca_visualization_only_spec.txt).

    Produces one PNG per tracked timestep:  {out_prefix}_t{t}.png

    Layout per figure:
        - rows    = models  (one row per entry in models_data)
        - columns = selected layers
        - each cell = RGB PCA feature map for that (model, layer) at timestep t

    Parameters
    ----------
    models_data   : list of (name: str, block_tokens_per_timestep: dict)
                    dict maps t_val -> list[L] of tensor [B, N, D]
    tracked_steps : ordered list of t values to visualize
    image_b       : index of image in batch to visualize (default 0)
    sel_layers    : 0-indexed layer indices; auto-selected sparse subset if None
                    (early + middle + late, matching REPA figure style)
    upsample_factor : nearest-neighbour upsampling for display
                      (default 8: 16×16 patch grid → 128×128 pixels)
    out_prefix    : filename prefix (default 'repa_pca')
    """
    all_btps = [btp for _, btp in models_data]
    tracked  = [t for t in tracked_steps if all(t in btp for btp in all_btps)]
    if not tracked:
        print("visualize_repa_pca: no tracked timesteps found in any model.")
        return

    L = len(all_btps[0][tracked[0]])

    # Sparse layer subset: early (0-6) + a mid point + late  (spec section "Which layers")
    if sel_layers is None:
        early = list(range(min(7, L)))
        mid   = [L // 2 - 1]
        late  = sorted({3 * L // 4 - 1, L - 2, L - 1})
        sel_layers = sorted(set(early + mid + late) & set(range(L)))

    n_rows = len(models_data)
    n_cols = len(sel_layers)

    for t in tracked:
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(n_cols * 1.6, n_rows * 1.9),
                                 squeeze=False)
        fig.suptitle(f't = {t}', fontsize=11, y=1.02)

        for row, (model_name, btp) in enumerate(models_data):
            block_list = btp[t]  # list[L] of tensor [B, N, D]
            for col, li in enumerate(sel_layers):
                tokens = block_list[li][image_b].cpu().numpy()  # [N, D]
                N, D   = tokens.shape
                Gh = Gw = int(round(N ** 0.5))
                panel  = pca_feature_map(tokens, Gh, Gw)       # [Gh, Gw, 3]
                if upsample_factor > 1:
                    panel = panel.repeat(upsample_factor, axis=0) \
                                 .repeat(upsample_factor, axis=1)
                ax = axes[row, col]
                ax.imshow(panel, interpolation='nearest')
                ax.axis('off')
                if row == 0:
                    ax.set_title(f'L{li}', fontsize=8)
            axes[row, 0].set_ylabel(model_name, fontsize=9)

        plt.tight_layout()
        fname = f'{out_prefix}_t{t}.png'
        fig.savefig(fname, dpi=120, bbox_inches='tight')
        plt.show()
        plt.close(fig)
        print(f"  Saved {fname}")


def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if args.ckpt is None:
        assert args.model == "DiT-XL/2", "Only DiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000

    # Load model:
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    ).to(device)
    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()
    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    # Labels to condition the model with:
    class_labels = [207, 360, 387, 974, 88, 979, 417, 279]

    # Create sampling noise:
    n = len(class_labels)
    z = torch.randn(n, 4, latent_size, latent_size, device=device)
    y = torch.tensor(class_labels, device=device)

    # Setup classifier-free guidance:
    z = torch.cat([z, z], 0)
    y_null = torch.tensor([args.num_classes] * n, device=device)
    y = torch.cat([y, y_null], 0)
    model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)

    # need_tracking when any analysis option is enabled
    need_tracking = args.track_similarity or args.run_pca or args.repa_pca

    # Sample images:
    if need_tracking:
        print(f"\nTracking block tokens at timesteps: {args.track_timesteps}")
        wrapper = ModelWrapper(model, model_kwargs, args.track_timesteps)
        samples = diffusion.p_sample_loop(
            wrapper, z.shape, noise=z, clip_denoised=False,
            model_kwargs={}, device=device, progress=True
        )
    else:
        samples = diffusion.p_sample_loop(
            model.forward_with_cfg, z.shape, z, clip_denoised=False,
            model_kwargs=model_kwargs, progress=True, device=device
        )

    samples, _ = samples.chunk(2, dim=0)
    samples = vae.decode(samples / 0.18215).sample

    # Save images:
    save_image(samples, "sample.png", nrow=4, normalize=True, value_range=(-1, 1))
    print("\nSaved samples to sample.png")

    # ── Cross-timestep similarity ─────────────────────────────────────────
    if args.track_similarity and need_tracking:
        print("\nComputing cross-timestep similarity...")
        cross_sim_4d, cross_sim_2d, tracked = compute_cross_timestep_similarity(
            wrapper.block_tokens_per_timestep, args.track_timesteps
        )
        T, L = cross_sim_4d.shape[0], cross_sim_4d.shape[1]
        print(f"  cross_sim_4d: {cross_sim_4d.shape}   cross_sim_2d: {cross_sim_2d.shape}")
        np.save("cross_sim_4d.npy", cross_sim_4d)
        np.save("cross_sim_2d.npy", cross_sim_2d)
        for ki, t in enumerate(tracked):
            diag_block = cross_sim_4d[ki, :, ki, :].copy()
            np.fill_diagonal(diag_block, np.nan)
            print(f"  t={t:3d}: within-timestep off-diag mean = {np.nanmean(diag_block):.3f}")
        visualize_cross_timestep_similarity(cross_sim_4d, cross_sim_2d, tracked,
                                            save_prefix="cross_sim")

        # Layer-wise cross-noise (layer-major reorder of the same data)
        print("\nComputing layer-wise cross-noise similarity...")
        layer_cross_4d, layer_cross_2d = compute_layer_cross_similarity(cross_sim_4d)
        np.save("layer_cross_4d.npy", layer_cross_4d)
        np.save("layer_cross_2d.npy", layer_cross_2d)
        print(f"  layer_cross_4d: {layer_cross_4d.shape}   layer_cross_2d: {layer_cross_2d.shape}")
        visualize_layer_cross_similarity(layer_cross_4d, layer_cross_2d, tracked,
                                         save_prefix="layer_cross")

    # ── REPA-style PCA feature map visualization ──────────────────────────
    if args.repa_pca and need_tracking:
        print("\nGenerating REPA-style PCA feature maps...")
        visualize_repa_pca(
            models_data=[('DiT', wrapper.block_tokens_per_timestep)],
            tracked_steps=args.track_timesteps,
            image_b=0,
            out_prefix='repa_pca',
        )

    # ── PCA analysis ──────────────────────────────────────────────────────
    if args.run_pca and need_tracking:
        print("\nRunning PCA analysis (token-wise, image-wise, trajectory)...")
        pca_results = run_pca_analysis(
            wrapper.block_tokens_per_timestep, args.track_timesteps
        )
        if pca_results:
            save_pca_metrics(pca_results)
            visualize_pca_results(pca_results)
            print(f"\nPCA analysis complete: {pca_results['L']} blocks × {pca_results['T']} timesteps")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download DiT-XL/2).")

    # Similarity / PCA tracking arguments
    parser.add_argument("--track-similarity", action="store_true",
                        help="Enable cross-timestep block cosine similarity tracking")
    parser.add_argument("--run-pca", action="store_true",
                        help="Run token-wise, image-wise, and trajectory PCA on block tokens")
    parser.add_argument("--repa-pca", action="store_true",
                        help="Generate REPA-style PCA RGB feature map panels per (timestep, layer)")
    parser.add_argument("--n-track-levels", type=int, default=10,
                        help="Number of evenly-spaced noise levels to track (default: 10). "
                             "Levels are auto-selected from the actual sampling schedule to "
                             "ensure they span the full denoising range.")
    parser.add_argument("--track-timesteps", type=int, nargs="+", default=None,
                        help="Explicit DDPM timestep values to track (overrides --n-track-levels). "
                             "Values must exist in the sampling schedule.")

    args = parser.parse_args()
    main(args)
