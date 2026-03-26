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
    H_list = []
    for t in tracked:
        tokens = block_tokens_per_timestep[t]   # list of L tensors, each (B, N, D)
        H_list.append(torch.stack(tokens, dim=0))   # [L, B, N, D]
    H = torch.stack(H_list, dim=0)   # [T, L, B, N, D]
    T, L, _, _, _ = H.shape

    # Normalize along hidden dimension
    h_norm = H / (torch.linalg.norm(H, dim=-1, keepdim=True) + 1e-8)

    # Broadcast pairwise over (T, L) x (T, L):
    #   h_norm[:, :, None, None, ...]  shape [T, L, 1, 1, B, N, D]
    # * h_norm[None, None, :, :, ...]  shape [1, 1, T, L, B, N, D]
    # dot-product over D -> [T, L, T, L, B, N]
    sim = (h_norm[:, :, None, None, :, :, :] * h_norm[None, None, :, :, :, :, :]).sum(dim=-1)

    # Average over batch and patch dimensions -> [T, L, T, L]
    cross_sim_4d = sim.mean(dim=(-2, -1))

    # Flatten (T, L) x (T, L) -> (T*L) x (T*L)
    # cross_sim_2d[k*L+a, k'*L+b] = cross_sim_4d[k, a, k', b]
    cross_sim_2d = cross_sim_4d.reshape(T * L, T * L)

    return cross_sim_4d.cpu().numpy(), cross_sim_2d.cpu().numpy(), tracked


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
##############################################################################

def _run_pca(X):
    """
    PCA on matrix X [M, F].
    Returns dict: evr, cum_evr, PR, top1, top5, top10, n80/90/95, components.
    """
    M, F = X.shape
    if M < 2:
        return None
    X_c = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(X_c, full_matrices=False)
    lambdas = S ** 2 / max(M - 1, 1)
    total = lambdas.sum() + 1e-12
    evr = lambdas / total
    cum_evr = np.cumsum(evr)
    PR = float(total ** 2 / (lambdas ** 2).sum())
    K = len(evr)
    top1  = float(evr[0])
    top5  = float(cum_evr[min(4,  K - 1)])
    top10 = float(cum_evr[min(9,  K - 1)])
    n80 = int(np.searchsorted(cum_evr, 0.80)) + 1
    n90 = int(np.searchsorted(cum_evr, 0.90)) + 1
    n95 = int(np.searchsorted(cum_evr, 0.95)) + 1
    return dict(evr=evr, cum_evr=cum_evr, PR=PR,
                top1=top1, top5=top5, top10=top10,
                n80=n80, n90=n90, n95=n95,
                components=Vt)   # Vt: [K, F]


def compute_token_pca(H):
    """Token-wise PCA.  H: numpy [B, N, D]  ->  PCA on [B*N, D]."""
    B, N, D = H.shape
    return _run_pca(H.reshape(B * N, D).astype(np.float32))


def compute_image_pca(H):
    """Image-wise PCA (mean-pool tokens).  H: numpy [B, N, D]  ->  PCA on [B, D]."""
    return _run_pca(H.mean(axis=1).astype(np.float32))


def compute_trajectory_pca(H_over_t):
    """
    Timestep-trajectory PCA (shared basis).
    H_over_t: list of T numpy arrays, each [B, N, D]

    Returns:
        pca_result: PCA metrics on stacked [B*T, D]
        traj_proj:  numpy [B, T, 3]  – first 3 PCs per image per timestep
    """
    T = len(H_over_t)
    B = H_over_t[0].shape[0]
    Z_list = [H.mean(axis=1).astype(np.float32) for H in H_over_t]  # T x [B, D]
    Z_all  = np.concatenate(Z_list, axis=0)                          # [B*T, D]
    pca_result = _run_pca(Z_all)
    if pca_result is None:
        return None, None
    mean_vec = Z_all.mean(axis=0)
    n_pc = min(3, len(pca_result['evr']))
    Vt3  = pca_result['components'][:n_pc]   # [3, D]
    proj_list = [(Z - mean_vec) @ Vt3.T for Z in Z_list]
    traj_proj = np.stack(proj_list, axis=1)  # [B, T, 3]
    return pca_result, traj_proj


def run_pca_analysis(block_tokens_per_timestep, tracked_steps):
    """
    Run all three PCA types on collected block tokens.

    Returns dict:
        tracked  : list of collected timesteps (ordered)
        L, T     : num blocks, num tracked timesteps
        token    : list[L] of list[T] of metric dicts
        image    : list[L] of list[T] of metric dicts
        traj     : list[L] of (pca_result, traj_proj [B, T, 3])
    """
    tracked = [t for t in tracked_steps if t in block_tokens_per_timestep]
    T = len(tracked)
    if T == 0:
        return None

    # Convert to numpy: block_np[ki][li] = [B, N, D]
    block_np = {}
    for ki, t in enumerate(tracked):
        block_np[ki] = [bt.cpu().numpy() for bt in block_tokens_per_timestep[t]]
    L = len(block_np[0])

    token_metrics = [[None] * T for _ in range(L)]
    image_metrics = [[None] * T for _ in range(L)]

    print("Running token-wise and image-wise PCA...")
    for li in tqdm(range(L), desc="Blocks (token+image PCA)"):
        for ki in range(T):
            H = block_np[ki][li]
            token_metrics[li][ki] = compute_token_pca(H)
            image_metrics[li][ki] = compute_image_pca(H)

    traj_results = []
    print("Running timestep-trajectory PCA...")
    for li in tqdm(range(L), desc="Blocks (trajectory PCA)"):
        H_over_t = [block_np[ki][li] for ki in range(T)]
        traj_results.append(compute_trajectory_pca(H_over_t))

    return dict(tracked=tracked, L=L, T=T,
                token=token_metrics, image=image_metrics, traj=traj_results)


def _extract_metric(metrics_2d, key, L, T):
    arr = np.zeros((L, T))
    for li in range(L):
        for ki in range(T):
            m = metrics_2d[li][ki]
            if m:
                arr[li, ki] = m[key]
    return arr


def save_pca_metrics(pca_results, prefix='pca'):
    """Save summary metrics as .npz files."""
    L, T = pca_results['L'], pca_results['T']
    tracked = np.array(pca_results['tracked'])

    for tag, metrics in [('token', pca_results['token']), ('image', pca_results['image'])]:
        np.savez(f'{prefix}_{tag}_pca_metrics.npz',
                 tracked=tracked,
                 top1 =_extract_metric(metrics, 'top1',  L, T),
                 top5 =_extract_metric(metrics, 'top5',  L, T),
                 top10=_extract_metric(metrics, 'top10', L, T),
                 PR   =_extract_metric(metrics, 'PR',    L, T),
                 n80  =_extract_metric(metrics, 'n80',   L, T),
                 n90  =_extract_metric(metrics, 'n90',   L, T),
                 n95  =_extract_metric(metrics, 'n95',   L, T))

    traj_PR   = np.array([r[0]['PR']   if r[0] else 0 for r in pca_results['traj']])
    traj_top1 = np.array([r[0]['top1'] if r[0] else 0 for r in pca_results['traj']])
    np.savez(f'{prefix}_traj_pca_metrics.npz',
             tracked=tracked, PR=traj_PR, top1=traj_top1)

    print(f"Saved {prefix}_token_pca_metrics.npz, {prefix}_image_pca_metrics.npz, "
          f"{prefix}_traj_pca_metrics.npz")


def visualize_pca_results(pca_results, save_prefix=None):
    """
    Produce:
      Fig 1 – 2×3 heatmaps (token top1/top5/PR, image top1/top5/PR)  over [L × T]
      Fig 2 – Trajectory plots (PC1 vs PC2) for selected blocks
      Fig 3 – Scree plots for selected (block, timestep) pairs
    """
    tracked = pca_results['tracked']
    L, T    = pca_results['L'], pca_results['T']
    xlabels = [f't={t}' for t in tracked]
    step    = max(1, L // 10)
    ylabels = [f'L{li}' if li % step == 0 else '' for li in range(L)]

    # ── Figure 1: heatmaps ────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    panels = [
        (_extract_metric(pca_results['token'], 'top1', L, T), 'Token-wise: top-1 var ratio', axes[0, 0]),
        (_extract_metric(pca_results['token'], 'top5', L, T), 'Token-wise: top-5 cum var',   axes[0, 1]),
        (_extract_metric(pca_results['token'], 'PR',   L, T), 'Token-wise: Participation Ratio', axes[0, 2]),
        (_extract_metric(pca_results['image'], 'top1', L, T), 'Image-wise: top-1 var ratio', axes[1, 0]),
        (_extract_metric(pca_results['image'], 'top5', L, T), 'Image-wise: top-5 cum var',   axes[1, 1]),
        (_extract_metric(pca_results['image'], 'PR',   L, T), 'Image-wise: Participation Ratio', axes[1, 2]),
    ]
    for mat, title, ax in panels:
        sns.heatmap(mat, ax=ax, cmap='viridis', annot=False,
                    xticklabels=xlabels, yticklabels=ylabels)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Block')
    fig.suptitle('PCA Summary: Token-wise (top) and Image-wise (bottom)', fontsize=13)
    plt.tight_layout()
    if save_prefix:
        fig.savefig(f'{save_prefix}_pca_heatmaps.png', dpi=100, bbox_inches='tight')
        print(f"  Saved {save_prefix}_pca_heatmaps.png")
    plt.show()

    # ── Figure 2: trajectory plots (selected blocks) ──────────────────────
    sel = sorted({0, L // 4, L // 2, 3 * L // 4, L - 1})
    fig2, axes2 = plt.subplots(1, len(sel), figsize=(5 * len(sel), 5))
    if len(sel) == 1:
        axes2 = [axes2]
    cmap = plt.cm.plasma
    for ax, li in zip(axes2, sel):
        traj_m, traj_p = pca_results['traj'][li]
        if traj_p is None:
            ax.set_title(f'Block {li}: no data')
            continue
        B_vis = traj_p.shape[0]
        for b in range(B_vis):
            xs, ys = traj_p[b, :, 0], traj_p[b, :, 1]
            for ki in range(T - 1):
                color = cmap(ki / max(T - 1, 1))
                ax.plot(xs[ki:ki+2], ys[ki:ki+2], '-o', color=color, markersize=4, linewidth=1.5)
        ax.set_title(f'Block {li}', fontsize=10)
        ax.set_xlabel('PC1')
        if li == sel[0]:
            ax.set_ylabel('PC2')
    sm = plt.cm.ScalarMappable(cmap=cmap,
                                norm=plt.Normalize(vmin=tracked[0], vmax=tracked[-1]))
    sm.set_array([])
    plt.colorbar(sm, ax=axes2, label='Timestep', shrink=0.6)
    fig2.suptitle('Timestep-Trajectory PCA – PC1 vs PC2 per Block', fontsize=12)
    plt.tight_layout()
    if save_prefix:
        fig2.savefig(f'{save_prefix}_pca_trajectories.png', dpi=100, bbox_inches='tight')
        print(f"  Saved {save_prefix}_pca_trajectories.png")
    plt.show()

    # ── Figure 3: scree plots ─────────────────────────────────────────────
    sel_pairs = [(0, 0), (L // 2, T // 2), (L - 1, T - 1)]
    fig3, axes3 = plt.subplots(1, len(sel_pairs), figsize=(6 * len(sel_pairs), 4))
    for ax, (li, ki) in zip(axes3, sel_pairs):
        m = pca_results['token'][li][ki]
        if m is None:
            continue
        n_show = min(30, len(m['evr']))
        ax.bar(range(1, n_show + 1), m['evr'][:n_show], alpha=0.7)
        ax.axhline(m['top1'], color='red', linestyle='--', alpha=0.6,
                   label=f"top1={m['top1']:.2f}  PR={m['PR']:.1f}")
        ax.set_xlabel('Component')
        ax.set_ylabel('Explained Var Ratio')
        ax.set_title(f'Scree – Block {li}, t={tracked[ki]}', fontsize=10)
        ax.legend(fontsize=8)
    fig3.suptitle('Scree Plots (Token-wise PCA)', fontsize=12)
    plt.tight_layout()
    if save_prefix:
        fig3.savefig(f'{save_prefix}_pca_scree.png', dpi=100, bbox_inches='tight')
        print(f"  Saved {save_prefix}_pca_scree.png")
    plt.show()


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

    # need_tracking when --track-similarity OR --run-pca
    need_tracking = args.track_similarity or args.run_pca

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

    # ── PCA analysis ──────────────────────────────────────────────────────
    if args.run_pca and need_tracking:
        print("\nRunning PCA analysis (token-wise, image-wise, trajectory)...")
        pca_results = run_pca_analysis(
            wrapper.block_tokens_per_timestep, args.track_timesteps
        )
        if pca_results:
            save_pca_metrics(pca_results, prefix='pca')
            visualize_pca_results(pca_results, save_prefix='pca')
            L_pca, T_pca = pca_results['L'], pca_results['T']
            print(f"\nPCA analysis complete: {L_pca} blocks × {T_pca} timesteps")
            print("Saved: pca_token_pca_metrics.npz, pca_image_pca_metrics.npz, "
                  "pca_traj_pca_metrics.npz")


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
    parser.add_argument("--track-timesteps", type=int, nargs="+",
                        default=[1, 4, 8, 16, 32, 48, 64, 80, 96, 127],
                        help="Timestep indices to track for similarity / PCA (default: 10 steps)")

    args = parser.parse_args()
    main(args)
