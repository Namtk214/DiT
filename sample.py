# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained DiT.
Optional: Track block-wise cosine similarity during sampling.
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
from tqdm import tqdm


def compute_block_cosine_matrix(block_tokens):
    """
    Compute pairwise cosine similarity matrix between all blocks.

    Args:
        block_tokens: List of L tensors, each with shape (B, N, D)
                      where L = num_blocks, B = batch_size,
                      N = num_tokens, D = hidden_dim

    Returns:
        sim_mat: (L, L) cosine similarity matrix averaged over batch and tokens
    """
    # Stack into single tensor: (L, B, N, D)
    H = torch.stack(block_tokens, dim=0)
    L, B, N, D = H.shape

    # Normalize along hidden dimension
    H_norm = H / (torch.linalg.norm(H, dim=-1, keepdim=True) + 1e-8)

    # Compute pairwise cosine: for each pair of blocks (i, j), compute dot product
    # H_norm shape: (L, B, N, D), output shape: (L, L, B, N)
    cosine_vals = torch.einsum('ibnd,jbnd->ijbn', H_norm, H_norm)

    # Average over batch and tokens: (L, L, B, N) -> (L, L)
    sim_mat = cosine_vals.mean(dim=(-2, -1))

    return sim_mat


class ModelWrapper:
    """Wrapper to track block similarities during sampling."""
    def __init__(self, model, model_kwargs, track_timesteps):
        self.model = model
        self.model_kwargs = model_kwargs
        self.track_timesteps = track_timesteps
        self.similarity_data = {}

    def __call__(self, x, t):
        # Get timestep value (t is a tensor)
        t_val = t[0].item() if isinstance(t, torch.Tensor) else t

        if t_val in self.track_timesteps:
            # Call forward_with_cfg with return_block_tokens=True
            out, block_tokens = self.model.forward_with_cfg(
                x, t, self.model_kwargs['y'], self.model_kwargs['cfg_scale'],
                return_block_tokens=True
            )

            # Compute cosine similarity matrix
            sim_mat = compute_block_cosine_matrix(block_tokens)

            # Store (convert to numpy for easier handling)
            self.similarity_data[t_val] = sim_mat.cpu().numpy()

            print(f"  [Timestep {t_val}] Computed similarity matrix, shape: {sim_mat.shape}")

            return out
        else:
            # Normal forward without block tokens
            return self.model.forward_with_cfg(
                x, t, self.model_kwargs['y'], self.model_kwargs['cfg_scale']
            )


def sample_with_similarity_tracking(model, diffusion, z, model_kwargs, device, track_timesteps):
    """
    Custom sampling loop with block-wise cosine similarity tracking.

    Args:
        model: DiT model
        diffusion: Diffusion object
        z: (B, C, H, W) initial noise
        model_kwargs: dict with 'y' (labels) and 'cfg_scale'
        device: torch device
        track_timesteps: list of timestep indices to track similarity

    Returns:
        samples: (B, C, H, W) generated samples
        similarity_data: dict with timestep -> sim_mat mapping
    """
    # Create wrapper that will track similarities
    wrapper = ModelWrapper(model, model_kwargs, track_timesteps)

    # Use standard p_sample_loop with our wrapper
    samples = diffusion.p_sample_loop(
        wrapper,
        z.shape,
        noise=z,
        clip_denoised=False,
        model_kwargs={},  # kwargs already in wrapper
        device=device,
        progress=True
    )

    return samples, wrapper.similarity_data


def main(args):
    # Initialize WandB if tracking similarity
    if args.track_similarity:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=f"sample_similarity_{args.run_name}",
            config=vars(args)
        )
        print("WandB initialized for similarity tracking")

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
    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()  # important!
    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    # Labels to condition the model with (feel free to change):
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

    # Sample images:
    if args.track_similarity:
        print(f"\nTracking similarity at timesteps: {args.track_timesteps}")
        samples, similarity_data = sample_with_similarity_tracking(
            model, diffusion, z, model_kwargs, device, args.track_timesteps
        )
        print(f"Collected similarity data at {len(similarity_data)} timesteps")
    else:
        samples = diffusion.p_sample_loop(
            model.forward_with_cfg, z.shape, z, clip_denoised=False,
            model_kwargs=model_kwargs, progress=True, device=device
        )

    samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
    samples = vae.decode(samples / 0.18215).sample

    # Save and display images:
    save_image(samples, "sample.png", nrow=4, normalize=True, value_range=(-1, 1))
    print("\nSaved samples to sample.png")

    # Log to WandB if tracking similarity
    if args.track_similarity:
        import wandb

        # Log generated images
        wandb_images = [wandb.Image(samples[i], caption=f"Class {class_labels[i]}")
                       for i in range(len(class_labels))]
        wandb.log({"generated_samples": wandb_images})

        # Log similarity matrices
        for timestep, sim_mat in similarity_data.items():
            # Log as heatmap
            fig = wandb.plot.heatmap(
                x_labels=[f"Block {i}" for i in range(sim_mat.shape[0])],
                y_labels=[f"Block {i}" for i in range(sim_mat.shape[1])],
                matrix_values=sim_mat.tolist(),
                show_text=True
            )
            wandb.log({f"similarity/timestep_{timestep:03d}": fig})

            # Also log raw matrix
            wandb.log({f"similarity_matrix/timestep_{timestep:03d}": wandb.Table(
                data=sim_mat.tolist(),
                columns=[f"Block_{i}" for i in range(sim_mat.shape[1])]
            )})

        # Compute and log summary statistics
        avg_sim_mat = np.mean(list(similarity_data.values()), axis=0)
        wandb.log({
            "similarity/average_across_timesteps": wandb.plot.heatmap(
                x_labels=[f"Block {i}" for i in range(avg_sim_mat.shape[0])],
                y_labels=[f"Block {i}" for i in range(avg_sim_mat.shape[1])],
                matrix_values=avg_sim_mat.tolist(),
                show_text=True
            )
        })

        print("\n✅ Logged similarity matrices to WandB")
        wandb.finish()


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
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")

    # Similarity tracking arguments
    parser.add_argument("--track-similarity", action="store_true",
                        help="Enable block-wise cosine similarity tracking")
    parser.add_argument("--track-timesteps", type=int, nargs="+",
                        default=[1, 4, 8, 32, 64, 127],
                        help="Timestep indices to track similarity (default: 1 4 8 32 64 127)")
    parser.add_argument("--wandb-project", type=str, default="dit-similarity",
                        help="WandB project name for similarity logging")
    parser.add_argument("--run-name", type=str, default="test",
                        help="WandB run name")

    args = parser.parse_args()
    main(args)
