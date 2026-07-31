"""
Generate a large batch of cell samples from a diffusion model and save them as a large
numpy array. This can be used to produce samples for evaluation or visualization.
"""
import argparse
import numpy as np
import torch as th
import torch.distributed as dist
import random

from guided_diffusion import dist_util, logger
from guided_diffusion.script_utilNoVAE import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)


def save_data(all_cells, traj, data_dir):
    np.savez(data_dir, cell_gen=all_cells)
    return


def infer_arch_from_ckpt(ckpt: dict):
    """
    Infer (input_dim, hidden_dim_list) from a Cell_Unet checkpoint.
    input_dim is inferred from layers.0.fc.weight shape [hidden0, input_dim]
    hidden_dim list is inferred from layers.{i}.fc.weight out_features.
    """
    key0 = "layers.0.fc.weight"
    if key0 not in ckpt:
        raise KeyError(f"Checkpoint missing '{key0}'. Can't infer architecture.")

    # input_dim = in_features of first fc
    input_dim = int(ckpt[key0].shape[1])

    # hidden_dim = out_features of each encoder fc
    hidden_dim = []
    i = 0
    while True:
        k = f"layers.{i}.fc.weight"
        if k not in ckpt:
            break
        out_features = int(ckpt[k].shape[0])
        hidden_dim.append(out_features)
        i += 1

    if len(hidden_dim) == 0:
        raise RuntimeError("Could not infer hidden_dim from checkpoint (no layers.*.fc.weight found).")

    return input_dim, hidden_dim


def main():
    # setup_seed(1234)
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure(dir="output/checkpoint/sample_logs")

    # -------------------------------
    # NoVAE FIX: infer input_dim + hidden_dim from checkpoint BEFORE creating the model
    # -------------------------------
    logger.log("loading checkpoint (to infer input_dim and hidden_dim)...")
    ckpt = dist_util.load_state_dict(args.model_path, map_location="cpu")

    args.input_dim, args.hidden_dim = infer_arch_from_ckpt(ckpt)
    print("NoVAE sampling: inferred args.input_dim  =", args.input_dim, flush=True)
    print("NoVAE sampling: inferred args.hidden_dim =", args.hidden_dim, flush=True)

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    print("diffusion.num_timesteps =", diffusion.num_timesteps)


    logger.log("loading model weights...")
    model.load_state_dict(ckpt, strict=True)

    model.to(dist_util.dev())
    model.eval()

    logger.log("sampling...")
    all_cells = []
    num_generated = 0
    world_size = dist.get_world_size()

    while num_generated < args.num_samples:
        current_batch_size = min(args.batch_size, args.num_samples - num_generated)

        model_kwargs = {}
        sample_fn = diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        print(f"timestep for sampling is {diffusion.betas.shape[0]}")
        sample, traj = sample_fn(
            model,
            (current_batch_size, args.input_dim),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            start_time=diffusion.betas.shape[0],
        )

        # sample, traj = sample_fn(
        #     model,
        #     (args.batch_size, args.input_dim),
        #     clip_denoised=args.clip_denoised,
        #     model_kwargs=model_kwargs,
        #     start_time=diffusion.num_timesteps,   # <-- IMPORTANT
        # )


        # Gather samples across processes
        gathered_samples = [th.zeros_like(sample) for _ in range(world_size)]
        dist.all_gather(gathered_samples, sample)  # works with GLOO; NCCL may need different handling
        gathered_samples = [s.cpu().numpy() for s in gathered_samples]
        gathered_samples = np.concatenate(gathered_samples, axis=0)

        needed = args.num_samples - num_generated
        trimmed = gathered_samples[:needed]
        all_cells.append(trimmed)
        num_generated += trimmed.shape[0]

        logger.log(f"created {num_generated} / {args.num_samples} samples")

    arr = np.concatenate(all_cells, axis=0)
    save_data(arr, traj, args.sample_dir)

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=False,
        num_samples=12000,
        batch_size=3000,
        use_ddim=False,
        model_path="output/checkpoint/backbone/open_problem/model800000.pt",
        sample_dir="output/simulated_samples/open_problem",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


def setup_seed(seed):
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    th.backends.cudnn.deterministic = True


if __name__ == "__main__":
    main()
