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
from guided_diffusion.script_util import (   
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)


def save_data(all_cells, traj, data_dir):
    cell_gen = all_cells
    np.savez(data_dir, cell_gen=cell_gen)
    return

def main():
    setup_seed(1234)
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure(dir='output/checkpoint/sample_logs')

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()

    logger.log("sampling...")
    all_cells = []
    num_generated = 0
    world_size = dist.get_world_size()

    while num_generated < args.num_samples:
        current_batch_size = min(args.batch_size, args.num_samples - num_generated)

        model_kwargs = {}
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample, traj = sample_fn(
            model,
            (current_batch_size, args.input_dim), 
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            start_time=diffusion.betas.shape[0],
        )

        # Gather samples across processes
        gathered_samples = [th.zeros_like(sample) for _ in range(world_size)]
        dist.all_gather(gathered_samples, sample)  # works with GLOO, not NCCL
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
        sample_dir="output/simulated_samples/open_problem"
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
