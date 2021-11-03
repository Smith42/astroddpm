import numpy as np
import torch
import time
import argparse

from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer

# Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8)
).to(device=DEVICE)

diffusion = GaussianDiffusion(
    model,
    timesteps = 1000,
    loss_type = 'l1'
).to(device=DEVICE)

parser = argparse.ArgumentParser("")
parser.add_argument('--dataset', default='probes', choices=['probes', 'sdss'], help='Which dataset?')
parser.add_argument('--milestone', default=750000, dest=milestone, type=int, help='start at this number')
parser.add_argument('--batches', default=105, dest=batches, type=int, help='Number of batches to process.')
args = parser.parse_args()

if args.dataset == 'probes':
    trainer = Trainer(
        diffusion,
        './data/probes/',
        logdir = './logs/probes/',
        image_size = 256,
        train_batch_size = 16,
        train_lr = 2e-5,
        train_num_steps = 750001,         # total training steps
        gradient_accumulate_every = 2,    # gradient accumulation steps
        ema_decay = 0.995,                # exponential moving average decay
        num_workers=32,
        rank = [0]
    )

if args.dataset == 'sdss':
    trainer = Trainer(
        diffusion,
        './data/sdss/',
        logdir = './logs/sdss/',
        image_size = 256,
        train_batch_size = 16,
        train_lr = 2e-5,
        train_num_steps = 750001,         # total training steps
        gradient_accumulate_every = 2,    # gradient accumulation steps
        ema_decay = 0.995,                # exponential moving average decay
        num_workers=32,
        rank = [0]
    )

trainer.load(args.milestone)

i = 0
for _ in range(args.batches):
    sampled_batch = diffusion.sample(256, batch_size=96)

    for sample in sampled_batch.detach().cpu().numpy():
        np.save(f"inferred/PROBES_2021-10-08/{int(time.time())}_{i:05d}.npy", sample)
        i = i + 1
