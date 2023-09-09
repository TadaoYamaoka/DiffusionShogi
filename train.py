import argparse
import numpy as np
import types
from tqdm import tqdm

from diffusers import (
    DDPMScheduler,
    UNet2DConditionModel,
)
from diffusers.utils import randn_tensor

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter

from dlshogi.common import *
from dlshogi.data_loader import Hcpe3DataLoader
from dlshogi.data_loader import DataLoader
from dlshogi import serializers
from dlshogi.network.policy_value_network import policy_value_network


parser = argparse.ArgumentParser()
parser.add_argument("cache")
parser.add_argument("test_data")
parser.add_argument("model")
parser.add_argument("--network", default="resnet30x384_relu")
parser.add_argument("-e", "--epoch", type=int, default=1)
parser.add_argument("-g", "--gpu", type=int, default=0)
parser.add_argument("-b", "--batch_size", type=int, default=32)
parser.add_argument("--eval_batch_size", type=int, default=16)
parser.add_argument('--eval_interval', type=int, default=1000)
parser.add_argument("--num_inference_steps", type=int, default=20)
args = parser.parse_args()


if args.gpu >= 0:
    device = torch.device(f"cuda:{args.gpu}")
else:
    device = torch.device("cpu")


train_len, actual_len = Hcpe3DataLoader.load_files([], cache=args.cache)
train_data = np.arange(train_len, dtype=np.uint32)
test_data = np.fromfile(args.test_data, dtype=HuffmanCodedPosAndEval)

train_dataloader = Hcpe3DataLoader(train_data, args.batch_size, device, shuffle=True)
test_dataloader = DataLoader(test_data, args.eval_batch_size, device)


model = policy_value_network(args.network)


def forward(self, x1, x2):
    u1_1_1 = self.l1_1_1(x1)
    u1_1_2 = self.l1_1_2(x1)
    u1_2 = self.l1_2(x2)
    u1 = self.act(self.norm1(u1_1_1 + u1_1_2 + u1_2))
    return self.blocks(u1)


model.forward = types.MethodType(forward, model)
model.to(device)
serializers.load_npz(args.model, model)
model.eval()


unet = UNet2DConditionModel(
    sample_size=9,
    in_channels=27,
    out_channels=27,
    encoder_hid_dim=model.policy.in_channels * 9 * 9,
)
unet.to(device)

optimizer = AdamW(
    unet.parameters(),
    lr=1e-4,
    betas=(0.9, 0.999),
    weight_decay=1e-2,
    eps=1e-08,
)

lr_scheduler = StepLR(optimizer=optimizer, step_size=1, gamma=0.5)

noise_scheduler = DDPMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
)

generator = torch.Generator(device=unet.device).manual_seed(0)

nl_loss = torch.nn.NLLLoss()
def accuracy(y, t):
    return (torch.max(y, 1)[1] == t).sum().detach().item() / len(t)

def eval():
    noise_scheduler = DDPMScheduler()

    x1, x2, policies, win, value = test_dataloader.sample()
    bsz = policies.shape[0]

    encoder_hidden_states = model(x1, x2).reshape(bsz, 1, -1)

    # inference (sample random noise and denoise)
    image = randn_tensor((bsz, 27, 9, 9), generator=generator, device=device)

    # set step values
    noise_scheduler.set_timesteps(args.num_inference_steps)

    unet.eval()
    for t in noise_scheduler.timesteps:
        # 1. predict noise model_output
        model_output = unet(image, t, encoder_hidden_states).sample

        # 2. compute previous image: x_t -> x_t-1
        image = noise_scheduler.step(model_output, t, image, generator=generator).prev_sample

    pred = image.reshape(bsz, -1)
    loss = nl_loss(torch.log(pred), policies)
    writer.add_scalar("loss/eval", loss.detach().item(), step)
    writer.add_scalar("accuracy/eval", accuracy(pred, policies), step)

    unet.train()

writer = SummaryWriter()


step = 0
for epoch in range(args.epoch):
    unet.train()
    for x1, x2, policies, win, value in tqdm(
        train_dataloader, total=train_len // args.batch_size, desc=f"epoch: {epoch}"
    ):
        bsz = policies.shape[0]
        policies = policies.reshape((bsz, 27, 9, 9))

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(policies)

        # Sample a random timestep for each image
        timesteps = torch.randint(
            0,
            noise_scheduler.config.num_train_timesteps,
            (bsz,),
            device=policies.device,
        )
        timesteps = timesteps.long()

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_policies = noise_scheduler.add_noise(policies, noise, timesteps)

        # Get the embedding for position
        encoder_hidden_states = model(x1, x2).reshape(bsz, 1, -1)

        target = noise

        # Predict the noise residual and compute loss
        model_pred = unet(noisy_policies, timesteps, encoder_hidden_states).sample

        loss = F.mse_loss(model_pred, target, reduction="mean")

        # Backpropagate
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        step += 1

        writer.add_scalar("loss/train", loss.detach().item(), step)

        if step % args.eval_interval == 0:
            eval()

    lr_scheduler.step()
