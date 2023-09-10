import types
import random

import lightning.pytorch as pl
from lightning.pytorch.cli import LightningCLI

from diffusers import (
    DDPMScheduler,
    UNet2DConditionModel,
)
from diffusers.utils import randn_tensor

from dlshogi.common import *
from dlshogi import cppshogi
from dlshogi import serializers
from dlshogi.network.policy_value_network import policy_value_network

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class HcpeDataset(Dataset):
    def __init__(self, hcpe):
        self.hcpe = np.fromfile(hcpe, dtype=HuffmanCodedPosAndEval)

    def __len__(self):
        return len(self.hcpe)

    def __getitem__(self, idx):
        features1 = torch.empty((FEATURES1_NUM, 9, 9), dtype=torch.float32)
        features2 = torch.empty((FEATURES2_NUM, 9, 9), dtype=torch.float32)
        move = torch.empty(1, dtype=torch.int64)
        result = torch.empty(1, dtype=torch.float32)
        value = torch.empty(1, dtype=torch.float32)

        cppshogi.hcpe_decode_with_value(
            self.hcpe[idx : idx + 1],
            features1.numpy(),
            features2.numpy(),
            move.numpy(),
            result.numpy(),
            value.numpy(),
        )

        return features1, features2, move, result, value
   


class HcpeSampleDataset(Dataset):
    def __init__(self, hcpe, batch_size):
        self.dataset = HcpeDataset(hcpe)
        self.batch_size = batch_size

    def __len__(self):
        return self.batch_size

    def __getitem__(self, idx):
        return self.dataset.__getitem__(random.randrange(len(self.dataset)))


class Hcpe3CacheDataset(Dataset):
    def __init__(self, cache):
        self.cache = cache
        self.load_cache()

    def load_cache(self):
        cppshogi.hcpe3_load_cache(self.cache)

    def __len__(self):
        return cppshogi.hcpe3_get_cache_num()

    def __getitem__(self, idx):
        index = np.array([idx], dtype=np.uint32)
        features1 = torch.empty((FEATURES1_NUM, 9, 9), dtype=torch.float32)
        features2 = torch.empty((FEATURES2_NUM, 9, 9), dtype=torch.float32)
        probability = torch.empty((9 * 9 * MAX_MOVE_LABEL_NUM), dtype=torch.float32)
        result = torch.empty(1, dtype=torch.float32)
        value = torch.empty(1, dtype=torch.float32)

        cppshogi.hcpe3_decode_with_value(
            index,
            features1.numpy(),
            features2.numpy(),
            probability.numpy(),
            result.numpy(),
            value.numpy(),
        )

        return features1, features2, probability, result, value

    @staticmethod
    def worker_init(worker_id):
        torch.utils.data.get_worker_info().dataset.load_cache()


def accuracy(y, t):
    return (torch.max(y, 1)[1] == t).sum().detach().item() / len(t)


class DiffusionPolicy(pl.LightningModule):
    def __init__(
        self,
        dlshogi_model,
        dlshogi_network="resnet30x384_relu",
        num_inference_steps=20,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.dlshogi_model = policy_value_network(dlshogi_network)

        def forward(self, x1, x2):
            u1_1_1 = self.l1_1_1(x1)
            u1_1_2 = self.l1_1_2(x1)
            u1_2 = self.l1_2(x2)
            u1 = self.act(self.norm1(u1_1_1 + u1_1_2 + u1_2))
            return self.blocks(u1)

        self.dlshogi_model.forward = types.MethodType(forward, self.dlshogi_model)
        serializers.load_npz(dlshogi_model, self.dlshogi_model)
        self.dlshogi_model.requires_grad_(False)

        self.unet = UNet2DConditionModel(
            sample_size=9,
            in_channels=27,
            out_channels=27,
            block_out_channels=(64, 128, 256, 256),
            cross_attention_dim=256,
            encoder_hid_dim=self.dlshogi_model.policy.in_channels * 9 * 9,
        )

        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
        )

    def training_step(self, batch, batch_idx):
        x1, x2, policies, win, value = batch
        self.dlshogi_model.eval()

        bsz = policies.shape[0]
        policies = policies.reshape((bsz, 27, 9, 9))

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(policies)

        # Sample a random timestep for each image
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bsz,),
            device=policies.device,
        )
        timesteps = timesteps.long()

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_policies = self.noise_scheduler.add_noise(policies, noise, timesteps)

        # Get the embedding for position
        encoder_hidden_states = self.dlshogi_model(x1, x2).reshape(bsz, 1, -1).detach()

        target = noise

        # Predict the noise residual and compute loss
        model_pred = self.unet(noisy_policies, timesteps, encoder_hidden_states).sample

        loss = F.mse_loss(model_pred, target, reduction="mean")

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        noise_scheduler = DDPMScheduler()

        x1, x2, policies, win, value = batch
        bsz = policies.shape[0]
        policies = policies.reshape(-1)

        encoder_hidden_states = self.dlshogi_model(x1, x2).reshape(bsz, 1, -1).detach()

        # inference (sample random noise and denoise)
        image = randn_tensor((bsz, 27, 9, 9), device=x1.device)

        # set step values
        noise_scheduler.set_timesteps(self.hparams.num_inference_steps)

        for t in noise_scheduler.timesteps:
            # 1. predict noise model_output
            model_output = self.unet(image, t, encoder_hidden_states).sample

            # 2. compute previous image: x_t -> x_t-1
            image = noise_scheduler.step(model_output, t, image).prev_sample

        pred = torch.clamp(image.reshape(bsz, -1), 1e-45, 1)
        loss = F.cross_entropy(torch.log(pred), policies)
        self.log_dict({"val_loss": loss, "val_acc": accuracy(pred, policies)}, sync_dist=True)

    def on_save_checkpoint(self, checkpoint):
        # dlshogi_modelは保存しない
        keys = list(checkpoint["state_dict"].keys())
        for key in keys:
            if "dlshogi_model" in key:
                del checkpoint["state_dict"][key]
        super().on_save_checkpoint(checkpoint)


class MyDataModule(pl.LightningDataModule):
    def __init__(self, train_cache, val_hcpe, batch_size, num_workers, val_batch_size):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            self.train = Hcpe3CacheDataset(self.hparams.train_cache)
            self.val = HcpeSampleDataset(
                self.hparams.val_hcpe, self.hparams.val_batch_size
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage == "predict":
            self.test = HcpeDataset(self.val_hcpe)

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
            worker_init_fn=Hcpe3CacheDataset.worker_init
        )

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.hparams.val_batch_size)

    def test_dataloader(self):
        return DataLoader(self.val, batch_size=self.hparams.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.val, batch_size=self.hparams.batch_size)


def main():
    LightningCLI(DiffusionPolicy, MyDataModule)


if __name__ == "__main__":
    main()
