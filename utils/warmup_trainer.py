"""Warm-up trainer."""

import logging
import math
import os
import sys
import time

import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

from utils.helper import AverageMeter, Save_Handle
from utils.trainer import Trainer

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from datasets.crowd_drone import Crowd_Drone
from datasets.crowd_rgbtcc import Crowd_RGBTCC
from models.RACANetWarmup import build_racanet_warmup


def train_collate(batch):
    """Collate training samples with variable point counts."""
    transposed_batch = list(zip(*batch))
    if type(transposed_batch[0][0]) == list:
        rgb_list = [item[0] for item in transposed_batch[0]]
        t_list = [item[1] for item in transposed_batch[0]]
        rgb = torch.stack(rgb_list, 0)
        t = torch.stack(t_list, 0)
        images = [rgb, t]
    else:
        images = torch.stack(transposed_batch[0], 0)
    points = transposed_batch[1]
    st_sizes = torch.FloatTensor(transposed_batch[2])
    return images, points, st_sizes


class WarmupTrainer(Trainer):
    def setup(self):
        args = self.args
        if torch.cuda.is_available():
            self.device_count = torch.cuda.device_count()
            logging.info("available gpu count: {}".format(self.device_count))
            self.device = torch.device("cuda:{}".format(args.device))
            logging.info("using gpu {}".format(args.device))
        else:
            raise Exception("gpu is not available")

        if args.dataset == "RGBTCC":
            train_set = Crowd_RGBTCC(
                os.path.join(args.data_dir, "train"),
                args.crop_size,
                args.downsample_ratio,
                "train",
            )
        elif args.dataset == "DroneRGBT":
            train_set = Crowd_Drone(
                os.path.join(args.data_dir, "train"),
                args.crop_size,
                args.downsample_ratio,
                "train",
            )
        else:
            raise Exception("The dataset is not implemented!")

        self.train_loader = DataLoader(
            train_set,
            collate_fn=train_collate,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers * self.device_count,
            pin_memory=True,
        )

        self.model = build_racanet_warmup(
            fusion_embed_dim=args.fusion_embed_dim,
            local_match_window=args.local_match_window,
        )
        logging.info("model params (M): {:.3f}".format(sum(p.numel() for p in self.model.parameters()) / 1e6))
        self.model.to(self.device)

        params = []
        for key, value in dict(self.model.named_parameters()).items():
            if "pvt_backbone_rgb" in key or "pvt_backbone_t" in key:
                params += [{"params": value, "lr": args.lr * 0.1}]
            else:
                params += [{"params": value}]
        self.optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)

        self.start_epoch = 0
        if args.resume:
            suf = args.resume.rsplit(".", 1)[-1]
            if suf == "tar":
                checkpoint = torch.load(args.resume, self.device)
                self.model.load_state_dict(checkpoint["model_state_dict"])
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                self.start_epoch = checkpoint["epoch"] + 1
            elif suf == "pth":
                self.model.load_state_dict(torch.load(args.resume, self.device))

        self.save_list = Save_Handle(max_num=args.max_model_num)
        self.gaussian_kernel = self._build_gaussian_kernel(args.car_sigma).to(self.device)

    @staticmethod
    def _build_gaussian_kernel(sigma):
        radius = max(1, int(math.ceil(3 * sigma)))
        ksize = radius * 2 + 1
        coords = torch.arange(ksize, dtype=torch.float32) - radius
        g1d = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
        g1d = g1d / g1d.sum()
        g2d = torch.outer(g1d, g1d)
        g2d = g2d / g2d.sum()
        return g2d.view(1, 1, ksize, ksize)

    def _build_candidate_target(self, points, image_h, image_w):
        b = len(points)
        target = torch.zeros((b, 1, image_h, image_w), dtype=torch.float32, device=self.device)

        for i, pts in enumerate(points):
            if pts.numel() == 0:
                continue
            x = pts[:, 0].round().long().clamp(0, image_w - 1)
            y = pts[:, 1].round().long().clamp(0, image_h - 1)
            target[i, 0, y, x] = 1.0

        padding = self.gaussian_kernel.shape[-1] // 2
        target = F.conv2d(target, self.gaussian_kernel, padding=padding)
        max_val = target.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        target = target / max_val
        return target

    @staticmethod
    def _candidate_loss(c_pred, c_gt):
        eps = 1e-6
        bce = F.binary_cross_entropy(c_pred, c_gt)
        inter = (c_pred * c_gt).sum(dim=(1, 2, 3))
        den = c_pred.sum(dim=(1, 2, 3)) + c_gt.sum(dim=(1, 2, 3))
        dice = (2 * inter + eps) / (den + eps)
        return bce + (1.0 - dice).mean()

    @staticmethod
    def _align_loss(stage_dict):
        eps = 1e-6
        q_rgb = stage_dict["q_rgb"]
        q_t = stage_dict["q_t"]
        q_t_aligned = stage_dict["q_t_aligned"]
        q_r_aligned = stage_dict["q_r_aligned"]
        c_map = stage_dict["c_map"]

        err_fw = torch.abs(q_rgb - q_t_aligned).mean(dim=1, keepdim=True)
        err_bw = torch.abs(q_t - q_r_aligned).mean(dim=1, keepdim=True)
        err = err_fw + err_bw

        weighted_err = (c_map * err).sum(dim=(1, 2, 3))
        normalizer = c_map.sum(dim=(1, 2, 3)).clamp_min(eps)
        return (weighted_err / normalizer).mean()

    def train(self):
        args = self.args
        for epoch in range(self.start_epoch, args.max_epoch):
            self.epoch = epoch
            logging.info("-----Epoch {}/{}-----".format(epoch, args.max_epoch - 1))
            self.train_epoch()

    def train_epoch(self):
        args = self.args
        self.model.train()

        epoch_total = AverageMeter()
        epoch_car = AverageMeter()
        epoch_align = AverageMeter()
        epoch_start = time.time()

        for inputs, points, _ in self.train_loader:
            inputs[0] = inputs[0].to(self.device)
            inputs[1] = inputs[1].to(self.device)
            points = [p.to(self.device) for p in points]

            with torch.set_grad_enabled(True):
                outputs = self.model(inputs)

                image_h, image_w = inputs[0].shape[2:]
                c_target_full = self._build_candidate_target(points, image_h, image_w)

                loss_car = 0.0
                loss_align = 0.0
                for stage_name in ["32x", "16x", "8x", "4x"]:
                    stage_out = outputs[stage_name]
                    stage_h, stage_w = stage_out["c_map"].shape[2:]
                    c_target_stage = F.interpolate(
                        c_target_full,
                        size=(stage_h, stage_w),
                        mode="bilinear",
                        align_corners=False,
                    )
                    loss_car = loss_car + self._candidate_loss(stage_out["c_map"], c_target_stage)
                    loss_align = loss_align + self._align_loss(stage_out)

                loss = args.lambda_car * loss_car + args.lambda_align * loss_align

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            batch_size = inputs[0].size(0)
            epoch_total.update(float(loss.item()), batch_size)
            epoch_car.update(float(loss_car.item()), batch_size)
            epoch_align.update(float(loss_align.item()), batch_size)

        logging.info(
            "Epoch {} Warmup Train, Total: {:.4f}, CAR: {:.4f}, ALIGN: {:.4f}, Cost {:.1f} sec".format(
                self.epoch,
                epoch_total.get_avg(),
                epoch_car.get_avg(),
                epoch_align.get_avg(),
                time.time() - epoch_start,
            )
        )

        save_path = os.path.join(self.save_dir, "{}_ckpt.tar".format(self.epoch))
        torch.save(
            {
                "epoch": self.epoch,
                "optimizer_state_dict": self.optimizer.state_dict(),
                "model_state_dict": self.model.state_dict(),
            },
            save_path,
        )
        self.save_list.append(save_path)

        # Save a plain state_dict for main training.
        torch.save(self.model.state_dict(), os.path.join(self.save_dir, "warmup_last.pth"))
