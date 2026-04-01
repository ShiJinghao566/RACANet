"""Warm-up pretraining entry point."""

import argparse
import os
import random

import numpy as np
import torch

from utils.warmup_trainer import WarmupTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Warm-up Pretraining")
    parser.add_argument("--exp-tag", default="", help="the tag of the experiment")
    parser.add_argument("--dataset", default="RGBTCC", choices=["RGBTCC", "DroneRGBT"], help="dataset name")
    parser.add_argument("--data-dir", default="/Crowd Counting/RGBT-CC", help="training data directory")
    parser.add_argument("--save-dir", default="/Crowd Counting/RGBT-CC", help="directory to save checkpoints")
    parser.add_argument("--resume", default="", help="resume checkpoint path")
    parser.add_argument("--device", default="0", help="assign device")

    parser.add_argument("--crop-size", type=int, default=256, help="training crop size")
    parser.add_argument("--downsample-ratio", type=int, default=8, help="kept consistent with main pipeline")

    parser.add_argument("--lr", type=float, default=1e-6, help="learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="weight decay")
    parser.add_argument("--max-epoch", type=int, default=200, help="max training epoch")
    parser.add_argument("--batch-size", type=int, default=8, help="train batch size")
    parser.add_argument("--num-workers", type=int, default=8, help="number of workers")
    parser.add_argument("--max-model-num", type=int, default=2, help="max checkpoints to keep")

    parser.add_argument("--fusion-embed-dim", type=int, default=32, help="projection dim in warm-up model")
    parser.add_argument("--local-match-window", type=int, default=3, help="local matching window size (odd)")
    parser.add_argument("--car-sigma", type=float, default=4.0, help="gaussian sigma for candidate region target")
    parser.add_argument("--lambda-car", type=float, default=1.0, help="candidate region loss weight")
    parser.add_argument("--lambda-align", type=float, default=1.0, help="local alignment loss weight")

    parser.add_argument("--seed", type=int, default=1, help="random seed")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device.strip()

    trainer = WarmupTrainer(args)
    trainer.setup()
    trainer.train()
