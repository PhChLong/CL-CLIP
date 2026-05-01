import argparse

import torch

from src.config import Config
from src.engine import Train
from src.methods import FineTune, LwF_LoRA, ZSCL
from src.models import CLIPWrapper


def str_to_bool(value):
    if isinstance(value, bool):
        return value

    value = value.lower()
    if value in ("true", "1", "yes", "y"):
        return True
    if value in ("false", "0", "no", "n"):
        return False

    raise argparse.ArgumentTypeError("Expected boolean value: true/false.")


def build_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--method", default=None)

    parser.add_argument("--model-num-layers", type=int, default=None)

    parser.add_argument("--root", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--pin-memory", type=str_to_bool, default=None)

    parser.add_argument("--train-name", default=None)
    parser.add_argument("--train-lr", type=float, default=None)
    parser.add_argument("--train-weight-decay", type=float, default=None)
    parser.add_argument("--train-max-epoch", type=int, default=None)
    parser.add_argument("--train-patience", type=int, default=None)
    parser.add_argument("--train-epsilon", type=float, default=None)
    parser.add_argument("--train-r", type=int, default=None)
    parser.add_argument("--train-distill-temp", type=float, default=None)
    parser.add_argument("--train-lambda-old", type=float, default=None)
    parser.add_argument("--train-alpha", type=float, default=None)

    parser.add_argument("--test-pipeline", action="store_true", default=None)
    return parser


def set_if_not_none(config, path, value):
    if value is None:
        return

    target = config
    for key in path[:-1]:
        target = getattr(target, key)
    setattr(target, path[-1], value)


def override_config(config, args):
    set_if_not_none(config, ("method",), args.method)

    set_if_not_none(config, ("model", "num_layers"), args.model_num_layers)

    set_if_not_none(config, ("datasets", "batch_size"), args.batch_size)
    set_if_not_none(config, ("datasets", "image_size"), args.image_size)
    set_if_not_none(config, ("datasets", "num_workers"), args.num_workers)
    set_if_not_none(config, ("datasets", "pin_memory"), args.pin_memory)

    set_if_not_none(config, ("train", "name"), args.train_name)
    set_if_not_none(config, ("train", "lr"), args.train_lr)
    set_if_not_none(config, ("train", "weight_decay"), args.train_weight_decay)
    set_if_not_none(config, ("train", "max_epoch"), args.train_max_epoch)
    set_if_not_none(config, ("train", "patience"), args.train_patience)
    set_if_not_none(config, ("train", "epsilon"), args.train_epsilon)
    set_if_not_none(config, ("train", "r"), args.train_r)
    set_if_not_none(config, ("train", "distill_temp"), args.train_distill_temp) #? lwf
    set_if_not_none(config, ("train", "lambda_old"), args.train_lambda_old) #? lwf
    set_if_not_none(config, ("train", "alpha"), args.train_alpha) #? c-clip

    return config


def build_method(method_name):
    methods = {
        "finetune": FineTune,
        "lwf": LwF_LoRA,
        "zscl": ZSCL,
    }

    if method_name not in methods:
        raise ValueError(f"Unknown or unsupported method: {method_name}")

    return methods[method_name]()


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    method_name = args.method or "lwf"
    config = override_config(Config(method_name), args)
    method = build_method(method_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip = CLIPWrapper(device=device)

    trainer = Train(clip, config, method)
    trainer.train_all_tasks(args.test_pipeline)
