# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from easydict import EasyDict
from .va_robotwin_train_cfg import va_robotwin_train_cfg
import os


def _env_bool(name, default=False):
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in ("1", "true", "yes", "y", "on")


va_robotwin_lingbot_train_cfg = EasyDict(
    __name__='Config: VA robotwin lingbot train'
)
va_robotwin_lingbot_train_cfg.update(va_robotwin_train_cfg)

va_robotwin_lingbot_train_cfg.dataset_path = os.getenv(
    "DATA_DIR",
    "/2023133163/datasets/lingbot/robotwin-clean-and-aug-lerobot",
)
va_robotwin_lingbot_train_cfg.empty_emb_path = os.getenv(
    "EMPTY_EMB_PATH",
    os.path.join(va_robotwin_lingbot_train_cfg.dataset_path, "empty_emb.pt"),
)
va_robotwin_lingbot_train_cfg.wan22_pretrained_model_name_or_path = os.getenv(
    "CKPT_DIR",
    "/2023133163/checkpoints/lingbot/lingbot-va-base",
)
va_robotwin_lingbot_train_cfg.save_root = os.getenv(
    "SAVE_ROOT",
    "/2023133163/checkpoints/lingbot/robotwin-posttrain",
)

va_robotwin_lingbot_train_cfg.enable_wandb = _env_bool("ENABLE_WANDB", False)
va_robotwin_lingbot_train_cfg.save_interval = 50000
