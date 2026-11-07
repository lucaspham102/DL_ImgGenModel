"""
Tiện ích dùng chung cho cả nhóm (VAE / DCGAN / DDPM).

- set_seed:        cố định seed để kết quả tái lập được
- Logger:          log scalar + image grid lên W&B hoặc TensorBoard (cùng một API)
- save_checkpoint: lưu nhiều model/optimizer/scheduler cùng lúc + epoch, step, config, RNG
- load_checkpoint: tải lại để train tiếp đúng chỗ hoặc để đánh giá
"""
import os
import random

import numpy as np
import torch
import torchvision.utils as vutils


# ---------------------------------------------------------------------------
# Seed
# ---------------------------------------------------------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Logger: một API chung cho W&B và TensorBoard
# ---------------------------------------------------------------------------
class Logger:
    def __init__(self, backend="wandb", project=None, entity=None,
                 run_name=None, config=None, log_dir="runs", sample_dir="samples"):
        """
        backend : "wandb" hoặc "tensorboard"
        project : tên project trên W&B (vd "gen-models-de4")
        entity  : tên team W&B (để mọi người log chung một chỗ); None = tài khoản cá nhân
        """
        self.backend = backend
        self.sample_dir = sample_dir
        os.makedirs(sample_dir, exist_ok=True)

        if backend == "wandb":
            import wandb
            self.wandb = wandb
            self.run = wandb.init(project=project, entity=entity,
                                  name=run_name, config=config)
        elif backend == "tensorboard":
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(os.path.join(log_dir, run_name or "run"))
            if config:
                self.writer.add_text("config", str(config))
        else:
            raise ValueError(f"backend không hợp lệ: {backend}")

    def log_scalars(self, metrics: dict, step: int):
        """metrics: {"loss/D": 0.5, "loss/G": 1.2, ...}"""
        if self.backend == "wandb":
            self.wandb.log(metrics, step=step)
        else:
            for k, v in metrics.items():
                self.writer.add_scalar(k, v, step)

    @torch.no_grad()
    def log_image_grid(self, images: torch.Tensor, step: int, tag="samples",
                       nrow=8, value_range=(-1, 1), save_local=True):
        """
        images      : tensor (N, C, H, W). Mặc định giả sử ảnh ở [-1, 1] (đầu ra Tanh).
                      Nếu model của bạn ra [0, 1] (vd VAE dùng Sigmoid) thì truyền value_range=(0, 1).
        save_local  : đồng thời lưu file PNG vào sample_dir (tiện chèn vào báo cáo).
        """
        grid = vutils.make_grid(images.detach().float().cpu(), nrow=nrow,
                                normalize=True, value_range=value_range)  # (3, H, W) trong [0, 1]

        if save_local:
            vutils.save_image(grid, os.path.join(self.sample_dir, f"{tag}_{step:07d}.png"))

        if self.backend == "wandb":
            self.wandb.log({tag: self.wandb.Image(grid.permute(1, 2, 0).numpy())}, step=step)
        else:
            self.writer.add_image(tag, grid, step)

    def finish(self):
        if self.backend == "wandb":
            self.wandb.finish()
        else:
            self.writer.close()


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------
def _unwrap(model):
    """Bỏ lớp bọc DataParallel / torch.compile để state_dict không bị tiền tố lạ."""
    if hasattr(model, "_orig_mod"):
        model = model._orig_mod
    if hasattr(model, "module"):
        model = model.module
    return model


def save_checkpoint(path, models: dict, optimizers: dict = None, schedulers: dict = None,
                    epoch: int = 0, step: int = 0, config: dict = None, extra: dict = None):
    """
    models     : {"G": netG, "D": netD}   hoặc {"vae": vae}   hoặc {"unet": unet, "ema": ema}
    optimizers : {"G": optG, "D": optD}
    extra      : dict tuỳ ý (vd fixed_noise để vẽ lại đúng batch nhiễu cũ)
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    state = {
        "models": {k: _unwrap(m).state_dict() for k, m in models.items()},
        "optimizers": {k: o.state_dict() for k, o in (optimizers or {}).items()},
        "schedulers": {k: s.state_dict() for k, s in (schedulers or {}).items()},
        "epoch": epoch,
        "step": step,
        "config": config,
        "extra": extra,
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    # Ghi ra file tạm rồi đổi tên: nếu Colab ngắt giữa chừng thì file cũ không bị hỏng
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_checkpoint(path, models: dict, optimizers: dict = None, schedulers: dict = None,
                    map_location="cpu", restore_rng=True):
    """
    Model phải được tạo (và chuyển lên device) TRƯỚC khi gọi hàm này.
    Trả về dict gồm epoch, step, config, extra để vòng train chạy tiếp.
    """
    # weights_only=False vì checkpoint có chứa RNG state của numpy/python (PyTorch >= 2.6 mặc định True)
    state = torch.load(path, map_location=map_location, weights_only=False)

    for k, m in models.items():
        _unwrap(m).load_state_dict(state["models"][k])
    for k, o in (optimizers or {}).items():
        o.load_state_dict(state["optimizers"][k])
    for k, s in (schedulers or {}).items():
        s.load_state_dict(state["schedulers"][k])

    if restore_rng and state.get("rng"):
        rng = state["rng"]
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"].cpu())
        if rng["cuda"] is not None and torch.cuda.is_available():
            try:
                torch.cuda.set_rng_state_all([s.cpu() for s in rng["cuda"]])
            except RuntimeError:
                print("[load_checkpoint] Số GPU khác lúc lưu, bỏ qua CUDA RNG state.")

    return {"epoch": state["epoch"], "step": state["step"],
            "config": state["config"], "extra": state["extra"]}
