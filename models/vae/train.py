"""
models/vae/train.py - Script huấn luyện độc lập cho mô hình VAE.
Phụ trách: Thành viên Nhánh VAE + Dữ liệu & Overleaf

Cách chạy từ thư mục gốc của project:
    python models/vae/train.py --dataset fashion_mnist --epochs 5 --batch-size 128 --backend tensorboard
    python models/vae/train.py --dataset celeba --epochs 30 --batch-size 128 --backend wandb

Cách chạy trực tiếp khi đang ở trong thư mục models/vae:
    cd models/vae
    python train.py --dataset celeba --epochs 30 --backend wandb
"""

import argparse
import os
import sys
import time

# Tự động định vị thư mục gốc của dự án (DL_ImgGenModel) và đưa vào sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.optim as optim

from data.dataloader import get_dataloaders
from models.vae.vae import VAE
from utils.logger import Logger, load_checkpoint, save_checkpoint, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình VAE (Variational Autoencoder)")

    # Dữ liệu
    parser.add_argument("--dataset", type=str, default="fashion_mnist", choices=["celeba", "fashion_mnist"],
                        help="Bộ dữ liệu (celeba hoặc fashion_mnist)")
    parser.add_argument("--data-root", type=str, default=os.path.join(PROJECT_ROOT, "data"),
                        help="Đường dẫn đến thư mục chứa dữ liệu")
    parser.add_argument("--subset", type=int, default=None,
                        help="Số lượng mẫu lấy để kiểm tra nhanh luồng train (None = toàn bộ)")

    # Siêu tham số
    parser.add_argument("--epochs", type=int, default=30, help="Tổng số epoch huấn luyện")
    parser.add_argument("--batch-size", type=int, default=128, help="Kích thước batch")
    parser.add_argument("--lr", type=float, default=5e-4, help="Tốc độ học (Learning rate cho Adam)")
    parser.add_argument("--latent-dim", type=int, default=128, help="Số chiều của vector tiềm ẩn z")
    parser.add_argument("--kl-weight", type=float, default=1.0, help="Hệ số beta cho KL Divergence (beta-VAE)")
    parser.add_argument("--kl-warmup-epochs", type=int, default=10,
                        help="Số epoch tăng tuyến tính KL weight đến --kl-weight (0 = tắt warm-up)")
    parser.add_argument("--seed", type=int, default=42, help="Seed cố định tính ngẫu nhiên")

    # Logging & Checkpoints
    parser.add_argument("--backend", type=str, default="wandb", choices=["wandb", "tensorboard", "none"],
                        help="Backend ghi log")
    parser.add_argument("--project", type=str, default="DL_ImageGenModel", help="Tên project W&B")
    parser.add_argument("--entity", type=str, default="24022367-vietnam-national-university-hanoi", help="Tên team W&B")
    parser.add_argument("--run-name", type=str, default=None, help="Tên run hiển thị trên W&B")
    parser.add_argument("--log-interval", type=int, default=50, help="Chu kỳ ghi log metric theo step")
    parser.add_argument("--ckpt-dir", type=str, default=os.path.join(PROJECT_ROOT, "checkpoints"),
                        help="Thư mục lưu các file checkpoint .pt")
    parser.add_argument("--resume", type=str, default=None,
                        help="Đường dẫn tới file checkpoint .pt để train tiếp khi Colab bị ngắt")

    return parser.parse_args()


def train_one_epoch(model, dataloader, optimizer, device, epoch, step, logger, args):
    """Huấn luyện mô hình VAE qua 1 epoch."""
    model.train()
    total_loss, total_recon, total_kl = 0.0, 0.0, 0.0
    num_batches = len(dataloader)
    warmup_ratio = (
        1.0
        if args.kl_warmup_epochs == 0
        else min(1.0, epoch / args.kl_warmup_epochs)
    )
    effective_kl_weight = args.kl_weight * warmup_ratio

    t0 = time.time()
    for batch_idx, batch in enumerate(dataloader):
        images = batch[0] if isinstance(batch, (list, tuple)) else batch
        images = images.to(device)

        optimizer.zero_grad()

        # Forward pass qua VAE: x -> Encoder -> (mu, logvar) -> Reparameterize -> Decoder -> recon_x
        recon_images, mu, logvar = model(images)

        # Tăng dần beta trong các epoch đầu để posterior học ổn định hơn.
        loss_dict = model.loss_function(
            recon_images,
            images,
            mu,
            logvar,
            kl_weight=effective_kl_weight,
        )
        loss = loss_dict["loss"]

        # Lan truyền ngược (Backpropagation)
        loss.backward()

        # Kẹp gradient (gradient clipping) chống bùng nổ gradient
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()
        total_recon += loss_dict["recon_loss"].item()
        total_kl += loss_dict["kl_loss"].item()
        step += 1

        # Ghi log W&B / TensorBoard
        if step % args.log_interval == 0 and logger is not None:
            metrics = {
                "train/loss": loss.item(),
                "train/recon_loss": loss_dict["recon_loss"].item(),
                "train/recon_mse": loss_dict["recon_mse"].item(),
                "train/kl_loss": loss_dict["kl_loss"].item(),
                "train/kl_weight": effective_kl_weight,
                "train/lr": optimizer.param_groups[0]["lr"],
            }
            logger.log_scalars(metrics, step=step)

        # In tiến độ ra màn hình
        if (batch_idx + 1) % max(1, num_batches // 5) == 0 or (batch_idx + 1) == num_batches:
            elapsed = time.time() - t0
            print(f"Epoch [{epoch}/{args.epochs}] | Batch [{batch_idx + 1}/{num_batches}] | "
                  f"Loss: {loss.item():.2f} (Recon: {loss_dict['recon_loss'].item():.2f}, "
                  f"KL: {loss_dict['kl_loss'].item():.2f}, beta: {effective_kl_weight:.3f}) | "
                  f"Thời gian: {elapsed:.1f}s")

    avg_loss = total_loss / max(1, num_batches)
    avg_recon = total_recon / max(1, num_batches)
    avg_kl = total_kl / max(1, num_batches)

    return avg_loss, avg_recon, avg_kl, step


@torch.no_grad()
def evaluate(model, dataloader, device, args):
    """Đánh giá loss trên tập validation."""
    model.eval()
    val_loss, val_recon, val_kl = 0.0, 0.0, 0.0
    num_batches = len(dataloader)

    for batch in dataloader:
        images = batch[0] if isinstance(batch, (list, tuple)) else batch
        images = images.to(device)

        recon_images, mu, logvar = model(images)
        loss_dict = model.loss_function(recon_images, images, mu, logvar, kl_weight=args.kl_weight)

        val_loss += loss_dict["loss"].item()
        val_recon += loss_dict["recon_loss"].item()
        val_kl += loss_dict["kl_loss"].item()

    avg_loss = val_loss / max(1, num_batches)
    avg_recon = val_recon / max(1, num_batches)
    avg_kl = val_kl / max(1, num_batches)

    return avg_loss, avg_recon, avg_kl


def main():
    args = parse_args()
    if args.kl_weight < 0:
        raise ValueError("kl_weight phải >= 0")
    if args.kl_warmup_epochs < 0:
        raise ValueError("kl_warmup_epochs phải >= 0")
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Thiết bị huấn luyện VAE: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    # 1. Tải DataLoaders từ data/dataloader.py
    print(f"[*] Đang nạp dữ liệu '{args.dataset}' từ '{args.data_root}'...")
    train_loader, val_loader, _ = get_dataloaders(
        name=args.dataset,
        root=args.data_root,
        batch_size=args.batch_size,
        subset=args.subset,
        seed=args.seed,
    )
    print(f"[+] Dữ liệu sẵn sàng: {len(train_loader)} batches train | {len(val_loader)} batches val")

    # 2. Khởi tạo Mô hình VAE, Optimizer & Scheduler
    model = VAE(in_channels=3, latent_dim=args.latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    start_epoch = 1
    global_step = 0
    best_val_loss = float("inf")

    # 3. Nạp Checkpoint nếu có cờ --resume (Chống đứt đoạn trên Google Colab)
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"[*] Đang tải checkpoint từ: {args.resume}")
            info = load_checkpoint(
                args.resume,
                models={"vae": model},
                optimizers={"vae": optimizer},
                schedulers={"vae": scheduler},
                map_location=device,
            )
            start_epoch = info["epoch"] + 1
            global_step = info["step"]
            if info.get("extra") and "best_val_loss" in info["extra"]:
                best_val_loss = info["extra"]["best_val_loss"]
            print(f"[+] Khôi phục thành công! Bắt đầu từ epoch {start_epoch}, step {global_step} (Best val loss: {best_val_loss:.2f})")
        else:
            print(f"[!] Cảnh báo: File '{args.resume}' không tồn tại. Bắt đầu huấn luyện mới.")

    # 4. Cấu hình Logger (W&B / TensorBoard)
    if args.run_name is None:
        args.run_name = f"vae-{args.dataset}-z{args.latent_dim}-bs{args.batch_size}"

    logger = None
    if args.backend != "none":
        print(f"[*] Khởi tạo Logger ({args.backend}) | Project: {args.project} | Run: {args.run_name}")
        logger = Logger(
            backend=args.backend,
            project=args.project,
            entity=args.entity,
            run_name=args.run_name,
            config=vars(args),
        )

    # 5. Cố định vector nhiễu để theo dõi chất lượng sinh ảnh qua từng epoch
    sample_size = min(64, args.batch_size)
    fixed_noise = torch.randn(sample_size, args.latent_dim, device=device)
    first_batch = next(iter(val_loader))
    fixed_real = (first_batch[0] if isinstance(first_batch, (list, tuple)) else first_batch)[:sample_size].to(device)

    if logger is not None:
        logger.log_image_grid(fixed_real, step=0, tag="val/ground_truth", value_range=(-1, 1))

    # 6. Vòng lặp huấn luyện chính
    print(f"\n{'='*30} BẮT ĐẦU HUẤN LUYỆN VAE ({args.epochs} EPOCHS) {'='*30}")
    os.makedirs(args.ckpt_dir, exist_ok=True)

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            t_epoch_start = time.time()

            # Train 1 epoch
            train_loss, train_recon, train_kl, global_step = train_one_epoch(
                model=model,
                dataloader=train_loader,
                optimizer=optimizer,
                device=device,
                epoch=epoch,
                step=global_step,
                logger=logger,
                args=args,
            )

            # Validation
            val_loss, val_recon, val_kl = evaluate(model, val_loader, device, args)
            scheduler.step(val_loss)

            epoch_time = time.time() - t_epoch_start
            print(f"[Epoch {epoch:02d}/{args.epochs:02d} - {epoch_time:.1f}s] "
                  f"Train Loss: {train_loss:.2f} (Recon: {train_recon:.2f}, KL: {train_kl:.2f}) | "
                  f"Val Loss: {val_loss:.2f} (Recon: {val_recon:.2f}, KL: {val_kl:.2f})")

            # Ghi log metrics cấp epoch
            if logger is not None:
                logger.log_scalars({
                    "epoch/train_loss": train_loss,
                    "epoch/val_loss": val_loss,
                    "epoch/val_recon": val_recon,
                    "epoch/val_kl": val_kl,
                }, step=epoch)

                with torch.no_grad():
                    gen_samples = model.decode(fixed_noise)
                    recon_samples = model.reconstruct(fixed_real)

                logger.log_image_grid(gen_samples, step=epoch, tag="epoch/samples_generated", value_range=(-1, 1))
                logger.log_image_grid(recon_samples, step=epoch, tag="epoch/reconstructed", value_range=(-1, 1))

            # Lưu checkpoint last (cho từng epoch)
            last_ckpt_path = os.path.join(args.ckpt_dir, f"vae_{args.dataset}_last.pt")
            save_checkpoint(
                path=last_ckpt_path,
                models={"vae": model},
                optimizers={"vae": optimizer},
                schedulers={"vae": scheduler},
                epoch=epoch,
                step=global_step,
                config=vars(args),
                extra={"best_val_loss": best_val_loss, "fixed_noise": fixed_noise.cpu()},
            )

            # Lưu checkpoint best (khi val loss thấp nhất)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_ckpt_path = os.path.join(args.ckpt_dir, f"vae_{args.dataset}_best.pt")
                save_checkpoint(
                    path=best_ckpt_path,
                    models={"vae": model},
                    optimizers={"vae": optimizer},
                    schedulers={"vae": scheduler},
                    epoch=epoch,
                    step=global_step,
                    config=vars(args),
                    extra={"best_val_loss": best_val_loss, "fixed_noise": fixed_noise.cpu()},
                )
                print(f"[*] Đã cập nhật Checkpoint tốt nhất: {best_ckpt_path} (Val Loss: {best_val_loss:.2f})")

    except KeyboardInterrupt:
        print("\n[!] Đã dừng huấn luyện theo yêu cầu. Checkpoint an toàn tại 'last.pt'.")

    finally:
        if logger is not None:
            logger.finish()
        print("\n[+] Quá trình hoàn tất.")


if __name__ == "__main__":
    main()
