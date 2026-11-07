"""
evaluate.py - Script chạy pipeline đánh giá cho VAE, DCGAN, DDPM.
Chạy kiểm tra nhanh với dữ liệu thật từ dataloader.py:
    python evaluate.py --dataset fashion_mnist --num-samples 128 --test-dummy
"""

import argparse
import json
import os
import torch

from data.dataloader import get_dataloaders
from utils.evaluation import evaluate_model


def parse_args():
    parser = argparse.ArgumentParser(description="Đánh giá mô hình sinh ảnh (FID, Speed, Diversity)")
    parser.add_argument("--model", type=str, default="ddpm", choices=["vae", "dcgan", "ddpm"])
    parser.add_argument("--dataset", type=str, default="fashion_mnist", choices=["celeba", "fashion_mnist"])
    parser.add_argument("--ckpt", type=str, default=None, help="Đường dẫn tới file checkpoint (.pt/.pth)")
    parser.add_argument("--num-samples", type=int, default=256, help="Số lượng ảnh dùng để đánh giá")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--no-fid", action="store_true", help="Bỏ qua tính FID khi chỉ muốn test nhanh tốc độ")
    parser.add_argument("--no-lpips", action="store_true", help="Bỏ qua tính LPIPS")
    parser.add_argument("--test-dummy", action="store_true", help="Chạy thử pipeline bằng bộ sinh nhiễu giả lập")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Thiết bị đánh giá: {device}")

    # Tải test_loader từ data/dataloader.py của Thành viên 1
    _, _, test_loader = get_dataloaders(
        name=args.dataset,
        root="./data/raw",
        batch_size=args.batch_size,
    )

    if args.test_dummy:
        # Bộ sinh giả lập trả về tensor (B, 3, 64, 64) trong miền [-1, 1] giống hệt mô hình thật
        sample_fn = lambda n, dev: torch.rand(n, 3, 64, 64, device=dev) * 2.0 - 1.0
        model_label = f" dummy_{args.model}"
    else:
        raise NotImplementedError(
            "Khi các thành viên code xong VAE/DCGAN/DDPM, nạp checkpoint và gán `sample_fn = model.sample` tại đây."
        )

    results = evaluate_model(
        model_name=model_label.strip(),
        sample_fn=sample_fn,
        real_dataloader=test_loader,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        device=device,
        compute_fid=not args.no_fid,
        use_lpips=not args.no_lpips,
    )

    print("\n[+] KẾT QUẢ ĐÁNH GIÁ:")
    print(json.dumps(results, indent=2, ensure_ascii=False))

    os.makedirs("outputs", exist_ok=True)
    out_file = os.path.join("outputs", f"eval_{model_label.strip()}_{args.dataset}.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[+] Đã lưu kết quả tại: {out_file}")


if __name__ == "__main__":
    main()