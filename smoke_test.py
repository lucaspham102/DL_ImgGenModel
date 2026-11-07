"""
Chạy thử môi trường GPU + kiểm tra utils.py hoạt động đúng.
Dùng dữ liệu ngẫu nhiên nên không cần tải dataset; mục đích là test pipeline, không phải train thật.

Cách chạy:
    python smoke_test.py --backend wandb --project gen-models-de4 --entity <ten-team>
    python smoke_test.py --backend tensorboard
"""
import argparse
import time

import torch
import torch.nn as nn

from utils.logger import Logger, load_checkpoint, save_checkpoint, set_seed


def tiny_G(nz=100):  # 100 -> 3x32x32
    return nn.Sequential(
        nn.ConvTranspose2d(nz, 128, 4, 1, 0, bias=False), nn.BatchNorm2d(128), nn.ReLU(True),  # 4x4
        nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(True),   # 8x8
        nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(True),    # 16x16
        nn.ConvTranspose2d(32, 3, 4, 2, 1, bias=False), nn.Tanh(),                             # 32x32
    )


def tiny_D():  # 3x32x32 -> logit
    return nn.Sequential(
        nn.Conv2d(3, 32, 4, 2, 1), nn.LeakyReLU(0.2, True),
        nn.Conv2d(32, 64, 4, 2, 1, bias=False), nn.BatchNorm2d(64), nn.LeakyReLU(0.2, True),
        nn.Conv2d(64, 1, 8, 1, 0),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", default="wandb", choices=["wandb", "tensorboard"])
    p.add_argument("--project", default="gen-models-de4")
    p.add_argument("--entity", default=None)
    p.add_argument("--steps", type=int, default=50)
    args = p.parse_args()

    # 1. Kiểm tra GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))
    print(f"PyTorch {torch.__version__}")

    set_seed(42)
    nz, bs = 100, 64
    G, D = tiny_G(nz).to(device), tiny_D().to(device)
    optG = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    optD = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
    crit = nn.BCEWithLogitsLoss()
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    logger = Logger(args.backend, project=args.project, entity=args.entity,
                    run_name="smoke-test", config={"nz": nz, "bs": bs, "lr": 2e-4})

    # 2. Vài bước train giả để test log scalar + đo tốc độ
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for step in range(1, args.steps + 1):
        real = torch.rand(bs, 3, 32, 32, device=device) * 2 - 1
        fake = G(torch.randn(bs, nz, 1, 1, device=device))

        lossD = crit(D(real).view(-1), torch.ones(bs, device=device)) + \
                crit(D(fake.detach()).view(-1), torch.zeros(bs, device=device))
        optD.zero_grad(); lossD.backward(); optD.step()

        lossG = crit(D(fake).view(-1), torch.ones(bs, device=device))  # non-saturating loss
        optG.zero_grad(); lossG.backward(); optG.step()

        if step % 10 == 0:
            logger.log_scalars({"loss/D": lossD.item(), "loss/G": lossG.item()}, step)
    if device == "cuda":
        torch.cuda.synchronize()
    print(f"{args.steps} bước: {time.time() - t0:.2f}s")

    # 3. Test log image grid
    G.eval()
    with torch.no_grad():
        samples = G(fixed_noise)
    logger.log_image_grid(samples, step=args.steps, tag="samples")

    # 4. Test checkpoint: lưu, tải vào model mới, so sánh đầu ra
    save_checkpoint("checkpoints/smoke.pt", {"G": G, "D": D}, {"G": optG, "D": optD},
                    epoch=0, step=args.steps, extra={"fixed_noise": fixed_noise.cpu()})

    G2, D2 = tiny_G(nz).to(device), tiny_D().to(device)
    optG2 = torch.optim.Adam(G2.parameters(), lr=2e-4, betas=(0.5, 0.999))
    optD2 = torch.optim.Adam(D2.parameters(), lr=2e-4, betas=(0.5, 0.999))
    info = load_checkpoint("checkpoints/smoke.pt", {"G": G2, "D": D2}, {"G": optG2, "D": optD2},
                           map_location=device)
    G2.eval()
    with torch.no_grad():
        samples2 = G2(info["extra"]["fixed_noise"].to(device))

    ok = torch.allclose(samples, samples2)
    print(f"Checkpoint round-trip: {'OK' if ok else 'LỖI'} (step={info['step']})")
    logger.finish()
    assert ok, "Đầu ra sau khi load khác trước khi lưu!"


if __name__ == "__main__":
    main()
