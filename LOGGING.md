# Hướng dẫn Logging (W&B) & Tiện ích dùng chung

Tài liệu này hướng dẫn cả nhóm log quá trình huấn luyện lên W&B và dùng các hàm trong `utils/logger.py`.
Người phụ trách: Thành viên 2.

## 1. Thông tin W&B của nhóm

| Mục | Giá trị |
|---|---|
| Entity (team) | `24022367-vietnam-national-university-hanoi` |
| Project | `DL_ImageGenModel` |
| Link project | https://wandb.ai/24022367-vietnam-national-university-hanoi/DL_ImageGenModel |

> Gõ đúng từng ký tự, kể cả chữ hoa. Sai tên project thì W&B sẽ tự tạo project mới thay vì báo lỗi, và log bị tách ra hai nơi.

## 2. Thiết lập lần đầu (mỗi người làm một lần)

1. Tạo tài khoản tại https://wandb.ai (nên dùng email trường) và **chấp nhận lời mời vào team** trong email.
2. Lấy API key tại https://wandb.ai/authorize (bấm **Generate** nếu chưa có).
3. Đăng nhập:

**Máy cá nhân**

```bash
python -m pip install wandb
python -m wandb login      # dán API key, ký tự có thể không hiện ra, cứ Enter
```

**Google Colab**

- Bấm biểu tượng chìa khóa (**Secrets**) ở thanh bên trái, thêm secret tên `WANDB_API_KEY`, bật *Notebook access*.
- Trong notebook:

```python
!pip install -q wandb
from google.colab import userdata
import wandb
wandb.login(key=userdata.get("WANDB_API_KEY"))
```

> **Không** dán API key vào code, không gửi vào nhóm chat, không commit lên GitHub (repo đang Public). Mỗi người dùng key của riêng mình.

## 3. Kiểm tra môi trường

Chạy từ thư mục gốc của repo:

```bash
python smoke_test.py --backend wandb --project DL_ImageGenModel --entity 24022367-vietnam-national-university-hanoi --steps 20
```

Kết quả đúng:

- Dòng đầu in `Device: cuda (...)` nếu có GPU (`cpu` nếu không).
- Có link `wandb.ai/.../runs/...` để xem run.
- Dòng cuối: `Checkpoint round-trip: OK`.

Script dùng dữ liệu ngẫu nhiên nên ảnh sinh ra chỉ là nhiễu — đó là bình thường.

## 4. Dùng trong code huấn luyện

```python
from utils.logger import Logger, save_checkpoint, load_checkpoint, set_seed

set_seed(42)

logger = Logger(
    "wandb",
    project="DL_ImageGenModel",
    entity="24022367-vietnam-national-university-hanoi",
    run_name="dcgan-fmnist-bs128",     # xem quy ước tên ở mục 5
    config=cfg,                        # dict hyperparameters, W&B tự lưu lại
)

for epoch in range(num_epochs):
    for x, _ in dataloader:
        ...
        step += 1
        if step % 100 == 0:
            logger.log_scalars({"loss/D": loss_D.item(), "loss/G": loss_G.item()}, step)

    # Lưới ảnh sinh từ một batch nhiễu CỐ ĐỊNH để so sánh được qua các epoch
    with torch.no_grad():
        samples = G(fixed_noise)
    logger.log_image_grid(samples, step=step, tag="samples")

    save_checkpoint(f"checkpoints/dcgan_epoch{epoch}.pt",
                    models={"G": G, "D": D},
                    optimizers={"G": opt_G, "D": opt_D},
                    epoch=epoch, step=step, config=cfg)

logger.finish()
```

### Lưu ý theo từng mô hình

- **Ảnh đầu ra ở [0, 1]** (vd VAE dùng Sigmoid): truyền `value_range=(0, 1)` vào `log_image_grid`, nếu không ảnh sẽ sai màu. Mặc định là `(-1, 1)` cho đầu ra Tanh.
- **Nhiều mạng / có EMA** (vd DDPM): truyền hết vào dict, vd `models={"unet": unet, "ema": ema}`.
- **Tên khóa trong dict** (`"G"`, `"D"`, `"vae"`...) phải giống nhau giữa lúc lưu và lúc tải.

### Train tiếp khi Colab bị ngắt

Tạo model và optimizer, chuyển lên device **trước**, rồi:

```python
info = load_checkpoint("checkpoints/dcgan_epoch10.pt",
                       models={"G": G, "D": D},
                       optimizers={"G": opt_G, "D": opt_D},
                       map_location=device)
start_epoch, step = info["epoch"] + 1, info["step"]
```

## 5. Quy ước đặt tên run

`<model>-<dataset>-<ghi chú>`, viết thường, nối bằng dấu gạch ngang. Ví dụ:

- `vae-fmnist-z64`
- `dcgan-fmnist-bs128`
- `ddpm-fmnist-t1000`
- `dcgan-celeba64-lr1e-4`

Quy ước tên metric để vẽ chung biểu đồ: loss đặt dạng `loss/<tên>` (vd `loss/D`, `loss/recon`, `loss/kl`), ảnh dùng tag `samples`.

## 6. Xem log

- Bấm link `View run ...` in ra trong terminal, hoặc vào link project ở mục 1.
- Trang project hiển thị biểu đồ của **mọi run chồng lên nhau** — tick/bỏ tick run ở cột trái để so sánh.
- Trong một run: **Charts** (loss, ảnh), **System** (GPU/CPU/RAM — dùng cho phần so sánh chi phí tính toán), **Logs** (output terminal), **Overview** (config, máy, thời gian chạy).

## 7. File sinh ra khi chạy (không commit)

| Thư mục | Nội dung |
|---|---|
| `checkpoints/` | trọng số `.pt` |
| `outputs/samples/` | ảnh PNG của lưới ảnh mẫu |
| `wandb/` | log cục bộ của W&B |
| `runs/` | log TensorBoard (nếu dùng backend `tensorboard`) |

Các thư mục này đã có trong `.gitignore`. Luôn chạy `git status` trước khi commit để chắc không đẩy nhầm lên.

## 8. Dự phòng: TensorBoard

Khi mất mạng hoặc test nhanh, đổi backend, phần code còn lại giữ nguyên:

```python
logger = Logger("tensorboard", run_name="dcgan-test")
```

Xem bằng `python -m tensorboard.main --logdir runs` rồi mở http://localhost:6006.
