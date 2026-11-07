# DL_ImgGenModel: Deep Learning Image Generation Models

> **Bài Tập Lớn Môn Học Sâu (Deep Learning)**  
> Nghiên cứu, cài đặt và so sánh 3 kiến trúc mô hình sinh ảnh kinh điển: **VAE (Variational Autoencoder)**, **DCGAN (Deep Convolutional GAN)**, và **DDPM (Denoising Diffusion Probabilistic Models)**.

---

## 📌 Mục Tiêu Dự Án

1. **Xây dựng & Huấn luyện:**
   - **VAE:** Học phân phối không gian tiềm ẩn (latent space) tối ưu hàm mục tiêu ELBO (Reconstruction Loss + KL Divergence).
   - **DCGAN:** Huấn luyện đối kháng (adversarial training) giữa Generator và Discriminator bằng các khối Convolution/Transposed Convolution.
   - **DDPM:** Quá trình khuếch tán thuận (thêm nhiễu Gaussian từng bước) và mô hình mạng U-Net học cách khử nhiễu ngược.
2. **Đánh giá & So sánh:**
   - **Định tính:** Trực quan hóa ảnh sinh từ nhiễu ngẫu nhiên qua các epoch, nội suy không gian tiềm ẩn (latent interpolation).
   - **Định lượng:** Tính toán chỉ số **FID (Fréchet Inception Distance)** trên tập Test để so sánh chất lượng và độ đa dạng giữa 3 mô hình.

---

## 🛠️ Cài Đặt Môi Trường

Khuyến nghị sử dụng **Python 3.10+**.

```powershell
# 1. Cài đặt các thư viện phụ thuộc
pip install -r requirements.txt

# 2. (Khuyến nghị nếu máy có GPU NVIDIA) Cài PyTorch hỗ trợ CUDA:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## 📊 Dữ Liệu & Quy Chuẩn Tiền Xử Lý (`dataloader.py`)

Dự án thiết kế một pipeline nạp dữ liệu dùng chung duy nhất tại [`dataloader.py`](./dataloader.py) nhằm đảm bảo cả 3 thành viên khi xây dựng VAE, GAN, DDPM đều sử dụng chung một chuẩn dữ liệu đầu vào.

### 1. Các tập dữ liệu hỗ trợ:

| Dataset | Mục đích | Số lượng ảnh | Kích thước | Phân chia Train / Val / Test |
| :--- | :--- | :--- | :--- | :--- |
| **CelebA** | Dữ liệu chính (Khuôn mặt người nổi tiếng) | 202,599 | $3 \times 64 \times 64$ | $162,770$ / $19,867$ / $19,962$ |
| **Fashion-MNIST** | Sanity Check (Kiểm tra nhanh mô hình) | 70,000 | $3 \times 64 \times 64$ *(đã nhân 3 kênh)* | $54,000$ / $6,000$ / $10,000$ |

> [!NOTE]
> **Điểm đặc biệt của Fashion-MNIST:** Ảnh gốc $1 \times 28 \times 28$ được tự động nhân 3 kênh RGB và resize lên $64 \times 64$. Điều này giúp toàn bộ kiến trúc mô hình (Conv, số kênh `in_channels=3`) được giữ nguyên 100%, có thể chuyển đổi qua lại giữa CelebA và Fashion-MNIST chỉ bằng 1 dòng lệnh mà không phải sửa code mô hình.

### 2. Quy chuẩn không gian màu & chuẩn hóa:
- **Ảnh đầu vào mô hình:** Chuẩn hóa về dải $[-1, 1]$ (phù hợp với hàm kích hoạt `Tanh` ở đầu ra của VAE/GAN và dải giá trị chuẩn của mô hình Diffusion).
- **Hàm phục hồi ảnh `denormalize(x)` / `to_01(x)`:** Chuyển tensor từ $[-1, 1] \rightarrow [0, 1]$ phục vụ việc lưu ảnh ra đĩa (`save_image`), trực quan hóa Matplotlib hoặc tính điểm FID.

### 3. Tối ưu hệ điều hành & Multi-processing:
- Mặc định `num_workers=0` trên Windows để chống lỗi nghẽn deadlock của `multiprocessing spawn`, tự động đặt `2` trên Linux/macOS.
- Tự động kích hoạt `pin_memory=True` khi phát hiện GPU CUDA.

---

## 🚀 Hướng Dẫn Sử Dụng

### 1. Kiểm tra DataLoader từ Command Line:

```powershell
# Chạy kiểm tra nhanh với Fashion-MNIST (Tải tự động 100% trong ~5s):
python dataloader.py fashion_mnist

# Chạy kiểm tra với CelebA:
python dataloader.py celeba

# Chạy kiểm tra cả hai bộ dữ liệu:
python dataloader.py
```
*Sau khi chạy, chương trình sẽ lưu ảnh lưới mẫu gồm 64 ảnh (`sample_fashion_mnist.png` hoặc `sample_celeba.png`) vào thư mục gốc để kiểm tra trực quan.*

### 2. Sử dụng trong code huấn luyện của từng thành viên:

```python
from dataloader import get_dataloaders, denormalize

# 1. Lấy DataLoaders chuẩn cho huấn luyện CelebA:
train_loader, val_loader, test_loader = get_dataloaders(
    name="celeba",
    batch_size=128,
    image_size=64
)

# 2. Sanity Check nhanh trên Fashion-MNIST:
train_loader, val_loader, test_loader = get_dataloaders("fashion_mnist")

# 3. Chạy thử nghiệm nhanh (chỉ lấy 2000 ảnh đầu để debug luồng train):
train_loader, _, _ = get_dataloaders("celeba", subset=2000)

# 4. Khi lưu ảnh dự đoán / tái tạo của mô hình:
# reconstructed: tensor đầu ra trong khoảng [-1, 1]
from torchvision.utils import save_image
save_image(denormalize(reconstructed[:64]), "epoch_1_sample.png", nrow=8)
```

---

## 📁 Cấu Trúc Thư Mục Dự Án

```text
DL_ImgGenModel/
├── data/                      # Thư mục chứa dữ liệu (tự động tạo, đã được bỏ qua trong .gitignore)
│   ├── FashionMNIST/          # Dữ liệu Fashion-MNIST
│   └── celeba/                # Dữ liệu ảnh CelebA (img_align_celeba/)
├── dataloader.py              # Module DataLoader dùng chung cho cả 3 mô hình
├── requirements.txt           # Danh sách các thư viện phụ thuộc
├── .gitignore                 # Bỏ qua data nặng, checkpoints .pth, cache và file rác
└── README.md                  # Tài liệu hướng dẫn dự án
```

---

## 🗓️ Lộ Trình Triển Khai (Roadmap)

- [x] Thiết kế pipeline dữ liệu chuẩn hóa (`dataloader.py`) dùng chung cho CelebA và Fashion-MNIST.
- [x] Cấu hình môi trường (`requirements.txt`, `.gitignore`).
- [ ] **Thành viên 1:** Xây dựng & huấn luyện **VAE** (Encoder, Decoder, Reparameterization, ELBO Loss).
- [ ] **Thành viên 2:** Xây dựng & huấn luyện **DCGAN** (Generator, Discriminator, Minimax GAN Loss).
- [ ] **Thành viên 3:** Xây dựng & huấn luyện **DDPM** (U-Net, Variance Schedule $\beta_t$, Denoising Loop).
- [ ] Đo đạc chỉ số FID trên tập Test, so sánh kết quả và hoàn thiện báo cáo.
