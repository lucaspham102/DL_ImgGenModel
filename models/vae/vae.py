"""
models/vae/vae.py - Cài đặt kiến trúc Variational Autoencoder (VAE) cho dữ liệu ảnh 64x64.

Cơ sở lý thuyết & tài liệu tham khảo:
    Bài báo gốc: "Auto-Encoding Variational Bayes" (Kingma & Welling, 2013).
    
    1. Encoder q_phi(z|x):
       Mô hình hóa phân phối hậu nghiệm gần đúng dưới dạng Gaussian phân phối chéo:
       q_phi(z|x) = N(z; mu(x), diag(sigma^2(x)))
       - mu: vector kỳ vọng (mean vector).
       - logvar = log(sigma^2): logarithm của phương sai, dùng logvar thay vì trực tiếp
         xuất sigma để đảm bảo tính ổn định số học và sigma = exp(0.5 * logvar) luôn dương.

    2. Reparameterization Trick:
       Để gradient có thể lan truyền ngược qua thao tác lấy mẫu ngẫu nhiên z ~ q_phi(z|x),
       tính ngẫu nhiên được tách sang biến phụ epsilon ~ N(0, I):
       z = mu + sigma * epsilon = mu + exp(0.5 * logvar) * epsilon
       Khi đó: dz/dmu = 1, dz/dsigma = epsilon.

    3. Decoder p_theta(x|z):
       Mô hình sinh nhận vector tiềm ẩn z và giải mã thành ảnh tái tạo x_hat qua mạng tích chập chuyển vị.
       Đầu ra sử dụng hàm kích hoạt Tanh() đưa giá trị pixel về [-1, 1], khớp với chuẩn hóa của dataloader.

    4. Hàm mục tiêu ELBO (Evidence Lower Bound):
       Loss = -ELBO = L_recon + beta * L_KL
       - L_recon: Negative log-likelihood (MSE giữa x và x_hat với giả định phân phối Gaussian).
       - L_KL: Khoảng cách Kullback-Leibler giữa q_phi(z|x) và tiên nghiệm chuẩn tắc p(z) ~ N(0, I):
         D_KL(q_phi(z|x) || p(z)) = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvEncoder(nn.Module):
    """
    Encoder mạng tích chập nén ảnh 64x64 về không gian tiềm ẩn z (latent_dim).

    Luồng xử lý không gian (Spatial Flow):
        Input: (B, in_channels, 64, 64)
        -> Conv1 (k=3, s=2, p=1): 64x64 -> 32x32 (kênh: 32)
        -> Conv2 (k=3, s=2, p=1): 32x32 -> 16x16 (kênh: 64)
        -> Conv3 (k=3, s=2, p=1): 16x16 -> 8x8   (kênh: 128)
        -> Conv4 (k=3, s=2, p=1): 8x8   -> 4x4   (kênh: 256)
        -> Flatten: 256 * 4 * 4 = 4096
        -> fc_mu, fc_logvar: 4096 -> latent_dim (128)
    """

    def __init__(self, in_channels: int = 3, latent_dim: int = 128) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim

        # 4 tầng tích chập giảm kích thước không gian: 64 -> 32 -> 16 -> 8 -> 4
        self.conv = nn.Sequential(
            # Tầng 1: (3, 64, 64) -> (32, 32, 32)
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            # Tầng 2: (32, 32, 32) -> (64, 16, 16)
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            # Tầng 3: (64, 16, 16) -> (128, 8, 8)
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # Tầng 4: (128, 8, 8) -> (256, 4, 4)
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # 2 nhánh tuyến tính dự đoán mu và log(sigma^2)
        self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(256 * 4 * 4, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.conv(x)
        h = torch.flatten(h, start_dim=1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class ConvDecoder(nn.Module):
    """
    Decoder mạng tích chập chuyển vị tái tạo ảnh 64x64 từ vector tiềm ẩn z.

    Luồng xử lý không gian (Spatial Flow):
        Input: z (B, latent_dim)
        -> Linear + LeakyReLU: latent_dim -> 256 * 4 * 4 = 4096
        -> Reshape: (B, 256, 4, 4)
        -> ConvTranspose1 (k=3, s=2, p=1, out_p=1): 4x4   -> 8x8   (kênh: 128)
        -> ConvTranspose2 (k=3, s=2, p=1, out_p=1): 8x8   -> 16x16 (kênh: 64)
        -> ConvTranspose3 (k=3, s=2, p=1, out_p=1): 16x16 -> 32x32 (kênh: 32)
        -> ConvTranspose4 (k=3, s=2, p=1, out_p=1): 32x32 -> 64x64 (kênh: out_channels)
        -> Tanh(): Dải giá trị [-1, 1] khớp với dataloader
    """

    def __init__(self, latent_dim: int = 128, out_channels: int = 3) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.out_channels = out_channels

        # Chiếu vector tiềm ẩn z về kích thước không gian nhỏ nhất 256 x 4 x 4
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256 * 4 * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # 4 tầng tích chập chuyển vị phóng to dần về ảnh 64x64
        self.deconv = nn.Sequential(
            # Tầng 1: (256, 4, 4) -> (128, 8, 8)
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # Tầng 2: (128, 8, 8) -> (64, 16, 16)
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            # Tầng 3: (64, 16, 16) -> (32, 32, 32)
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            # Tầng 4: (32, 32, 32) -> (out_channels, 64, 64)
            nn.ConvTranspose2d(32, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Tanh(),  # Đưa giá trị pixel về dải [-1, 1]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z)
        h = h.view(-1, 256, 4, 4)
        recon_x = self.deconv(h)
        return recon_x


class VAE(nn.Module):
    """
    Mô hình Variational Autoencoder (VAE) hoàn chỉnh cho dữ liệu ảnh 64x64.
    
    Tương thích với:
        - dataloader.py: Dải chuẩn hóa [-1, 1], kích thước 64x64, 3 kênh.
        - evaluate.py: Cung cấp phương thức sample(num_samples, device).
    """

    def __init__(self, in_channels: int = 3, latent_dim: int = 128) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim

        self.encoder = ConvEncoder(in_channels=in_channels, latent_dim=latent_dim)
        self.decoder = ConvDecoder(latent_dim=latent_dim, out_channels=in_channels)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Trích xuất tham số phân phối tiềm ẩn (mu, logvar) từ ảnh đầu vào."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Tái tạo ảnh từ vector tiềm ẩn z."""
        return self.decoder(z)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization Trick (Kingma & Welling, 2013).
        
        Tính toán:
            sigma = exp(0.5 * logvar)
            epsilon ~ N(0, I)
            z = mu + sigma * epsilon
        
        Cho phép gradient truyền ngược từ z về mu và logvar.
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        # Khi đánh giá / suy luận cố định, có thể sử dụng kỳ vọng mu để tái tạo sắc nét
        return mu

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass hoàn chỉnh của VAE:
            x -> Encoder -> (mu, logvar) -> Reparameterize -> z -> Decoder -> recon_x

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: (recon_x, mu, logvar)
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

    @staticmethod
    def loss_function(
        recon_x: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        kl_weight: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Tính Negative ELBO Loss cho VAE:
            Total_Loss = Recon_Loss (MSE) + kl_weight * KL_Divergence

        Công thức:
            Loss = Reconstruction_Loss + kl_weight * KL_Divergence
            - Reconstruction_Loss: Sai số bình phương trung bình (MSE) giữa ảnh gốc và ảnh tái tạo.
              Tính tổng trên không gian pixel và trung bình theo batch:
              MSE = sum((recon_x - x)^2) / B
            - KL_Divergence:
              D_KL = -0.5 * sum(1 + logvar - mu^2 - exp(logvar)) / B

        Args:
            recon_x (torch.Tensor): Ảnh tái tạo từ Decoder, kích thước (B, C, H, W).
            x (torch.Tensor): Ảnh gốc, kích thước (B, C, H, W).
            mu (torch.Tensor): Kỳ vọng phân phối tiềm ẩn, kích thước (B, latent_dim).
            logvar (torch.Tensor): Log-variance phân phối tiềm ẩn, kích thước (B, latent_dim).
            kl_weight (float): Hệ số beta (KL weight). Mặc định: 1.0.

        Returns:
            Dict[str, torch.Tensor]:
                - "loss": Tổng loss dùng trực tiếp cho loss.backward().
                - "recon_loss": Thành phần sai số tái tạo (MSE).
                - "kl_loss": Thành phần phân kỳ Kullback-Leibler.
        """
        batch_size = x.size(0)

        # 1. Reconstruction Loss: Tổng sai số pixel của mỗi ảnh, trung bình theo batch
        recon_loss = F.mse_loss(recon_x, x, reduction="sum") / batch_size

        # 2. KL Divergence: D_KL(q(z|x) || N(0, I)) theo công thức giải tích đóng của Kingma & Welling (2013)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size

        total_loss = recon_loss + kl_weight * kl_loss

        return {
            "loss": total_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
        }

    @torch.no_grad()
    def sample(self, num_samples: int, device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Giao diện chuẩn sinh ảnh phục vụ evaluate.py:
        Lấy mẫu z ~ N(0, I) và giải mã qua Decoder để sinh ảnh mới.

        Args:
            num_samples (int): Số lượng ảnh cần sinh.
            device (torch.device, optional): Thiết bị tính toán. Nếu None, tự động lấy theo model.

        Returns:
            torch.Tensor: Tensor ảnh sinh (num_samples, C, H, W) trong miền [-1, 1].
        """
        self.eval()
        if device is None:
            device = next(self.parameters()).device
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.decode(z)

    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """
        Tái tạo ảnh từ ảnh đầu vào bằng kỳ vọng mu (deterministic reconstruction),
        phục vụ việc trực quan hóa chất lượng mô hình hoặc log ảnh qua logger.
        """
        self.eval()
        mu, _ = self.encode(x)
        return self.decode(mu)


if __name__ == "__main__":
    batch_size = 16
    in_channels = 3
    latent_dim = 128
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"=== KIỂM THỬ TOÀN DIỆN VAE 64x64 TRÊN {device} ===")
    model = VAE(in_channels=in_channels, latent_dim=latent_dim).to(device)

    # 1. Kiểm tra kích thước tensor đầu vào/đầu ra
    dummy_input = (torch.rand(batch_size, in_channels, 64, 64, device=device) * 2.0) - 1.0
    recon, mu, logvar = model(dummy_input)

    print(f"[+] Encoder:")
    print(f"    - Ảnh gốc   : {dummy_input.shape}")
    print(f"    - Vector mu : {mu.shape} (kỳ vọng: [{batch_size}, {latent_dim}])")
    print(f"    - Logvar    : {logvar.shape} (kỳ vọng: [{batch_size}, {latent_dim}])")

    print(f"[+] Decoder (Tái tạo về 64x64):")
    print(f"    - Ảnh tái tạo: {recon.shape} (kỳ vọng: [{batch_size}, {in_channels}, 64, 64])")
    print(f"    - Miền giá trị: [{recon.min().item():.3f}, {recon.max().item():.3f}] (kỳ vọng nằm trong [-1, 1])")

    # 2. Kiểm tra hàm mất mát ELBO & lan truyền ngược Autograd
    loss_dict = model.loss_function(recon, dummy_input, mu, logvar, kl_weight=1.0)
    total_loss = loss_dict["loss"]
    total_loss.backward()

    # Xác nhận mọi tham số đều nhận được gradient hợp lệ (không bị None hoặc NaN/Inf)
    all_grads_ok = True
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is None or torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                all_grads_ok = False
                print(f"[!] Lỗi gradient tại: {name}")
                break
    print(f"[+] Kiểm tra Autograd backward: {'HOÀN HẢO (100% tham số có gradient hợp lệ)' if all_grads_ok else 'LỖI'}")
    print(f"    - Total Loss      : {total_loss.item():.4f}")
    print(f"    - Recon Loss (MSE): {loss_dict['recon_loss'].item():.4f}")
    print(f"    - KL Divergence   : {loss_dict['kl_loss'].item():.4f}")

    # 3. Kiểm tra hàm sample() cho evaluate.py
    sampled_imgs = model.sample(num_samples=8, device=device)
    print(f"[+] Hàm sample(8):")
    print(f"    - Kích thước ảnh sinh: {sampled_imgs.shape}")
    print(f"    - Miền giá trị       : [{sampled_imgs.min().item():.3f}, {sampled_imgs.max().item():.3f}]")

    # 4. Kiểm tra hàm reconstruct() phục vụ trực quan hóa
    recon_deterministic = model.reconstruct(dummy_input[:4])
    print(f"[+] Hàm reconstruct(4):")
    print(f"    - Kích thước: {recon_deterministic.shape}")

    # 5. Đếm số tham số
    enc_params = sum(p.numel() for p in model.encoder.parameters())
    dec_params = sum(p.numel() for p in model.decoder.parameters())
    print(f"[+] Tổng tham số:")
    print(f"    - Encoder : {enc_params:,}")
    print(f"    - Decoder : {dec_params:,}")
    print(f"    - Toàn VAE: {sum(p.numel() for p in model.parameters()):,}")
