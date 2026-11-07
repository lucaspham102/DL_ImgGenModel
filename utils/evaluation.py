"""
Pipeline đánh giá dùng chung cho VAE, DCGAN, ĐDPM

Tiêu chí:
1. FID: Chất lượng và độ chân thực của ảnh sinh
2. Generation Speed( ms/img và img/sec): Chi phí tính toán và tốc độ suy luận
3. Diversit( LPIPS, Pairwise L2, Pixel Std): Độ đa dạng, phát hiện Mode collapse
"""

import time
from typing import Callable, Dict, Optional
import torch
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")

def ensure_01_float(images: torch.Tensor) -> torch.Tensor:
    """
    Đưa tensor về miền float [0, 1], do sai số dấu phẩy động trên GPU nên cần đặt tolerance là -0.01 giúp phân biệt chính xác 100%
    if images.min() < -0.01:
    """
    if images.min() < -0.01:
        return (images.clamp(-1.0, 1.0) + 1.0) / 2.0
    return images.clamp(0.0, 1.0)

def ensure_01_unit8_rgb(images: torch.Tensor) -> torch.Tensor:
    """
    Chuẩn hóa tensor ảnh bất kỳ về định dạng chuẩn mà mạng InceptionV3 của TorchMetrics yêu cầu:
    - Data type: torch.unit8 trong đoạn [0, 255] 
    (*) Số nguyên k dấu chỉ nhận từ 0 đến 255, ngoài khoảng sẽ bị tràn số làm điểm ảnh đang trắng biến thành đen sì và làm lệch hoàn toàn chỉ số FID
    - Kích thước: (B, 3, H, W)
    """
    images = ensure_01_float(images)

    if images.shape[1] == 1:
        images = images.repeat(1, 3, 1, 1)

    return (images * 255.0).to(torch.uint8)

@torch.no_grad()
def compute_fid_score(
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
    device: torch.device,
    batch_size: int = 64,
) -> float:
    """
    1. Tính chỉ số FID giữa tập ảnh thật và tập ảnh sinh. 
    Sử dụng tầng đặc trưng 2048 chiều của mạng InceptionV3. 
    FID càng thấp thì phân phối ảnh sinh càng giống thật.
    """
    from torchmetrics.image.fid import FrechetInceptionDistance

    fid_metrics = FrechetInceptionDistance(feature=2048, normalize=False).to(device)

    #Real pics
    for i in range(0, len(real_images), batch_size):
        batch_real = ensure_01_unit8_rgb(real_images[i : i + batch_size]).to(device)
        fid_metrics.update(batch_real, real=True)

    #Fake pics
    for i in range(0, len(fake_images), batch_size):
        batch_fake = ensure_01_unit8_rgb(fake_images[i : i + batch_size]).to(device)
        fid_metrics.update(batch_fake, real=False)

    return float(fid_metrics.compute().item())

@torch.no_grad()
def measure_generation_speed(
    sample_fn: Callable[[int, torch.device], torch.Tensor],
    num_samples: int = 128,
    batch_size: int = 32,
    device: torch.device = torch.device("cpu"),
    warmup_batches: int = 1,
) -> Dict[str, float]:
    """
    2. Đo tốc độ sinh ảnh thực tế trên phần cứng hiện tại.
    Có bước chạy làm nóng và đồng bộ hóa luồng CUDA trước/ sau khi bấm giờ để tránh sai số do thực thi bất đồng bộ trên GPU
    """
    for _ in range(warmup_batches):
        _ = sample_fn(min(batch_size, 8), device)

    if device.type == "cuda":
        torch.cuda.synchronize()

    start_time = time.perf_counter()
    generated = 0
    while generated < num_samples:
        cur_bs = min(batch_size, num_samples -generated)
        _ = sample_fn(cur_bs, device)
        generated += cur_bs

    if device.type == "cuda":
        torch.cuda.synchronize()

    total_sec = time.perf_counter() - start_time
    ms_per_img = (total_sec * 1000.0) / num_samples
    img_per_sec = num_samples / total_sec if total_sec > 0 else 0.0

    return {
        "total_time_sec": round(total_sec, 4),
        "ms_per_image": round(ms_per_img, 3),
        "image_per_sec": round(img_per_sec, 2),
    }


@torch.no_grad()
def compute_diversity_score(
    fake_images: torch.Tensor,
    num_pairs: int = 500,
    device: torch.device = torch.device("cpu"),
    use_lpips: bool = True
) -> Dict[str, Optional[float]]:
    """
    3. Đánh giá độ đa dạng giữa các mẫu ảnh sinh ra để phát hiện Mode collapse.
    Lấy ngẫu nhiên 'num_pairs' cặp ảnh khác nhau( x_i, x_j) trong tập ảnh sinh và đo:
        - Pairwise LPIPS: Khoảng cách cảm nhận trên đặc trưng mạng học sâu:
        - Pairwise L2: Khoảng cách Euclid trung bình trên không gian pixel.
        - Pixel Std: Độ lệch chuẩn trung bình giữa các ảnh tại từng vị trí điểm ảnh>
    Điểm càng cao chứng tỏ các ảnh sinh ra càng phong phú, không bị lặp mẫu.
    """
    imgs_01 = ensure_01_float(fake_images)
    n = imgs_01.shape[0]
    if n < 2:
        raise ValueError("Cần ít nhất 2 ảnh sinh để tính độ đa dạng")

    #Random num_pairs (idx1 != idx2)
    idx1 = torch.randint(0, n, (num_pairs,))
    idx2 = torch.randint(0, n, (num_pairs,))
    idx2 = torch.where(idx1 == idx2, (idx2 + 1) % n, idx2)

    p1 = imgs_01[idx1].to(device)
    p2 = imgs_01[idx2].to(device)

    #3.1 Euclid
    flat1 = p1.view(num_pairs, -1)
    flat2 = p2.view(num_pairs, -1)
    l2_dist = torch.norm(flat1 - flat2, p=2, dim=1).mean().item()

    #3.2 Độ lệch chuẩn liên mẫu
    pixel_std = imgs_01.std(dim=0).mean().item()

    metrics: Dict[str, Optional[float]] = {
        "pairwise_L2": round(l2_dist, 4),
        "pixel_std": round(pixel_std, 4),
        "pairwise_lpips": None
    }

    #3.3 Khoảng cách cảm nhận LPIPS (req input [-1,1], 3 kênh)
    if use_lpips:
        try:
            import lpips

            loss_fn = lpips.LPIPS(net="squeeze", verbose=False).to(device)
            in1 = p1 * 2.0 - 1.0
            in2 = p2 * 2.0 - 1.0
            if in1.shape[1] == 1:
                in1 = in1.repeat(1, 3, 1, 1)
                in2 = in2.repeat(1, 3, 1, 1)
            lpips_dist = loss_fn(in1, in2).mean().item()
            metrics["pairwise_lpips"] = round(lpips_dist, 4)
        except Exception as e:
            print(f"[!] Bỏ qua LPIPS ({e.__class__.__name__}: {e})")
    return metrics

@torch.no_grad()
def evaluate_model(
    model_name: str,
    sample_fn: Callable[[int, torch.device], torch.Tensor],
    real_dataloader: torch.utils.data.DataLoader,
    num_samples: int = 1000,
    batch_size: int = 64,
    device: torch.device = torch.device("cpu"),
    compute_fid: bool = True,
    use_lpips: bool = True,
) -> Dict[str, object]:
    """
    Hàm điều phối chính: Nhận Dataloader tập test, rút "num_samples" ảnh thật, sinh "num_samples" ảnh giả từ mô hình và trả về bảng kết quả tổng hợp.
    """
    print(f"\n========== ĐÁNH GIÁ MÔ HÌNH: {model_name.upper()} ==========")

    # B1: đo tốc độ sinh sảnh
    print("[1/3] Đang đo tốc độ sinh ảnh...")
    speed_res = measure_generation_speed(
        sample_fn=sample_fn,
        num_samples=min(num_samples, 128),
        batch_size=batch_size,
        device=device,
    )

    # B2: Sinh ảnh
    print(f"[2/3] Đang thu thập {num_samples} ảnh thật và sinh {num_samples} ảnh mới...")
    real_list, fake_list = [], []
    collected = 0
    for x_real, _ in real_dataloader:
        real_list.append(x_real.cpu())
        collected += x_real.shape[0]
        if collected >= num_samples:
            break
    real_images = torch.cat(real_list, dim=0)[:num_samples]

    generated = 0
    while generated < num_samples:
        cur_bs = min(batch_size, num_samples - generated)
        fake_batch = sample_fn(cur_bs, device).cpu()
        fake_list.append(fake_batch)
        generated += cur_bs
    fake_images = torch.cat(fake_list, dim=0)[:num_samples]

    # B3: Tính độ đa dạng và FID
    print("[3/3] Đang tính toán Độ đa dạng (Diversity) và FID...")
    diversity_res = compute_diversity_score(
        fake_images=fake_images,
        num_pairs=min(500, num_samples),
        device=device,
        use_lpips=use_lpips,
    )

    fid_score = None
    if compute_fid:
        fid_score = round(
            compute_fid_score(
                real_images=real_images,
                fake_images=fake_images,
                device=device,
                batch_size=batch_size,
            ),
            3,
        )
    return {
        "model": model_name,
        "num_samples": num_samples,
        "FID": fid_score,
        "speed": speed_res,
        "diversity": diversity_res
    }

