"""
dataloader.py - Pipeline nạp dữ liệu dùng chung cho cả 3 mô hình (VAE, DCGAN, DDPM).

Đã thiết kế:
    - Sử dụng chính thức metadata phân chia `list_eval_partition.txt` của CelebA:
        + partition == "0" -> Train (162,770 ảnh)
        + partition == "1" -> Val   (19,867 ảnh)
        + partition == "2" -> Test  (19,962 ảnh)

Fashion-MNIST: dùng để sanity check nhanh -> lưu tại ./data/FashionMNIST
  -> mặc định được resize 64x64 và nhân ra 3 kênh để dùng CHUNG kiến trúc,
     không phải sửa code mô hình khi chuyển qua lại.

Ảnh chuẩn hóa về [-1, 1] (hợp với tanh của VAE/GAN và DDPM).
Hàm to_01() / denormalize() chuyển về [0, 1] để lưu ảnh, tính FID hoặc dùng BCE loss.

Ví dụ:
    from data.dataloader import get_dataloaders, denormalize
    train_dl, val_dl, test_dl = get_dataloaders("celeba")              # CelebA 64x64
    train_dl, val_dl, test_dl = get_dataloaders("fashion_mnist")       # sanity check
    train_dl, _, _ = get_dataloaders("celeba", subset=2000)            # chạy thử nhanh
"""
import glob
import os
import subprocess
import sys
from typing import Dict, List, Optional, Tuple
from PIL import Image

import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets, transforms
from torchvision.utils import save_image

SEED = 42

# Số lượng ảnh chuẩn theo metadata chính thức của CelebA
CELEBA_EXPECTED_COUNTS = {
    "0": 162770,  # Train
    "1": 19867,   # Validation
    "2": 19962,   # Test
}


# ---------------------------------------------------------------------------
# SỬA LỖI 3: Định nghĩa Custom Exception CorruptedImageError
# ---------------------------------------------------------------------------
class CorruptedImageError(Exception):
    """Ngoại lệ khi phát hiện file ảnh bị lỗi, hỏng hoặc không thể đọc."""
    pass


# ---------------------------------------------------------------------------
# Transform & Phục hồi ảnh
# ---------------------------------------------------------------------------
def build_transform(name: str, image_size: int, rgb: bool = True, is_train: bool = False):
    """Xây dựng transform đồng bộ cho toàn bộ mô hình: resize 64x64 và chuẩn hóa [-1, 1]."""
    tfs = []
    if name == "celeba":
        tfs.append(transforms.CenterCrop(148))  # Cắt vùng mặt trung tâm từ ảnh gốc 178x218
    elif name == "fashion_mnist" and rgb:
        tfs.append(transforms.Grayscale(num_output_channels=3))

    tfs += [
        transforms.Resize(image_size, antialias=True),
        transforms.CenterCrop(image_size),
    ]

    # Data augmentation lật ngang ngẫu nhiên trên tập train của CelebA
    if is_train and name == "celeba":
        tfs.append(transforms.RandomHorizontalFlip(p=0.5))

    tfs.append(transforms.ToTensor())  # Chuyển về float tensor trong [0, 1]
    channels = 1 if (name == "fashion_mnist" and not rgb) else 3
    tfs.append(transforms.Normalize([0.5] * channels, [0.5] * channels))  # -> [-1, 1]
    return transforms.Compose(tfs)


def denormalize(x: torch.Tensor, clamp: bool = False) -> torch.Tensor:
    """
    Chuyển đổi tensor từ miền chuẩn hóa [-1, 1] về miền [0, 1].

    Args:
        x (torch.Tensor): Tensor ảnh từ mô hình (kỳ vọng trong [-1, 1]).
        clamp (bool):
            - False (Mặc định): Áp dụng công thức tỉ lệ toán học thuần túy (x + 1.0) / 2.0.
              Không clamp giúp pipeline kiểm thử phát hiện ngay nếu mô hình sinh giá trị bất thường
              (bùng nổ gradient, vượt ngưỡng [-1, 1] hoặc NaN/Inf) thay vì bị âm thầm che giấu.
            - True: Giới hạn giá trị trong [0, 1] khi cần xuất file ảnh an toàn ra đĩa (save_image).
    """
    out = (x + 1.0) / 2.0
    if clamp:
        out = out.clamp(0.0, 1.0)
    return out


to_01 = denormalize


# ---------------------------------------------------------------------------
# Dataset dựa trên Manifest
# ---------------------------------------------------------------------------
class CelebAManifestDataset(Dataset):
    """Dataset đọc ảnh CelebA từ danh sách đường dẫn file (Manifest)."""

    def __init__(self, img_paths: List[str], transform=None):
        self.img_paths = img_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, 0


# ---------------------------------------------------------------------------
# SỬA LỖI 2: Xây dựng Image Index quét toàn diện, hỗ trợ mọi cấu trúc Kaggle
# ---------------------------------------------------------------------------
def _build_image_index(celeba_dir: str) -> Dict[str, str]:
    """
    Quét toàn bộ thư mục celeba_dir một lần duy nhất và tạo dictionary ánh xạ
    từ tên file (vd '000001.jpg') sang đường dẫn tuyệt đối thực tế trên đĩa.
    Hỗ trợ mọi cấu trúc lồng thư mục phức tạp từ Kaggle/Google Drive.

    Báo lỗi nếu phát hiện file trùng tên giữa các thư mục khác nhau.
    """
    image_index: Dict[str, str] = {}
    for current_root, _, files in os.walk(celeba_dir):
        for filename in files:
            if filename.lower().endswith(".jpg"):
                full_path = os.path.join(current_root, filename)
                if filename in image_index:
                    raise ValueError(
                        f"Phát hiện tên ảnh bị trùng: '{filename}'\n"
                        f"  - Vị trí 1: '{image_index[filename]}'\n"
                        f"  - Vị trí 2: '{full_path}'\n"
                        f"Vui lòng xác minh và dọn dẹp cấu trúc thư mục dataset."
                    )
                image_index[filename] = full_path
    return image_index


# ---------------------------------------------------------------------------
# SỬA LỖI 1: Tìm kiếm metadata chính thức; KHÔNG tự tạo file giả
# ---------------------------------------------------------------------------
def _find_partition_file(celeba_dir: str) -> Optional[str]:
    """Tìm file metadata chính thức list_eval_partition.txt (hoặc .csv)."""
    candidates = ["list_eval_partition.txt", "list_eval_partition.csv"]
    for root, _, files in os.walk(celeba_dir):
        for f in files:
            if f.lower() in candidates:
                return os.path.join(root, f)
    return None


def _ensure_celeba_available(root: str) -> str:
    """Đảm bảo thư mục dữ liệu CelebA tồn tại (tự động tải từ Kaggle nếu chưa có)."""
    celeba_dir = os.path.abspath(os.path.join(root, "celeba"))
    os.makedirs(celeba_dir, exist_ok=True)

    has_jpg = False
    for _, _, files in os.walk(celeba_dir):
        if any(f.lower().endswith(".jpg") for f in files):
            has_jpg = True
            break

    if not has_jpg:
        print("[*] Chưa phát hiện ảnh CelebA. Đang tự động tải bộ CelebA từ Kaggle...")
        os.environ["KAGGLEHUB_CACHE"] = celeba_dir
        try:
            import kagglehub
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "kagglehub"])
            import kagglehub
        kagglehub.dataset_download("jessicali9530/celeba-dataset")
        print(f"[+] Tải hoàn tất tại: {celeba_dir}")

    return celeba_dir


# ---------------------------------------------------------------------------
# SỬA LỖI 4: Kiểm tra tính toàn vẹn của toàn bộ ảnh và lưu cache
# ---------------------------------------------------------------------------
def _verify_single_image(path: str) -> Tuple[bool, str]:
    """Kiểm tra 1 ảnh bằng PIL verify()."""
    try:
        with Image.open(path) as img:
            img.verify()
        return True, path
    except Exception:
        return False, path


def validate_images(
    image_paths: List[str],
    cache_path: Optional[str] = None,
    max_workers: int = 16,
) -> Tuple[List[str], List[str]]:
    """
    Kiểm tra tính toàn vẹn của toàn bộ ảnh và lưu cache kết quả.
    Báo cáo chi tiết số ảnh hợp lệ và số ảnh lỗi bị loại bỏ.
    """
    # 1. Đọc từ cache nếu có
    if cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cache_data = json.load(f)
            corrupted_set = set(cache_data.get("corrupted_paths", []))
            total_checked = cache_data.get("total_checked", 0)

            if total_checked == len(image_paths):
                valid_paths = [p for p in image_paths if p not in corrupted_set]
                corrupted_paths = [p for p in image_paths if p in corrupted_set]
                if len(corrupted_paths) > 0:
                    print(f"[!] Tải từ cache: Đã loại bỏ {len(corrupted_paths):,} ảnh lỗi.")
                return valid_paths, corrupted_paths
        except Exception as e:
            print(f"[!] Cache xác thực bị lỗi hoặc không khớp ({e}). Tiến hành xác thực lại...")

    # 2. Xác thực đa luồng (multi-threaded verification)
    print(f"[*] Đang xác thực tính toàn vẹn của {len(image_paths):,} ảnh (chỉ chạy 1 lần và sẽ lưu cache)...")
    valid_paths: List[str] = []
    corrupted_paths: List[str] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(_verify_single_image, image_paths)
        for is_valid, path in results:
            if is_valid:
                valid_paths.append(path)
            else:
                corrupted_paths.append(path)

    # 3. Báo cáo cụ thể số ảnh lỗi bị loại bỏ (không âm thầm bỏ qua)
    if len(corrupted_paths) > 0:
        print(f"[!] CẢNH BÁO: Phát hiện và loại bỏ {len(corrupted_paths):,} ảnh bị hỏng:")
        for p in corrupted_paths[:5]:
            print(f"    - Ảnh hỏng: {p}")
        if len(corrupted_paths) > 5:
            print(f"    - ... và {len(corrupted_paths) - 5} ảnh khác.")
    else:
        print(f"[+] Toàn bộ {len(valid_paths):,} ảnh đều hợp lệ và toàn vẹn.")

    # 4. Lưu cache kết quả xác thực ra đĩa
    if cache_path:
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump({
                    "total_checked": len(image_paths),
                    "corrupted_paths": corrupted_paths,
                }, f, indent=2)
            print(f"[+] Đã lưu cache xác thực tính toàn vẹn vào: '{cache_path}'")
        except Exception as err:
            print(f"[!] Không thể ghi cache xác thực: {err}")

    return valid_paths, corrupted_paths


# ---------------------------------------------------------------------------
# SỬA LỖI 1 & 5: Đọc metadata chính thức & Kiểm tra nghiêm ngặt số lượng
# ---------------------------------------------------------------------------
def load_celeba_splits_from_partition(
    root: str,
    tf_train,
    tf_eval,
    validate_integrity: bool = True,
) -> Tuple[CelebAManifestDataset, CelebAManifestDataset, CelebAManifestDataset]:
    """
    Phân chia CelebA thành 3 tập Train / Val / Test bằng metadata chính thức `list_eval_partition.txt`.

    Kiểm tra nghiêm ngặt:
        - Không tìm thấy metadata: raise FileNotFoundError (không tự tạo file giả).
        - Trùng lặp filename trong metadata: raise ValueError.
        - Partition ID ngoài {'0', '1', '2'}: raise ValueError.
        - Ảnh bị thiếu trên đĩa: raise FileNotFoundError với danh sách cụ thể.
        - Không đủ số lượng ảnh theo phân chia: raise RuntimeError.
    """
    celeba_dir = _ensure_celeba_available(root)

    # LỖI 1: Tìm metadata chính thức; KHÔNG tự tạo file giả
    partition_file = _find_partition_file(celeba_dir)
    if partition_file is None:
        raise FileNotFoundError(
            "Không tìm thấy list_eval_partition.txt. "
            "Vui lòng tải metadata chính thức của CelebA và đặt vào thư mục data/celeba."
        )

    # LỖI 2: Xây dựng image_index toàn diện một lần duy nhất
    print(f"[*] Đang quét và lập chỉ mục ảnh từ: '{celeba_dir}'...")
    image_index = _build_image_index(celeba_dir)
    print(f"[+] Đã lập chỉ mục {len(image_index):,} ảnh trên đĩa.")

    train_imgs: List[str] = []
    val_imgs: List[str] = []
    test_imgs: List[str] = []

    seen_metadata_files: Dict[str, str] = {}
    missing_files: List[str] = []

    # LỖI 5: Đọc và kiểm tra nghiêm ngặt từng dòng metadata
    print(f"[*] Đang đọc metadata phân chia chính thức: '{partition_file}'...")
    with open(partition_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            parts = line.strip().replace(",", " ").split()
            if len(parts) < 2:
                continue

            filename, partition = parts[0], parts[1]

            # 5.1 Kiểm tra partition ID hợp lệ
            if partition not in ("0", "1", "2"):
                raise ValueError(
                    f"Dòng {line_num}: Partition ID không hợp lệ '{partition}' cho ảnh '{filename}'. "
                    f"Chỉ chấp nhận các giá trị: '0' (Train), '1' (Val), '2' (Test)."
                )

            # 5.2 Kiểm tra file ảnh có xuất hiện nhiều lần hoặc bị gán 2 partition không
            if filename in seen_metadata_files:
                prev_part = seen_metadata_files[filename]
                raise ValueError(
                    f"Dòng {line_num}: File ảnh '{filename}' xuất hiện nhiều lần trong metadata! "
                    f"(Lần 1 gán partition {prev_part}, lần 2 gán partition {partition})."
                )
            seen_metadata_files[filename] = partition

            # 5.3 Tra cứu đường dẫn file qua image_index
            img_path = image_index.get(filename)
            if img_path is None:
                missing_files.append(filename)
                continue

            # Phân bổ vào 3 tập
            if partition == "0":
                train_imgs.append(img_path)
            elif partition == "1":
                val_imgs.append(img_path)
            elif partition == "2":
                test_imgs.append(img_path)

    # 5.4 Thống kê file bị thiếu và báo lỗi nếu có
    if len(missing_files) > 0:
        sample_missing = missing_files[:5]
        raise FileNotFoundError(
            f"Phát hiện {len(missing_files):,} file ảnh được liệt kê trong metadata nhưng không tồn tại trên đĩa!\n"
            f"Mẫu các file bị thiếu: {sample_missing}\n"
            f"Vui lòng kiểm tra lại bộ dữ liệu CelebA trong thư mục '{celeba_dir}'."
        )

    # 5.5 Kiểm tra chính xác số lượng ảnh sau khi đọc metadata
    actual_counts = {
        "0": len(train_imgs),
        "1": len(val_imgs),
        "2": len(test_imgs),
    }
    for partition, expected in CELEBA_EXPECTED_COUNTS.items():
        actual = actual_counts[partition]
        if actual != expected:
            part_name = "Train (0)" if partition == "0" else "Val (1)" if partition == "1" else "Test (2)"
            raise RuntimeError(
                f"Số lượng ảnh của Partition {part_name} không hợp lệ: "
                f"tìm thấy {actual:,} ảnh, yêu cầu chính xác {expected:,} ảnh theo chuẩn CelebA."
            )

    # LỖI 4: Xác thực tính toàn vẹn của toàn bộ ảnh có lưu cache
    if validate_integrity:
        cache_file = os.path.join(celeba_dir, ".celeba_validation_cache.json")
        all_imgs = train_imgs + val_imgs + test_imgs
        valid_paths, corrupted_paths = validate_images(all_imgs, cache_path=cache_file)

        if len(corrupted_paths) > 0:
            corrupted_set = set(corrupted_paths)
            train_imgs = [p for p in train_imgs if p not in corrupted_set]
            val_imgs = [p for p in val_imgs if p not in corrupted_set]
            test_imgs = [p for p in test_imgs if p not in corrupted_set]
            print(f"[!] Đã loại bỏ tổng cộng {len(corrupted_paths):,} ảnh lỗi khỏi tập dữ liệu huấn luyện.")

    print(f"[+] Phân chia tập dữ liệu CelebA hoàn tất (Khớp 100% chuẩn metadata):")
    print(f"    - Train (partition 0): {len(train_imgs):,} ảnh")
    print(f"    - Val   (partition 1): {len(val_imgs):,} ảnh")
    print(f"    - Test  (partition 2): {len(test_imgs):,} ảnh")

    train_dataset = CelebAManifestDataset(train_imgs, transform=tf_train)
    val_dataset = CelebAManifestDataset(val_imgs, transform=tf_eval)
    test_dataset = CelebAManifestDataset(test_imgs, transform=tf_eval)

    return train_dataset, val_dataset, test_dataset


# ---------------------------------------------------------------------------
# API DÙNG CHUNG: get_dataloaders()
# ---------------------------------------------------------------------------
def get_dataloaders(
    name: str = "celeba",
    root: str = "./data",
    image_size: int = 64,
    batch_size: int = 128,
    num_workers: Optional[int] = None,
    val_size: int = 6000,
    seed: int = SEED,
    subset: Optional[int] = None,
    fmnist_rgb: bool = True,
    validate_integrity: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Trả về bộ 3 DataLoaders (train, val, test) chuẩn hóa dùng chung cho cả 3 mô hình.

    Args:
        name (str): 'celeba' hoặc 'fashion_mnist'.
        root (str): Thư mục chứa dữ liệu ('./data').
        image_size (int): Kích thước cạnh ảnh vuông (mặc định 64).
        batch_size (int): Kích thước batch huấn luyện (mặc định 128).
        num_workers (int, optional): Số luồng tải dữ liệu (0 trên Windows để chống deadlock, 2 trên Linux).
        val_size (int): Số lượng ảnh val trích xuất cho Fashion-MNIST (mặc định 6,000).
        seed (int): Seed ngẫu nhiên để đảm bảo tính tái lập.
        subset (int, optional): Lấy mẫu NGẪU NHIÊN N ảnh train để debug (Đã sửa lỗi 3).
        fmnist_rgb (bool): Nhân 3 kênh RGB cho Fashion-MNIST (mặc định True).
        validate_integrity (bool): Xác thực tính toàn vẹn của file ảnh kèm lưu cache (mặc định True).

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: (train_loader, val_loader, test_loader)
    """
    os.makedirs(root, exist_ok=True)
    if num_workers is None:
        num_workers = 0 if sys.platform == "win32" else 2

    name = name.lower()
    tf_train = build_transform(name, image_size, fmnist_rgb, is_train=True)
    tf_eval = build_transform(name, image_size, fmnist_rgb, is_train=False)
    g = torch.Generator().manual_seed(seed)

    if name == "celeba":
        train, val, test = load_celeba_splits_from_partition(root, tf_train, tf_eval, validate_integrity=validate_integrity)
    elif name == "fashion_mnist":
        full = datasets.FashionMNIST(root, train=True, download=True, transform=tf_train)
        test = datasets.FashionMNIST(root, train=False, download=True, transform=tf_eval)
        train, val = random_split(full, [len(full) - val_size, val_size], generator=g)
    else:
        raise ValueError(f"Bộ dữ liệu không được hỗ trợ: '{name}'. Chọn 'celeba' hoặc 'fashion_mnist'.")

    # LỖI 3: Lấy mẫu ngẫu nhiên có seed cố định, KHÔNG lấy tuần tự N ảnh đầu
    if subset is not None:
        subset = min(subset, len(train))
        g_sub = torch.Generator().manual_seed(seed)
        random_indices = torch.randperm(len(train), generator=g_sub)[:subset].tolist()
        train = Subset(train, random_indices)
        print(f"[*] Subset mode: Lấy ngẫu nhiên {subset:,} mẫu từ tập train (Seed={seed}).")

    pin_memory = torch.cuda.is_available()
    loader_kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)

    return (
        DataLoader(train, shuffle=True, drop_last=True, **loader_kwargs),
        DataLoader(val, shuffle=False, **loader_kwargs),
        DataLoader(test, shuffle=False, **loader_kwargs),
    )


def _verify_and_save_sample(ds_name: str, root: str = "./data", sample_dir: str = "./outputs/samples"):
    """Kiểm tra pipeline và lưu ảnh mẫu 64 ảnh ra đĩa để nghiệm thu."""
    print("=" * 65)
    print(f"[*] Đang kiểm tra DataLoader: '{ds_name}'...")
    tr, va, te = get_dataloaders(ds_name, root=root)
    x = next(iter(tr))[0]
    print(f"[+] Số lượng mẫu (Train / Val / Test): {len(tr.dataset):,} / {len(va.dataset):,} / {len(te.dataset):,}")
    print(f"[+] Kích thước Batch : {tuple(x.shape)} | Min/Max: {x.min().item():.2f} / {x.max().item():.2f}")

    # Kiểm tra hàm denormalize() không che giấu lỗi
    x_denorm_raw = denormalize(x, clamp=False)
    print(f"[+] Kiểm tra denormalize(clamp=False): Min/Max = [{x_denorm_raw.min().item():.3f}, {x_denorm_raw.max().item():.3f}]")

    os.makedirs(sample_dir, exist_ok=True)
    out_path = os.path.join(sample_dir, f"sample_{ds_name}.png")
    # Khi lưu file ảnh ra đĩa thì dùng clamp=True để an toàn
    save_image(denormalize(x[:64], clamp=True), out_path, nrow=8)
    print(f"[+] Đã lưu 64 ảnh mẫu vào: '{out_path}'")


if __name__ == "__main__":
    target = [sys.argv[1]] if len(sys.argv) > 1 else ["fashion_mnist", "celeba"]
    for ds in target:
        try:
            _verify_and_save_sample(ds, root="./data")
        except Exception as err:
            print(f"[!] Lỗi khi xử lý '{ds}': {err}")