"""
dataloader.py - Pipeline nạp dữ liệu dùng chung cho cả 3 mô hình (VAE, DCGAN, DDPM).

Các nguồn dữ liệu:
    - CelebA: chia train/validation/test theo list_eval_partition.txt (hoặc .csv).
      Số lượng chuẩn trước khi loại ảnh hỏng lần lượt là 162770 / 19867 / 19962.
      Mặc định yêu cầu đủ ảnh theo metadata và đúng số lượng chuẩn.
    - Fashion-MNIST: tải qua torchvision, tách validation từ 60000 ảnh train.
      Theo mặc định ảnh được resize thành 64x64 và chuyển thành 3 kênh.

Ảnh đầu ra của dataset được chuẩn hóa vào [-1, 1].
Hàm denormalize() (bí danh to_01) đổi từ [-1, 1] về miền tỷ lệ [0, 1];
clamp=True chỉ cắt ngưỡng khi cần. Giá trị ngoài miền và NaN không được
bảo đảm phát hiện chỉ bằng denormalize().

Lưu ý: Nếu bật validate_integrity, ảnh hỏng bị loại bỏ sau khi kiểm tra
số lượng ảnh CelebA theo metadata, nên kích thước các tập có thể giảm.

Ví dụ:
    from data.dataloader import get_dataloaders, denormalize
    train_dl, val_dl, test_dl = get_dataloaders("celeba")              # CelebA 64x64
    train_dl, val_dl, test_dl = get_dataloaders("fashion_mnist")       # sanity check
    train_dl, _, _ = get_dataloaders("celeba", subset=2000)            # chạy thử nhanh
"""
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from copy import copy
from typing import Dict, List, Optional, Tuple
from PIL import Image

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from torchvision.utils import save_image

SEED = 42

# Số lượng ảnh ở từng partition trong bộ CelebA đầy đủ (trước khi lọc ảnh hỏng)
CELEBA_EXPECTED_COUNTS = {
    "0": 162770,  # Train
    "1": 19867,   # Validation
    "2": 19962,   # Test
}


# ---------------------------------------------------------------------------
# Biến đổi ảnh đầu vào và chuyển đổi miền giá trị
# ---------------------------------------------------------------------------
class _SeededHorizontalFlip(transforms.RandomHorizontalFlip):
    """Dùng RNG riêng khi DataLoader chạy transform trong tiến trình chính."""

    def __init__(self, generator: torch.Generator):
        super().__init__(p=0.5)
        self.generator = generator

    def forward(self, img):
        if torch.rand(1, generator=self.generator) < self.p:
            return transforms.functional.hflip(img)
        return img


def build_transform(
    name: str,
    image_size: int,
    rgb: bool = True,
    is_train: bool = False,
    *,
    flip_generator: Optional[torch.Generator] = None,
):
    """Tạo transform theo image_size, loại dữ liệu và chế độ train/eval."""
    tfs = []
    if name == "celeba":
        tfs.append(transforms.CenterCrop(148))  # Crop giữa ảnh CelebA về 148x148 trước khi resize
    elif name == "fashion_mnist" and rgb:
        tfs.append(transforms.Grayscale(num_output_channels=3))

    tfs += [
        transforms.Resize(image_size, antialias=True),
        transforms.CenterCrop(image_size),
    ]

    # Chỉ lật ngang ngẫu nhiên với ảnh CelebA ở tập train
    if is_train and name == "celeba":
        if flip_generator is None:
            tfs.append(transforms.RandomHorizontalFlip(p=0.5))
        else:
            tfs.append(_SeededHorizontalFlip(flip_generator))

    tfs.append(transforms.ToTensor())  # Chuyển về float tensor trong [0, 1]
    channels = 1 if (name == "fashion_mnist" and not rgb) else 3
    tfs.append(transforms.Normalize([0.5] * channels, [0.5] * channels))  # -> [-1, 1]
    return transforms.Compose(tfs)


def denormalize(x: torch.Tensor, clamp: bool = False) -> torch.Tensor:
    """
    Tính (x + 1) / 2 để đưa giá trị đã chuẩn hóa về thang [0, 1].

    Args:
        x: Tensor đầu vào, thông thường có giá trị trong [-1, 1].
        clamp: Nếu True, cắt giá trị hữu hạn về [0, 1]. Nếu False (mặc định),
            giữ nguyên kết quả phép biến đổi kể cả khi vượt [0, 1].

    Returns:
        Tensor sau biến đổi; hàm không tự kiểm tra NaN/Inf hay cảnh báo lỗi.
    """
    out = (x + 1.0) / 2.0
    if clamp:
        out = out.clamp(0.0, 1.0)
    return out


to_01 = denormalize


# ---------------------------------------------------------------------------
# Dataset CelebA từ danh sách đường dẫn ảnh
# ---------------------------------------------------------------------------
class CelebAManifestDataset(Dataset):
    """Đọc ảnh RGB theo đường dẫn; trả về (ảnh đã transform, nhãn giả 0)."""

    def __init__(self, img_paths: List[str], transform=None):
        self.img_paths = img_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.img_paths[idx]
        with Image.open(img_path) as source:
            img = source.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, 0


# ---------------------------------------------------------------------------
# Lập chỉ mục ảnh theo tên file trong cây thư mục CelebA
# ---------------------------------------------------------------------------
def _build_image_index(celeba_dir: str) -> Dict[str, str]:
    """
    Quét đệ quy thư mục và ánh xạ tên file .jpg sang đường dẫn trên đĩa.
    Báo ValueError nếu hai file .jpg có cùng tên (phân biệt hoa/thường).
    Không xác minh nội dung ảnh ở bước này.
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
# Tìm metadata chia tập CelebA có sẵn trên đĩa
# ---------------------------------------------------------------------------
def _find_partition_file(celeba_dir: str) -> Optional[str]:
    """Tìm đệ quy file list_eval_partition.txt/.csv; không tạo metadata."""
    candidates = ["list_eval_partition.txt", "list_eval_partition.csv"]
    for root, _, files in os.walk(celeba_dir):
        for f in files:
            if f.lower() in candidates:
                return os.path.join(root, f)
    return None


def _ensure_celeba_available(root: str) -> str:
    """Dùng ảnh có sẵn dưới root/celeba hoặc tải bằng kagglehub nếu chưa có .jpg.

    Trả về thư mục chứa ảnh; dữ liệu Kaggle có thể ở ngoài root/celeba.
    Hàm chưa kiểm tra metadata hoặc số lượng ảnh.
    """
    celeba_dir = os.path.abspath(os.path.join(root, "celeba"))
    os.makedirs(celeba_dir, exist_ok=True)

    def has_images(directory: str) -> bool:
        return any(
            filename.lower().endswith(".jpg")
            for _, _, files in os.walk(directory)
            for filename in files
        )

    if has_images(celeba_dir):
        return celeba_dir

    print("[*] Chưa phát hiện ảnh CelebA. Đang tải từ Kaggle...")
    try:
        import kagglehub
    except ImportError as exc:
        raise ImportError(
            "Cần cài kagglehub trước khi tải CelebA: pip install kagglehub"
        ) from exc

    # Dùng đường dẫn do kagglehub trả về, không giả định là root/celeba.
    downloaded_dir = os.path.abspath(
        kagglehub.dataset_download("jessicali9530/celeba-dataset")
    )
    if not has_images(downloaded_dir):
        raise FileNotFoundError(
            f"Đã tải dataset nhưng không tìm thấy ảnh .jpg tại: {downloaded_dir}"
        )
    print(f"[+] CelebA nằm tại: {downloaded_dir}")
    return downloaded_dir


# ---------------------------------------------------------------------------
# Kiểm tra khả năng đọc ảnh và dùng cache trạng thái
# ---------------------------------------------------------------------------
def _verify_single_image(path: str) -> Tuple[bool, str]:
    """Thử giải mã ảnh bằng PIL; trả về (thành công, đường_dẫn)."""
    try:
        with Image.open(path) as img:
            img.load()
        return True, path
    except Exception:
        return False, path


def validate_images(
    image_paths: List[str],
    cache_path: Optional[str] = None,
    max_workers: int = 16,
) -> Tuple[List[str], List[str]]:
    """Kiểm tra ảnh và lưu cache theo đường dẫn, kích thước, mtime_ns.

    Cache được làm mới khi các thuộc tính trên đổi; không dùng hash nội dung,
    vì vậy không bảo đảm phát hiện sửa file mà giữ nguyên size/mtime.
    """
    if max_workers < 1:
        raise ValueError("max_workers phải >= 1")

    # Lấy chữ ký metadata để so với cache; không hash hoặc đọc lại nội dung
    # nếu size và mtime_ns của file không thay đổi.
    signatures = {}
    for path in image_paths:
        try:
            st = os.stat(path)
            signatures[path] = [st.st_size, st.st_mtime_ns]
        except OSError:
            signatures[path] = None

    cached = {}
    if cache_path and os.path.isfile(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("version") == 2 and isinstance(data.get("entries"), dict):
                cached = data["entries"]
        except (OSError, ValueError, TypeError) as exc:
            print(f"[!] Cache không đọc được ({exc}); kiểm tra lại ảnh.")

    status = {}
    to_verify = []
    for path in image_paths:
        entry = cached.get(path)
        if (
            signatures[path] is not None
            and isinstance(entry, dict)
            and entry.get("signature") == signatures[path]
            and isinstance(entry.get("valid"), bool)
        ):
            status[path] = entry["valid"]
        else:
            to_verify.append(path)

    if to_verify:
        print(f"[*] Kiểm tra {len(to_verify):,}/{len(image_paths):,} ảnh...")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for ok, path in executor.map(_verify_single_image, to_verify):
                status[path] = ok
    else:
        print(f"[+] Sử dụng cache cho {len(image_paths):,} ảnh.")

    valid_paths = [p for p in image_paths if status[p]]
    corrupted_paths = [p for p in image_paths if not status[p]]
    if corrupted_paths:
        print(f"[!] Phát hiện {len(corrupted_paths):,} ảnh hỏng / không thể đọc:")
        for path in corrupted_paths[:5]:
            print(f"    - {path}")
    else:
        print(f"[+] {len(valid_paths):,} ảnh đều hợp lệ.")

    if cache_path:
        cache_data = {
            "version": 2,
            "entries": {
                path: {"signature": signatures[path], "valid": status[path]}
                for path in image_paths
            },
        }
        try:
            temp_path = cache_path + ".tmp"
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(cache_data, f)
            os.replace(temp_path, cache_path)
        except OSError as exc:
            print(f"[!] Không ghi được cache: {exc}")

    return valid_paths, corrupted_paths


# ---------------------------------------------------------------------------
# Tách CelebA theo metadata và đối chiếu số lượng ảnh
# ---------------------------------------------------------------------------
def load_celeba_splits_from_partition(
    root: str,
    tf_train,
    tf_eval,
    validate_integrity: bool = True,
    *,
    strict_celeba_counts: bool = True,
) -> Tuple[CelebAManifestDataset, CelebAManifestDataset, CelebAManifestDataset]:
    """
    Tách tập theo list_eval_partition.txt hoặc .csv của CelebA.

    Kiểm tra trùng tên, dòng sai định dạng, partition không hợp lệ, ảnh thiếu
    và số lượng của từng partition theo bộ CelebA đầy đủ. Khi đặt
    strict_celeba_counts=False, chỉ dùng ảnh có trên đĩa được liệt kê trong
    metadata và cảnh báo nếu thiếu ảnh hoặc lệch số lượng. Khi bật kiểm tra
    tính toàn vẹn, loại ảnh không đọc được sau bước đối chiếu số lượng.
    """
    celeba_dir = _ensure_celeba_available(root)

    # Yêu cầu file metadata có sẵn; không tự sinh phân chia ngẫu nhiên.
    partition_file = _find_partition_file(celeba_dir)
    if partition_file is None:
        raise FileNotFoundError(
            "Không tìm thấy list_eval_partition.txt hoặc list_eval_partition.csv. "
            f"Vui lòng đặt metadata chính thức của CelebA trong thư mục '{celeba_dir}'."
        )

    # Lập bảng tra cứu tên ảnh -> đường dẫn.
    print(f"[*] Đang quét và lập chỉ mục ảnh từ: '{celeba_dir}'...")
    image_index = _build_image_index(celeba_dir)
    print(f"[+] Đã lập chỉ mục {len(image_index):,} ảnh trên đĩa.")

    train_imgs: List[str] = []
    val_imgs: List[str] = []
    test_imgs: List[str] = []

    seen_metadata_files: Dict[str, str] = {}
    missing_files: List[str] = []

    # Phân tích từng dòng metadata, chấp nhận header CSV phổ biến.
    print(f"[*] Đang đọc metadata phân chia chính thức: '{partition_file}'...")
    with open(partition_file, "r", encoding="utf-8-sig") as f:
        for line_num, line in enumerate(f, start=1):
            parts = line.strip().replace(",", " ").split()
            if not parts:
                continue
            if parts == ["image_id", "partition"] or parts == ["image", "partition"]:
                continue  # header CSV (nếu có)
            if len(parts) != 2:
                raise ValueError(f"Dòng {line_num} metadata sai định dạng: {line.strip()!r}")

            filename, partition = parts

            # Chỉ chấp nhận partition 0, 1, 2.
            if partition not in ("0", "1", "2"):
                raise ValueError(
                    f"Dòng {line_num}: Partition ID không hợp lệ '{partition}' cho ảnh '{filename}'. "
                    f"Chỉ chấp nhận các giá trị: '0' (Train), '1' (Val), '2' (Test)."
                )

            # Không cho phép một tên file xuất hiện nhiều lần trong metadata.
            if filename in seen_metadata_files:
                prev_part = seen_metadata_files[filename]
                raise ValueError(
                    f"Dòng {line_num}: File ảnh '{filename}' xuất hiện nhiều lần trong metadata! "
                    f"(Lần 1 gán partition {prev_part}, lần 2 gán partition {partition})."
                )
            seen_metadata_files[filename] = partition

            # Ánh xạ tên file sang đường dẫn ảnh thực tế.
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

    if missing_files:
        sample_missing = missing_files[:5]
        message = (
            f"Phát hiện {len(missing_files):,} file ảnh được liệt kê trong metadata nhưng không tồn tại trên đĩa!\n"
            f"Mẫu các file bị thiếu: {sample_missing}\n"
            f"Vui lòng kiểm tra lại bộ dữ liệu CelebA trong thư mục '{celeba_dir}'."
        )
        if strict_celeba_counts:
            raise FileNotFoundError(message)
        print(f"[!] {message}")

    # Đối chiếu số lượng trước khi loại các ảnh bị hỏng.
    actual_counts = {
        "0": len(train_imgs),
        "1": len(val_imgs),
        "2": len(test_imgs),
    }
    for partition, expected in CELEBA_EXPECTED_COUNTS.items():
        actual = actual_counts[partition]
        if actual != expected:
            part_name = "Train (0)" if partition == "0" else "Val (1)" if partition == "1" else "Test (2)"
            message = (
                f"Số lượng ảnh của Partition {part_name} lệch chuẩn CelebA: "
                f"tìm thấy {actual:,} ảnh, số lượng chuẩn là {expected:,} ảnh."
            )
            if strict_celeba_counts:
                raise RuntimeError(message)
            print(f"[!] {message}")

    # Tùy chọn lọc ảnh không đọc được, tái sử dụng cache nếu còn hợp lệ.
    if validate_integrity:
        cache_file = os.path.join(celeba_dir, ".celeba_validation_cache.json")
        all_imgs = train_imgs + val_imgs + test_imgs
        _, corrupted_paths = validate_images(all_imgs, cache_path=cache_file)

        if len(corrupted_paths) > 0:
            corrupted_set = set(corrupted_paths)
            train_imgs = [p for p in train_imgs if p not in corrupted_set]
            val_imgs = [p for p in val_imgs if p not in corrupted_set]
            test_imgs = [p for p in test_imgs if p not in corrupted_set]
            print(f"[!] Đã loại bỏ tổng cộng {len(corrupted_paths):,} ảnh lỗi khỏi các tập CelebA.")

    print("[+] Số ảnh CelebA trong mỗi tập:")
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
    *,
    strict_celeba_counts: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Trả về bộ 3 DataLoaders (train, val, test) chuẩn hóa dùng chung cho cả 3 mô hình.

    Args:
        name (str): 'celeba' hoặc 'fashion_mnist'.
        root (str): Thư mục chứa dữ liệu ('./data').
        image_size (int): Kích thước cạnh ảnh vuông (mặc định 64).
        batch_size (int): Kích thước batch huấn luyện (mặc định 128).
        num_workers (int, optional): Số worker của DataLoader. Mặc định 0 trên
            Windows và tối đa 4 trên các nền tảng khác, tùy số CPU.
            Khi num_workers > 0, worker được giữ lại qua các epoch.
        val_size (int): Số lượng ảnh val trích xuất cho Fashion-MNIST (mặc định 6,000).
        seed (int): Seed để tái lập phép chia, subset, shuffle và lật ảnh
            giữa các lần chạy có cùng cấu hình DataLoader.
        subset (int, optional): Lấy ngẫu nhiên tối đa N mẫu train, seed cố định.
            Phải >= 1; số mẫu train cuối cùng phải >= batch_size.
        fmnist_rgb (bool): Nếu True, nhân kênh xám thành 3 kênh ở Fashion-MNIST.
        validate_integrity (bool): Kiểm tra và lọc ảnh CelebA không đọc được;
            cache dựa trên đường dẫn, kích thước và thời gian sửa file.
        strict_celeba_counts (bool): Nếu True, yêu cầu số lượng chuẩn và đủ
            ảnh theo metadata. Đặt False để chạy thử với ảnh hiện có. Khi đó
            tập validation hoặc test có thể rỗng.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: (train_loader, val_loader, test_loader)
    """
    os.makedirs(root, exist_ok=True)
    if num_workers is None:
        num_workers = 0 if sys.platform == "win32" else min(4, os.cpu_count() or 1)

    name = name.lower()
    if image_size < 1:
        raise ValueError("image_size phải >= 1")
    flip_generator = (
        torch.Generator().manual_seed(seed)
        if name == "celeba" and num_workers == 0
        else None
    )
    tf_train = build_transform(
        name, image_size, fmnist_rgb, is_train=True, flip_generator=flip_generator
    )
    tf_eval = build_transform(name, image_size, fmnist_rgb, is_train=False)

    if name == "celeba":
        train, val, test = load_celeba_splits_from_partition(
            root,
            tf_train,
            tf_eval,
            validate_integrity=validate_integrity,
            strict_celeba_counts=strict_celeba_counts,
        )
    elif name == "fashion_mnist":
        if not 0 < val_size < 60000:
            raise ValueError("val_size phải nằm trong khoảng 1..59999")
        full_train = datasets.FashionMNIST(root, train=True, download=True, transform=tf_train)
        # Chia sẻ tensor ảnh và nhãn, nhưng dùng transform riêng cho validation.
        full_eval = copy(full_train)
        full_eval.transform = tf_eval
        test = datasets.FashionMNIST(root, train=False, download=True, transform=tf_eval)
        split_generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(full_train), generator=split_generator).tolist()
        train = Subset(full_train, indices[:-val_size])
        val = Subset(full_eval, indices[-val_size:])
    else:
        raise ValueError(f"Bộ dữ liệu không được hỗ trợ: '{name}'. Chọn 'celeba' hoặc 'fashion_mnist'.")

    # Chọn subset ngẫu nhiên, có thể tái lập theo seed.
    if subset is not None:
        if subset < 1:
            raise ValueError("subset phải là số nguyên dương")
        subset = min(subset, len(train))
        g_sub = torch.Generator().manual_seed(seed)
        random_indices = torch.randperm(len(train), generator=g_sub)[:subset].tolist()
        train = Subset(train, random_indices)
        print(f"[*] Subset mode: Lấy ngẫu nhiên {subset:,} mẫu từ tập train (Seed={seed}).")

    if batch_size < 1:
        raise ValueError("batch_size phải là số nguyên dương")
    if len(train) < batch_size:
        raise ValueError(
            f"Tập train chỉ có {len(train)} ảnh, ít hơn batch_size={batch_size}; "
            "hãy giảm batch_size hoặc tăng subset."
        )

    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )

    return (
        DataLoader(
            train, shuffle=True, drop_last=True,
            generator=torch.Generator().manual_seed(seed), **loader_kwargs,
        ),
        DataLoader(val, shuffle=False, generator=torch.Generator().manual_seed(seed), **loader_kwargs),
        DataLoader(test, shuffle=False, generator=torch.Generator().manual_seed(seed), **loader_kwargs),
    )


def _verify_and_save_sample(ds_name: str, root: str = "./data", sample_dir: str = "./outputs/samples"):
    """Chạy thử DataLoader và lưu tối đa 64 ảnh đầu batch ra PNG."""
    print("=" * 65)
    print(f"[*] Đang kiểm tra DataLoader: '{ds_name}'...")
    tr, va, te = get_dataloaders(ds_name, root=root)
    x = next(iter(tr))[0]
    print(f"[+] Số lượng mẫu (Train / Val / Test): {len(tr.dataset):,} / {len(va.dataset):,} / {len(te.dataset):,}")
    print(f"[+] Kích thước Batch : {tuple(x.shape)} | Min/Max: {x.min().item():.2f} / {x.max().item():.2f}")

    # In khoảng giá trị sau khi chuyển thang, không tự kiểm tra NaN/Inf.
    x_denorm_raw = denormalize(x, clamp=False)
    print(f"[+] Kiểm tra denormalize(clamp=False): Min/Max = [{x_denorm_raw.min().item():.3f}, {x_denorm_raw.max().item():.3f}]")

    os.makedirs(sample_dir, exist_ok=True)
    out_path = os.path.join(sample_dir, f"sample_{ds_name}.png")
    # Cắt ngưỡng khi tạo ảnh mẫu để pixel nằm trong [0, 1].
    save_image(denormalize(x[:64], clamp=True), out_path, nrow=8)
    print(f"[+] Đã lưu 64 ảnh mẫu vào: '{out_path}'")


if __name__ == "__main__":
    target = [sys.argv[1]] if len(sys.argv) > 1 else ["fashion_mnist", "celeba"]
    for ds in target:
        try:
            _verify_and_save_sample(ds, root="./data")
        except Exception as err:
            print(f"[!] Lỗi khi xử lý '{ds}': {err}")
