"""
dataloader.py - DataLoader dùng chung cho VAE, DCGAN, DDPM.

Dữ liệu chính: CelebA (64x64, 3 kênh) -> lưu và tự động chia thư mục tại:
    ./data/celeba/train/  (162,770 ảnh)
    ./data/celeba/val/    (19,867 ảnh)
    ./data/celeba/test/   (19,962 ảnh)

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
import shutil
import subprocess
import sys
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets, transforms
from torchvision.utils import save_image

SEED = 42


def build_transform(name: str, image_size: int, rgb: bool = True, is_train: bool = False):
    tfs = []
    if name == "celeba":
        tfs.append(transforms.CenterCrop(148))       # ảnh gốc 178x218 -> crop vùng mặt
    elif name == "fashion_mnist" and rgb:
        tfs.append(transforms.Grayscale(num_output_channels=3))

    tfs += [
        transforms.Resize(image_size, antialias=True),
        transforms.CenterCrop(image_size),
    ]
    # Chỉ bật lật ngang ngẫu nhiên trên tập train của CelebA để tăng độ đa dạng
    if is_train and name == "celeba":
        tfs.append(transforms.RandomHorizontalFlip(p=0.5))

    tfs.append(transforms.ToTensor())                # [0, 1]
    channels = 1 if (name == "fashion_mnist" and not rgb) else 3
    tfs.append(transforms.Normalize([0.5] * channels, [0.5] * channels))  # -> [-1, 1]
    return transforms.Compose(tfs)


def denormalize(x: torch.Tensor) -> torch.Tensor:
    """[-1, 1] -> [0, 1] (lưu ảnh, tính FID, BCE cho VAE)."""
    return (x.clamp(-1, 1) + 1) / 2


to_01 = denormalize


class ImageFolderFlat(Dataset):
    """Dataset đọc toàn bộ ảnh .jpg từ một thư mục đã chia sẵn (train / val / test)."""
    def __init__(self, folder_path: str, transform=None):
        self.imgs = sorted(glob.glob(os.path.join(folder_path, "*.jpg")))
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = Image.open(self.imgs[idx]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, 0


def _find_raw_celeba_dir(root: str):
    """Quét tìm thư mục gốc chứa >= 200,000 ảnh .jpg chưa được chia bên trong `./data`."""
    if not os.path.exists(root):
        return None, None

    candidates = [
        os.path.join(root, "celeba", "img_align_celeba"),
        os.path.join(root, "celeba", "img_align_celeba", "img_align_celeba"),
        os.path.join(root, "img_align_celeba"),
        os.path.join(root, "img_align_celeba", "img_align_celeba"),
    ]
    for dirpath, _, _ in os.walk(root):
        if os.path.basename(dirpath) == "img_align_celeba" and dirpath not in candidates:
            candidates.append(dirpath)

    for cand in candidates:
        if os.path.isdir(cand):
            jpgs = glob.glob(os.path.join(cand, "*.jpg"))
            if len(jpgs) >= 200000:
                return cand, sorted(jpgs)
    return None, None


def _is_already_split(celeba_base: str) -> bool:
    """Kiểm tra xem 3 thư mục train, val, test đã được chia đủ số lượng ảnh chưa."""
    tr_dir = os.path.join(celeba_base, "train")
    va_dir = os.path.join(celeba_base, "val")
    te_dir = os.path.join(celeba_base, "test")
    if not (os.path.isdir(tr_dir) and os.path.isdir(va_dir) and os.path.isdir(te_dir)):
        return False

    n_tr = len(glob.glob(os.path.join(tr_dir, "*.jpg")))
    n_va = len(glob.glob(os.path.join(va_dir, "*.jpg")))
    n_te = len(glob.glob(os.path.join(te_dir, "*.jpg")))
    return n_tr == 162770 and n_va == 19867 and n_te == 19962


def _split_into_folders(all_imgs: list, celeba_base: str):
    """Chia 202,599 ảnh gốc vào 3 thư mục vật lý: train (162,770), val (19,867), test (19,962)."""
    splits = {
        "train": all_imgs[:162770],
        "val": all_imgs[162770:182637],
        "test": all_imgs[182637:202599],
    }
    print(f"[*] Đang chia 202,599 ảnh CelebA vào 3 thư mục train / val / test tại '{celeba_base}'...")
    for split_name, file_list in splits.items():
        split_dir = os.path.join(celeba_base, split_name)
        os.makedirs(split_dir, exist_ok=True)
        for src_path in file_list:
            dst_path = os.path.join(split_dir, os.path.basename(src_path))
            if not os.path.exists(dst_path):
                os.replace(src_path, dst_path)
        print(f"    -> Đã tạo thư mục '{split_name}/' với {len(file_list):,} ảnh.")

    # Dọn dẹp các thư mục tạm rỗng sau khi đã di chuyển toàn bộ ảnh sang train/val/test
    for leftover in ["datasets", "img_align_celeba"]:
        p = os.path.join(celeba_base, leftover)
        if os.path.exists(p):
            shutil.rmtree(p, ignore_errors=True)


def _download_celeba_from_kaggle_to_data(root: str) -> str:
    """Tải và giải nén bộ CelebA từ Kaggle trực tiếp vào thư mục `./data/celeba`."""
    target_celeba_dir = os.path.abspath(os.path.join(root, "celeba"))
    os.makedirs(target_celeba_dir, exist_ok=True)
    os.environ["KAGGLEHUB_CACHE"] = target_celeba_dir

    try:
        import kagglehub
    except ImportError:
        print("[*] Đang tự động cài đặt thư viện 'kagglehub' để tải từ Kaggle...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "kagglehub"])
        import kagglehub

    print(f"[*] Đang tải và giải nén CelebA từ Kaggle vào '{target_celeba_dir}'...")
    kagglehub.dataset_download("jessicali9530/celeba-dataset")
    print(f"[+] Đã tải xong dữ liệu CelebA tại: {target_celeba_dir}")
    return target_celeba_dir


def _load_celeba_splits(root: str, tf_train, tf_eval):
    """
    Đảm bảo CelebA được tải về `./data/celeba` và chia thành 3 thư mục:
        ./data/celeba/train
        ./data/celeba/val
        ./data/celeba/test
    """
    celeba_base = os.path.join(root, "celeba")
    os.makedirs(celeba_base, exist_ok=True)

    # 1. Nếu đã chia sẵn thành 3 thư mục train/val/test thì nạp ngay
    if _is_already_split(celeba_base):
        print(f"[*] Phát hiện CelebA đã chia sẵn train/val/test tại: {celeba_base}")
        return (
            ImageFolderFlat(os.path.join(celeba_base, "train"), transform=tf_train),
            ImageFolderFlat(os.path.join(celeba_base, "val"), transform=tf_eval),
            ImageFolderFlat(os.path.join(celeba_base, "test"), transform=tf_eval),
        )

    # 2. Nếu có thư mục ảnh gốc chưa chia (img_align_celeba) thì tiến hành chia vào train/val/test
    raw_dir, all_imgs = _find_raw_celeba_dir(root)

    # 3. Nếu chưa có ảnh trên máy: thử tải qua torchvision, nếu lỗi thì tải qua Kaggle
    if raw_dir is None:
        try:
            print("[*] Thử tải CelebA qua torchvision vào thư mục './data'...")
            datasets.CelebA(root, split="train", download=True)
            raw_dir, all_imgs = _find_raw_celeba_dir(root)
        except Exception as tv_err:
            print(f"[!] Torchvision Google Drive bị giới hạn ({tv_err.__class__.__name__}). Tự động chuyển sang Kaggle...")
            try:
                _download_celeba_from_kaggle_to_data(root)
                raw_dir, all_imgs = _find_raw_celeba_dir(root)
            except Exception as kg_err:
                raise RuntimeError(
                    f"\n[LỖI TẢI CELEBA]: Cả torchvision và Kaggle tự động đều thất bại ({kg_err}).\n"
                    "-> CÁCH KHẮC PHỤC THỦ CÔNG:\n"
                    "1. Tải bộ CelebA từ Kaggle: https://www.kaggle.com/datasets/jessicali9530/celeba-dataset\n"
                    "2. Giải nén thư mục 'img_align_celeba' vào: ./data/celeba/img_align_celeba/\n"
                    "3. Chạy lại `python dataloader.py` để code tự động chia vào train/val/test."
                ) from kg_err

    if raw_dir is None or all_imgs is None:
        raise FileNotFoundError("Không tìm thấy đủ 202,599 ảnh .jpg trong ./data/celeba để chia tập.")

    # Thực hiện chia ảnh vào 3 thư mục vật lý train/, val/, test/
    _split_into_folders(all_imgs, celeba_base)

    return (
        ImageFolderFlat(os.path.join(celeba_base, "train"), transform=tf_train),
        ImageFolderFlat(os.path.join(celeba_base, "val"), transform=tf_eval),
        ImageFolderFlat(os.path.join(celeba_base, "test"), transform=tf_eval),
    )


def get_dataloaders(name="celeba", root="./data/raw", image_size=64, batch_size=128,
                    num_workers=None, val_size=6000, seed=SEED, subset=None,
                    fmnist_rgb=True):
    """
    subset: nếu đặt (vd 2000) chỉ lấy N ảnh đầu của tập train để debug nhanh.
    num_workers: mặc định 0 trên Windows để tránh lỗi multiprocessing / deadlock, 2 trên Linux/macOS.
    """
    os.makedirs(root, exist_ok=True)
    if num_workers is None:
        num_workers = 0 if sys.platform == "win32" else 2

    name = name.lower()
    tf_train = build_transform(name, image_size, fmnist_rgb, is_train=True)
    tf_eval = build_transform(name, image_size, fmnist_rgb, is_train=False)
    g = torch.Generator().manual_seed(seed)

    if name == "celeba":
        train, val, test = _load_celeba_splits(root, tf_train, tf_eval)
    elif name == "fashion_mnist":
        full = datasets.FashionMNIST(root, train=True, download=True, transform=tf_train)
        test = datasets.FashionMNIST(root, train=False, download=True, transform=tf_eval)
        train, val = random_split(full, [len(full) - val_size, val_size], generator=g)
    else:
        raise ValueError(f"Dataset không hỗ trợ: {name}")

    if subset is not None:
        train = Subset(train, range(min(subset, len(train))))

    pin_memory = torch.cuda.is_available()
    kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
    return (DataLoader(train, shuffle=True, drop_last=True, **kw),
            DataLoader(val, shuffle=False, **kw),
            DataLoader(test, shuffle=False, **kw))


def _verify_and_save_sample(ds_name: str, root: str = "./data/raw", sample_dir: str = "./outputs/samples"):
    print("=" * 65)
    print(f"[*] Đang khởi tạo & kiểm tra DataLoader cho: '{ds_name}' (lưu tại {root})...")
    tr, va, te = get_dataloaders(ds_name, root=root)
    x = next(iter(tr))[0]
    print(f"[+] {ds_name} train/val/test: {len(tr.dataset):,} / {len(va.dataset):,} / {len(te.dataset):,}")
    print(f"[+] Batch shape: {tuple(x.shape)} | min/max: {x.min().item():.2f} / {x.max().item():.2f}")
    
    # Tạo thư mục chứa ảnh mẫu nếu chưa tồn tại và lưu ảnh vào đó
    os.makedirs(sample_dir, exist_ok=True)
    out_path = os.path.join(sample_dir, f"sample_{ds_name}.png")
    save_image(denormalize(x[:64]), out_path, nrow=8)
    print(f"[+] Hoàn tất! Đã lưu 64 ảnh mẫu vào: '{out_path}'")


if __name__ == "__main__":
    # Chạy `python dataloader.py` không tham số sẽ tự động tải, chia tập và kiểm tra cả hai bộ dữ liệu
    target_datasets = [sys.argv[1]] if len(sys.argv) > 1 else ["fashion_mnist", "celeba"]

    for ds in target_datasets:
        try:
            _verify_and_save_sample(ds, root="./data/raw")
        except Exception as err:
            print(f"[!] Lỗi khi xử lý '{ds}': {err}")