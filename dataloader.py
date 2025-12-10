import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import pandas as pd
from experiment_config import config

# --- IMPORT MONAI ---
from monai.transforms import (
    Compose,
    EnsureChannelFirst,
    ScaleIntensityRange,
    RandAffine,
    RandGaussianNoise,
    RandGaussianSmooth,
    RandAdjustContrast,
    SpatialPad,
    CenterSpatialCrop,
    Resize,
    RandScaleIntensity,
    RandShiftIntensity,
    RandFlip,
    RandHistogramShift,
    RandBiasField,
    ToTensor,
)

# Hàm khởi tạo seed cho worker để đảm bảo tái lập kết quả
def worker_init_fn(worker_id):
    seed = int(torch.utils.data.get_worker_info().seed) % (2**32)
    np.random.seed(seed=seed)

class CTCaseDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        dataset: pd.DataFrame,
        mode: str = "3D",
        is_train: bool = True,  # Cờ để bật/tắt Augmentation
    ):
        self.data_dir = Path(data_dir)
        self.dataset = dataset
        self.mode = mode
        self.is_train = is_train
        
        # Kích thước crop mong muốn (từ config)
        # Lưu ý: MONAI nhận kích thước theo thứ tự (Spatial dims)
        # Ví dụ: (64, 64, 64)
        self.local_size = (config.SIZE_PX, config.SIZE_PX, config.SIZE_PX)
        
        # Global size: Lấy rộng hơn gấp đôi (hoặc theo config PATCH_SIZE nếu bạn muốn)
        # Ở đây tôi set cứng logic Global = 2x Local để bao quát ngữ cảnh
        self.global_crop_size = (config.SIZE_PX, 128, 128) 
        
        # --- ĐỊNH NGHĨA CÁC PHÉP BIẾN ĐỔI (TRANSFORMS) ---
        
        # 1. Tiền xử lý cơ bản (Luôn chạy)
        # - Cắt HU từ -1000 đến 400 (Lung Window)
        # - Chuẩn hóa về [0, 1]
        self.base_transforms = Compose([
            EnsureChannelFirst(channel_dim="no_channel"), # Tạo kênh channel đầu tiên (1, D, H, W)
            ScaleIntensityRange(
                a_min=-1000.0, a_max=400.0, 
                b_min=0.0, b_max=1.0, 
                clip=True
            ),
        ])

        # 2. Augmentation (Chỉ chạy khi Train)
        # Lưu ý: RandAffine xử lý cả Xoay (Rotate) và Dịch chuyển (Translate)
        if self.is_train:
            self.aug_transforms = Compose([
                # --- 1. BIẾN ĐỔI HÌNH HỌC (Dùng MONAI thay vì Scipy cho nhanh) ---
                RandAffine(
                    prob=0.5,
                    rotate_range=(np.pi/9, np.pi/9, np.pi/9), # +/- 20 độ
                    translate_range=(3, 3, 3), 
                    scale_range=(0.1, 0.1, 0.1),
                    mode="bilinear", padding_mode="border"
                ),
                RandFlip(prob=0.5, spatial_axis=0), # Lật ảnh
                RandFlip(prob=0.5, spatial_axis=1),
                RandFlip(prob=0.5, spatial_axis=2),

                # --- 2. BIẾN ĐỔI CƯỜNG ĐỘ (Học từ Code mẫu - Rất quan trọng) ---
                # Giúp model chống lại sự khác biệt giữa các máy chụp CT
                RandGaussianNoise(prob=0.4, mean=0.0, std=0.1),
                RandGaussianSmooth(prob=0.2, sigma_x=(0.5, 1.0), sigma_y=(0.5, 1.0), sigma_z=(0.5, 1.0)),
                RandScaleIntensity(prob=0.5, factors=0.25),
                RandShiftIntensity(prob=0.5, offsets=0.1),
                RandAdjustContrast(prob=0.5, gamma=(0.7, 1.5)),
                
                # Những cái này code mẫu có, rất tốt cho CT:
                RandHistogramShift(prob=0.3, num_control_points=10),
                RandBiasField(prob=0.3, degree=2, coeff_range=(0.0, 0.1)),
            ])
        else:
            self.aug_transforms = None

    def __getitem__(self, idx):
        # 1. Lấy thông tin từ DataFrame
        row = self.dataset.iloc[idx]
        annotation_id = row.AnnotationID
        label = row.label

        # 2. Load dữ liệu thô (Numpy)
        # Lưu ý: Code này giả định file .npy đã được crop sơ bộ (128x128x64) từ bước extract_nodules
        # và đã được resample về 1mm/pixel chuẩn.
        image_path = self.data_dir / "image" / f"{annotation_id}.npy"
        
        try:
            img_data = np.load(image_path) # Shape gốc: (D, H, W) ví dụ (64, 128, 128)
        except Exception as e:
            # Fallback nếu lỗi file (trả về tensor 0 để không crash luồng train)
            print(f"Error loading {image_path}: {e}")
            img_data = np.zeros((64, 128, 128), dtype=np.float32)

        # Chuyển sang Float32
        img_data = img_data.astype(np.float32)

        # 3. Tiền xử lý cơ bản (Windowing + Normalize + Channel First)
        # Input: (D, H, W) -> Output: (1, D, H, W)
        img_tensor = self.base_transforms(img_data)

        # 4. Cắt Local & Global (Dual Path)
        
        # --- LOCAL PATH (64x64x64) ---
        # Cắt chính giữa tâm
        local_cropper = CenterSpatialCrop(roi_size=self.local_size)
        patch_local = local_cropper(img_tensor)

        # --- GLOBAL PATH (64x128x128 -> Resize về 64x64x64) ---
        # Cắt rộng hơn (theo kích thước global định nghĩa ở init)
        global_cropper = CenterSpatialCrop(roi_size=self.global_crop_size)
        # Nếu ảnh gốc nhỏ hơn global crop size, MONAI sẽ tự pad (nếu dùng SpatialPad trước đó)
        # Nhưng ở đây ta giả định input extract_nodules đã đủ lớn.
        
        # Nếu ảnh gốc nhỏ hơn crop size thì phải Pad trước
        if img_tensor.shape[1] < self.global_crop_size[0] or \
           img_tensor.shape[2] < self.global_crop_size[1] or \
           img_tensor.shape[3] < self.global_crop_size[2]:
            padder = SpatialPad(spatial_size=self.global_crop_size)
            img_tensor_padded = padder(img_tensor)
            patch_global_raw = global_cropper(img_tensor_padded)
        else:
            patch_global_raw = global_cropper(img_tensor)

        # 5. Augmentation (Chỉ áp dụng khi Train)
        if self.is_train and self.aug_transforms is not None:
            # --- ĐỒNG BỘ AUGMENTATION (Sync Transforms) ---
            # Để Local và Global xoay cùng một hướng, ta nối chúng lại, augment, rồi tách ra.
            # Tuy nhiên, do kích thước khác nhau, ta augmentation trên Local, 
            # còn Global ta chỉ áp dụng Intensity Augmentation (hoặc giữ nguyên) để giữ ngữ cảnh.
            
            # Cách 1: Chỉ Augment Local (Decoupled - Dễ làm, hiệu quả cao)
            patch_local = self.aug_transforms(patch_local)
            
            # Cách 2 (Nâng cao): Nếu muốn Global cũng xoay, cần dùng set_random_state của MONAI
            # Nhưng ở đây ta dùng Cách 1 cho đơn giản và ổn định như đã thảo luận.
            
            # Thêm nhiễu nhẹ cho Global (Intensity only)
            intensity_aug = Compose([
                RandGaussianNoise(prob=0.1, std=0.02),
                RandAdjustContrast(prob=0.1, gamma=(0.8, 1.2))
            ])
            patch_global_raw = intensity_aug(patch_global_raw)

        # Resize Global về cùng kích thước với Local (để đưa vào Dual Path Model)
        # Resize từ (1, 64, 128, 128) -> (1, 64, 64, 64)
        resizer = Resize(spatial_size=self.local_size, mode="trilinear")
        patch_global = resizer(patch_global_raw)

        # 6. Chuẩn bị đầu ra
        # Tạo nhãn Tensor
        target = torch.tensor([label], dtype=torch.float32) # Shape (1,)

        return {
            "image_local": patch_local,   # (1, 64, 64, 64)
            "image_global": patch_global, # (1, 64, 64, 64) - đã resize
            "label": target,
            "ID": annotation_id
        }

    def __len__(self):
        return len(self.dataset)

def get_data_loader(
    data_dir,
    dataset,
    mode="3D",
    sampler=None,
    workers=4,
    batch_size=16,
    # Các tham số cũ (rotations, translations...) không cần nữa vì đã xử lý trong MONAI
    rotations=None, 
    translations=None,
    size_mm=50, # Không dùng, giữ lại để tương thích signature cũ
    size_px=64, # Dùng cho local size
):
    # Xác định xem đây là tập Train hay Valid dựa vào việc có Sampler hay không
    # (Thường train mới dùng sampler, valid dùng sequential)
    # Tuy nhiên, chuẩn nhất là truyền cờ is_train từ bên ngoài. 
    # Ở đây ta dùng mẹo: nếu có sampler (WeightedRandomSampler) -> Train.
    is_train = (sampler is not None)

    # Khởi tạo Dataset
    data_set = CTCaseDataset(
        data_dir=data_dir,
        dataset=dataset,
        mode=mode,
        is_train=is_train # Bật tắt Augmentation dựa vào đây
    )

    shuffle = False
    if sampler is None:
        # Nếu không có sampler (tức là Valid hoặc Train shuffle thường), ta bật shuffle
        # Lưu ý: Valid set thường không cần shuffle, nhưng shuffle cũng không sao.
        # Để an toàn cho logic "is_train", ta set shuffle=True nếu là train_loader không sampler.
        # Nhưng đơn giản nhất: Train -> Shuffle/Sampler, Valid -> No Shuffle.
        shuffle = is_train 

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        sampler=sampler,
        worker_init_fn=worker_init_fn,
    )

    return data_loader

# --- TEST BLOCK ---
if __name__ == "__main__":
    # Test thử xem chạy được không
    print("Testing MONAI Dataloader...")
    # Giả lập config và data
    from experiment_config import config
    
    # Load thử file csv valid
    try:
        df = pd.read_csv(config.CSV_DIR_VALID)
        print(f"Loaded {len(df)} samples from valid csv.")
        
        # Tạo loader (giả lập chế độ Train để test Augmentation)
        loader = get_data_loader(
            data_dir=config.DATADIR,
            dataset=df.iloc[:5], # Lấy 5 mẫu thôi
            sampler=torch.utils.data.RandomSampler(df.iloc[:5]) # Giả lập sampler
        )
        
        for batch in loader:
            loc = batch["image_local"]
            glo = batch["image_global"]
            lbl = batch["label"]
            print(f"Local Shape: {loc.shape} | Range: [{loc.min():.2f}, {loc.max():.2f}]")
            print(f"Global Shape: {glo.shape}")
            print(f"Label: {lbl}")
            break
        print("✅ Test Passed!")
    except Exception as e:
        print(f"❌ Test Failed: {e}")