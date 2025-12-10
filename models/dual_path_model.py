import torch
import torch.nn as nn
import torch.nn.functional as F
# Import class ResNet cũ của bạn
from models.model_3d_resnet import LungNodule3DResNet
from models.model_3d_resnet import LungNodule3DResNet

class DualPathLungNoduleNet(nn.Module):
    def __init__(self, num_classes=1, pretrained_path=None, input_channels=1):
        super(DualPathLungNoduleNet, self).__init__()
        
        print("🌟 Initializing DUAL PATH Model...")
        
        # --- NHÁNH 1: LOCAL PATH (Chi tiết nốt) ---
        # Khởi tạo model cũ, load weights đầy đủ
        self.local_net = LungNodule3DResNet(
            num_classes=num_classes, 
            pretrained_path=pretrained_path, 
            input_channels=input_channels
        )
        # Loại bỏ lớp Linear cuối cùng (Classifier) để lấy Feature Vector (512 chiều)
        # Trong torchvision r3d_18, lớp cuối tên là 'fc'. Ta thay bằng Identity (giữ nguyên output trước đó)
        self.local_net.model.fc = nn.Identity()
        
        # --- NHÁNH 2: GLOBAL PATH (Ngữ cảnh) ---
        # Khởi tạo model thứ 2 y hệt
        self.global_net = LungNodule3DResNet(
            num_classes=num_classes, 
            pretrained_path=pretrained_path, 
            input_channels=input_channels
        )
        self.global_net.model.fc = nn.Identity()
        
        # --- FUSION HEAD (Đầu ra) ---
        # Đầu ra của ResNet18 là 512. Hai nhánh cộng lại là 1024.
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512 * 2, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x_local, x_global):
        """
        x_local:  Tensor [Batch, 1, 64, 64, 64] (Crop chặt)
        x_global: Tensor [Batch, 1, 64, 128, 128] (Crop rộng gốc)
        """
        
        # 1. Xử lý Global Input
        # Resize Global Input từ [64, 128, 128] xuống [64, 64, 64] để khớp với pre-trained scale
        # và giảm VRAM. Dùng chế độ 'area' hoặc 'trilinear' cho 3D.
        if x_global.shape[-1] != x_local.shape[-1]:
            x_global = F.interpolate(x_global, size=x_local.shape[2:], mode='trilinear', align_corners=False)
        
        # 2. Forward pass qua 2 nhánh backbone
        # Output sẽ là vectors [Batch, 512]
        feat_local = self.local_net(x_local)
        feat_global = self.global_net(x_global)
        
        # 3. Fusion (Nối đặc trưng)
        # Kết quả: [Batch, 1024]
        combined = torch.cat([feat_local, feat_global], dim=1)
        
        # 4. Classification
        out = self.classifier(combined)
        
        return out