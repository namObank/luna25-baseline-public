import logging
import torch
import torch.nn as nn
import torchvision.models.video as models
import os

class LungNodule3DResNet(nn.Module):
    def __init__(
        self,
        num_classes=1,
        pretrained_path=None,
        input_channels=1,
    ):
        super(LungNodule3DResNet, self).__init__()
        
        self.input_channels = input_channels
        logging.info(f"🏗️ Initializing 3D ResNet-18 | Input: {input_channels}ch | Output: {num_classes} classes")
        
        # 1. Khởi tạo backbone chuẩn
        self.model = models.r3d_18(weights=None)

        self._modify_first_layer(input_channels)

        if pretrained_path:
            if os.path.exists(pretrained_path):
                self._load_pretrained_weights(pretrained_path)
            else:
                logging.warning(f"⚠️ Pretrained path '{pretrained_path}' not found. Initializing randomly.")

        # 3. Sửa lớp cuối cùng (FC)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)
        
        logging.info("✅ Model ready.")

    def _modify_first_layer(self, new_in_channels):
        """
        Thay thế layer đầu tiên để khớp với kiến trúc của MedicalNet (ResNet23).
        MedicalNet: Kernel=(7, 7, 7), Stride=(1, 2, 2), Padding=(3, 3, 3)
        """
        old_conv = self.model.stem[0]
        
        # Tạo layer mới đúng chuẩn MedicalNet (Kernel 7x7x7)
        # Lưu ý: Padding phải là (3, 3, 3) để giữ kích thước với kernel 7
        new_conv = nn.Conv3d(
            in_channels=new_in_channels,
            out_channels=old_conv.out_channels,
            kernel_size=(7, 7, 7),  # MedicalNet dùng 7x7x7
            stride=(1, 2, 2),       # Giữ nguyên stride của r3d_18
            padding=(3, 3, 3),      # Padding cho kernel 7
            bias=False
        )
        
        # Thay thế vào model
        self.model.stem[0] = new_conv
        logging.info(f"   -> Modified Stem: Kernel updated to (7, 7, 7) to match MedicalNet.")

    def _load_pretrained_weights(self, path):
        logging.info(f"⏳ Loading weights from: {path}")
        try:
            checkpoint = torch.load(path, map_location='cpu')
            state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

            new_state_dict = {}
            for k, v in state_dict.items():
                # Xóa prefix module.
                name = k.replace("module.", "")
                
                # Mapping tên layer từ MedicalNet sang Torchvision
                if name.startswith("conv1"):
                    name = name.replace("conv1", "stem.0")
                elif name.startswith("bn1"):
                    name = name.replace("bn1", "stem.1")
                
                new_state_dict[name] = v
            
            # Load weight (strict=False để bỏ qua lớp fc cuối cùng vì khác số class)
            msg = self.model.load_state_dict(new_state_dict, strict=False)
            logging.info(f"   -> Weights loaded successfully. Missing keys (expected for fc): {len(msg.missing_keys)}")
            
        except Exception as e:
            logging.error(f"❌ Error loading weights: {e}")
            # Nếu lỗi thì raise luôn để dừng chương trình, không train với random weight
            raise e 

    def forward(self, x):
        return self.model(x)