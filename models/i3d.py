import math
import os
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from experiment_config import config

# =============================================================================
# PHẦN 1: CÁC HÀM HỖ TRỢ (HELPER FUNCTIONS)
# =============================================================================

def get_padding_shape(filter_shape, stride):
    def _pad_top_bottom(filter_dim, stride_val):
        pad_along = max(filter_dim - stride_val, 0)
        pad_top = pad_along // 2
        pad_bottom = pad_along - pad_top
        return pad_top, pad_bottom

    padding_shape = []
    for filter_dim, stride_val in zip(filter_shape, stride):
        pad_top, pad_bottom = _pad_top_bottom(filter_dim, stride_val)
        padding_shape.append(pad_top)
        padding_shape.append(pad_bottom)
    depth_top = padding_shape.pop(0)
    depth_bottom = padding_shape.pop(0)
    padding_shape.append(depth_top)
    padding_shape.append(depth_bottom)

    return tuple(padding_shape)


def simplify_padding(padding_shapes):
    all_same = True
    padding_init = padding_shapes[0]
    for pad in padding_shapes[1:]:
        if pad != padding_init:
            all_same = False
    return all_same, padding_init


class Unit3Dpy(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=(1, 1, 1),
        stride=(1, 1, 1),
        activation="relu",
        padding="SAME",
        use_bias=False,
        use_bn=True,
    ):
        super(Unit3Dpy, self).__init__()

        self.padding = padding
        self.activation = activation
        self.use_bn = use_bn
        if padding == "SAME":
            padding_shape = get_padding_shape(kernel_size, stride)
            simplify_pad, pad_size = simplify_padding(padding_shape)
            self.simplify_pad = simplify_pad
        elif padding == "VALID":
            padding_shape = 0
        else:
            raise ValueError(
                "padding should be in [VALID|SAME] but got {}".format(padding)
            )

        if padding == "SAME":
            if not simplify_pad:
                self.pad = torch.nn.ConstantPad3d(padding_shape, 0)
                self.conv3d = torch.nn.Conv3d(
                    in_channels, out_channels, kernel_size, stride=stride, bias=use_bias
                )
            else:
                self.conv3d = torch.nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    padding=pad_size,
                    bias=use_bias,
                )
        elif padding == "VALID":
            self.conv3d = torch.nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding_shape,
                stride=stride,
                bias=use_bias,
            )
        else:
            raise ValueError(
                "padding should be in [VALID|SAME] but got {}".format(padding)
            )

        if self.use_bn:
            self.batch3d = torch.nn.BatchNorm3d(out_channels)

        if activation == "relu":
            self.activation = torch.nn.functional.relu

    def forward(self, inp):
        if self.padding == "SAME" and self.simplify_pad is False:
            inp = self.pad(inp)
        out = self.conv3d(inp)
        if self.use_bn:
            out = self.batch3d(out)
        if self.activation is not None:
            out = torch.nn.functional.relu(out)
        return out


class MaxPool3dTFPadding(torch.nn.Module):
    def __init__(self, kernel_size, stride=None, padding="SAME"):
        super(MaxPool3dTFPadding, self).__init__()
        if padding == "SAME":
            padding_shape = get_padding_shape(kernel_size, stride)
            self.padding_shape = padding_shape
            self.pad = torch.nn.ConstantPad3d(padding_shape, 0)
        self.pool = torch.nn.MaxPool3d(kernel_size, stride, ceil_mode=True)

    def forward(self, inp):
        inp = self.pad(inp)
        out = self.pool(inp)
        return out


class Mixed(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Mixed, self).__init__()
        # Branch 0
        self.branch_0 = Unit3Dpy(in_channels, out_channels[0], kernel_size=(1, 1, 1))

        # Branch 1
        branch_1_conv1 = Unit3Dpy(in_channels, out_channels[1], kernel_size=(1, 1, 1))
        branch_1_conv2 = Unit3Dpy(
            out_channels[1], out_channels[2], kernel_size=(3, 3, 3)
        )
        self.branch_1 = torch.nn.Sequential(branch_1_conv1, branch_1_conv2)

        # Branch 2
        branch_2_conv1 = Unit3Dpy(in_channels, out_channels[3], kernel_size=(1, 1, 1))
        branch_2_conv2 = Unit3Dpy(
            out_channels[3], out_channels[4], kernel_size=(3, 3, 3)
        )
        self.branch_2 = torch.nn.Sequential(branch_2_conv1, branch_2_conv2)

        # Branch3
        branch_3_pool = MaxPool3dTFPadding(
            kernel_size=(3, 3, 3), stride=(1, 1, 1), padding="SAME"
        )
        branch_3_conv2 = Unit3Dpy(in_channels, out_channels[5], kernel_size=(1, 1, 1))
        self.branch_3 = torch.nn.Sequential(branch_3_pool, branch_3_conv2)

    def forward(self, inp):
        out_0 = self.branch_0(inp)
        out_1 = self.branch_1(inp)
        out_2 = self.branch_2(inp)
        out_3 = self.branch_3(inp)
        out = torch.cat((out_0, out_1, out_2, out_3), 1)
        return out

# =============================================================================
# PHẦN 2: I3D MODEL CHÍNH (UPDATED FOR DUAL PATH & CT SCANS)
# =============================================================================

class I3D(torch.nn.Module):
    def __init__(
        self,
        num_classes,
        input_channels=1, # Mặc định là 1 cho ảnh CT
        modality="rgb",
        dropout_prob=0,
        name="inception",
        pre_trained=True,
        freeze_bn=True,
        return_features=False, # Cờ mới: True để lấy vector đặc trưng (1024)
    ):
        super(I3D, self).__init__()
        self.name = name
        self.num_classes = num_classes
        self.freeze_bn = freeze_bn
        self.return_features = return_features 
        self.input_channels = input_channels
        
        if modality == "rgb":
            in_channels = 3
        elif modality == "flow":
            in_channels = 2
        else:
            raise ValueError(f"{modality} not among known modalities [rgb|flow]")
        
        self.modality = modality

        # --- KHỞI TẠO CÁC LAYER INCEPTION ---
        conv3d_1a_7x7 = Unit3Dpy(out_channels=64, in_channels=in_channels, kernel_size=(7, 7, 7), stride=(2, 2, 2), padding="SAME")
        self.conv3d_1a_7x7 = conv3d_1a_7x7
        self.maxPool3d_2a_3x3 = MaxPool3dTFPadding(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding="SAME")
        self.conv3d_2b_1x1 = Unit3Dpy(out_channels=64, in_channels=64, kernel_size=(1, 1, 1), padding="SAME")
        self.conv3d_2c_3x3 = Unit3Dpy(out_channels=192, in_channels=64, kernel_size=(3, 3, 3), padding="SAME")
        self.maxPool3d_3a_3x3 = MaxPool3dTFPadding(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding="SAME")
        self.mixed_3b = Mixed(192, [64, 96, 128, 16, 32, 32])
        self.mixed_3c = Mixed(256, [128, 128, 192, 32, 96, 64])
        self.maxPool3d_4a_3x3 = MaxPool3dTFPadding(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding="SAME")
        self.mixed_4b = Mixed(480, [192, 96, 208, 16, 48, 64])
        self.mixed_4c = Mixed(512, [160, 112, 224, 24, 64, 64])
        self.mixed_4d = Mixed(512, [128, 128, 256, 24, 64, 64])
        self.mixed_4e = Mixed(512, [112, 144, 288, 32, 64, 64])
        self.mixed_4f = Mixed(528, [256, 160, 320, 32, 128, 128])
        self.maxPool3d_5a_2x2 = MaxPool3dTFPadding(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding="SAME")
        self.mixed_5b = Mixed(832, [256, 160, 320, 32, 128, 128])
        self.mixed_5c = Mixed(832, [384, 192, 384, 48, 128, 128])

        # --- MODIFICATION: Sửa Pooling để chạy được với input nhỏ (64x64x64) ---
        # Dùng AdaptiveAvgPool3d(1) thay vì AvgPool3d((2, 7, 7)) cố định
        self.avg_pool = torch.nn.AdaptiveAvgPool3d(1)
        
        self.dropout = torch.nn.Dropout(dropout_prob)

        # Lớp Classifier (Chỉ dùng khi return_features=False)
        self.conv3d_0c_1x1 = Unit3Dpy(
            in_channels=1024, 
            out_channels=self.num_classes, 
            kernel_size=(1, 1, 1), 
            activation=None, 
            use_bias=True, 
            use_bn=False
        )

        # --- LOGIC LOAD WEIGHTS ĐẦY ĐỦ ---
        if pre_trained:
            logging.info(f"⏳ Loading I3D weights from {config.MODEL_RGB_I3D}...")
            try:
                # Load file checkpoint
                pretrained_dict = torch.load(config.MODEL_RGB_I3D, map_location='cpu')
                model_dict = self.state_dict()
                
                new_state_dict = {}
                loaded_count = 0

                for old_key, weights in pretrained_dict.items():
                    # Mapping tên biến từ file gốc sang cấu trúc class hiện tại
                    new_key = old_key.replace('Conv3d', 'conv3d').replace('Mixed', 'mixed')
                    new_key = new_key.replace('.b0.', '.branch_0.')
                    new_key = new_key.replace('.b1a.', '.branch_1.0.')
                    new_key = new_key.replace('.b1b.', '.branch_1.1.')
                    new_key = new_key.replace('.b2a.', '.branch_2.0.')
                    new_key = new_key.replace('.b2b.', '.branch_2.1.')
                    new_key = new_key.replace('.b3b.', '.branch_3.1.')
                    
                    # Mapping BN
                    new_key = new_key.replace('.bn.', '.batch3d.')
                    
                    # Mapping Classifier
                    new_key = new_key.replace('logits.conv3d.', 'conv3d_0c_1x1.')

                    # Kiểm tra và nạp
                    if new_key in model_dict:
                        if model_dict[new_key].shape == weights.shape:
                            new_state_dict[new_key] = weights
                            loaded_count += 1
                        else:
                            # Bỏ qua lớp cuối nếu số class không khớp (ví dụ 400 vs 1)
                            pass
                
                model_dict.update(new_state_dict)
                self.load_state_dict(model_dict, strict=False)
                logging.info(f"✅ Successfully loaded {loaded_count} layers from I3D Pre-trained.")
                
            except Exception as e:
                logging.error(f"❌ Weight loading failed: {e}")

        # Freeze Batch Norm nếu cần
        if self.freeze_bn:
            self.train() # Gọi hàm train custom bên dưới

    def train(self, mode=True):
        """Override train để đóng băng BN layers"""
        super(I3D, self).train(mode)
        if self.freeze_bn:
            for m in self.modules():
                if isinstance(m, torch.nn.BatchNorm3d):
                    m.eval()
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False

    def forward(self, inp):
        # 1. Tự động Expand channel nếu đầu vào là 1 kênh (CT) nhưng model cần 3 (RGB)
        if self.input_channels == 3 and inp.shape[1] == 1:
            inp = inp.expand(-1, 3, -1, -1, -1)
            
        # 2. Forward Pass qua backbone
        out = self.conv3d_1a_7x7(inp)
        out = self.maxPool3d_2a_3x3(out)
        out = self.conv3d_2b_1x1(out)
        out = self.conv3d_2c_3x3(out)
        out = self.maxPool3d_3a_3x3(out)
        out = self.mixed_3b(out)
        out = self.mixed_3c(out)
        out = self.maxPool3d_4a_3x3(out)
        out = self.mixed_4b(out)
        out = self.mixed_4c(out)
        out = self.mixed_4d(out)
        out = self.mixed_4e(out)
        out = self.mixed_4f(out)
        out = self.maxPool3d_5a_2x2(out)
        out = self.mixed_5b(out)
        out = self.mixed_5c(out) # Output shape tại đây: [Batch, 1024, D', H', W']

        # 3. Pooling & Dropout
        out = self.avg_pool(out) # [Batch, 1024, 1, 1, 1]
        out = self.dropout(out)

        # 4. LOGIC CHO DUAL PATH: Trả về Features
        if self.return_features:
            # Trả về vector đặc trưng [Batch, 1024] để nối với nhánh kia
            return out.view(out.size(0), -1)
        
        # 5. Logic Cũ: Classification (Dùng khi chạy đơn lẻ)
        out = self.conv3d_0c_1x1(out)
        out = out.mean(2).reshape(out.shape[0]) # [Batch]
        return out

if __name__ == "__main__":
    # Test thử
    model = I3D(num_classes=1, input_channels=1, return_features=True)
    x = torch.rand(2, 1, 64, 64, 64)
    out = model(x)
    print("Testing I3D with 1-channel input:")
    print(f"Input shape: {x.shape}")
    print(f"Output shape (Features): {out.shape}") # Mong đợi: [2, 1024]

class DualPathI3DNet(nn.Module):
    def __init__(self, num_classes=1, input_channels=1, dropout_prob=0.5):
        super(DualPathI3DNet, self).__init__()
        
        print("🌟 Initializing DUAL PATH I3D Model...")
        
        # --- NHÁNH 1: LOCAL PATH ---
        # Bật cờ return_features=True để lấy vector 1024
        self.local_net = I3D(
            num_classes=num_classes,
            input_channels=3, # I3D gốc dùng 3 kênh (code sẽ tự expand từ 1->3)
            modality='rgb',
            dropout_prob=0.5,
            return_features=True 
        )
        
        # --- NHÁNH 2: GLOBAL PATH ---
        self.global_net = I3D(
            num_classes=num_classes,
            input_channels=3,
            modality='rgb',
            dropout_prob=0.5,
            return_features=True
        )
        
        # --- FUSION HEAD ---
        # I3D feature dim = 1024. Hai nhánh = 2048.
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(1024 * 2, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob),
            nn.Linear(512, num_classes)
        )

    def forward(self, x_local, x_global):
        """
        x_local:  [Batch, 1, 64, 64, 64]
        x_global: [Batch, 1, 64, 128, 128]
        """
        
        # 1. Resize Global về cùng kích thước với Local (64^3)
        # Để giảm VRAM và đảm bảo I3D chạy mượt
        if x_global.shape[-1] != x_local.shape[-1]:
            x_global = F.interpolate(x_global, size=x_local.shape[2:], mode='trilinear', align_corners=False)
        
        # 2. Forward qua backbone -> Lấy features [Batch, 1024]
        feat_local = self.local_net(x_local)
        feat_global = self.global_net(x_global)
        
        # 3. Nối đặc trưng
        combined = torch.cat([feat_local, feat_global], dim=1) # [Batch, 2048]
        
        # 4. Phân loại
        out = self.classifier(combined)
        
        return out