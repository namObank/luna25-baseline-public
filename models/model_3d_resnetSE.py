import torch
import torch.nn as nn
import logging

# =========================================================================================
# PHẦN 1: CÁC KHỐI CƠ BẢN (BUILDING BLOCKS)
# =========================================================================================

def conv3x3x3(in_planes, out_planes, stride=1):
    """3x3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv1x1x1(in_planes, out_planes, stride=1):
    """1x1x1 convolution"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride,
                     bias=False)

class SELayer3D(nn.Module):
    """Squeeze-and-Excitation Module cho 3D"""
    def __init__(self, channel, reduction=16):
        super(SELayer3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)

class BasicBlockSE(nn.Module):
    """
    BasicBlock chuẩn của ResNet-18, tích hợp thêm SE.
    Tên biến (conv1, bn1...) được đặt khớp 100% với MedicalNet.
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlockSE, self).__init__()
        # Conv1 + BN1
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        
        # Conv2 + BN2
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        
        # SE Module
        self.se = SELayer3D(planes)
        
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Thêm SE vào trước khi cộng residual
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

# =========================================================================================
# PHẦN 2: MODEL CHÍNH (REBUILT FROM SCRATCH)
# =========================================================================================

class LungNodule3DSEResNet(nn.Module):
    def __init__(self, num_classes=1, input_channels=1, pretrained_path=None):
        super(LungNodule3DSEResNet, self).__init__()
        
        logging.info(f"🏗️ Initializing Custom SE-ResNet-18 3D | Input: {input_channels}ch")
        
        self.inplanes = 64
        
        # --- Layer đầu tiên (Stem) ---
        # MedicalNet dùng kernel 7x7x7, stride 1,2,2 (để giữ độ sâu)
        # Input channels mặc định là 1 cho MedicalNet gốc
        self.conv1 = nn.Conv3d(input_channels, 64, kernel_size=7, stride=(1, 2, 2),
                               padding=(3, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        # --- Các lớp ResNet ---
        self.layer1 = self._make_layer(BasicBlockSE, 64, 2)
        self.layer2 = self._make_layer(BasicBlockSE, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlockSE, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlockSE, 512, 2, stride=2)

        # --- Đầu ra (Head) ---
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # Classifier
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512 * BasicBlockSE.expansion, num_classes)
        )

        # Khởi tạo trọng số
        self._init_weights()

        # Load Pretrained
        if pretrained_path:
            self._load_pretrained_weights(pretrained_path, input_channels)

        logging.info("✅ Model ready.")

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _load_pretrained_weights(self, path, input_channels):
        logging.info(f"⏳ Loading backbone weights from {path}...")
        try:
            # Load file
            checkpoint = torch.load(path, map_location='cpu')
            state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

            # Tạo dict mới để chứa weight đã lọc
            new_state_dict = {}
            model_state = self.state_dict()
            
            loaded_layers = 0

            for k, v in state_dict.items():
                # Xóa prefix 'module.' nếu có
                name = k.replace("module.", "")
                
                # Xử lý lớp đầu tiên (conv1) nếu số kênh khác nhau
                if name == "conv1.weight":
                    # MedicalNet gốc là 1 kênh.
                    # Nếu input_channels > 1 (ví dụ 3 kênh), ta phải nhân bản weight
                    if v.shape[1] != input_channels:
                        logging.info(f"   -> Converting conv1 weights from {v.shape[1]}ch to {input_channels}ch")
                        # Kỹ thuật: copy weights sang các kênh mới
                        # (Với CT 1 kênh -> 1 kênh thì đoạn này không chạy, giữ nguyên v)
                        if input_channels == 1 and v.shape[1] == 3:
                             # Trường hợp hiếm: file weight là 3 kênh (video), model là 1 kênh
                             v = torch.mean(v, dim=1, keepdim=True)
                    
                # Mapping keys: 
                # MedicalNet keys rất chuẩn: conv1, bn1, layer1.0.conv1 ...
                # Code này viết khớp tên biến nên gần như map 1-1
                
                if name in model_state:
                    # Kiểm tra size lần cuối cho chắc
                    if model_state[name].shape == v.shape:
                        new_state_dict[name] = v
                        loaded_layers += 1
                    else:
                        logging.warning(f"   ⚠️ Shape mismatch for {name}: File{v.shape} vs Model{model_state[name].shape}")
            
            # Load vào model (strict=False để bỏ qua SE layers và FC head)
            self.load_state_dict(new_state_dict, strict=False)
            
            # Kiểm tra sơ bộ
            if loaded_layers > 100:
                logging.info(f"✅ SUCCESS: Loaded {loaded_layers} layers from MedicalNet.")
            else:
                logging.warning(f"⚠️ Loaded only {loaded_layers} layers. Something might be wrong with key names.")

        except Exception as e:
            logging.error(f"❌ Error loading weights: {e}")

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.fc(x)

        return x