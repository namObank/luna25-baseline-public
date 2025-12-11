import logging
import numpy as np
import torch
import torch.nn as nn
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
    
class I3D_Backbone(torch.nn.Module):
    def __init__(
        self,
        input_channels=1,
        modality="rgb",
        dropout_prob=0,
        name="inception",
        pre_trained=True,
        freeze_bn=True
    ):
        super(I3D_Backbone, self).__init__()
        self.name = name
        self.freeze_bn = freeze_bn
        self.input_channels = input_channels
        
        if modality == "rgb": in_channels = 3
        elif modality == "flow": in_channels = 2
        else: raise ValueError(f"{modality} not among known modalities")
        self.modality = modality

        # --- KHỞI TẠO CÁC LAYER INCEPTION ---
        self.conv3d_1a_7x7 = Unit3Dpy(out_channels=64, in_channels=in_channels, kernel_size=(7, 7, 7), stride=(2, 2, 2), padding="SAME")
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

        # Load Weights
        if pre_trained:
            self._load_pretrained_weights()

        if self.freeze_bn:
            self.train()

    def _load_pretrained_weights(self):
        logging.info(f"⏳ Loading I3D weights from {config.MODEL_RGB_I3D}...")
        try:
            pretrained_dict = torch.load(config.MODEL_RGB_I3D, map_location='cpu')
            model_dict = self.state_dict()
            new_state_dict = {}
            for old_key, weights in pretrained_dict.items():
                new_key = old_key.replace('Conv3d', 'conv3d').replace('Mixed', 'mixed')
                new_key = new_key.replace('.b0.', '.branch_0.').replace('.b1a.', '.branch_1.0.')
                new_key = new_key.replace('.b1b.', '.branch_1.1.').replace('.b2a.', '.branch_2.0.')
                new_key = new_key.replace('.b2b.', '.branch_2.1.').replace('.b3b.', '.branch_3.1.')
                new_key = new_key.replace('.bn.', '.batch3d.')
                if new_key in model_dict and model_dict[new_key].shape == weights.shape:
                    new_state_dict[new_key] = weights
            model_dict.update(new_state_dict)
            self.load_state_dict(model_dict, strict=False)
            logging.info(f"✅ I3D Backbone Loaded.")
        except Exception as e:
            logging.error(f"❌ Weight loading failed: {e}")

    def train(self, mode=True):
        super(I3D_Backbone, self).train(mode)
        if self.freeze_bn:
            for m in self.modules():
                if isinstance(m, torch.nn.BatchNorm3d):
                    m.eval()
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False

    def forward(self, inp):
        if inp.shape[1] == 1:
            inp = inp.expand(-1, 3, -1, -1, -1)
        
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
        # Output shape: [Batch, 1024, Depth, Height, Width]
        out = self.mixed_5c(out) 
        return out

# =============================================================================
# PHẦN 2: TRANSFORMER I3D (Pulse3D Style)
# =============================================================================

class TransformerI3D(nn.Module):
    def __init__(
        self,
        num_classes=1,
        input_channels=1,
        input_size=(64, 64, 64), # Kích thước đầu vào chuẩn
        num_heads=8,
        num_layers=5,
        embed_dim=1024, # Output channel của I3D là 1024
        dropout=0.1
    ):
        super(TransformerI3D, self).__init__()
        logging.info("🌟 Initializing I3D + Transformer Head...")

        # 1. Backbone: I3D (Feature Extractor)
        self.backbone = I3D_Backbone(input_channels=input_channels, pre_trained=False)
        
        # 2. Tính toán kích thước chuỗi token (Sequence Length)
        # Chạy thử một pass giả để biết feature map ra bao nhiêu
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, *input_size)
            features = self.backbone(dummy)
            # features shape: [1, 1024, D', H', W']
            self.feat_shape = features.shape[2:]
            self.num_tokens = self.feat_shape[0] * self.feat_shape[1] * self.feat_shape[2]
            logging.info(f"   -> Feature Map size: {self.feat_shape}, Num Tokens: {self.num_tokens}")

        # 3. Transformer Components
        # [CLS] Token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Positional Embedding (Learnable)
        # Kích thước: [1, Num_Tokens + 1 (CLS), Embed_Dim]
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_tokens + 1, embed_dim) * 0.02)
        self.pos_drop = nn.Dropout(p=dropout)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True # Pre-Norm (giống Pulse3D)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 4. Classification Head (MLP)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes)
        )

        # Init weights cho phần mới
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # 1. CNN Backbone
        # Input: [B, 1, 64, 64, 64] -> Output: [B, 1024, D, H, W]
        x = self.backbone(x)
        
        # 2. Flatten & Permute
        # [B, 1024, D, H, W] -> [B, 1024, N] -> [B, N, 1024]
        x = x.flatten(2).transpose(1, 2)
        
        # 3. Add CLS Token & Positional Embedding
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1) # [B, 1, 1024]
        x = torch.cat((cls_tokens, x), dim=1)         # [B, N+1, 1024]
        
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # 4. Transformer Encoder
        x = self.transformer(x)

        # 5. Classification Head
        # Lấy token đầu tiên (CLS token) làm đại diện
        cls_out = x[:, 0] 
        cls_out = self.norm(cls_out)
        out = self.head(cls_out)
        
        return out

if __name__ == "__main__":
    # Test
    model = TransformerI3D(num_classes=1, input_channels=1)
    x = torch.randn(2, 1, 64, 64, 64)
    out = model(x)
    print(f"Input: {x.shape}")
    print(f"Output: {out.shape}")