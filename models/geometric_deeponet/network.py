import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import warnings
from typing import List

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class InceptionStyleCNNEncoder(nn.Module):
    def __init__(self, input_channels: int, c1: int, c2: int, c3: int, fc1: int, fc2: int):
        super().__init__()
        # Branch 1
        self.b1_conv1 = ConvBlock(input_channels, c1, 1)
        self.b1_pool1 = nn.MaxPool2d(2, 2)
        self.b1_conv2 = ConvBlock(c1, c2, 1)
        self.b1_pool2 = nn.MaxPool2d(2, 2)
        self.b1_conv3 = ConvBlock(c2, c3, 1)
        self.b1_pool3 = nn.MaxPool2d(2, 2)
        # Branch 2
        self.b2_conv1a = ConvBlock(input_channels, c1, 1)
        self.b2_conv1b = ConvBlock(c1, c1, 3, padding=1)
        self.b2_pool1  = nn.MaxPool2d(2, 2)
        self.b2_conv2a = ConvBlock(c1, c2, 1)
        self.b2_conv2b = ConvBlock(c2, c2, 3, padding=1)
        self.b2_pool2  = nn.MaxPool2d(2, 2)
        self.b2_conv3a = ConvBlock(c2, c3, 1)
        self.b2_conv3b = ConvBlock(c3, c3, 3, padding=1)
        self.b2_pool3  = nn.MaxPool2d(2, 2)
        # Branch 3
        self.b3_conv1a = ConvBlock(input_channels, c1, 1)
        self.b3_conv1b = ConvBlock(c1, c1, 5, padding=2)
        self.b3_pool1  = nn.MaxPool2d(2, 2)
        self.b3_conv2a = ConvBlock(c1, c2, 1)
        self.b3_conv2b = ConvBlock(c2, c2, 5, padding=2)
        self.b3_pool2  = nn.MaxPool2d(2, 2)
        self.b3_conv3a = ConvBlock(c2, c3, 1)
        self.b3_conv3b = ConvBlock(c3, c3, 5, padding=2)
        self.b3_pool3  = nn.MaxPool2d(2, 2)
        # Fusion
        concat_channels = 3 * c3
        self.fusion_conv1 = ConvBlock(concat_channels, fc1, 1)
        self.fusion_pool1 = nn.MaxPool2d(2, 2)
        self.fusion_conv2 = ConvBlock(fc1, fc2, 1)
        self.fusion_pool2 = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.final_cnn_channels = fc2

    def forward(self, x):
        p1 = self.b1_pool3(self.b1_conv3(self.b1_pool2(self.b1_conv2(self.b1_pool1(self.b1_conv1(x))))))
        p2 = self.b2_pool3(self.b2_conv3b(self.b2_conv3a(self.b2_pool2(self.b2_conv2b(self.b2_conv2a(self.b2_pool1(self.b2_conv1b(self.b2_conv1a(x)))))))))
        p3 = self.b3_pool3(self.b3_conv3b(self.b3_conv3a(self.b3_pool2(self.b3_conv2b(self.b3_conv2a(self.b3_pool1(self.b3_conv1b(self.b3_conv1a(x)))))))))
        c  = torch.cat((p1, p2, p3), dim=1)
        f  = self.fusion_pool2(self.fusion_conv2(self.fusion_pool1(self.fusion_conv1(c))))
        return self.flatten(f)

class LinearMLP(nn.Module):
    def __init__(self, dims: List[int], nonlin):
        super().__init__()
        layers = []
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims)-2:
                layers.append(nonlin())
        self.mlp = nn.Sequential(*layers)
    def forward(self, x):
        return self.mlp(x)

class torchSine(nn.Module):
    def forward(self, x): return torch.sin(x)

class GeoDeepONetTime(nn.Module):
    def __init__(
        self, height: int, width: int, num_input_timesteps: int,
        input_channels_loc: int, effective_output_channels: int,
        modes: int,
        branch_stage1_layers: List[int], trunk_stage1_layers: List[int],
        branch_stage2_layers: List[int], trunk_stage2_layers: List[int],
        cnn_c1: int, cnn_c2: int, cnn_c3: int, cnn_fc1: int, cnn_fc2: int
    ):
        super().__init__()
        if input_channels_loc != 2:
            warnings.warn("GeoDeepONetTime expects input_channels_loc=2 (x,y). SDF will be added.")

        self.input_channels_loc_base = input_channels_loc
        self.input_channels_loc_effective = input_channels_loc + 1
        self.effective_output_channels = effective_output_channels
        self.modes = modes
        self.height = height; self.width = width
        self.num_points = height * width

        # --- Branch ---
        channels_per_step = self.effective_output_channels
        cnn_in_ch = num_input_timesteps * channels_per_step
        self.cnn_encoder = InceptionStyleCNNEncoder(cnn_in_ch, cnn_c1, cnn_c2, cnn_c3, cnn_fc1, cnn_fc2)
        with torch.no_grad():
            dummy = torch.zeros(1, cnn_in_ch, height, width)
            flat  = self.cnn_encoder(dummy)
            cnn_flat = flat.shape[1]
        branch_dims1 = [cnn_flat] + branch_stage1_layers + [modes]
        self.branch_stage_1 = LinearMLP(branch_dims1, nn.ReLU)
        branch_dims2 = [modes] + branch_stage2_layers + [modes * effective_output_channels]
        self.branch_stage_2 = LinearMLP(branch_dims2, nn.ReLU)

        # --- Trunk ---
        trunk_dims1 = [self.input_channels_loc_effective] + trunk_stage1_layers + [modes]
        self.trunk_stage_1 = LinearMLP(trunk_dims1, nn.ReLU)
        trunk_dims2 = [modes]                                 + trunk_stage2_layers + [modes * effective_output_channels]
        self.trunk_stage_2 = LinearMLP(trunk_dims2, torchSine)

        # --- Bias ---
        self.b = nn.Parameter(torch.tensor(0.0))

    def forward(self, inputs: tuple):
        x1, _, coords, sdf = inputs[:4]
        # --- Branch ---
        feat = self.cnn_encoder(x1)
        glob = self.branch_stage_1(feat)

        # --- Trunk ---
        # coords: [b, 2, h, w] → [b, h*w, 2]
        c2 = rearrange(coords, 'b c h w -> b (h w) c')
        # sdf:    [b, 1, h, w] → [b, h*w, 1]
        sdf_flat = rearrange(sdf, 'b 1 h w -> b (h w) 1')
        # combine into [b, h*w, 3]
        trunk_in = torch.cat((c2, sdf_flat), dim=-1)
        # pass each point through the trunk MLP → [b, h*w, modes]
        local = self.trunk_stage_1(trunk_in)

        # --- Merge & Stage2 ---
        # glob: [b, modes] → [b, 1, modes], local: [b, h*w, modes]
        merged = glob.unsqueeze(1) * local
        avg = merged.mean(dim=1)

        out_b = self.branch_stage_2(avg)       # [b, modes*eff_out]
        out_t = self.trunk_stage_2(merged)     # [b, h*w, modes*eff_out]

        # reshape for tensor contraction
        b_r = rearrange(out_b, 'b (m c) -> b m c', m=self.modes, c=self.effective_output_channels)
        t_r = rearrange(out_t, 'b p (m c) -> b p m c', m=self.modes, c=self.effective_output_channels)

        # compute solution and add bias
        sol_flat = torch.einsum('bmc,bpmc->bpc', b_r, t_r) + self.b

        # final shape [b, 1, p, c]
        return rearrange(sol_flat, 'b p c -> b 1 p c')
