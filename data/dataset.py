import torch
import numpy as np
from torch.utils.data import Dataset
from einops import rearrange

class LidDrivenDataset2DTime(Dataset):
    def __init__( self,
        x_path: str,
        y_path: str,
        num_input_timesteps: int,
        final_timestep: int,
        every_nth_timestep: int,
        H: int, W: int,
        max_output_channels: int,
        domain_length_x: float,
        domain_length_y: float):
        
        self.H, self.W = H, W
        self.domain_length_x, self.domain_length_y = domain_length_x, domain_length_y
        self.eff_C = max_output_channels - 1

        # --- load once, to numpy ---
        arr_x = np.load(x_path, mmap_mode='r')
        data_x = arr_x['data'] if hasattr(arr_x, 'files') else arr_x  # shape [N,2,H,W]
        arr_y = np.load(y_path, mmap_mode='r')
        data_y = arr_y['data'] if hasattr(arr_y, 'files') else arr_y  # shape [N,T,C_out,H,W]

        # re, sdf (each [N,1,H,W])
        self.re_tensor  = torch.tensor(data_x[:, 0:1], dtype=torch.float32)
        self.sdf_tensor = torch.tensor(data_x[:, 1:2], dtype=torch.float32)

        # shape [N, T, C_out, H, W]
        self.y_tensor   = torch.tensor(data_y, dtype=torch.float32)
        
        x_lin = torch.linspace(0., self.domain_length_x, self.W)
        y_lin = torch.linspace(0., self.domain_length_y, self.H)
        yv, xv = torch.meshgrid(y_lin, x_lin, indexing='ij')
        self.coords = torch.stack([xv, yv], dim=0)     # [2, H, W]

        # time‐index bookkeeping
        max_t = min(final_timestep, self.y_tensor.shape[1])
        self.starts = list(range(0, max_t - num_input_timesteps, every_nth_timestep))
        self.num_input_timesteps = num_input_timesteps
        self.N = self.y_tensor.shape[0]
        
        # for postprocess
        self.num_samples_orig = self.N
        self.valid_start_indices = self.starts
        self.total_timesteps_available = self.y_tensor.shape[1]

    def __len__(self):
        return self.N * len(self.starts)

    def __getitem__(self, idx):
        # which sample & which time
        s = idx // len(self.starts)
        t0 = self.starts[idx % len(self.starts)]

        # grab the static fields (no new allocation)
        re   = self.re_tensor[s]     # [1,H,W]
        sdf  = self.sdf_tensor[s]    # [1,H,W]
        coords = self.coords         # [2,H,W]

        # branch input: stack num_input_timesteps frames of [C_flow,H,W]
        # yields shape [nb, C_flow, H, W]
        seq = self.y_tensor[s, t0 : t0 + self.num_input_timesteps, : self.eff_C ]
        # flatten to [(nb*C) , H, W]
        branch = rearrange(seq, 'nb c h w -> (nb c) h w')

        # target at t0+nb: shape [C_flow, H, W] → [1, H*W, C_flow]
        tgt = self.y_tensor[s, t0 + self.num_input_timesteps, :self.eff_C]
        tgt = rearrange(tgt, 'c h w -> 1 (h w) c')

        return (branch, re, coords, sdf), tgt

    def sdf_map_for_sample(self, s: int) -> np.ndarray:
        sdf = self.sdf_tensor[s].cpu().numpy()[0]  # numpy [H,W]
        return sdf >= 0

    def get_ground_truth_sequence(
        self, sample_idx: int, t_start: int, t_end: int) -> torch.Tensor:
        arr = self.y_tensor[sample_idx, t_start:t_end, :self.eff_C]
        # [nb, C, H, W] → [1, C, H, W]
        t = arr.detach() 
        t = t.permute(0, 2, 3, 1) # → [1, H, W, C]
        t = t.reshape(1, self.H * self.W, self.eff_C) # → [1, H*W, C]
        return t
