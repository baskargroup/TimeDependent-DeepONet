import torch
import numpy as np
from torch.utils.data import Dataset
from einops import rearrange

class LidDrivenDataset2DTime(Dataset):
    def __init__(
        self,
        x_path,
        y_path,
        num_input_timesteps,
        final_timestep,
        every_nth_timestep,
        H,
        W,
        data_type,
        includePressure,
        max_output_channels
    ):
        self.H, self.W = H, W
        self.eff_C = (
            max_output_channels
            if includePressure
            else max_output_channels - 1
        )
        arr_x = np.load(x_path, mmap_mode='r')
        self.x = arr_x['data'] if hasattr(arr_x, 'files') else arr_x
        arr_y = np.load(y_path, mmap_mode='r')
        self.y = arr_y['data'] if hasattr(arr_y, 'files') else arr_y

        max_t = min(final_timestep, self.y.shape[1])
        self.starts = list(
            range(0, max_t - num_input_timesteps, every_nth_timestep)
        )
        self.num_input_timesteps = num_input_timesteps
        self.N = self.x.shape[0]

        # For postprocess compatibility
        self.num_samples_orig = self.N
        self.valid_start_indices = self.starts
        self.total_timesteps_available = self.y.shape[1]

    def __len__(self):
        return self.N * len(self.starts)

    def __getitem__(self, idx):
        s = idx // len(self.starts)
        t0 = self.starts[idx % len(self.starts)]
        # static fields
        re = torch.tensor(self.x[s, 0:1], dtype=torch.float32)
        sdf = torch.tensor(self.x[s, 1:2], dtype=torch.float32)

        # coords
        yv, xv = torch.meshgrid(
            torch.linspace(0, 4.0, self.H),
            torch.linspace(0, 16.0, self.W),
            indexing='ij'
        )
        coords = torch.stack([xv, yv], dim=0).float()

        # input sequence
        seq_np = self.y[
            s,
            t0 : t0 + self.num_input_timesteps,
            : self.eff_C
        ]
        seq = torch.tensor(seq_np.copy(), dtype=torch.float32)
        branch = rearrange(seq, 'nb c h w -> (nb c) h w')

        # target
        tgt_np = self.y[s, t0 + self.num_input_timesteps, : self.eff_C]
        tgt = torch.tensor(tgt_np.copy(), dtype=torch.float32)
        tgt = rearrange(tgt, 'c h w -> 1 (h w) c')

        return (branch, re, coords, sdf), tgt

    def sdf_map_for_sample(self, s: int) -> np.ndarray:
        """
        Return a boolean mask where `True` indicates outside of geometry (SDF >= 0).
        """
        sdf = self.x[s, 1]  # numpy array [H, W]
        return sdf >= 0

    def get_ground_truth_sequence(
        self, sample_idx: int, t_start: int, t_end: int) -> torch.Tensor:
        """
        Returns a tensor of shape [1, H*W, C_out] for the ground-truth at steps [t_start:t_end].
        """
        # slice y: shape [1, C_out, H, W]
        arr = self.y[sample_idx, t_start:t_end, : self.eff_C]
        t = torch.tensor(arr.copy(), dtype=torch.float32)
        # [1, C_out, H, W] → [1, H, W, C_out]
        t = t.permute(0, 2, 3, 1)
        # flatten spatial dims → [1, H*W, C_out]
        t = t.reshape(1, self.H * self.W, self.eff_C)
        return t