import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from itertools import product
from typing import Dict


def gauss_pt_eval(tensor: torch.Tensor, kernels: nn.ParameterList, stride: int = 1) -> torch.Tensor:
    if not kernels:
        raise ValueError("No Gauss kernels provided.")
    conv = F.conv2d
    B, C = tensor.shape[0], tensor.shape[1]
    device = tensor.device
    # determine output spatial shape
    with torch.no_grad():
        sample_out = conv(tensor[:, :1], kernels[0].to(device), stride=stride)
        out_spatial = sample_out.shape[2:]

    results = []
    for k in kernels:
        k = k.to(device)
        # apply convolution per channel
        out_ch = [conv(tensor[:, i:i+1], k, stride=stride) for i in range(C)]
        results.append(torch.cat(out_ch, dim=1).unsqueeze(1))

    out = torch.cat(results, dim=1)
    expected = (B, len(kernels), C) + out_spatial
    if out.shape != expected:
        warnings.warn(f"Shape mismatch in gauss_pt_eval: {out.shape} != {expected}")
    return out


class FEM2D(nn.Module):
    """
    Builds 2D FEM convolution kernels and evaluates derivatives.
    """
    def __init__(
        self,
        height: int,
        width: int,
        domain_length_x: float,
        domain_length_y: float,
        device: torch.device
    ):
        super().__init__()
        self.height, self.width = height, width
        self.device = device
        # 2-point Gauss quadrature
        self.gpx = [-0.57735, 0.57735]
        self.kernels_dx = nn.ParameterList()
        self.kernels_dy = nn.ParameterList()
        self._build_kernels(domain_length_x, domain_length_y)

    def _build_kernels(self, Lx: float, Ly: float):
        hx = Lx / (self.width - 1)
        hy = Ly / (self.height - 1)
        # linear basis functions on [-1,1]
        bf = lambda x: [0.5 * (1 - x), 0.5 * (1 + x)]
        dbf = lambda x: [-0.5, 0.5]

        for gx, gy in product(self.gpx, repeat=2):
            dx = torch.zeros(2, 2, device=self.device)
            dy = torch.zeros(2, 2, device=self.device)
            for i, bf_x in enumerate(bf(gx)):
                for j, bf_y in enumerate(bf(gy)):
                    dx[j, i] = dbf(gx)[i] * (2 / hx) * bf_y
                    dy[j, i] = bf_x * (dbf(gy)[j] * (2 / hy))
            # store kernels with shape [1,1,2,2]
            self.kernels_dx.append(nn.Parameter(dx.unsqueeze(0).unsqueeze(0), requires_grad=False))
            self.kernels_dy.append(nn.Parameter(dy.unsqueeze(0).unsqueeze(0), requires_grad=False))

    def eval_derivative_x(self, tensor: torch.Tensor) -> torch.Tensor:
        return gauss_pt_eval(tensor, self.kernels_dx)

    def eval_derivative_y(self, tensor: torch.Tensor) -> torch.Tensor:
        return gauss_pt_eval(tensor, self.kernels_dy)


class DerivativeCalculator(nn.Module):
    """
    Computes first spatial derivatives for 'u' and 'v' channels.
    """
    def __init__(
        self,
        height: int,
        width: int,
        domain_length_x: float,
        domain_length_y: float,
        device: torch.device,
        channels: int = 2  # number of channels: 2 for (u,v)
    ):
        super().__init__()
        self.channels = channels
        self.fem = FEM2D(height, width, domain_length_x, domain_length_y, device)

    def calculate_first_derivatives(self, y_spatial: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        y_spatial: [B, C, H, W] tensor where C == channels
        Returns a dict with keys 'u_x','u_y','v_x','v_y'.
        """
        deriv = {}
        names = ['u', 'v'][:self.channels]
        for idx, name in enumerate(names):
            field = y_spatial[:, idx:idx+1]
            deriv[f'{name}_x'] = self.fem.eval_derivative_x(field)
            deriv[f'{name}_y'] = self.fem.eval_derivative_y(field)
        return deriv

    forward = calculate_first_derivatives
