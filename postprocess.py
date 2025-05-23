import os
import glob
import time
from typing import Set
from collections import defaultdict

import torch
import numpy as np
import numpy.ma as ma
from omegaconf import OmegaConf
from tqdm import tqdm
from einops import rearrange

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

import warnings
warnings.filterwarnings('ignore', message='networkx backend defined more than once')

from data.dataset import LidDrivenDataset2DTime
from models.geometric_deeponet.geometric_deeponet import GeometricDeepONetTime


# color / norm presets
FIELD_CMAP    = plt.get_cmap('turbo')
ERROR_CMAP    = plt.get_cmap('turbo')
FIELD_CMAP_T  = FIELD_CMAP.copy();  FIELD_CMAP_T.set_bad(color='none')
ERROR_CMAP_T  = ERROR_CMAP.copy();  ERROR_CMAP_T.set_bad(color='none')
FIELD_VMIN    = {'U': 0, 'V': -1}
FIELD_VMAX    = {'U': 2, 'V': 1}
ERROR_VMIN, ERROR_VMAX = 1e-4, 1.0


def load_config(path: str):
    cfg = OmegaConf.load(path)
    return OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))


def load_model(ckpt_path: str, device: torch.device):
    if os.path.isdir(ckpt_path):
        ckpts = sorted(glob.glob(os.path.join(ckpt_path, '**/*.ckpt'), recursive=True))
        ckpt_path = ckpts[-1]
    model = GeometricDeepONetTime.load_from_checkpoint(ckpt_path, map_location=device)
    return model.eval().to(device)


def compute_metrics(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray):
    mets, diffs, gts = {}, [], []
    fluid = ~mask.flatten()
    for i, ch in enumerate(['U', 'V']):
        d = pred[i].flatten()[fluid] - gt[i].flatten()[fluid]
        mets[f'rel_{ch}']   = np.linalg.norm(d) / np.linalg.norm(gt[i].flatten()[fluid])
        mets[f'rmse_{ch}']  = np.sqrt((d**2).mean())
        mets[f'linf_{ch}'] = np.max(np.abs(d))
        diffs.append(d); gts.append(gt[i].flatten()[fluid])
    all_d = np.concatenate(diffs)
    all_g = np.concatenate(gts)
    mets['rel_Overall']   = np.linalg.norm(all_d)  / np.linalg.norm(all_g)
    mets['rmse_Overall']  = np.sqrt((all_d**2).mean())
    mets['linf_Overall'] = np.max(np.abs(all_d))
    return mets


def _plot_grid(pred, gt, mask, out_dir, s, t):
    C, H, W = gt.shape
    data = [gt, pred, np.abs(pred - gt)]
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(3, C, figsize=(6*C, 7.5))
    fig.subplots_adjust(wspace=0.3, hspace=0.02)

    for row_idx, (cmap, norm) in enumerate([
        (FIELD_CMAP, None),
        (FIELD_CMAP, None),
        (ERROR_CMAP, LogNorm(ERROR_VMIN, ERROR_VMAX))
    ]):
        for j in range(C):
            ax = axes[row_idx][j]
            im = ax.imshow(
                ma.masked_where(mask, data[row_idx][j]),
                cmap=cmap, norm=norm,
                vmin=None if row_idx == 2 else FIELD_VMIN[['U','V'][j]],
                vmax=None if row_idx == 2 else FIELD_VMAX[['U','V'][j]],
                origin='lower'
            )
            ax.set_xticks([]); ax.set_yticks([])
            cax = make_axes_locatable(ax).append_axes('right', size='5%', pad=0.05)
            fmt = '%.1e' if row_idx == 2 else '%.1f'
            fig.colorbar(im, cax=cax, format=fmt)

    fig.suptitle(f"Sample {s}, Timestep {t}", fontsize=16)
    fig.savefig(os.path.join(out_dir, f"grid_s{s}_t{t}.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)


def _plot_individual(pred, gt, mask, sample_dir, s, t):
    data = [gt, pred, np.abs(pred - gt)]
    indiv = os.path.join(sample_dir, 'individual_plots')
    os.makedirs(indiv, exist_ok=True)

    def save_cb(cmap, norm, fname, ticks=None, ticklabels=None):
        fig, ax = plt.subplots(figsize=(1.2, 4))
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cb = plt.colorbar(sm, cax=ax, orientation='vertical')
        if ticks:     cb.set_ticks(ticks)
        if ticklabels: cb.set_ticklabels(ticklabels)
        fig.savefig(os.path.join(indiv, fname), dpi=150, transparent=True, bbox_inches='tight')
        plt.close(fig)

    save_cb(FIELD_CMAP_T, None,   'u_colorbar.png')
    save_cb(FIELD_CMAP_T, None,   'v_colorbar.png')
    save_cb(ERROR_CMAP_T, LogNorm(ERROR_VMIN, ERROR_VMAX),
            'error_colorbar.png', ticks=[1,1e-2,1e-4], ticklabels=['0','-2','-4'])

    for i, kind in enumerate(['GT','Pred','Err']):
        cmap = ERROR_CMAP_T if i==2 else FIELD_CMAP_T
        norm = LogNorm(ERROR_VMIN, ERROR_VMAX) if i==2 else None
        for j, lbl in enumerate(['U','V']):
            fig, ax = plt.subplots(figsize=(5, 4), facecolor='none')
            ax.imshow(
                ma.masked_where(mask, data[i][j]),
                cmap=cmap, norm=norm,
                vmin=None if i==2 else FIELD_VMIN[lbl],
                vmax=None if i==2 else FIELD_VMAX[lbl],
                origin='lower'
            )
            ax.set_xticks([]); ax.set_yticks([])
            fig.savefig(os.path.join(indiv, f"{lbl}_{kind}_s{s}_t{t}.png"),
                        dpi=150, transparent=True, bbox_inches='tight')
            plt.close(fig)


def evaluate(model, device, cfg, mode: str, save_ids: Set[int]):
    ds = LidDrivenDataset2DTime(
        cfg.data.file_path_test_x,
        cfg.data.file_path_test_y,
        cfg.model.num_input_timesteps,
        cfg.model.final_timestep,
        cfg.data.every_nth_timestep,
        cfg.model.height,
        cfg.model.width,
        cfg.model.output_channels,
        cfg.model.domain_length_x,
        cfg.model.domain_length_y
    )
    N      = ds.num_samples_orig
    starts = ds.valid_start_indices
    num_in = cfg.model.num_input_timesteps
    eff_C  = ds.eff_C
    base   = cfg.model.plot_path
    selected_ts = [20, 30, 40, 50, min(cfg.model.final_timestep, ds.total_timesteps_available)]
    results      = defaultdict(list)
    selected     = {t: [] for t in selected_ts}
    total_time = count = 0

    # physical-coordinate grid
    yv, xv = torch.meshgrid(
        torch.linspace(0., cfg.model.domain_length_y, ds.H),
        torch.linspace(0., cfg.model.domain_length_x, ds.W),
        indexing='ij'
    )
    coord_grid = torch.stack([xv, yv], dim=0).unsqueeze(0).to(device)

    os.makedirs(os.path.join(base, mode), exist_ok=True)

    for s in tqdm(range(N), desc=mode):
        sample_plot = s in save_ids
        if sample_plot:
            sd = os.path.join(base, mode, f"sample_{s}")
            os.makedirs(sd, exist_ok=True)
            fpt = open(os.path.join(sd, f"{mode}_point_pred.csv"), 'w')
            fpt.write("time,u_gt1,u_pr1,v_gt1,v_pr1,u_gt2,u_pr2,v_gt2,v_pr2\n")

        # single Re+SDF per sample
        (_, re0, _, sdf0), _ = ds[s * len(starts)]
        re0, sdf0 = re0.unsqueeze(0).to(device), sdf0.unsqueeze(0).to(device)
        mask = ~ds.sdf_map_for_sample(s)

        for i, t0 in enumerate(starts):
            (branch, _, _, _), tgt = ds[s * len(starts) + i]
            branch = branch.unsqueeze(0).to(device)

            with torch.no_grad():
                t_start = time.time()
                pred    = model((branch, re0, coord_grid, sdf0))
                total_time += time.time() - t_start
                count += 1

            arrp = pred[0,0].cpu().numpy().reshape(ds.H, ds.W, eff_C).transpose(2,0,1)
            arrg = rearrange(tgt, '1 (h w) c -> c h w', h=ds.H, w=ds.W).cpu().numpy()
            t    = t0 + num_in

            mets = compute_metrics(arrp, arrg, mask)
            results[t].append(mets)
            if t in selected_ts:
                selected[t].append((s, mets))

            if sample_plot:
                i1, j1 = int(3.5*ds.W/16), ds.H//2
                i2, j2 = int(4.5*ds.W/16), ds.H//2
                vals = [
                    arrg[0,j1,i1], arrp[0,j1,i1],
                    arrg[1,j1,i1], arrp[1,j1,i1],
                    arrg[0,j2,i2], arrp[0,j2,i2],
                    arrg[1,j2,i2], arrp[1,j2,i2]
                ]
                fpt.write(f"{t}," + ",".join(f"{v:.6e}" for v in vals) + "\n")
                _plot_grid(arrp, arrg, mask, sd, s, t)
                _plot_individual(arrp, arrg, mask, sd, s, t)

        if sample_plot:
            fpt.close()

    if count:
        avg = total_time / count
        with open(os.path.join(base, mode, 'inference_time.txt'), 'w') as f:
            f.write(f"{avg:.6e}\n")

    hdr = ['timestep','rel_U','rel_V','rel_Overall',
           'rmse_U','rmse_V','rmse_Overall',
           'linf_U','linf_V','linf_Overall']
    with open(os.path.join(base, f'{mode}_errors.csv'), 'w') as f:
        f.write(','.join(hdr) + '\n')
        for t in sorted(results):
            avgm = {k: np.mean([m[k] for m in results[t]]) for k in results[t][0]}
            row = [str(t)] + [f"{avgm[k]:.6e}" for k in hdr[1:]]
            f.write(','.join(row) + '\n')

    for t in selected_ts:
        with open(os.path.join(base, f'{mode}_t{t}_errors.csv'), 'w') as f:
            f.write(','.join(['sample_id'] + hdr[1:]) + '\n')
            for s, mets in selected[t]:
                row = [str(s)] + [f"{mets[k]:.6e}" for k in hdr[1:]]
                f.write(','.join(row) + '\n')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', required=True)
    parser.add_argument('--config_path', required=True)
    parser.add_argument('--sample_ids', default='', help='comma-separated IDs')
    args = parser.parse_args()

    cfg    = load_config(args.config_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = load_model(args.checkpoint_path, device)
    saves  = {int(i) for i in args.sample_ids.split(',') if i.strip().isdigit()}
    evaluate(model, device, cfg, 'single_step', saves)
    evaluate(model, device, cfg, 'rollout',    saves)
