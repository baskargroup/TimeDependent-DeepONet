import os
import glob
from typing import Set
from collections import defaultdict, deque
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

from data.dataset import LidDrivenDataset2DTime
from models.geometric_deeponet.geometric_deeponet import GeometricDeepONetTime

# Styles
FIELD_CMAP, ERROR_CMAP = plt.get_cmap('jet'), plt.get_cmap('jet')
FIELD_VMIN, FIELD_VMAX = {'U':0,'V':-1}, {'U':2,'V':1}
ERROR_VMIN, ERROR_VMAX = 1e-6, 1.0
FIELD_CMAP_T = FIELD_CMAP.copy(); FIELD_CMAP_T.set_bad(color='none')
ERROR_CMAP_T = ERROR_CMAP.copy(); ERROR_CMAP_T.set_bad(color='none')


def load_config(path):
    cfg = OmegaConf.load(path)
    return OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))


def load_model(ckpt_path, device):
    if os.path.isdir(ckpt_path):
        ckpts = sorted(glob.glob(os.path.join(ckpt_path, '**/*.ckpt'), recursive=True))
        ckpt_path = ckpts[-1]
    m = GeometricDeepONetTime.load_from_checkpoint(ckpt_path, map_location=device)
    return m.eval()


def load_dataset(cfg, is_test):
    x = cfg.data.file_path_test_x if is_test else cfg.data.file_path_train_x
    y = cfg.data.file_path_test_y if is_test else cfg.data.file_path_train_y
    return LidDrivenDataset2DTime(
        x, y,
        cfg.model.num_input_timesteps, cfg.model.final_timestep,
        cfg.data.every_nth_timestep,
        cfg.model.height, cfg.model.width,
        cfg.data.type, cfg.model.includePressure,
        cfg.model.output_channels
    )


def compute_metrics(pred, gt, mask):
    mets, diffs, gts = {}, [], []
    fluid = ~mask.flatten()
    for i, ch in enumerate(['U','V']):
        d = pred[i].flatten()[fluid] - gt[i].flatten()[fluid]
        mets[f'rel_{ch}']  = np.linalg.norm(d)/np.linalg.norm(gt[i].flatten()[fluid])
        mets[f'rmse_{ch}'] = np.sqrt((d**2).mean())
        mets[f'linf_{ch}'] = np.max(np.abs(d))
        diffs.append(d); gts.append(gt[i].flatten()[fluid])
    all_d, all_g = np.concatenate(diffs), np.concatenate(gts)
    mets['rel_Overall']  = np.linalg.norm(all_d)/np.linalg.norm(all_g)
    mets['rmse_Overall'] = np.sqrt((all_d**2).mean())
    mets['linf_Overall'] = np.max(np.abs(all_d))
    return mets


def _plot_grid(pred, gt, mask, out_dir, s, t):
    C,H,W = gt.shape
    data = [gt, pred, np.abs(pred-gt)]
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(3,C,figsize=(6*C,7.5),squeeze=False)
    fig.subplots_adjust(wspace=0.3,hspace=0.02)
    for i in range(3):
        is_err = (i==2)
        for j in range(C):
            ax, arr = axes[i][j], data[i][j]
            im = ax.imshow(
                ma.masked_where(mask, arr),
                cmap=ERROR_CMAP if is_err else FIELD_CMAP,
                norm=LogNorm(ERROR_VMIN,ERROR_VMAX) if is_err else None,
                vmin=(None if is_err else FIELD_VMIN[['U','V'][j]]),
                vmax=(None if is_err else FIELD_VMAX[['U','V'][j]]),
                origin='lower'
            )
            ax.set_xticks([]); ax.set_yticks([])
            cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.05)
            cbar = fig.colorbar(im,cax=cax,format='%.1e' if is_err else '%.1f')
            if is_err:
                cbar.set_ticks([1,1e-2,1e-4,1e-6]); cbar.set_ticklabels(['0','-2','-4','-6'])
    fig.suptitle(f"Sample {s}, Timestep {t}", fontsize=16)
    fig.savefig(os.path.join(out_dir,f"grid_s{s}_t{t}.png"),dpi=150,bbox_inches='tight')
    plt.close(fig)


def _plot_individual(pred, gt, mask, sample_dir, s, t):
    C,H,W = gt.shape
    data = [gt, pred, np.abs(pred-gt)]
    indiv = os.path.join(sample_dir,"individual_plots")
    os.makedirs(indiv, exist_ok=True)
    for i in range(3):
        is_err = (i==2)
        kind = ['GT','Pred','Err'][i]
        for j,lbl in enumerate(['U','V']):
            arr = data[i][j]
            fig, ax = plt.subplots(1,1,figsize=(5,4),facecolor='none')
            ax.set_facecolor('none')
            im = ax.imshow(
                ma.masked_where(mask,arr),
                cmap=ERROR_CMAP_T if is_err else FIELD_CMAP_T,
                norm=LogNorm(ERROR_VMIN,ERROR_VMAX) if is_err else None,
                vmin=None if is_err else FIELD_VMIN[lbl],
                vmax=None if is_err else FIELD_VMAX[lbl],
                origin='lower'
            )
            ax.set_xticks([]); ax.set_yticks([])
            cax = make_axes_locatable(ax).append_axes("right",size="5%",pad=0.05)
            cbar = fig.colorbar(im,cax=cax,format='%.1e' if is_err else '%.1f')
            if is_err:
                cbar.set_ticks([1,1e-2,1e-4,1e-6]); cbar.set_ticklabels(['0','-2','-4','-6'])
            fig.savefig(os.path.join(indiv,f"{lbl}_{kind}_s{s}_t{t}.png"),
                        dpi=150,transparent=True,bbox_inches='tight')
            plt.close(fig)


def evaluate(model, ds, device, cfg, mode: str, save_ids: Set[int]):
    num_in,H,W = cfg.model.num_input_timesteps, cfg.model.height, cfg.model.width
    base = cfg.model.plot_path
    results = defaultdict(list)
    os.makedirs(os.path.join(base,mode), exist_ok=True)

    csv_path = os.path.join(base,mode,f"{mode}_point_pred.csv")
    with open(csv_path,'w') as f_pt:
        f_pt.write("time,u_gt1,u_pr1,v_gt1,v_pr1,u_gt2,u_pr2,v_gt2,v_pr2\n")
        for s in tqdm(range(ds.num_samples_orig),desc=mode):
            gather = (s not in save_ids)

            if mode=='single_step':
                mask = ~ds.sdf_map_for_sample(s)  #  inside
                T = min(cfg.model.final_timestep, ds.total_timesteps_available)
                inp0, _ = ds[s*len(ds.valid_start_indices)]
                _, r0, coords0, sdf0 = inp0

                for t0 in range(T-num_in):
                    seq = ds.y[s,t0:t0+num_in,:ds.eff_C]
                    branch = rearrange(
                        torch.tensor(seq.copy(),dtype=torch.float32),
                        'nb c h w->1 (nb c) h w'
                    ).to(device)

                    with torch.no_grad():
                        pred = model((branch,
                                      r0.unsqueeze(0).to(device),
                                      coords0.unsqueeze(0).to(device),
                                      sdf0.unsqueeze(0).to(device)))
                    arrp = pred[0,0].cpu().numpy().reshape(H,W,-1).transpose(2,0,1)
                    arrg = ds.y[s,t0+num_in,:ds.eff_C].copy()
                    mets = compute_metrics(arrp,arrg,mask)
                    t = t0+num_in
                    results[t].append(mets)

                    i1,j1=W//4,H//2; i2,j2=W//2,H//2
                    u1,v1=arrg[0,j1,i1],arrp[0,j1,i1]
                    u2,v2=arrg[0,j2,i2],arrp[0,j2,i2]
                    f_pt.write(f"{t},{u1},{v1},{u1},{v1},{u2},{v2},{u2},{v2}\n")

                    if not gather:
                        ddir=os.path.join(base,mode,f"sample_{s}")
                        _plot_grid(arrp,arrg,mask,ddir,s,t)
                        _plot_individual(arrp,arrg,mask,ddir,s,t)

            else:  # rollout
                inp0,_ = ds[s*len(ds.valid_start_indices)]
                x0,r0,coords0,sdf0 = inp0
                mask = ~ds.sdf_map_for_sample(s)
                C_out = ds.eff_C
                dq = deque([x0[i*C_out:(i+1)*C_out] for i in range(num_in)],maxlen=num_in)
                max_s = min(cfg.model.final_timestep,ds.total_timesteps_available)-num_in

                for step in range(max_s):
                    xb = torch.cat(list(dq),dim=0).unsqueeze(0).to(device)
                    with torch.no_grad():
                        pred = model((xb,
                                      r0.unsqueeze(0).to(device),
                                      coords0.unsqueeze(0).to(device),
                                      sdf0.unsqueeze(0).to(device)))
                    gt = ds.get_ground_truth_sequence(s,num_in+step,num_in+step+1)
                    arrp = pred[0,0].cpu().numpy().reshape(H,W,-1).transpose(2,0,1)
                    arrg = gt[0].cpu().numpy().reshape(H,W,-1).transpose(2,0,1)
                    mets=compute_metrics(arrp,arrg,mask)
                    t = num_in+step
                    results[t].append(mets)

                    i1,j1=W//4,H//2; i2,j2=W//2,H//2
                    u1,v1=arrg[0,j1,i1],arrp[0,j1,i1]
                    u2,v2=arrg[0,j2,i2],arrp[0,j2,i2]
                    f_pt.write(f"{t},{u1},{v1},{u1},{v1},{u2},{v2},{u2},{v2}\n")

                    if not gather:
                        ddir=os.path.join(base,mode,f"sample_{s}")
                        _plot_grid(arrp,arrg,mask,ddir,s,t)
                        _plot_individual(arrp,arrg,mask,ddir,s,t)

                    p = pred[0,0].cpu()
                    dq.append(p.reshape(H,W,C_out).permute(2,0,1))

    os.makedirs(base, exist_ok=True)
    hdr=['timestep','rel_U','rel_V','rel_Overall',
         'rmse_U','rmse_V','rmse_Overall',
         'linf_U','linf_V','linf_Overall']
    with open(os.path.join(base,f"{mode}_errors.csv"),'w') as f:
        f.write(",".join(hdr)+"\n")
        for t in sorted(results):
            avg = {k:np.mean([m[k] for m in results[t]]) for k in results[t][0]}
            row=[str(t),
                 f"{avg['rel_U']:.6e}",f"{avg['rel_V']:.6e}",f"{avg['rel_Overall']:.6e}",
                 f"{avg['rmse_U']:.6e}",f"{avg['rmse_V']:.6e}",f"{avg['rmse_Overall']:.6e}",
                 f"{avg['linf_U']:.6e}",f"{avg['linf_V']:.6e}",f"{avg['linf_Overall']:.6e}"]
            f.write(",".join(row)+"\n")


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint_path',required=True)
    p.add_argument('--config_path',    required=True)
    p.add_argument('--sample_ids',     default='',help="IDs")
    args=p.parse_args()

    cfg=load_config(args.config_path)
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=load_model(args.checkpoint_path,dev)
    ds=load_dataset(cfg,is_test=True)
    saves={int(i) for i in args.sample_ids.split(',') if i.strip().isdigit()}

    evaluate(model,ds,dev,cfg,'single_step',saves)
    evaluate(model,ds,dev,cfg,'rollout',    saves)
