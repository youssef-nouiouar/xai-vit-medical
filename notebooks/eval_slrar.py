"""
Evaluate SLR-AR vs. standard attention rollout on a CRC-histology ViT-B/16.

Metrics (no region annotations required):
  * Spectral Stability Index (SSI)  -- go/no-go signal, label-free
  * Insertion AUC  (higher better)
  * Deletion  AUC  (lower  better)
  * Qualitative overlays

Ablation grid: rollout | snb | gts | slrar   (Table 3 of the paper)

Example
-------
python eval_slrar.py \
    --ckpt /path/to/vit_crc.pth \
    --data /path/to/CRC-VAL-HE-7K \
    --n-images 200 --smoothing applied --out ./results
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
import timm
from timm.data import create_transform
from torchvision import datasets
from torch.utils.data import DataLoader, Subset

from slr_ar import (
    AttentionCatcher, SLRARConfig, build_rollout, cls_attribution_map,
    get_attentions, spectral_stability_index, ssi_by_depth,
)

VARIANTS = ["rollout", "snb", "gts", "slrar"]


# --------------------------------------------------------------------------
# Model / data
# --------------------------------------------------------------------------

def load_model(ckpt, num_classes=9, device="cuda"):
    model = timm.create_model(
        "vit_base_patch16_224",
        pretrained=False,
        num_classes=num_classes,
        global_pool="token",
        drop_path_rate=0.0,       # inference
    )
    sd = torch.load(ckpt, map_location="cpu")
    for key in ("state_dict", "model", "model_state_dict"):
        if isinstance(sd, dict) and key in sd:
            sd = sd[key]
            break
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"[load_model] missing={list(missing)[:5]} unexpected={list(unexpected)[:5]}")
    return model.eval().to(device)


def build_loader(root, mean, std, n_images, batch_size, seed=0):
    tf = create_transform(input_size=224, is_training=False, mean=mean, std=std,
                          crop_pct=1.0)
    ds = datasets.ImageFolder(root, transform=tf)
    g = np.random.default_rng(seed)
    idx = g.choice(len(ds), size=min(n_images, len(ds)), replace=False)
    return DataLoader(Subset(ds, idx.tolist()), batch_size=batch_size,
                      shuffle=False, num_workers=4), ds.classes


# --------------------------------------------------------------------------
# Insertion / Deletion
# --------------------------------------------------------------------------

def _blur(x, sigma_px=11):
    """Gaussian-blurred baseline. For H&E patches a blurred image is a far more
    sensible 'absent' reference than black or gray: zeroing pixels creates an
    out-of-distribution artefact the model reacts to for the wrong reasons."""
    k = sigma_px if sigma_px % 2 == 1 else sigma_px + 1
    coords = torch.arange(k, dtype=x.dtype, device=x.device) - k // 2
    g = torch.exp(-(coords ** 2) / (2 * (k / 3.0) ** 2))
    g = (g / g.sum()).view(1, 1, 1, k)
    C = x.shape[1]
    x = F.conv2d(F.pad(x, (k // 2,) * 2 + (0, 0), mode="reflect"),
                 g.expand(C, 1, 1, k), groups=C)
    x = F.conv2d(F.pad(x, (0, 0) + (k // 2,) * 2, mode="reflect"),
                 g.transpose(-1, -2).expand(C, 1, k, 1), groups=C)
    return x


@torch.no_grad()
def insertion_deletion_auc(model, x, sal, target, mode, n_steps=49,
                           patch=16, grid=14, fwd_bs=32):
    """AUC over patch-level insertion or deletion, one image at a time.

    x      : (1, 3, 224, 224) normalised input
    sal    : (1, grid, grid) saliency
    target : int, class index the curve is scored against (use the model's own
             prediction, not the label -- these metrics measure faithfulness to
             the model, not correctness)
    """
    device = x.device
    order = torch.argsort(sal.reshape(-1), descending=True)   # most salient first
    n_patch = grid * grid
    per_step = max(1, n_patch // n_steps)
    steps = list(range(0, n_patch + 1, per_step))
    if steps[-1] != n_patch:
        steps.append(n_patch)

    base = _blur(x) if mode == "insertion" else torch.zeros_like(x)
    start, fill = (base, x) if mode == "insertion" else (x, base)

    masks = []
    for s in steps:
        m = torch.zeros(n_patch, device=device)
        m[order[:s]] = 1.0
        m = m.reshape(1, 1, grid, grid)
        masks.append(F.interpolate(m, scale_factor=patch, mode="nearest"))
    masks = torch.cat(masks, dim=0)                            # (S, 1, 224, 224)

    imgs = start * (1 - masks) + fill * masks
    probs = []
    for i in range(0, imgs.shape[0], fwd_bs):
        logits = model(imgs[i:i + fwd_bs])
        probs.append(logits.softmax(-1)[:, target])
    probs = torch.cat(probs).cpu().numpy()
    frac = np.array(steps) / n_patch
    return float(np.trapz(probs, frac))


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True, help="CRC-VAL-HE-7K root (ImageFolder)")
    ap.add_argument("--out", default="./results")
    ap.add_argument("--n-images", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-classes", type=int, default=9)
    ap.add_argument("--tau", type=float, default=0.95)
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--M", type=int, default=8)
    ap.add_argument("--smoothing", choices=["paper", "applied"], default="applied")
    ap.add_argument("--beta-mode", choices=["adaptive", "fixed"], default="adaptive")
    ap.add_argument("--exact-exp", action="store_true")
    ap.add_argument("--renorm-rows", action="store_true")
    ap.add_argument("--mean", type=float, nargs=3, default=[0.485, 0.456, 0.406])
    ap.add_argument("--std", type=float, nargs=3, default=[0.229, 0.224, 0.225])
    ap.add_argument("--n-viz", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cfg = SLRARConfig(tau=args.tau, K=args.K, M=args.M, smoothing=args.smoothing,
                      beta_mode=args.beta_mode, exact_exp=args.exact_exp,
                      renorm_rows=args.renorm_rows)

    model = load_model(args.ckpt, args.num_classes, args.device)
    loader, classes = build_loader(args.data, tuple(args.mean), tuple(args.std),
                                   args.n_images, args.batch_size)
    print(f"[data] {len(classes)} classes, {args.n_images} images sampled")

    catcher = AttentionCatcher(model)
    acc = {v: {"ssi": [], "ins": [], "del": [], "qnorm": []} for v in VARIANTS}
    depth_curves = {v: [] for v in VARIANTS}
    viz_bank = []

    for bi, (x, y) in enumerate(loader):
        x = x.to(args.device)
        logits, attns = get_attentions(model, x, catcher)
        pred = logits.argmax(-1)

        for v in VARIANTS:
            R, diags = build_rollout(attns, v, cfg)
            sal = cls_attribution_map(R, grid=14)                 # (B,14,14)
            acc[v]["ssi"].append(spectral_stability_index(R).cpu())
            acc[v]["qnorm"].append(torch.stack(diags["q_norm"]).mean(1))

            for i in range(x.shape[0]):
                t = int(pred[i])
                acc[v]["ins"].append(insertion_deletion_auc(
                    model, x[i:i+1], sal[i:i+1], t, "insertion"))
                acc[v]["del"].append(insertion_deletion_auc(
                    model, x[i:i+1], sal[i:i+1], t, "deletion"))

            if bi == 0:
                depth_curves[v].append(ssi_by_depth(attns, v, cfg))
                if len(viz_bank) < args.n_viz:
                    viz_bank.append((v, sal[:args.n_viz].cpu()))

        if bi == 0:
            viz_x = x[:args.n_viz].cpu()
            viz_y = [classes[int(c)] for c in y[:args.n_viz]]
        print(f"  batch {bi+1}/{len(loader)} done")

    catcher.remove()

    # ---- report ----------------------------------------------------------
    summary = {}
    print("\n" + "=" * 68)
    print(f"{'Variant':<10} {'SSI':>10} {'Insertion':>12} {'Deletion':>12} {'mean|Q|_2':>10}")
    print("-" * 68)
    for v in VARIANTS:
        ssi = torch.cat(acc[v]["ssi"]).mean().item()
        ins = float(np.mean(acc[v]["ins"]))
        dele = float(np.mean(acc[v]["del"]))
        qn = torch.cat(acc[v]["qnorm"]).mean().item()
        summary[v] = {"ssi": ssi, "insertion_auc": ins, "deletion_auc": dele,
                      "mean_q_spectral_norm": qn}
        print(f"{v:<10} {ssi:>10.4f} {ins:>12.4f} {dele:>12.4f} {qn:>10.4f}")
    print("=" * 68)
    print("Higher SSI = less spectral collapse | Higher insertion, lower deletion = better")

    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump({"config": vars(args), "results": summary}, f, indent=2)

    # ---- figures ---------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Figure 4 style: SSI vs depth
        fig, ax = plt.subplots(figsize=(6, 4))
        for v in VARIANTS:
            c = depth_curves[v][0].mean(1).numpy()
            ax.plot(range(1, len(c) + 1), c, marker="o", label=v)
        ax.set_xlabel("Transformer depth (layer)")
        ax.set_ylabel("Spectral Stability Index  |λ₂|/|λ₁|")
        ax.set_yscale("log")
        ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "ssi_by_depth.png"), dpi=150)

        # Qualitative overlays
        mean = torch.tensor(args.mean).view(1, 3, 1, 1)
        std = torch.tensor(args.std).view(1, 3, 1, 1)
        imgs = (viz_x * std + mean).clamp(0, 1).permute(0, 2, 3, 1).numpy()
        n = imgs.shape[0]
        fig, axes = plt.subplots(n, 5, figsize=(13, 2.6 * n))
        axes = np.atleast_2d(axes)
        for r in range(n):
            axes[r, 0].imshow(imgs[r]); axes[r, 0].set_ylabel(viz_y[r], fontsize=8)
            axes[r, 0].set_title("input" if r == 0 else "")
            for c, (v, sal) in enumerate(viz_bank):
                up = F.interpolate(sal[r:r+1].unsqueeze(1), size=224,
                                   mode="bilinear", align_corners=False)[0, 0]
                axes[r, c+1].imshow(imgs[r]); axes[r, c+1].imshow(up, cmap="jet", alpha=0.5)
                axes[r, c+1].set_title(v if r == 0 else "")
            for a in axes[r]:
                a.set_xticks([]); a.set_yticks([])
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "qualitative.png"), dpi=150)
        print(f"[figures] written to {args.out}")
    except Exception as e:
        print(f"[figures] skipped: {e}")


if __name__ == "__main__":
    main()
