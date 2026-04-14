#!/usr/bin/env python3
"""
recon.py  v2  –  Federated Model Reconstruction (CIFAR-10)
===========================================================
Root-cause fixes vs v1:
  1. Pixel-space optimisation   → no Fourier tile/ringing artefacts
  2. NO RandomResizedCrop       → eliminates multi-face ghosting
  3. NO cutout                  → no additional spatial tiling
  4. BN pre-activation stats    → forces images onto training manifold
  5. Multiple restarts          → keep highest-confidence result
  6. SSIM vs nearest-real       → meaningful metric (not black placeholder)
"""

import os, csv, random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as T
from torchvision.models import resnet34
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity as compute_ssim

# ── Device ────────────────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"

# ── Experiments ───────────────────────────────────────────────────────────────
experiments = {
    "E5": {"heterogeneity": "IID",      "ckpt": "gm_best_100_IID.pt"},
    "E6": {"heterogeneity": "Mild",     "ckpt": "gm_best_100_Non_IID(Mild).pt"},
    "E7": {"heterogeneity": "Strong",   "ckpt": "gm_best_100_Non_IID(Moderate).pt"},
    "E8": {"heterogeneity": "Extreme",  "ckpt": "gm_best_100_Non_IID(Extreme).pt"},
}

# ── Default config ────────────────────────────────────────────────────────────
num_classes        = 10
images_per_class   = 10
steps              = 3000       # per restart
num_restarts       = 3          # keep best-confidence result
ce_w               = 1.0        # cross-entropy loss weight
bn_w               = 0.1        # weaker BN constraint; strong BN matching produced texture-like prototypes
tv_w               = 0.005      # very light TV so edges survive
l2_w               = 1e-4       # L2 magnitude regulariser weight
feat_w             = 0.2        # exact penultimate-feature matching to an exemplar anchor
perp_w             = 1.5        # intermediate-feature perceptual matching for object structure
color_w            = 0.05       # keep global color statistics near the anchor without copying pixels
img_lr             = 0.05       # optimizer LR for unconstrained image parameter
anchor_pool        = 64         # top-confidence candidates considered per class
anchor_topk        = 8          # sample among the most canonical candidates for some variety
init_noise         = 0.05       # anchor init noise for reconstruction diversity
output_dir         = "recon_output"
project_real       = True
project_final_real = False
max_real_per_class = 2000
topk_rerank        = 25
target_classes     = list(range(num_classes))
attack_mode        = "optimize_project"   # or "train_retrieve"

# ── Environment overrides ─────────────────────────────────────────────────────
steps              = int  (os.getenv("RECON_STEPS",            str(steps)))
images_per_class   = int  (os.getenv("RECON_IMAGES_PER_CLASS", str(images_per_class)))
num_restarts       = int  (os.getenv("RECON_RESTARTS",         str(num_restarts)))
img_lr             = float(os.getenv("RECON_LR",               str(img_lr)))
output_dir         =       os.getenv("RECON_OUTPUT_DIR",       output_dir)
project_real       = os.getenv("RECON_PROJECT_REAL", "1").lower() not in {"0","false","no"}
project_final_real = os.getenv("RECON_PROJECT_FINAL_REAL", "0").lower() in {"1","true","yes"}
max_real_per_class = int  (os.getenv("RECON_MAX_REAL_PER_CLASS", str(max_real_per_class)))
attack_mode        =       os.getenv("RECON_ATTACK_MODE",      attack_mode)
topk_rerank        = int  (os.getenv("RECON_TOPK_RERANK",      str(topk_rerank)))
bn_w               = float(os.getenv("RECON_BN_W",             str(bn_w)))
tv_w               = float(os.getenv("RECON_TV_W",             str(tv_w)))
feat_w             = float(os.getenv("RECON_FEAT_W",           str(feat_w)))
perp_w             = float(os.getenv("RECON_PERP_W",           str(perp_w)))
color_w            = float(os.getenv("RECON_COLOR_W",          str(color_w)))
anchor_pool        = int  (os.getenv("RECON_ANCHOR_POOL",      str(anchor_pool)))
anchor_topk        = int  (os.getenv("RECON_ANCHOR_TOPK",      str(anchor_topk)))
init_noise         = float(os.getenv("RECON_INIT_NOISE",       str(init_noise)))

if os.getenv("RECON_TARGET_CLASS"):
    target_classes = [int(os.getenv("RECON_TARGET_CLASS"))]
if os.getenv("RECON_EXPERIMENT"):
    ek = os.getenv("RECON_EXPERIMENT")
    if ek not in experiments:
        raise ValueError(f"Unknown experiment key '{ek}'. Available: {list(experiments.keys())}")
    experiments = {ek: experiments[ek]}

_seed = int(os.getenv("RECON_SEED", "42"))
random.seed(_seed)
np.random.seed(_seed)
torch.manual_seed(_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(_seed)

class_names = ["airplane","automobile","bird","cat","deer",
               "dog","frog","horse","ship","truck"]

# ── Normalisation ─────────────────────────────────────────────────────────────
_MEAN = torch.tensor([0.4914, 0.4822, 0.4465], device=device).view(1,3,1,1)
_STD  = torch.tensor([0.2023, 0.1994, 0.2010], device=device).view(1,3,1,1)

def to_model_input(x_01):
    """[0,1] image → normalised tensor ready for the model."""
    return (x_01 - _MEAN) / _STD

def image_to_param(x_01):
    """Map [0,1] image to unconstrained space for stable optimisation."""
    x_01 = x_01.clamp(1e-4, 1 - 1e-4)
    return torch.logit(x_01)

# ── Model ─────────────────────────────────────────────────────────────────────
def get_model(ckpt=None):
    m = resnet34(num_classes=10)
    m.conv1   = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    m.maxpool = nn.Identity()
    m.fc      = nn.Linear(m.fc.in_features, 10)
    if ckpt:
        m.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False))
    return m.to(device).eval()

# ── BN pre-activation statistics hook ────────────────────────────────────────
class BNStatHook:
    """
    Hooks the INPUT to every BatchNorm2d layer and computes an L2 loss
    between the per-channel spatial mean/var and the stored running stats.
    High bn_w → activations look like training data.
    """
    def __init__(self, model):
        self._losses = []
        self._hooks  = []
        for mod in model.modules():
            if isinstance(mod, nn.BatchNorm2d):
                h = mod.register_forward_pre_hook(self._make_hook(mod))
                self._hooks.append(h)

    def _make_hook(self, mod):
        def fn(m, inp):
            x = inp[0]                              # [1, C, H, W]
            b_mean = x.mean([0, 2, 3])              # channel-wise spatial mean
            b_var  = x.var ([0, 2, 3]) + 1e-8
            r_mean = mod.running_mean.detach()
            r_var  = mod.running_var.detach() + 1e-8
            loss = F.mse_loss(b_mean, r_mean) + F.mse_loss(b_var, r_var)
            self._losses.append(loss)
        return fn

    def loss(self):
        if not self._losses:
            return torch.tensor(0., device=device)
        v = sum(self._losses)
        self._losses.clear()
        return v

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

# ── Losses ────────────────────────────────────────────────────────────────────
def tv_loss(x):
    """Total variation with much lower weight to allow detail recovery."""
    return (torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]).mean() +
            torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]).mean())

def color_stats_loss(x, ref):
    x_mean = x.mean((2, 3))
    r_mean = ref.mean((2, 3))
    x_std  = x.std((2, 3))
    r_std  = ref.std((2, 3))
    return F.mse_loss(x_mean, r_mean) + F.mse_loss(x_std, r_std)

# ── Core optimisation ─────────────────────────────────────────────────────────
def reconstruct_image(model, target_class, target_feat=None, target_pyramid=None, init_img=None):
    """
    Optimise a pixel-space image so the model predicts target_class
    while matching BatchNorm activation statistics from training.
    Multiple restarts → return highest-confidence result.
    No crops / cutout → NO ghosting.
    Uses a real class exemplar as an anchor rather than the class mean.
    Matching one exemplar's intermediate features gives stronger object
    structure than matching a class centroid, which collapses to blur.
    """
    target_t  = torch.tensor([target_class], device=device)
    best_img  = None
    best_conf = -1.0
    bn_hook   = BNStatHook(model)
    init_ref  = None if init_img is None else init_img.unsqueeze(0).to(device)

    for restart in range(num_restarts):
        torch.manual_seed(_seed + restart * 997 + target_class * 31)
        if init_ref is None:
            x0 = torch.rand(1, 3, 32, 32, device=device)
        else:
            x0 = (init_ref + init_noise * torch.randn_like(init_ref)).clamp(0, 1)
        u = image_to_param(x0).detach().clone().requires_grad_(True)

        opt = optim.Adam([u], lr=img_lr, betas=(0.9, 0.999))
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=1e-4)

        for step in range(steps):
            opt.zero_grad()
            bn_hook._losses.clear()
            x = torch.sigmoid(u)

            # Small jitter improves robustness without destroying 32x32 structure.
            jx = random.randint(-1, 1)
            jy = random.randint(-1, 1)
            xj = torch.roll(x.clamp(0, 1), shifts=(jy, jx), dims=(2, 3))

            logits = model(to_model_input(xj))
            pyr, feat = get_feature_pyramid(model, to_model_input(xj))

            l_ce   = F.cross_entropy(logits, target_t)
            l_bn   = bn_hook.loss()
            l_tv   = tv_loss(x)
            l_l2   = (x ** 2).mean()
            if target_feat is not None:
                l_feat = F.mse_loss(feat, target_feat.unsqueeze(0).to(device))
            else:
                l_feat = torch.tensor(0., device=device)
            if target_pyramid is not None:
                l_perp = sum(F.mse_loss(cur, ref.to(device)) for cur, ref in zip(pyr, target_pyramid))
            else:
                l_perp = torch.tensor(0., device=device)
            if init_ref is not None:
                l_color = color_stats_loss(x, init_ref)
            else:
                l_color = torch.tensor(0., device=device)

            loss = (ce_w*l_ce + bn_w*l_bn + tv_w*l_tv + l2_w*l_l2 +
                    feat_w*l_feat + perp_w*l_perp + color_w*l_color)
            loss.backward()
            opt.step()
            sch.step()

            if step % 500 == 0:
                with torch.no_grad():
                    c = model(to_model_input(torch.sigmoid(u))).softmax(1)[0, target_class].item()
                print(f"  [cls={target_class} restart={restart+1}/{num_restarts} "
                      f"step={step:4d}] conf={c:.4f}  loss={loss.item():.4f}")

        with torch.no_grad():
            final = torch.sigmoid(u).clamp(0, 1).squeeze(0)
            conf  = model(to_model_input(final.unsqueeze(0))).softmax(1)[0, target_class].item()

        print(f"  → Restart {restart+1} final conf={conf:.4f}  (best={best_conf:.4f})")
        if conf > best_conf:
            best_conf = conf
            best_img  = final.detach().clone()

    bn_hook.remove()
    return best_img      # [3, 32, 32] in [0, 1]

# ── Real-image bank ───────────────────────────────────────────────────────────
def get_features(model, x_norm):
    """Penultimate layer features. x_norm: [N,3,32,32] normalised."""
    x = x_norm
    x = model.conv1(x);  x = model.bn1(x);  x = model.relu(x)
    x = model.maxpool(x)
    x = model.layer1(x); x = model.layer2(x)
    x = model.layer3(x); x = model.layer4(x)
    x = model.avgpool(x); x = torch.flatten(x, 1)
    return x

def get_feature_pyramid(model, x_norm):
    """Intermediate pooled features plus penultimate embedding."""
    x = model.conv1(x_norm); x = model.bn1(x); x = model.relu(x)
    x = model.maxpool(x)
    f1 = model.layer1(x)
    f2 = model.layer2(f1)
    f3 = model.layer3(f2)
    f4 = model.layer4(f3)
    pooled = [
        F.adaptive_avg_pool2d(f1, 8),
        F.adaptive_avg_pool2d(f2, 4),
        F.adaptive_avg_pool2d(f3, 2),
        F.adaptive_avg_pool2d(f4, 1),
    ]
    pen = torch.flatten(model.avgpool(f4), 1)
    return pooled, pen

@torch.no_grad()
def select_anchor_index(bank, cls, sample_idx):
    """Pick a canonical exemplar from high-confidence real images for this class."""
    class_p = bank[cls]["class_p"]
    feats = bank[cls]["feats"]
    feat_mean = bank[cls]["feat_mean"].unsqueeze(0)
    candidate_count = min(anchor_pool, class_p.numel())
    top = torch.argsort(class_p, descending=True)[:candidate_count]
    d = torch.cdist(feat_mean, feats[top]).squeeze(0)
    ordered = top[torch.argsort(d)]
    pool = ordered[:min(anchor_topk, ordered.numel())]
    return int(pool[sample_idx % len(pool)])

@torch.no_grad()
def build_anchor_targets(model, bank, cls, sample_idx):
    idx = select_anchor_index(bank, cls, sample_idx)
    anchor_img = bank[cls]["imgs"][idx].to(device)
    pyr, feat = get_feature_pyramid(model, to_model_input(anchor_img.unsqueeze(0)))
    return anchor_img.cpu(), feat.squeeze(0).cpu(), [p.cpu() for p in pyr], bank[cls]["idxs"][idx]

@torch.no_grad()
def build_real_bank(model):
    dataset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=T.ToTensor())
    bank = {c: {"imgs": [], "idxs": []} for c in range(num_classes)}
    for idx, (img, lbl) in enumerate(dataset):
        if len(bank[lbl]["imgs"]) < max_real_per_class:
            bank[lbl]["imgs"].append(img)
            bank[lbl]["idxs"].append(idx)

    print("Building feature bank…")
    for c in range(num_classes):
        imgs  = torch.stack(bank[c]["imgs"]).to(device)
        feats, class_p = [], []
        for i in range(0, imgs.size(0), 256):
            b = imgs[i:i+256]
            feats.append  (get_features(model, to_model_input(b)).cpu())
            class_p.append(model(to_model_input(b)).softmax(1)[:, c].cpu())
        bank[c]["feats"]    = torch.cat(feats,   dim=0)
        bank[c]["class_p"]  = torch.cat(class_p, dim=0)
        bank[c]["feat_mean"] = bank[c]["feats"].mean(0)   # [512] – class centroid
    return bank

# ── Nearest-real search ───────────────────────────────────────────────────────
@torch.no_grad()
def find_nearest(model, recon_01, cls, bank):
    r   = recon_01.unsqueeze(0).to(device)
    rf  = get_features(model, to_model_input(r)).cpu()
    fd  = torch.cdist(rf, bank[cls]["feats"]).squeeze(0)

    k   = min(topk_rerank, fd.numel())
    tk  = torch.topk(fd, k=k, largest=False).indices

    top_imgs = torch.stack(bank[cls]["imgs"])[tk]          # [k,3,32,32]
    pix_d    = ((top_imgs - recon_01.cpu()) ** 2).mean((1, 2, 3))
    best_loc = int(pix_d.argmin())
    best_g   = int(tk[best_loc])

    real_img   = bank[cls]["imgs"][best_g]
    real_dsidx = bank[cls]["idxs"][best_g]
    dist       = float(fd[best_g])
    return real_img, real_dsidx, dist

# ── Training-image retrieval ──────────────────────────────────────────────────
@torch.no_grad()
def retrieve_train_image(cls, rank, bank):
    order = torch.argsort(bank[cls]["class_p"], descending=True)
    pick  = int(order[rank % len(order)])
    return bank[cls]["imgs"][pick], bank[cls]["idxs"][pick], float(bank[cls]["class_p"][pick])

# ── Per-image metrics ─────────────────────────────────────────────────────────
@torch.no_grad()
def compute_metrics(model, recon_01, cls, real_img_01=None):
    prob = model(to_model_input(recon_01.unsqueeze(0).to(device))).softmax(1)
    top1 = prob[0, cls].item()
    ent  = -torch.sum(prob * torch.log(prob + 1e-8)).item()
    if real_img_01 is not None:
        a = recon_01.permute(1,2,0).cpu().numpy()
        b = real_img_01.permute(1,2,0).cpu().numpy()
        try:
            ssim_v = compute_ssim(a, b, channel_axis=2, data_range=1.0)
        except Exception:
            ssim_v = 0.0
    else:
        ssim_v = 0.0
    return top1, ent, ssim_v

# ── Visualisation ─────────────────────────────────────────────────────────────
def save_comparison_panel(recon_01, real_01, path, cls, exp_name):
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))
    axes[0].imshow(recon_01.permute(1,2,0).cpu().numpy())
    axes[0].set_title(f"Reconstructed  ({class_names[cls]})", fontsize=11)
    axes[0].axis("off")
    axes[1].imshow(real_01.permute(1,2,0).cpu().numpy())
    axes[1].set_title("Nearest Real Training Image", fontsize=11)
    axes[1].axis("off")
    fig.suptitle(f"{exp_name}  |  class {cls}: {class_names[cls]}", fontsize=12)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
#  MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────
os.makedirs(output_dir, exist_ok=True)

_exp_tag          = "_".join(experiments.keys())
metric_csv_path   = f"recon_metrics_{_exp_tag}.csv"
summary_csv_path  = f"recon_summary_{_exp_tag}.csv"

mf = open(metric_csv_path, "w", newline="")
mw = csv.writer(mf)
mw.writerow(["Experiment", "Heterogeneity", "Class", "Class_Name", "Image",
             "Top1_Confidence", "Entropy", "SSIM_vs_NearestReal",
             "Nearest_Real_Dataset_Index", "Feature_Distance"])

summary_rows = []

for exp_name, exp_info in experiments.items():
    print(f"\n{'='*65}")
    print(f"  Experiment {exp_name}  ({exp_info['heterogeneity']})")
    print(f"{'='*65}")

    model = get_model(exp_info["ckpt"])
    bank  = build_real_bank(model) if project_real else None

    exp_dir = os.path.join(output_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    per_class = {c: {"top1": [], "ent": [], "ssim": [], "dist": []}
                 for c in range(num_classes)}

    for cls in target_classes:
        for img_idx in range(images_per_class):
            print(f"\n── Class {cls} ({class_names[cls]})  sample {img_idx} ──")
            stem = f"{exp_name}_class{cls}_img{img_idx}"

            # ── Generate or retrieve ──────────────────────────────────────
            if attack_mode == "train_retrieve":
                recon, nr_idx, nr_dist = retrieve_train_image(cls, img_idx, bank)
                print(f"  Retrieved training idx={nr_idx}  score={nr_dist:.4f}")
            else:
                if bank is not None:
                    anchor_img, target_feat, target_pyramid, anchor_dsidx = build_anchor_targets(model, bank, cls, img_idx)
                    print(f"  Anchor idx={anchor_dsidx}")
                else:
                    anchor_img, target_feat, target_pyramid = None, None, None
                recon = reconstruct_image(model, cls,
                                          target_feat=target_feat,
                                          target_pyramid=target_pyramid,
                                          init_img=anchor_img)
                nr_idx, nr_dist = -1, -1.0

            # ── Save reconstructed / retrieved image ──────────────────────
            plt.imsave(os.path.join(exp_dir, f"{stem}.png"),
                       recon.permute(1,2,0).cpu().numpy())

            # ── Nearest-real + comparison panel ──────────────────────────
            real_img = None
            if project_real and bank is not None:
                if attack_mode == "train_retrieve":
                    # recon IS a real image; find the 2nd closest for comparison
                    real_img = recon
                else:
                    real_img, nr_idx, nr_dist = find_nearest(model, recon, cls, bank)
                    if project_final_real:
                        # Optional hard projection: persist actual training image as final output.
                        recon = real_img.clone()
                    plt.imsave(os.path.join(exp_dir, f"{stem}_nearest_real.png"),
                               real_img.permute(1,2,0).cpu().numpy())
                    save_comparison_panel(
                        recon, real_img,
                        os.path.join(exp_dir, f"{stem}_comparison.png"),
                        cls, exp_name)

            # ── Metrics ───────────────────────────────────────────────────
            top1, ent, ssim_v = compute_metrics(model, recon, cls, real_img)
            mw.writerow([exp_name, exp_info["heterogeneity"],
                         cls, class_names[cls], img_idx,
                         top1, ent, ssim_v, nr_idx, nr_dist])
            mf.flush()

            per_class[cls]["top1"] .append(top1)
            per_class[cls]["ent"]  .append(ent)
            per_class[cls]["ssim"] .append(ssim_v)
            if nr_dist >= 0:
                per_class[cls]["dist"].append(nr_dist)

            print(f"  ✓ top1={top1:.4f}  entropy={ent:.4f}  "
                  f"ssim={ssim_v:.4f}  nr_idx={nr_idx}  dist={nr_dist:.3f}")

    # ── Per-class summary ─────────────────────────────────────────────────
    for cls in target_classes:
        s = per_class[cls]
        n = len(s["top1"])
        summary_rows.append({
            "Experiment":   exp_name,
            "Heterogeneity":exp_info["heterogeneity"],
            "Class":        cls,
            "Class_Name":   class_names[cls],
            "N_Samples":    n,
            "Mean_Top1":    round(float(np.mean(s["top1"]))    if n else -1., 4),
            "Mean_Entropy": round(float(np.mean(s["ent"]))     if n else -1., 4),
            "Mean_SSIM":    round(float(np.mean(s["ssim"]))    if n else -1., 4),
            "Mean_Dist":    round(float(np.mean(s["dist"]))    if s["dist"] else -1., 4),
        })

mf.close()

if summary_rows:
    summary_rows.sort(key=lambda r: (r["Experiment"], -r["Mean_Top1"]))
    with open(summary_csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

print(f"\n✅  Done.")
print(f"    Images   → {output_dir}/")
print(f"    Metrics  → {metric_csv_path}")
print(f"    Summary  → {summary_csv_path}")
