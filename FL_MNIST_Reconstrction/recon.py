# recon_full_mnist.py
# ----------------------------------------------------------
# Federated Model Reconstruction & Metrics (MNIST)
# Works for multiple FL models (IID & Non-IID)
# ----------------------------------------------------------

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import torchvision
from torchvision.models import resnet34
import matplotlib.pyplot as plt
import numpy as np
import os
import csv
from skimage.metrics import structural_similarity as ssim

# ----------------------------
# DEVICE
# ----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------
# EXPERIMENTS (ALL MODELS)
# ----------------------------
experiments = {
    "E1": {
        "alpha": float("inf"),
        "heterogeneity": "IID",
        "ckpt": "gm_best_100_MNIST_IID.pt"
    },
    #"E2": {
    #    "alpha": 0.5,
     #   "heterogeneity": "Mild",
      #  "ckpt": "gm_best_20_MNIST_Non_IID(Mild).pt"
    #},
    #"E3": {
     #   "alpha": 0.3,
      #  "heterogeneity": "Strong",
      #  "ckpt": "gm_best_20_MNIST_Non_IID(Strong).pt"
    #},
    #"E4": {
     #   "alpha": 0.1,
      #  "heterogeneity": "Extreme",
       # "ckpt": "gm_best_20_MNIST_Non_IID(Extreme).pt"
    #},
}

# ----------------------------
# CONFIG
# ----------------------------
num_classes = 10
images_per_class = 10
steps = 20000
tv_w = 0.001
freq_w = 0.001
img_size = 28
channels = 1
output_dir = "recon_output_mnist"
project_real_images = True
max_real_per_class = 3000
class_names = [str(i) for i in range(10)]

# ----------------------------
# MODEL LOADER (MNIST)
# ----------------------------
def get_model(pretrained_ckpt=None):
    model = resnet34(num_classes=10)

    model.conv1 = nn.Conv2d(
        1, 64, kernel_size=3, stride=1, padding=1, bias=False
    )
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, 10)

    if pretrained_ckpt:
        model.load_state_dict(torch.load(pretrained_ckpt, map_location=device))

    return model.to(device).eval()

# ----------------------------
# FOURIER PARAMETERIZATION
# ----------------------------
def init_fourier(shape=(1, 1, 28, 28), scale=0.01):
    spectrum = torch.randn(shape, device=device) * scale
    spectrum = torch.complex(spectrum, spectrum.clone())
    spectrum.requires_grad_(True)
    return spectrum

def fourier_to_img(spectrum):
    img = torch.fft.ifft2(spectrum, norm="ortho").real
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return img

# ----------------------------
# AUGMENTATION & NORMALIZATION
# ----------------------------
normalize = T.Normalize((0.1307,), (0.3081,))

mean = torch.tensor([0.1307], device=device).view(1, 1, 1, 1)
std = torch.tensor([0.3081], device=device).view(1, 1, 1, 1)

def normalize_batch(x):
    return (x - mean) / std

augmentation = T.Compose([
    T.RandomResizedCrop(28, scale=(0.5, 1.0)),
    T.RandomRotation(15),
])

# ----------------------------
# CUTOUT
# ----------------------------
def cutout(img, hole_size=6):
    _, h, w = img.shape
    y = np.random.randint(0, h - hole_size)
    x = np.random.randint(0, w - hole_size)
    img[:, y:y + hole_size, x:x + hole_size] = 0
    return img

# ----------------------------
# TV LOSS
# ----------------------------
def tv_loss(img):
    return (
        torch.sum(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :])) +
        torch.sum(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]))
    )

@torch.no_grad()
def penultimate_features(model, x):
    x = model.conv1(x)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)
    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)
    x = model.avgpool(x)
    x = torch.flatten(x, 1)
    return x

@torch.no_grad()
def build_real_image_bank(model, max_per_class=3000):
    dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=T.ToTensor())
    by_class = {c: {"images": [], "indices": []} for c in range(num_classes)}

    for idx, (img, label) in enumerate(dataset):
        if len(by_class[label]["images"]) >= max_per_class:
            continue
        by_class[label]["images"].append(img)
        by_class[label]["indices"].append(idx)

    for c in range(num_classes):
        imgs = torch.stack(by_class[c]["images"], dim=0).to(device)
        feats = []
        bs = 512
        for i in range(0, imgs.size(0), bs):
            batch = imgs[i:i + bs]
            feat = penultimate_features(model, normalize_batch(batch)).cpu()
            feats.append(feat)
        by_class[c]["features"] = torch.cat(feats, dim=0)

    return by_class

@torch.no_grad()
def find_nearest_real_image(model, recon_img, target_class, real_bank):
    recon_batch = recon_img.unsqueeze(0).to(device)
    recon_feat = penultimate_features(model, normalize_batch(recon_batch)).cpu()

    class_feats = real_bank[target_class]["features"]
    dists = torch.cdist(recon_feat, class_feats, p=2).squeeze(0)
    nearest_idx = int(torch.argmin(dists).item())

    nearest_img = real_bank[target_class]["images"][nearest_idx]
    nearest_dataset_idx = real_bank[target_class]["indices"][nearest_idx]
    nearest_dist = float(dists[nearest_idx].item())
    return nearest_img, nearest_dataset_idx, nearest_dist

def save_comparison_panel(recon_img, real_img, save_path, class_id, exp_name):
    recon_np = recon_img.squeeze(0).cpu().numpy()
    real_np = real_img.squeeze(0).cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    axes[0].imshow(recon_np, cmap="gray")
    axes[0].set_title(f"Recon ({class_names[class_id]})")
    axes[0].axis("off")

    axes[1].imshow(real_np, cmap="gray")
    axes[1].set_title("Nearest Real")
    axes[1].axis("off")

    fig.suptitle(f"{exp_name} | class {class_id}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)

# ----------------------------
# RECONSTRUCTION FUNCTION
# ----------------------------
def reconstruct_image(model, target_class, steps=steps):
    spectrum = init_fourier()
    optimizer = optim.Adam([spectrum], lr=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=steps, eta_min=1e-4
    )

    target_logits = torch.zeros((1, num_classes), device=device)
    target_logits[0, target_class] = 12.0

    for step in range(steps):
        optimizer.zero_grad()

        recon = fourier_to_img(spectrum).squeeze(0)
        aug = augmentation(recon.cpu()).to(device)
        aug = cutout(aug)
        inp = normalize(aug).unsqueeze(0)

        pred = model(inp)
        logit_loss = nn.MSELoss()(pred, target_logits)

        loss = (
            logit_loss +
            tv_w * tv_loss(recon.unsqueeze(0)) +
            freq_w * torch.mean(torch.abs(spectrum) ** 0.7)
        )

        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % 5000 == 0:
            print(f"[Class {target_class} | Step {step}/{steps}] Loss={loss.item():.4f}")

    return recon.detach()

# ----------------------------
# METRICS
# ----------------------------
def compute_metrics(model, recon_img, target_class):
    inp = normalize(recon_img).unsqueeze(0).to(device)
    pred = model(inp).softmax(dim=1)

    top1 = pred[0, target_class].item()
    entropy = -torch.sum(pred * torch.log(pred + 1e-8)).item()

    recon_np = recon_img.squeeze(0).cpu().numpy()
    target_np = np.zeros_like(recon_np)

    try:
        ssim_val = ssim(target_np, recon_np, data_range=1.0)
    except:
        ssim_val = 0.0

    return top1, entropy, ssim_val

# ----------------------------
# MAIN LOOP (ALL MODELS)
# ----------------------------
os.makedirs(output_dir, exist_ok=True)

csv_file = open("recon_metrics_mnist.csv", "w", newline="")
writer_csv = csv.writer(csv_file)
writer_csv.writerow([
    "Experiment", "Class", "Image", "Top1_Logit", "Entropy", "SSIM",
    "Nearest_Real_Index", "Nearest_Real_Feature_Distance"
])

summary_rows = []

for exp_name, exp_info in experiments.items():
    print(f"\n==== Experiment {exp_name} ({exp_info['heterogeneity']}) ====")

    model = get_model(exp_info["ckpt"])
    real_bank = build_real_image_bank(model, max_per_class=max_real_per_class) if project_real_images else None
    per_class_stats = {c: {"top1": [], "entropy": [], "ssim": [], "dist": []} for c in range(num_classes)}

    exp_dir = f"{output_dir}/{exp_name}"
    os.makedirs(exp_dir, exist_ok=True)

    for cls in range(num_classes):
        for img_idx in range(images_per_class):

            recon_img = reconstruct_image(model, cls)

            plt.imshow(recon_img.squeeze(0).cpu().numpy(), cmap="gray")
            plt.axis("off")
            plt.savefig(
                f"{exp_dir}/{exp_name}_class{cls}_img{img_idx}.png",
                bbox_inches="tight",
                pad_inches=0
            )
            plt.close()

            nearest_idx, nearest_dist = -1, -1.0
            if project_real_images:
                real_img, nearest_idx, nearest_dist = find_nearest_real_image(model, recon_img, cls, real_bank)
                plt.imshow(real_img.squeeze(0).cpu().numpy(), cmap="gray")
                plt.axis("off")
                plt.savefig(
                    f"{exp_dir}/{exp_name}_class{cls}_img{img_idx}_nearest_real.png",
                    bbox_inches="tight",
                    pad_inches=0
                )
                plt.close()
                save_comparison_panel(
                    recon_img,
                    real_img,
                    f"{exp_dir}/{exp_name}_class{cls}_img{img_idx}_comparison.png",
                    cls,
                    exp_name
                )

            top1, ent, ssim_val = compute_metrics(model, recon_img, cls)

            writer_csv.writerow([
                exp_name, cls, img_idx, top1, ent, ssim_val,
                nearest_idx, nearest_dist
            ])
            per_class_stats[cls]["top1"].append(top1)
            per_class_stats[cls]["entropy"].append(ent)
            per_class_stats[cls]["ssim"].append(ssim_val)
            if nearest_dist >= 0:
                per_class_stats[cls]["dist"].append(nearest_dist)

            print(
                f"{exp_name} | Class {cls} | Img {img_idx} → "
                f"Top1={top1:.4f}, Ent={ent:.4f}, SSIM={ssim_val:.4f}, "
                f"NearestRealIdx={nearest_idx}, Dist={nearest_dist:.4f}"
            )

    for cls in range(num_classes):
        dvals = per_class_stats[cls]["dist"]
        summary_rows.append({
            "Experiment": exp_name,
            "Class": cls,
            "Class_Name": class_names[cls],
            "Samples": len(per_class_stats[cls]["top1"]),
            "Mean_Top1_Logit": float(np.mean(per_class_stats[cls]["top1"])),
            "Mean_Entropy": float(np.mean(per_class_stats[cls]["entropy"])),
            "Mean_SSIM": float(np.mean(per_class_stats[cls]["ssim"])),
            "Mean_Nearest_Real_Distance": float(np.mean(dvals)) if len(dvals) > 0 else -1.0,
        })

csv_file.close()
summary_rows = sorted(summary_rows, key=lambda x: (x["Experiment"], x["Mean_Nearest_Real_Distance"]))
with open("recon_class_summary_mnist.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow([
        "Experiment", "Class", "Class_Name", "Samples",
        "Mean_Top1_Logit", "Mean_Entropy", "Mean_SSIM",
        "Mean_Nearest_Real_Distance"
    ])
    for row in summary_rows:
        w.writerow([
            row["Experiment"], row["Class"], row["Class_Name"], row["Samples"],
            row["Mean_Top1_Logit"], row["Mean_Entropy"], row["Mean_SSIM"],
            row["Mean_Nearest_Real_Distance"]
        ])

print("✅ MNIST reconstruction completed for ALL models.")
