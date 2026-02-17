# recon_full_mnist.py
# ----------------------------------------------------------
# Federated Model Reconstruction & Metrics (MNIST)
# Works for multiple FL models (IID & Non-IID)
# ----------------------------------------------------------

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
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
os.makedirs("recon_output_mnist", exist_ok=True)

csv_file = open("recon_metrics_mnist.csv", "w", newline="")
writer_csv = csv.writer(csv_file)
writer_csv.writerow(["Experiment", "Class", "Image", "Top1_Logit", "Entropy", "SSIM"])

for exp_name, exp_info in experiments.items():
    print(f"\n==== Experiment {exp_name} ({exp_info['heterogeneity']}) ====")

    model = get_model(exp_info["ckpt"])

    exp_dir = f"recon_output_mnist/{exp_name}"
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

            top1, ent, ssim_val = compute_metrics(model, recon_img, cls)

            writer_csv.writerow([
                exp_name, cls, img_idx, top1, ent, ssim_val
            ])

            print(
                f"{exp_name} | Class {cls} | Img {img_idx} → "
                f"Top1={top1:.4f}, Ent={ent:.4f}, SSIM={ssim_val:.4f}"
            )

csv_file.close()
print("✅ MNIST reconstruction completed for ALL models.")
