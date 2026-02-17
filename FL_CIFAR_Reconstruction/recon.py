# recon_full.py
# ----------------------------------------------------------
# Federated Model Reconstruction & Metrics
# Works for multiple FL models (IID and Non-IID)
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
# EXPERIMENTS
# ----------------------------
experiments = {
    #"E5": {"alpha": float('inf'), "heterogeneity": "IID", "ckpt": "gm_best_100_IID.pt"},
    #"E6": {"alpha": 0.5, "heterogeneity": "Mild", "ckpt": "gm_best_100_Non_IID(Mild).pt"},
    #"E7": {"alpha": 0.3, "heterogeneity": "Strong", "ckpt": "gm_best_100_Non_IID(Moderate).pt"},
    "E8": {"alpha": 0.1, "heterogeneity": "Extreme", "ckpt": "gm_best_100_Non_IID(Extreme).pt"},
}

# ----------------------------
# CONFIG
# ----------------------------
num_classes = 10
images_per_class = 10
steps = 50000
tv_w = 0.001
freq_w = 0.001

# ----------------------------
# MODEL LOADER
# ----------------------------
def get_model(pretrained_ckpt=None):
    model = resnet34(num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, 10)
    if pretrained_ckpt:
        model.load_state_dict(torch.load(pretrained_ckpt, map_location=device))
    return model.to(device).eval()

# ----------------------------
# FOURIER PARAMETERIZATION
# ----------------------------
def init_fourier(shape=(1, 3, 32, 32), scale=0.01):
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
normalize = T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
augmentation = T.Compose([
    T.RandomResizedCrop(32, scale=(0.4,1.0)),
    T.RandomHorizontalFlip(),
    T.ColorJitter(0.3,0.3,0.3,0.1),
])

# ----------------------------
# CUTOUT
# ----------------------------
def cutout(img, hole_size=8):
    # img: [C,H,W]
    c,h,w = img.shape
    y = np.random.randint(0,h-hole_size)
    x = np.random.randint(0,w-hole_size)
    img[:, y:y+hole_size, x:x+hole_size] = 0
    return img

# ----------------------------
# TV LOSS
# ----------------------------
def tv_loss(img):
    # img: [1,3,32,32]
    return (torch.sum(torch.abs(img[:,:,:-1,:] - img[:,:,1:,:])) +
            torch.sum(torch.abs(img[:,:,:,:-1] - img[:,:,:,1:])))

# ----------------------------
# RECONSTRUCTION FUNCTION
# ----------------------------
def reconstruct_image(model, target_class, steps=steps):
    spectrum = init_fourier()
    optimizer = optim.Adam([spectrum], lr=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=0.0001)

    target_logits = torch.zeros((1, num_classes), device=device)
    target_logits[0, target_class] = 12.0

    for step in range(steps):
        optimizer.zero_grad()

        recon = fourier_to_img(spectrum).squeeze(0)      # [3,32,32]
        aug = augmentation(recon.cpu()).to(device)        # still [3,32,32]
        aug = cutout(aug)                                 # apply cutout
        inp = normalize(aug).unsqueeze(0).to(device)     # [1,3,32,32]

        pred = model(inp)
        logit_loss = nn.MSELoss()(pred, target_logits)
        loss = logit_loss + tv_w*tv_loss(recon.unsqueeze(0)) + freq_w*torch.mean(torch.abs(spectrum)**0.7)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % 5000 == 0:
            print(f"[Class {target_class} Step {step}/{steps}] Loss={loss.item():.4f}")

    return recon.detach()

# ----------------------------
# METRICS
# ----------------------------
def compute_metrics(model, recon_img, target_class):
    inp = normalize(recon_img).unsqueeze(0).to(device)
    pred = model(inp).softmax(dim=1)
    top1_logit = pred[0,target_class].item()
    entropy = -torch.sum(pred*torch.log(pred+1e-8)).item()
    
    # SSIM
    recon_np = recon_img.permute(1,2,0).cpu().numpy()
    target_np = np.zeros_like(recon_np)  # dummy blank target for placeholder
    try:
        ssim_val = ssim(target_np, recon_np, channel_axis=2)
    except:
        ssim_val = 0.0
    return top1_logit, entropy, ssim_val

# ----------------------------
# MAIN LOOP
# ----------------------------
os.makedirs("recon_outputE8", exist_ok=True)
csv_file = open("recon_metricsE8.csv", "w", newline="")
writer_csv = csv.writer(csv_file)
writer_csv.writerow(["Experiment","Class","Image","Top1_Logit","Entropy","SSIM"])

for exp_name, exp_info in experiments.items():
    print(f"\n==== Starting Experiment {exp_name} ({exp_info['heterogeneity']}) ====")
    model = get_model(exp_info["ckpt"])
    
    for cls in range(num_classes):
        for img_idx in range(images_per_class):
            recon_img = reconstruct_image(model, cls, steps=50000)  # test steps = 100 for speed, adjust as needed
            plt.imshow(recon_img.permute(1,2,0).cpu().numpy())
            plt.axis("off")
            plt.savefig(f"recon_output/{exp_name}_class{cls}_img{img_idx}.png")
            plt.close()

            top1, ent, ssim_val = compute_metrics(model, recon_img, cls)
            writer_csv.writerow([exp_name, cls, img_idx, top1, ent, ssim_val])
            print(f"{exp_name} Class {cls} Img {img_idx} → Top1:{top1:.4f}, Ent:{ent:.4f}, SSIM:{ssim_val:.4f}")

csv_file.close()
print("All reconstructions completed. Metrics saved to recon_metrics.csv")
