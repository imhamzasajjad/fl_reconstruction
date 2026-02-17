# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import resnet34
from PIL import Image
import os
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ================= CONFIG =================
device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATHS = {
    "E5": "gm_best_100_IID.pt",
    "E6": "gm_best_100_Non_IID(Mild).pt",
    "E7": "gm_best_100_Non_IID(Moderate).pt",
    "E8": "gm_best_100_Non_IID(Extreme).pt"
}

IMAGE_DIR = "recon_output"  # folder with all reconstructed images
CLASS_NAMES = ["airplane","automobile","bird","cat","deer",
               "dog","frog","horse","ship","truck"]
NUM_CLASSES = len(CLASS_NAMES)

# ================= MODEL LOADING =================
def load_model(path):
    model = resnet34(num_classes=NUM_CLASSES)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model

# ================= TRANSFORM =================
transform = T.Compose([
    T.Resize((32,32)),
    T.ToTensor(),
    T.Normalize((0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010))
])

# ================= LOAD IMAGE FILES =================
image_files = [f for f in sorted(os.listdir(IMAGE_DIR)) if f.lower().endswith((".png",".jpg",".jpeg"))]

true_labels = []
for f in image_files:
    try:
        cls = int(f.split("_")[1].replace("class",""))
    except:
        cls = -1
    true_labels.append(cls)

# ================= PREDICTIONS =================
predictions_per_model = {}
detailed_predictions = []

for model_name, model_path in MODEL_PATHS.items():
    print(f"\n=== Evaluating model {model_name} ===")
    model = load_model(model_path)
    preds = []

    for idx, fname in enumerate(image_files):
        img_path = os.path.join(IMAGE_DIR, fname)
        img = Image.open(img_path).convert("RGB")
        x = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
        pred_class = probs.argmax(dim=1).item()
        preds.append(pred_class)

        # store detailed per-image CSV
        detailed_predictions.append([model_name, fname, true_labels[idx], pred_class, probs[0].cpu().tolist()])

    predictions_per_model[model_name] = preds

# ================= SAVE DETAILED IMAGE PREDICTIONS =================
detailed_csv = "image_predictions.csv"
with open(detailed_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Eval_Model", "Image_Name", "True_Class", "Predicted_Class", "Probabilities"])
    for row in detailed_predictions:
        writer.writerow(row)

# ================= COMPUTE PER-CLASS METRICS =================
overall_csv = "cross_model_metrics.csv"
with open(overall_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "Eval_Model", "Accuracy", "Precision_Macro", "Recall_Macro", "F1_Macro",
        "Class", "TP", "TN", "FP", "FN"
    ])

    for model_name, preds in predictions_per_model.items():
        acc = accuracy_score(true_labels, preds)
        prec = precision_score(true_labels, preds, average='macro', zero_division=0)
        rec = recall_score(true_labels, preds, average='macro', zero_division=0)
        f1 = f1_score(true_labels, preds, average='macro', zero_division=0)
        cm = confusion_matrix(true_labels, preds, labels=range(NUM_CLASSES))

        for i in range(NUM_CLASSES):
            TP = cm[i, i]
            FP = cm[:, i].sum() - TP
            FN = cm[i, :].sum() - TP
            TN = cm.sum() - (TP + FP + FN)
            writer.writerow([model_name, acc, prec, rec, f1, i, TP, TN, FP, FN])

print("\n✅ Evaluation complete!")
print(f"Overall metrics saved to: {overall_csv}")
print(f"Detailed image predictions saved to: {detailed_csv}")

# ================= CROSS-MODEL CONFUSION HEATMAP =================
heatmap_dir = "heatmaps"
os.makedirs(heatmap_dir, exist_ok=True)

for model_name, preds in predictions_per_model.items():
    cm = confusion_matrix(true_labels, preds, labels=range(NUM_CLASSES))
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, cmap="Blues")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.title(f"Confusion Matrix for {model_name} on All Reconstructions")
    plt.tight_layout()
    plt.savefig(os.path.join(heatmap_dir, f"confusion_{model_name}.png"))
    plt.close()

print(f"✅ Confusion heatmaps saved in folder: {heatmap_dir}")
