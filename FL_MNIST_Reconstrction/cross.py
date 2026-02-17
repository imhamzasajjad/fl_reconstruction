# -*- coding: utf-8 -*-
#!/usr/bin/env python3

"""
FINAL CLEAN EVALUATION SCRIPT

For each model:
    - Test on all 4 datasets
    - Print prediction per image
    - Save ONE predictions CSV
    - Save ONE per-class metrics CSV
    - Save ONE overall model performance CSV
"""

import os
import re
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from PIL import Image
import torchvision.transforms as T
from torchvision.models import resnet34

# =========================================================
# CONFIG
# =========================================================
device = "cuda" if torch.cuda.is_available() else "cpu"

experiments = {
    "E1_IID": "gm_best_100_MNIST_IID.pt",
    "E2_Mild": "gm_best_20_MNIST_Non_IID(Mild).pt",
    "E3_Strong": "gm_best_20_MNIST_Non_IID(Strong).pt",
    "E4_Extreme": "gm_best_20_MNIST_Non_IID(Extream).pt",
}

dataset_root = "./recon_pub_mnist"

output_predictions = "ALL_predictions.csv"
output_metrics = "ALL_metrics.csv"
output_overall = "OVERALL_MODEL_PERFORMANCE.csv"

# =========================================================
# TRANSFORM (same as training test transform)
# =========================================================
transform = T.Compose([
    T.Grayscale(num_output_channels=1),
    T.Resize((28, 28)),
    T.ToTensor(),
    T.Normalize((0.1307,), (0.3081,))
])

# =========================================================
# MODEL ARCHITECTURE
# =========================================================
def get_model():
    model = resnet34(num_classes=10)
    model.conv1 = nn.Conv2d(
        1, 64, kernel_size=3, stride=1, padding=1, bias=False
    )
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model

# =========================================================
# EXTRACT LABEL FROM FILENAME
# Example: E1_IID_c3_4.png -> 3
# =========================================================
def extract_label(filename):
    match = re.search(r'c(\d+)', filename)
    if match:
        return int(match.group(1))
    else:
        raise ValueError(f"Cannot extract label from {filename}")

# =========================================================
# STORAGE
# =========================================================
all_predictions = []
all_metrics = []

# =========================================================
# MAIN EVALUATION LOOP
# =========================================================
for model_name, model_path in experiments.items():

    print("\n========================================")
    print(f"Loading Model: {model_name}")
    print("========================================")

    model = get_model().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    for dataset_name in experiments.keys():

        print(f"\n--- Testing {model_name} on {dataset_name} ---")

        dataset_path = os.path.join(dataset_root, dataset_name)

        y_true = []
        y_pred = []

        for fname in sorted(os.listdir(dataset_path)):

            if fname.endswith(".png"):

                img_path = os.path.join(dataset_path, fname)

                img = Image.open(img_path)
                img = transform(img).unsqueeze(0).to(device)

                true_label = extract_label(fname)

                with torch.no_grad():
                    output = model(img)
                    pred_label = torch.argmax(output, dim=1).item()

                y_true.append(true_label)
                y_pred.append(pred_label)

                # 🔥 PRINT LIVE RESULT
                print(f"[{model_name} | {dataset_name}] "
                      f"{fname} -> True: {true_label} | Pred: {pred_label}")

                # Store prediction
                all_predictions.append({
                    "model": model_name,
                    "dataset": dataset_name,
                    "filename": fname,
                    "true_label": true_label,
                    "predicted_label": pred_label,
                    "correct": int(true_label == pred_label)
                })

        # =====================================================
        # PER-CLASS METRICS FOR THIS MODEL × DATASET
        # =====================================================
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        accuracy = np.mean(y_true == y_pred)
        print(f"Accuracy on {dataset_name} = {accuracy:.4f}")

        for c in range(10):

            TP = np.sum((y_pred == c) & (y_true == c))
            TN = np.sum((y_pred != c) & (y_true != c))
            FP = np.sum((y_pred == c) & (y_true != c))
            FN = np.sum((y_pred != c) & (y_true == c))

            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            all_metrics.append({
                "model": model_name,
                "dataset": dataset_name,
                "class": c,
                "TP": TP,
                "TN": TN,
                "FP": FP,
                "FN": FN,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "dataset_accuracy": accuracy
            })

# =========================================================
# SAVE PREDICTIONS FILE
# =========================================================
df_predictions = pd.DataFrame(all_predictions)
df_predictions.to_csv(output_predictions, index=False)

print("\nSaved:", output_predictions)

# =========================================================
# SAVE PER-CLASS METRICS FILE
# =========================================================
df_metrics = pd.DataFrame(all_metrics)
df_metrics.to_csv(output_metrics, index=False)

print("Saved:", output_metrics)

# =========================================================
# OVERALL PERFORMANCE PER MODEL (ALL DATASETS COMBINED)
# =========================================================
print("\n========================================")
print("Computing Overall Performance Per Model")
print("========================================")

overall_results = []

for model_name in experiments.keys():

    model_data = df_predictions[df_predictions["model"] == model_name]

    y_true = model_data["true_label"].values
    y_pred = model_data["predicted_label"].values

    accuracy = np.mean(y_true == y_pred)

    precisions = []
    recalls = []
    f1s = []

    for c in range(10):

        TP = np.sum((y_pred == c) & (y_true == c))
        FP = np.sum((y_pred == c) & (y_true != c))
        FN = np.sum((y_pred != c) & (y_true == c))

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    macro_precision = np.mean(precisions)
    macro_recall = np.mean(recalls)
    macro_f1 = np.mean(f1s)

    overall_results.append({
        "Model": model_name,
        "Accuracy": round(accuracy, 4),
        "Precision": round(macro_precision, 4),
        "Recall": round(macro_recall, 4),
        "F1": round(macro_f1, 4)
    })

    print(f"\n{model_name}")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {macro_precision:.4f}")
    print(f"Recall   : {macro_recall:.4f}")
    print(f"F1 Score : {macro_f1:.4f}")

# Save final summary table
df_overall = pd.DataFrame(overall_results)
df_overall.to_csv(output_overall, index=False)

print("\nSaved:", output_overall)
print("\n✅ Evaluation Complete!")
