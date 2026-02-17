# Can Non-IID Data Prevent Privacy Leakage in Federated Learning?

Official implementation of:

**Can Non-IID Data Prevent Privacy Leakage in Federated Learning?  
An Empirical Study of Post-Hoc Model Reconstruction Attacks**

---

## 📖 Overview

This repository contains the official implementation of our empirical study analyzing whether statistical heterogeneity (non-IID client data) mitigates privacy leakage in federated learning.

We evaluate post-hoc model reconstruction attacks under varying Dirichlet heterogeneity levels:

α ∈ {∞ (IID), 0.5 (Mild), 0.3 (Strong), 0.1 (Extreme)}

Experiments are conducted on:

- MNIST
- CIFAR-10
- 10 federated clients
- ResNet-34 backbone
- Weighted FedAvg aggregation

---

## 🏗 Federated Learning Setup

- Cross-device federated learning
- 10 clients
- 100 rounds (CIFAR-10)
- 20 rounds (MNIST)
- 3 local epochs per round
- Batch size: 128
- SGD (lr=0.05, weight decay=5e-4)
- Cosine annealing scheduler
- Mixup augmentation (α=0.8)
- ImageNet-pretrained ResNet-34 initialization

---

## 🔓 Reconstruction Attack

After federated training, we perform post-hoc reconstruction from the trained global model.

Evaluation metrics include:

- Classification accuracy
- Precision / Recall / F1
- Target-class confidence
- Predictive entropy
- Cross-model evaluation
- Structural similarity (SSIM)

---

## 📂 Repository Structure

```
fl_reconstruction/
│
├── cifar/
│   ├── fl_cifar.py
│   ├── recon.py
│   └── cross.py
│   └── Results    
│
├── mnist/
│   ├── fl_mnist.py
│   ├── recon.py
│   └── cross.py
│   └── Results 
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Installation

It is recommended to create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate      # Linux / Mac
venv\Scripts\activate         # Windows
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Running Experiments

### 1️⃣ Train Federated Model

CIFAR-10:

```bash
cd cifar
python fl_cifar.py
```

MNIST:

```bash
cd mnist
python fl_mnist.py
```

---

### 2️⃣ Run Reconstruction Attack

```bash
python recon.py
```

---

### 3️⃣ Run Cross-Model Evaluation

```bash
python cross.py
```

This generates the evaluation metrics reported in the paper.

---

## 📊 Reproducibility

- Datasets are automatically downloaded via torchvision.
- Model weights (`*.pt`) are not included to reduce repository size.
- Results are saved in `results/`.
- Experiments were conducted using CUDA-enabled GPUs.
- Random seeds can be fixed inside training scripts for deterministic behavior.

---

## 📜 Citation

If you use this code in your research, please cite our work.  
Citation details will be updated upon publication.

---

## ⚠️ Disclaimer

This code is provided strictly for academic research purposes.
