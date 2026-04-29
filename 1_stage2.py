# ==========================================
# Stage 2: ConvNeXt-Small 5-fold CV + 최종 학습
# stage2_train.py
#
# 구조:
#   1. train.csv (899장)으로 5-fold stratified CV → mean ± std 보고
#   2. train.csv 전체로 최종 모델 학습 → best_model.pth
#   3. test.csv (225장)으로 최종 평가 (XAI 분석용)
# ==========================================

import gc
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, f1_score,
                              confusion_matrix, classification_report)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

CFG = {
    "split_dir":       "/data/JSM/convnext/0_stage1",
    "save_dir":        "/data/JSM/convnext/1_train",
    "model_name":      "convnext_small",
    "num_classes":     5,
    "pretrained":      True,
    "img_size":        224,
    "batch_size":      32,
    "epochs":          100,
    "lr":              4e-4,
    "weight_decay":    1e-2,
    "label_smoothing": 0.1,
    "warmup_epochs":   5,
    "min_lr":          1e-6,
    "patience":        15,
    "seed":            42,
    "n_folds":         5,
}

torch.manual_seed(CFG["seed"])
np.random.seed(CFG["seed"])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"device: {device}")

save_root = Path(CFG["save_dir"])
save_root.mkdir(parents=True, exist_ok=True)

# ==========================================
# Transform
# ==========================================
train_transform = transforms.Compose([
    transforms.Resize((CFG["img_size"], CFG["img_size"])),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(45),
    transforms.ColorJitter(brightness=0.3, contrast=0.3,
                           saturation=0.2, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2),
])

val_transform = transforms.Compose([
    transforms.Resize((CFG["img_size"], CFG["img_size"])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ==========================================
# Dataset
# ==========================================
class SeedDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df        = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        img   = Image.open(row["filepath"]).convert("RGB")
        label = int(row["label"])
        if self.transform:
            img = self.transform(img)
        return img, label

# 데이터 로드
train_df = pd.read_csv(Path(CFG["split_dir"]) / "train.csv")
test_df  = pd.read_csv(Path(CFG["split_dir"]) / "test.csv")

with open(Path(CFG["split_dir"]) / "split_config.json") as f:
    split_cfg = json.load(f)
classes   = split_cfg["classes"]
short_cls = [c.replace("_crop", "") for c in classes]

print(f"Train: {len(train_df)}장 | Test: {len(test_df)}장")

# ==========================================
# 학습/평가 함수
# ==========================================
criterion = nn.CrossEntropyLoss(label_smoothing=CFG["label_smoothing"])

def build_model():
    m = timm.create_model(CFG["model_name"], pretrained=CFG["pretrained"],
                           num_classes=CFG["num_classes"])
    return m.to(device)

def train_epoch(model, loader, optimizer):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(labels)
        correct    += (out.argmax(1) == labels).sum().item()
        total      += len(labels)
    return total_loss / total, correct / total

def eval_epoch(model, loader):
    model.eval()
    preds, trues, total_loss = [], [], 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out  = model(imgs)
            loss = criterion(out, labels)
            total_loss += loss.item() * len(labels)
            preds.extend(out.argmax(1).cpu().numpy())
            trues.extend(labels.cpu().numpy())
    acc = accuracy_score(trues, preds)
    return total_loss / len(loader.dataset), acc, preds, trues

def run_training(model, tr_loader, va_loader, save_path, tag=""):
    optimizer = torch.optim.AdamW(model.parameters(),
                                   lr=CFG["lr"],
                                   weight_decay=CFG["weight_decay"])
    warmup_sch = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda e: (e+1)/CFG["warmup_epochs"] if e < CFG["warmup_epochs"] else 1.0)
    cos_sch = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CFG["epochs"]-CFG["warmup_epochs"], eta_min=CFG["min_lr"])

    best_acc, best_epoch, patience_cnt = 0, 0, 0
    history = {"train_loss": [], "train_acc": [],
               "val_loss":   [], "val_acc":   []}

    for epoch in range(1, CFG["epochs"] + 1):
        tr_loss, tr_acc       = train_epoch(model, tr_loader, optimizer)
        va_loss, va_acc, _, _ = eval_epoch(model, va_loader)

        if epoch <= CFG["warmup_epochs"]:
            warmup_sch.step()
        else:
            cos_sch.step()

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)

        if va_acc > best_acc:
            best_acc, best_epoch, patience_cnt = va_acc, epoch, 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_cnt += 1

        print(f"  [{tag}] Epoch {epoch:3d} | "
              f"Tr {tr_acc:.4f} | Va {va_acc:.4f} | "
              f"Best {best_acc:.4f}@{best_epoch}")

        if patience_cnt >= CFG["patience"]:
            print(f"  Early stopping @ epoch {epoch}")
            break

        if epoch % 10 == 0:
            torch.cuda.empty_cache(); gc.collect()

    return best_acc, best_epoch, history

# ==========================================
# 1. 5-Fold Cross Validation
# ==========================================
print("\n" + "=" * 65)
print(" 5-Fold Stratified Cross Validation (train.csv 899장)")
print("=" * 65)

skf    = StratifiedKFold(n_splits=CFG["n_folds"], shuffle=True,
                          random_state=CFG["seed"])
labels = train_df["label"].values
fold_results = []

for fold, (tr_idx, va_idx) in enumerate(skf.split(train_df, labels), 1):
    print(f"\n[Fold {fold}/{CFG['n_folds']}]  "
          f"train={len(tr_idx)} val={len(va_idx)}")

    tr_loader = DataLoader(
        SeedDataset(train_df.iloc[tr_idx], train_transform),
        batch_size=CFG["batch_size"], shuffle=True,
        num_workers=4, pin_memory=True)
    va_loader = DataLoader(
        SeedDataset(train_df.iloc[va_idx], val_transform),
        batch_size=CFG["batch_size"], shuffle=False,
        num_workers=4, pin_memory=True)

    model     = build_model()
    fold_path = save_root / f"fold{fold}_best.pth"
    best_acc, best_epoch, _ = run_training(
        model, tr_loader, va_loader, fold_path, f"Fold{fold}")

    model.load_state_dict(torch.load(fold_path, map_location=device))
    _, acc, preds, trues = eval_epoch(model, va_loader)
    f1   = f1_score(trues, preds, average='macro', zero_division=0)
    prec = precision_score(trues, preds, average='macro', zero_division=0)
    rec  = recall_score(trues, preds, average='macro', zero_division=0)

    fold_results.append({
        "fold": fold, "best_epoch": best_epoch,
        "accuracy": acc, "f1": f1,
        "precision": prec, "recall": rec,
    })
    print(f"  → Fold {fold}: Acc={acc:.4f}  F1={f1:.4f}")
    del model; torch.cuda.empty_cache(); gc.collect()

cv_df = pd.DataFrame(fold_results)
cv_df.to_csv(save_root / "cv_results.csv", index=False)

print("\n" + "=" * 65)
print(" 5-Fold CV 요약")
print("=" * 65)
for _, row in cv_df.iterrows():
    print(f"  Fold {int(row.fold)}: Acc={row.accuracy:.4f}  F1={row.f1:.4f}")
print(f"\n  Mean Acc: {cv_df.accuracy.mean():.4f} ± {cv_df.accuracy.std():.4f}")
print(f"  Mean F1:  {cv_df.f1.mean():.4f} ± {cv_df.f1.std():.4f}")

# CV 결과 시각화
fig, ax = plt.subplots(figsize=(7, 4))
x = [f"Fold {i}" for i in range(1, CFG["n_folds"]+1)]
ax.bar(x, cv_df["accuracy"], color="#0D9488", edgecolor='white',
       label="Accuracy", alpha=0.85)
ax.bar(x, cv_df["f1"], color="#F59E0B", edgecolor='white',
       label="F1", alpha=0.7)
ax.axhline(cv_df.accuracy.mean(), color="#0D9488", linestyle='--',
           label=f"Mean Acc={cv_df.accuracy.mean():.4f}±{cv_df.accuracy.std():.4f}")
ax.axhline(cv_df.f1.mean(), color="#F59E0B", linestyle='--',
           label=f"Mean F1={cv_df.f1.mean():.4f}±{cv_df.f1.std():.4f}")
ax.set_ylim(0.9, 1.01)
ax.set_title("5-Fold CV — ConvNeXt-Small", fontsize=12)
ax.legend(fontsize=9); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(save_root / "cv_results.png", dpi=150, bbox_inches='tight')
plt.close()

# ==========================================
# 2. 최종 모델 학습 (train.csv 전체 899장)
# ==========================================
print("\n" + "=" * 65)
print(" 최종 모델 학습 (train.csv 전체 899장)")
print("=" * 65)

final_tr_loader = DataLoader(
    SeedDataset(train_df, train_transform),
    batch_size=CFG["batch_size"], shuffle=True,
    num_workers=4, pin_memory=True)
final_va_loader = DataLoader(
    SeedDataset(test_df, val_transform),
    batch_size=CFG["batch_size"], shuffle=False,
    num_workers=4, pin_memory=True)

final_model = build_model()
_, _, final_history = run_training(
    final_model, final_tr_loader, final_va_loader,
    save_root / "best_model.pth", "Final")

# ==========================================
# 3. 최종 Test 평가
# ==========================================
print("\n" + "=" * 65)
print(" 최종 Test 평가 (test.csv 225장)")
print("=" * 65)

final_model.load_state_dict(
    torch.load(save_root / "best_model.pth", map_location=device))
_, final_acc, final_preds, final_trues = eval_epoch(final_model, final_va_loader)

prec = precision_score(final_trues, final_preds, average='macro', zero_division=0)
rec  = recall_score(final_trues, final_preds, average='macro', zero_division=0)
f1   = f1_score(final_trues, final_preds, average='macro', zero_division=0)
cm   = confusion_matrix(final_trues, final_preds)

print(f"  Accuracy:  {final_acc:.4f}")
print(f"  Precision: {prec:.4f}")
print(f"  Recall:    {rec:.4f}")
print(f"  F1-Score:  {f1:.4f}")
print(f"\n  5-Fold CV: {cv_df.accuracy.mean():.4f} ± {cv_df.accuracy.std():.4f}")
print("\n[클래스별 성능]")
print(classification_report(final_trues, final_preds,
                             target_names=short_cls, digits=4))

# 결과 저장
results = {
    "model":          CFG["model_name"],
    "cv_acc_mean":    round(float(cv_df.accuracy.mean()), 4),
    "cv_acc_std":     round(float(cv_df.accuracy.std()),  4),
    "cv_f1_mean":     round(float(cv_df.f1.mean()), 4),
    "cv_f1_std":      round(float(cv_df.f1.std()),  4),
    "test_accuracy":  round(final_acc, 4),
    "test_precision": round(prec, 4),
    "test_recall":    round(rec,  4),
    "test_f1":        round(f1,   4),
    "cfg": CFG,
}
with open(save_root / "results.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
pd.DataFrame(final_history).to_csv(
    save_root / "final_training_history.csv", index=False)

# 시각화: Loss/Acc 곡선
epochs_ran = list(range(1, len(final_history["train_loss"]) + 1))
fig, axes  = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(epochs_ran, final_history["train_loss"], label="Train", color="#0D9488")
axes[0].plot(epochs_ran, final_history["val_loss"],   label="Test",  color="#EF4444")
axes[0].set_title("Loss Curve"); axes[0].set_xlabel("Epoch")
axes[0].legend(); axes[0].grid(alpha=0.3)
axes[1].plot(epochs_ran, final_history["train_acc"], label="Train", color="#0D9488")
axes[1].plot(epochs_ran, final_history["val_acc"],   label="Test",  color="#EF4444")
axes[1].axhline(final_acc, color='blue', linestyle=':',
                label=f"Best={final_acc:.4f}")
axes[1].set_title("Accuracy Curve"); axes[1].set_xlabel("Epoch")
axes[1].legend(); axes[1].grid(alpha=0.3)
plt.suptitle(
    f"ConvNeXt-Small | CV: {cv_df.accuracy.mean():.4f}±{cv_df.accuracy.std():.4f} | "
    f"Test: {final_acc:.4f}", fontsize=11)
plt.tight_layout()
plt.savefig(save_root / "training_curve.png", dpi=150, bbox_inches='tight')
plt.close()

# 시각화: Confusion Matrix
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=short_cls, yticklabels=short_cls, ax=ax)
ax.set_title(
    f"Confusion Matrix — ConvNeXt-Small\nAcc: {final_acc:.4f}  F1: {f1:.4f}",
    fontsize=11)
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
plt.tight_layout()
plt.savefig(save_root / "confusion_matrix.png", dpi=150, bbox_inches='tight')
plt.close()

print(f"\n완료. 저장 경로: {save_root}")
print("생성 파일: best_model.pth, results.json, cv_results.csv, cv_results.png")
print("          training_curve.png, confusion_matrix.png")