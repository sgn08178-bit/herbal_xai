# ==========================================
# Stage 4: Sanity Check
# stage4_sanity.py
#
# 4-1. Model randomization test
#      학습된 모델 vs 랜덤 초기화 모델 IG 비교
#      → SSIM, Spearman r로 유의미한 차이 확인
#
# 4-2. Data randomization test
#      정상 모델 vs 라벨 셔플 재학습 모델 IG 비교
#      → 라벨이 의미 없으면 IG도 달라져야 함
#
# 참고: Adebayo et al. (2018) "Sanity Checks for Saliency Maps"
# ==========================================

import gc
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from scipy.stats import spearmanr
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

import torch
import torch.nn as nn
import timm
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

CFG = {
    "split_dir":     "/data/JSM/convnext/0_stage1",
    "model_path":    "/data/JSM/convnext/1_train/best_model.pth",
    "ig_cache_dir": "/data/JSM/convnext/2_ig_cache/ig_blur",
    "save_dir":      "/data/JSM/convnext/3_sanity",
    "model_name":    "convnext_small",
    "num_classes":   5,
    "img_size":      224,
    "ig_steps":      100,
    "ig_chunk":      10,
    "seed":          42,
    "n_sample":      30,    # sanity check용 이미지 수 (test set에서)

    # 4-2 data randomization 재학습 설정 (가볍게)
    "shuffle_epochs":  50,
    "shuffle_batch":   32,
    "shuffle_lr":      4e-4,
    "shuffle_patience": 999,
}

torch.manual_seed(CFG["seed"])
np.random.seed(CFG["seed"])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"device: {device}")

save_root = Path(CFG["save_dir"])
save_root.mkdir(parents=True, exist_ok=True)

# ==========================================
# 공통 설정
# ==========================================
_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
_std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

def normalize_batch(t):
    return (t - _mean) / _std

load_transform = transforms.Compose([
    transforms.Resize((CFG["img_size"], CFG["img_size"])),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.Resize((CFG["img_size"], CFG["img_size"])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

test_df  = pd.read_csv(Path(CFG["split_dir"]) / "test.csv")
train_df = pd.read_csv(Path(CFG["split_dir"]) / "train.csv")

with open(Path(CFG["split_dir"]) / "split_config.json") as f:
    split_cfg = json.load(f)
classes = split_cfg["classes"]

# sanity check용 샘플 (test set에서)
sample_df = test_df.sample(n=CFG["n_sample"], random_state=CFG["seed"])
print(f"Sanity check 샘플: {len(sample_df)}장")

# ==========================================
# IG 계산 함수
# ==========================================
def compute_ig(model, input_tensor, target_class, steps=100, chunk_size=10):
    baseline  = torch.zeros_like(input_tensor)
    diff      = input_tensor - baseline
    alphas    = torch.linspace(0, 1, steps + 1, device=device)
    grads_sum = torch.zeros_like(input_tensor.squeeze(0))

    for start in range(0, steps + 1, chunk_size):
        end   = min(start + chunk_size, steps + 1)
        a_sub = alphas[start:end].view(-1, 1, 1, 1)
        scaled  = (baseline + a_sub * diff).detach().requires_grad_(True)
        normed  = normalize_batch(scaled)
        outputs = model(normed)
        outputs[:, target_class].sum().backward()
        grads   = scaled.grad.detach()
        weights = torch.ones(grads.shape[0], device=device)
        if start == 0:   weights[0]  = 0.5
        if end == steps + 1: weights[-1] = 0.5
        weights = weights.view(-1, 1, 1, 1)
        grads_sum += (grads * weights).sum(dim=0)
        del scaled, normed, outputs, a_sub, grads, weights
        torch.cuda.empty_cache()

    avg_grads = grads_sum / steps
    ig_signed = diff.squeeze(0).detach() * avg_grads
    ig_map    = ig_signed.abs().sum(dim=0).cpu().numpy().astype(np.float32)
    del avg_grads, ig_signed, grads_sum, diff, baseline
    torch.cuda.empty_cache()
    return ig_map

def compare_maps(map_a, map_b):
    """두 IG 맵 비교: Spearman r + SSIM"""
    r, _    = spearmanr(map_a.flatten(), map_b.flatten())
    s       = ssim(map_a, map_b,
                   data_range=max(map_a.max(), map_b.max()) - 
                              min(map_a.min(), map_b.min()) + 1e-8)
    return float(r), float(s)

# ==========================================
# 4-1. Model Randomization Test
# ==========================================
print("\n" + "=" * 60)
print(" Stage 4-1: Model Randomization Test")
print("=" * 60)

# 학습된 모델 로드
trained_model = timm.create_model(CFG["model_name"], pretrained=False,
                                   num_classes=CFG["num_classes"])
trained_model.load_state_dict(
    torch.load(CFG["model_path"], map_location=device))
trained_model = trained_model.to(device)
trained_model.eval()

# 랜덤 초기화 모델 (pretrained=False, 가중치 초기화)
random_model = timm.create_model(CFG["model_name"], pretrained=False,
                                  num_classes=CFG["num_classes"])
random_model = random_model.to(device)
random_model.eval()

model_rand_records = []

for _, row in tqdm(sample_df.iterrows(), total=len(sample_df),
                   desc="Model Rand Test"):
    try:
        pil_img = Image.open(row["filepath"]).convert("RGB")
        inp     = load_transform(pil_img).unsqueeze(0).to(device)
        cls_name = row["class"]
        stem     = Path(row["filepath"]).stem

        # 학습된 모델의 IG (캐시에서 로드)
        cache_path = Path(CFG["ig_cache_dir"]) / cls_name / (stem + ".npz")
        if cache_path.exists():
            ig_trained = np.load(cache_path)["ig_map"]
            pred_class = int(np.load(cache_path)["pred_class"])
        else:
            with torch.no_grad():
                pred_class = trained_model(
                    normalize_batch(inp)).argmax(1).item()
            ig_trained = compute_ig(trained_model, inp, pred_class,
                                     CFG["ig_steps"], CFG["ig_chunk"])

        # 랜덤 모델의 IG (같은 pred_class 사용)
        ig_random = compute_ig(random_model, inp, pred_class,
                                CFG["ig_steps"], CFG["ig_chunk"])

        r, s = compare_maps(ig_trained, ig_random)
        model_rand_records.append({
            "filename": stem, "class": cls_name,
            "spearman_r": round(r, 4), "ssim": round(s, 4),
        })
        del inp, pil_img, ig_random
        gc.collect(); torch.cuda.empty_cache()

    except Exception as e:
        tqdm.write(f"실패: {row['filepath']} — {e}")

mr_df = pd.DataFrame(model_rand_records)
mr_df.to_csv(save_root / "model_rand_results.csv", index=False)

print(f"\n  Spearman r (학습 vs 랜덤): "
      f"{mr_df.spearman_r.mean():.4f} ± {mr_df.spearman_r.std():.4f}")
print(f"  SSIM       (학습 vs 랜덤): "
      f"{mr_df.ssim.mean():.4f} ± {mr_df.ssim.std():.4f}")

if mr_df.spearman_r.mean() < 0.5:
    print("  ✅ 학습 모델과 랜덤 모델 IG가 유의미하게 다름")
    print("     → IG가 실제 학습된 특징을 반영함")
else:
    print("  ⚠ 학습 모델과 랜덤 모델 IG가 유사함")
    print("     → IG가 입력 이미지 구조만 반영할 가능성")

del random_model; gc.collect(); torch.cuda.empty_cache()

# ==========================================
# 4-2. Data Randomization Test
# ==========================================
print("\n" + "=" * 60)
print(" Stage 4-2: Data Randomization Test")
print(" (라벨 셔플 후 재학습 → IG 비교)")
print("=" * 60)

# 라벨 셔플
class ShuffledDataset(Dataset):
    def __init__(self, df, transform=None, seed=42):
        self.df        = df.reset_index(drop=True)
        self.transform = transform
        rng            = np.random.RandomState(seed)
        self.labels    = rng.permutation(df["label"].values)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        img   = Image.open(row["filepath"]).convert("RGB")
        label = int(self.labels[idx])
        if self.transform:
            img = self.transform(img)
        return img, label

train_transform = transforms.Compose([
    transforms.Resize((CFG["img_size"], CFG["img_size"])),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

shuffled_loader = DataLoader(
    ShuffledDataset(train_df, train_transform, seed=CFG["seed"]),
    batch_size=CFG["shuffle_batch"], shuffle=True,
    num_workers=4, pin_memory=True)

val_loader = DataLoader(
    ShuffledDataset(test_df, val_transform, seed=CFG["seed"]),
    batch_size=CFG["shuffle_batch"], shuffle=False,
    num_workers=4, pin_memory=True)

# 셔플 모델 학습
shuffled_model = timm.create_model(CFG["model_name"], pretrained=True,
                                    num_classes=CFG["num_classes"])
shuffled_model = shuffled_model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(shuffled_model.parameters(),
                               lr=CFG["shuffle_lr"])

print(f"  라벨 셔플 모델 학습 ({CFG['shuffle_epochs']} epochs)...")
best_loss   = float('inf')
patience_cnt = 0

for epoch in range(1, CFG["shuffle_epochs"] + 1):
    shuffled_model.train()
    total_loss = 0
    for imgs, labels in shuffled_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out  = shuffled_model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(shuffled_loader)
    print(f"    Epoch {epoch:2d} | Loss {avg_loss:.4f}")

    if avg_loss < best_loss:
        best_loss = avg_loss
        patience_cnt = 0
        torch.save(shuffled_model.state_dict(),
                   save_root / "shuffled_model.pth")
    else:
        patience_cnt += 1
        if patience_cnt >= CFG["shuffle_patience"]:
            print(f"    Early stopping @ epoch {epoch}")
            break

shuffled_model.load_state_dict(
    torch.load(save_root / "shuffled_model.pth", map_location=device))
shuffled_model.eval()

# 셔플 모델 IG 계산 및 비교
data_rand_records = []

for _, row in tqdm(sample_df.iterrows(), total=len(sample_df),
                   desc="Data Rand Test"):
    try:
        pil_img  = Image.open(row["filepath"]).convert("RGB")
        inp      = load_transform(pil_img).unsqueeze(0).to(device)
        cls_name = row["class"]
        stem     = Path(row["filepath"]).stem

        # 학습된 모델 IG (캐시)
        cache_path = Path(CFG["ig_cache_dir"]) / cls_name / (stem + ".npz")
        if cache_path.exists():
            ig_trained = np.load(cache_path)["ig_map"]
            pred_class = int(np.load(cache_path)["pred_class"])
        else:
            with torch.no_grad():
                pred_class = trained_model(
                    normalize_batch(inp)).argmax(1).item()
            ig_trained = compute_ig(trained_model, inp, pred_class,
                                     CFG["ig_steps"], CFG["ig_chunk"])

        # 셔플 모델 IG
        with torch.no_grad():
            shuf_pred = shuffled_model(
                normalize_batch(inp)).argmax(1).item()
        ig_shuffled = compute_ig(shuffled_model, inp, shuf_pred,
                                  CFG["ig_steps"], CFG["ig_chunk"])

        r, s = compare_maps(ig_trained, ig_shuffled)
        data_rand_records.append({
            "filename": stem, "class": cls_name,
            "spearman_r": round(r, 4), "ssim": round(s, 4),
        })
        del inp, pil_img, ig_shuffled
        gc.collect(); torch.cuda.empty_cache()

    except Exception as e:
        tqdm.write(f"실패: {row['filepath']} — {e}")

dr_df = pd.DataFrame(data_rand_records)
dr_df.to_csv(save_root / "data_rand_results.csv", index=False)

print(f"\n  Spearman r (학습 vs 셔플): "
      f"{dr_df.spearman_r.mean():.4f} ± {dr_df.spearman_r.std():.4f}")
print(f"  SSIM       (학습 vs 셔플): "
      f"{dr_df.ssim.mean():.4f} ± {dr_df.ssim.std():.4f}")

if dr_df.spearman_r.mean() < 0.5:
    print("  ✅ 학습 모델과 셔플 모델 IG가 유의미하게 다름")
    print("     → IG가 데이터 패턴을 반영함")
else:
    print("  ⚠ 학습 모델과 셔플 모델 IG가 유사함")

del shuffled_model, trained_model
gc.collect(); torch.cuda.empty_cache()

# ==========================================
# 시각화
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Model randomization
axes[0].hist(mr_df["spearman_r"], bins=15, color="#0D9488",
             edgecolor='white', alpha=0.85)
axes[0].axvline(mr_df.spearman_r.mean(), color='red', linestyle='--',
                label=f"Mean={mr_df.spearman_r.mean():.4f}")
axes[0].axvline(0.5, color='gray', linestyle=':', alpha=0.7,
                label="0.5 threshold")
axes[0].set_title("4-1 Model Randomization\n(학습 vs 랜덤 모델 IG Spearman r)",
                  fontsize=10)
axes[0].set_xlabel("Spearman r"); axes[0].set_ylabel("Count")
axes[0].legend(); axes[0].grid(alpha=0.3)

# Data randomization
axes[1].hist(dr_df["spearman_r"], bins=15, color="#F59E0B",
             edgecolor='white', alpha=0.85)
axes[1].axvline(dr_df.spearman_r.mean(), color='red', linestyle='--',
                label=f"Mean={dr_df.spearman_r.mean():.4f}")
axes[1].axvline(0.5, color='gray', linestyle=':', alpha=0.7,
                label="0.5 threshold")
axes[1].set_title("4-2 Data Randomization\n(학습 vs 셔플 모델 IG Spearman r)",
                  fontsize=10)
axes[1].set_xlabel("Spearman r"); axes[1].set_ylabel("Count")
axes[1].legend(); axes[1].grid(alpha=0.3)

plt.suptitle("Stage 4: Sanity Checks for IG Attribution Maps",
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(save_root / "sanity_check_results.png",
            dpi=150, bbox_inches='tight')
plt.close()

# ==========================================
# 결과 저장
# ==========================================
summary = {
    "model_rand_spearman_mean": round(float(mr_df.spearman_r.mean()), 4),
    "model_rand_spearman_std":  round(float(mr_df.spearman_r.std()),  4),
    "model_rand_ssim_mean":     round(float(mr_df.ssim.mean()), 4),
    "data_rand_spearman_mean":  round(float(dr_df.spearman_r.mean()), 4),
    "data_rand_spearman_std":   round(float(dr_df.spearman_r.std()),  4),
    "data_rand_ssim_mean":      round(float(dr_df.ssim.mean()), 4),
    "n_sample":                 CFG["n_sample"],
    "passed": bool(mr_df.spearman_r.mean() < 0.5 and
               dr_df.spearman_r.mean() < 0.5),
}
with open(save_root / "sanity_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\n" + "=" * 60)
print(" Stage 4 Sanity Check 최종 결과")
print("=" * 60)
print(f"  4-1 Model Rand: r={mr_df.spearman_r.mean():.4f} "
      f"({'✅ PASS' if mr_df.spearman_r.mean() < 0.5 else '⚠ FAIL'})")
print(f"  4-2 Data  Rand: r={dr_df.spearman_r.mean():.4f} "
      f"({'✅ PASS' if dr_df.spearman_r.mean() < 0.5 else '⚠ FAIL'})")
print(f"\n완료. 저장 경로: {save_root}")
print("생성 파일: sanity_check_results.png, sanity_summary.json")
print("          model_rand_results.csv, data_rand_results.csv")