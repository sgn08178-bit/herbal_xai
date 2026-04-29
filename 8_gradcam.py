# ==========================================
# Stage 8: GradCAM vs IG 비교
# stage8_gradcam.py
#
# - GradCAM 맵 생성 (마지막 conv layer)
# - IG 맵과 정성적(시각) + 정량적(Spearman r, SSIM) 비교
# - 클래스별 평균 맵 비교
# - Supplementary material용
# ==========================================

import gc
import json
import warnings
import numpy as np
import cv2
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from scipy.stats import spearmanr
from skimage.metrics import structural_similarity as ssim

import torch
import torch.nn as nn
import timm
from torchvision import transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

CFG = {
    "split_dir":    "/data/JSM/convnext/0_stage1",
    "model_path":   "/data/JSM/convnext/1_train/best_model.pth",
    "ig_cache_dir": "/data/JSM/convnext/2_ig_cache/ig_zero",
    "save_dir":     "/data/JSM/convnext/7_gradcam",
    "model_name":   "convnext_small",
    "num_classes":  5,
    "img_size":     224,
    "seed":         42,
    "n_sample":     20,   # 정량 비교용 샘플 수
}

torch.manual_seed(CFG["seed"])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"device: {device}")

save_root  = Path(CFG["save_dir"])
cache_root = Path(CFG["ig_cache_dir"])
save_root.mkdir(parents=True, exist_ok=True)

# ==========================================
# 모델 로드
# ==========================================
model = timm.create_model(CFG["model_name"], pretrained=False,
                           num_classes=CFG["num_classes"])
model.load_state_dict(torch.load(CFG["model_path"], map_location=device))
model = model.to(device)
model.eval()
print("모델 로드 완료")

_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1,3,1,1)
_std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1,3,1,1)

def normalize_batch(t):
    return (t - _mean) / _std

img_transform = transforms.Compose([
    transforms.Resize((CFG["img_size"], CFG["img_size"])),
    transforms.ToTensor(),
])

# ==========================================
# 데이터
# ==========================================
test_df = pd.read_csv(Path(CFG["split_dir"]) / "test.csv")
with open(Path(CFG["split_dir"]) / "split_config.json") as f:
    split_cfg = json.load(f)
classes   = split_cfg["classes"]
short_cls = [c.replace("_crop","") for c in classes]
H = W     = CFG["img_size"]

# ==========================================
# GradCAM 구현
# ==========================================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model        = model
        self.target_layer = target_layer
        self.gradients    = None
        self.activations  = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, target_class):
        """
        input_tensor: (1, 3, H, W) normalized
        반환: (H, W) float32 GradCAM 맵
        """
        self.model.zero_grad()
        output = self.model(input_tensor)
        output[0, target_class].backward()

        # Global average pooling of gradients
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)  # (1, C, 1, 1)
        cam     = (weights * self.activations).sum(dim=1).squeeze(0)  # (h, w)
        cam     = torch.relu(cam).cpu().numpy()

        # Resize to input size
        cam = cv2.resize(cam, (W, H), interpolation=cv2.INTER_LINEAR)

        # Normalize
        mn, mx = cam.min(), cam.max()
        if mx - mn > 1e-8:
            cam = (cam - mn) / (mx - mn)
        else:
            cam = np.zeros_like(cam)

        # Clear gradients
        self.gradients   = None
        self.activations = None
        torch.cuda.empty_cache()

        return cam.astype(np.float32)

def normalize_map(m):
    mn, mx = m.min(), m.max()
    if mx - mn < 1e-8:
        return np.zeros_like(m)
    return (m - mn) / (mx - mn)

# ConvNeXt-Small의 마지막 stage layer 찾기
# timm ConvNeXt 구조: model.stages[-1].blocks[-1].conv_dw
target_layer = model.head.norm
gradcam      = GradCAM(model, target_layer)
print(f"GradCAM target layer: stages[-1].blocks[-1].conv_dw")

# ==========================================
# 전체 test set GradCAM 계산 + IG 비교
# ==========================================
print("\nGradCAM 계산 + IG 비교 중...")
records = []

for _, row in tqdm(test_df.iterrows(), total=len(test_df),
                   desc="GradCAM", unit="img"):
    try:
        cls_name   = row["class"]
        stem       = Path(row["filepath"]).stem
        cache_path = cache_root / cls_name / (stem + ".npz")
        if not cache_path.exists():
            continue

        # IG 로드
        loaded  = np.load(cache_path)
        ig_raw  = loaded["ig_map"].astype(float)
        p99     = np.percentile(ig_raw, 99)
        ig_map  = np.clip(ig_raw / (p99 + 1e-8), 0, 1).astype(np.float32)
        pred_class = int(loaded["pred_class"])
        true_class = int(loaded["true_class"])
        is_correct = bool(loaded["is_correct"])

        # GradCAM 계산
        pil_img = Image.open(row["filepath"]).convert("RGB")
        inp     = img_transform(pil_img).unsqueeze(0).to(device)
        inp_norm = normalize_batch(inp)
        gc_map  = gradcam.generate(inp_norm, pred_class)

        # 정량 비교
        r, _   = spearmanr(ig_map.flatten(), gc_map.flatten())
        s      = ssim(ig_map, gc_map,
                      data_range=max(ig_map.max(), gc_map.max()) -
                                 min(ig_map.min(), gc_map.min()) + 1e-8)

        records.append({
            "filename":   stem,
            "class":      cls_name,
            "class_short": cls_name.replace("_crop",""),
            "pred_class": pred_class,
            "true_class": true_class,
            "is_correct": is_correct,
            "spearman_r": round(float(r), 4),
            "ssim":       round(float(s), 4),
        })

        del inp, inp_norm, gc_map
        gc.collect(); torch.cuda.empty_cache()

    except Exception as e:
        tqdm.write(f"실패: {row['filepath']} — {e}")

df = pd.DataFrame(records)
df.to_csv(save_root / "gradcam_ig_comparison.csv", index=False)

# ==========================================
# 결과 출력
# ==========================================
print("\n" + "=" * 60)
print(" GradCAM vs IG 정량 비교 결과")
print("=" * 60)
print(f"  전체 평균 Spearman r: {df.spearman_r.mean():.4f} ± {df.spearman_r.std():.4f}")
print(f"  전체 평균 SSIM:       {df.ssim.mean():.4f} ± {df.ssim.std():.4f}")

print("\n  [클래스별]")
for short in short_cls:
    sub = df[df["class_short"] == short]
    print(f"  {short}: r={sub.spearman_r.mean():.4f}  SSIM={sub.ssim.mean():.4f}")

# ==========================================
# 시각화 A: 클래스별 평균 IG vs GradCAM
# ==========================================
print("\n클래스별 평균 맵 계산 중...")
avg_ig  = {cls: np.zeros((H,W)) for cls in classes}
avg_gc  = {cls: np.zeros((H,W)) for cls in classes}
avg_ori = {cls: np.zeros((H,W,3)) for cls in classes}
cnt     = {cls: 0 for cls in classes}

for _, row in tqdm(test_df.iterrows(), total=len(test_df),
                   desc="평균 맵", unit="img"):
    try:
        cls_name   = row["class"]
        stem       = Path(row["filepath"]).stem
        cache_path = cache_root / cls_name / (stem + ".npz")
        if not cache_path.exists():
            continue

        loaded     = np.load(cache_path)
        ig_raw     = loaded["ig_map"].astype(float)
        p99        = np.percentile(ig_raw, 99)
        ig_map     = np.clip(ig_raw / (p99 + 1e-8), 0, 1).astype(np.float32)
        pred_class = int(loaded["pred_class"])

        pil_img = Image.open(row["filepath"]).convert("RGB")
        img_np  = img_transform(pil_img).permute(1,2,0).numpy()
        inp     = torch.tensor(img_np).permute(2,0,1).unsqueeze(0).to(device)
        inp_norm = normalize_batch(inp)
        gc_map  = gradcam.generate(inp_norm, pred_class)

        avg_ig[cls_name]  += ig_map
        avg_gc[cls_name]  += gc_map
        avg_ori[cls_name] += img_np
        cnt[cls_name]     += 1

        del inp, inp_norm, gc_map
        gc.collect(); torch.cuda.empty_cache()

    except Exception as e:
        tqdm.write(f"실패: {e}")

for cls in classes:
    n = cnt[cls]
    if n == 0: continue
    avg_ig[cls]  /= n
    avg_gc[cls]  /= n
    avg_ori[cls] /= n

# Figure: 원본 / IG / GradCAM 나란히
fig, axes = plt.subplots(len(classes), 3,
                          figsize=(9, len(classes) * 2.8))
col_titles = ["원본 이미지", "IG Attribution\n(Integrated Gradients)",
              "GradCAM\n(Gradient-weighted CAM)"]

for j, title in enumerate(col_titles):
    axes[0, j].set_title(title, fontsize=10, fontweight='bold', pad=4)

for i, (cls, short) in enumerate(zip(classes, short_cls)):
    axes[i, 0].set_ylabel(short, fontsize=10, fontweight='bold',
                           rotation=90, labelpad=4)
    axes[i, 0].imshow(np.clip(avg_ori[cls], 0, 1))
    axes[i, 0].axis('off')
    axes[i, 1].imshow(avg_ig[cls], cmap='jet', vmin=0, vmax=1)
    axes[i, 1].axis('off')
    axes[i, 2].imshow(avg_gc[cls], cmap='jet', vmin=0, vmax=1)
    axes[i, 2].axis('off')

plt.suptitle(
    "ConvNeXt-Small — IG vs GradCAM 클래스별 평균 Attribution 맵",
    fontsize=12, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(save_root / "gradcam_vs_ig_avgmaps.png",
            dpi=200, bbox_inches='tight')
plt.close()
print("클래스별 평균 맵 저장: gradcam_vs_ig_avgmaps.png")

# ==========================================
# 시각화 B: 클래스별 Spearman r 분포
# ==========================================
fig, ax = plt.subplots(figsize=(8, 4))
colors_cls = ["#0D9488","#F59E0B","#10B981","#8B5CF6","#EF4444"]
for short, color in zip(short_cls, colors_cls):
    sub = df[df["class_short"] == short]["spearman_r"]
    ax.hist(sub, bins=15, alpha=0.6, color=color,
            label=f"{short} (μ={sub.mean():.3f})", edgecolor='white')

ax.axvline(df.spearman_r.mean(), color='black', linestyle='--',
           linewidth=2, label=f"전체 평균 r={df.spearman_r.mean():.3f}")
ax.set_xlabel("Spearman r (IG vs GradCAM)")
ax.set_ylabel("Count")
ax.set_title("IG vs GradCAM 공간적 상관계수 분포 (이미지 단위)", fontsize=11)
ax.legend(fontsize=8); ax.grid(alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(save_root / "gradcam_ig_spearman_dist.png",
            dpi=200, bbox_inches='tight')
plt.close()

# ==========================================
# 시각화 C: 대표 이미지 샘플 비교 (클래스별 2장)
# ==========================================
sample_rows = []
for cls in classes:
    sub = test_df[test_df["class"] == cls]
    sub = sub[sub["filepath"].apply(
        lambda p: (cache_root / cls / (Path(p).stem+".npz")).exists())]
    sample_rows.append(sub.iloc[:2])
sample_df = pd.concat(sample_rows).reset_index(drop=True)

n_samples = len(sample_df)
fig, axes = plt.subplots(n_samples, 3,
                          figsize=(8, n_samples * 2.5))
if n_samples == 1:
    axes = axes[np.newaxis, :]

axes[0, 0].set_title("원본", fontsize=10, fontweight='bold')
axes[0, 1].set_title("IG Attribution", fontsize=10, fontweight='bold')
axes[0, 2].set_title("GradCAM", fontsize=10, fontweight='bold')

for idx, (_, row) in enumerate(sample_df.iterrows()):
    cls_name   = row["class"]
    stem       = Path(row["filepath"]).stem
    cache_path = cache_root / cls_name / (stem + ".npz")

    loaded     = np.load(cache_path)
    ig_raw     = loaded["ig_map"].astype(float)
    p99        = np.percentile(ig_raw, 99)
    ig_map     = np.clip(ig_raw / (p99 + 1e-8), 0, 1)
    pred_class = int(loaded["pred_class"])

    pil_img = Image.open(row["filepath"]).convert("RGB")
    img_np  = img_transform(pil_img).permute(1,2,0).numpy()
    inp     = torch.tensor(img_np).permute(2,0,1).unsqueeze(0).to(device)
    gc_map  = gradcam.generate(normalize_batch(inp), pred_class)

    # r, ssim
    r, _ = spearmanr(ig_map.flatten(), gc_map.flatten())

    axes[idx, 0].imshow(np.clip(img_np, 0, 1))
    axes[idx, 0].set_ylabel(
        cls_name.replace("_crop",""), fontsize=8, rotation=90)
    axes[idx, 0].axis('off')

    axes[idx, 1].imshow(ig_map, cmap='jet', vmin=0, vmax=1)
    axes[idx, 1].axis('off')

    axes[idx, 2].imshow(gc_map, cmap='jet', vmin=0, vmax=1)
    axes[idx, 2].set_title(f"r={r:.3f}", fontsize=8)
    axes[idx, 2].axis('off')

    del inp, gc_map
    gc.collect(); torch.cuda.empty_cache()

plt.suptitle("IG vs GradCAM 이미지별 비교 (클래스별 2장)",
             fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(save_root / "gradcam_ig_samples.png",
            dpi=200, bbox_inches='tight')
plt.close()
print("샘플 비교 저장: gradcam_ig_samples.png")

# ==========================================
# 결과 JSON 저장
# ==========================================
summary = {
    "spearman_r_mean":  round(float(df.spearman_r.mean()), 4),
    "spearman_r_std":   round(float(df.spearman_r.std()),  4),
    "ssim_mean":        round(float(df.ssim.mean()), 4),
    "ssim_std":         round(float(df.ssim.std()),  4),
    "n_images":         len(df),
    "by_class": {
        short: {
            "spearman_r": round(float(
                df[df["class_short"]==short].spearman_r.mean()), 4),
            "ssim": round(float(
                df[df["class_short"]==short].ssim.mean()), 4),
        }
        for short in short_cls
    }
}
with open(save_root / "gradcam_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n완료. 저장 경로: {save_root}")
print("생성 파일:")
print("  gradcam_vs_ig_avgmaps.png    (클래스별 평균 맵 비교)")
print("  gradcam_ig_spearman_dist.png (상관계수 분포)")
print("  gradcam_ig_samples.png       (이미지별 샘플 비교)")
print("  gradcam_ig_comparison.csv    (전체 수치)")
print("  gradcam_summary.json         (요약)")