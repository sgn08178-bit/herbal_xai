# ==========================================
# Stage 6: Occlusion Ablation
# stage6_occlusion.py
#
# IG attribution 상위 영역을 마스킹 → 정확도 변화 측정
# → IG가 실제로 분류에 중요한 영역을 잡는지 인과적 증거
#
# 실험 구성:
#   A. IG 기반 마스킹 (top 10%, 20%, 30%)
#   B. 랜덤 마스킹 (동일 비율) — 대조군
#   C. 핵심 필터 기반 마스킹 (FFT_LowPass, Gabor_f0.3_t0° top 30%)
#
# 판정:
#   IG 마스킹 > 랜덤 마스킹 정확도 하락 → IG가 중요 영역 포착
# ==========================================

import gc
import json
import warnings
import numpy as np
import pandas as pd
import cv2
import pywt
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from scipy.ndimage import gaussian_filter, distance_transform_edt
from skimage.filters import gabor

import torch
import timm
from torchvision import transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

CFG = {
    "split_dir":    "/data/JSM/convnext/0_stage1",
    "model_path":   "/data/JSM/convnext/1_train/best_model.pth",
    "ig_cache_dir": "/data/JSM/convnext/2_ig_cache/ig_zero",
    "save_dir":     "/data/JSM/convnext/5_occlusion",
    "model_name":   "convnext_small",
    "num_classes":  5,
    "img_size":     224,
    "mask_ratios":  [0.1, 0.2, 0.3],   # 마스킹 비율
    "mask_value":   0.0,                # 마스킹 값 (0 = black)
    "n_random_trials": 10,              # 랜덤 마스킹 반복 횟수
    "seed":         42,
    # 필터 마스킹용
    "gabor_freq":   0.3,
    "gabor_theta":  0,
    "fft_low_pass_ratio": 0.3,
}

np.random.seed(CFG["seed"])
torch.manual_seed(CFG["seed"])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"device: {device}")

save_root = Path(CFG["save_dir"])
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
# 데이터 로드
# ==========================================
test_df = pd.read_csv(Path(CFG["split_dir"]) / "test.csv")
with open(Path(CFG["split_dir"]) / "split_config.json") as f:
    split_cfg = json.load(f)
classes   = split_cfg["classes"]
short_cls = [c.replace("_crop","") for c in classes]
print(f"Test set: {len(test_df)}장")

# ==========================================
# 헬퍼 함수
# ==========================================
def predict(img_tensor):
    """(1,3,H,W) → pred_class, confidence"""
    with torch.no_grad():
        logits = model(normalize_batch(img_tensor.to(device)))
        probs  = torch.softmax(logits, dim=1)
        pred   = logits.argmax(dim=1).item()
        conf   = probs[0, pred].item()
    return pred, conf

def make_ig_mask(ig_map, ratio, fg_mask=None):
    """IG 상위 ratio 영역 마스크 (1=마스킹할 곳)"""
    if fg_mask is not None:
        ig_fg = ig_map * (fg_mask > 0).astype(float)
    else:
        ig_fg = ig_map
    threshold = np.percentile(ig_fg[ig_fg > 0], (1 - ratio) * 100) \
                if (ig_fg > 0).sum() > 0 else 0
    return (ig_fg >= threshold).astype(np.float32)

def make_random_mask(ig_map, ratio, fg_mask=None, seed=None):
    """랜덤 위치 동일 비율 마스크"""
    rng = np.random.RandomState(seed)
    if fg_mask is not None:
        fg_idx = np.where(fg_mask.flatten() > 0)[0]
        n_mask = int(len(fg_idx) * ratio)
        chosen = rng.choice(fg_idx, size=n_mask, replace=False)
    else:
        n_total = ig_map.size
        n_mask  = int(n_total * ratio)
        chosen  = rng.choice(n_total, size=n_mask, replace=False)
    mask = np.zeros(ig_map.size, dtype=np.float32)
    mask[chosen] = 1.0
    return mask.reshape(ig_map.shape)

def apply_mask(img_tensor, mask_2d, value=0.0):
    """(1,3,H,W) 텐서에 (H,W) 마스크 적용"""
    img = img_tensor.clone()
    m   = torch.tensor(mask_2d, dtype=torch.float32).unsqueeze(0)  # (1,H,W)
    img[0] = img[0] * (1 - m) + value * m
    return img

def get_fg_mask(img_u8):
    img_bgr = cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR)
    gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask    = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask    = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    return mask

def get_filter_mask(img_u8, filter_name, ratio):
    """특정 필터 응답 상위 영역 마스크"""
    img_bgr = cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR)
    gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(float) / 255.0

    if filter_name == "FFT_LowPass":
        H, W  = gray.shape
        fft_s = np.fft.fftshift(np.fft.fft2(gray))
        cy, cx = H//2, W//2
        r_c   = int(min(H,W) * CFG["fft_low_pass_ratio"] / 2)
        ys, xs = np.ogrid[:H, :W]
        cm    = ((ys-cy)**2 + (xs-cx)**2) <= r_c**2
        lp    = np.zeros((H,W), dtype=complex)
        lp[cm] = fft_s[cm]
        fmap  = np.abs(np.fft.ifft2(np.fft.ifftshift(lp)))

    elif filter_name == "Gabor":
        fr, fi = gabor(gray, frequency=CFG["gabor_freq"],
                       theta=np.radians(CFG["gabor_theta"]))
        fmap   = np.sqrt(fr**2 + fi**2)

    elif filter_name == "DistanceTransform":
        fg_mask = get_fg_mask(img_u8)
        fmap    = distance_transform_edt(fg_mask > 0).astype(float)

    mn, mx = fmap.min(), fmap.max()
    if mx - mn > 1e-8:
        fmap = (fmap - mn) / (mx - mn)

    threshold = np.percentile(fmap[fmap > 0], (1-ratio)*100) \
                if (fmap > 0).sum() > 0 else 0
    return (fmap >= threshold).astype(np.float32)

# ==========================================
# 메인 실험
# ==========================================
records = []
cache_root = Path(CFG["ig_cache_dir"])

for _, row in tqdm(test_df.iterrows(), total=len(test_df),
                   desc="Occlusion", unit="img"):
    try:
        cls_name   = row["class"]
        true_label = int(row["label"])
        stem       = Path(row["filepath"]).stem
        cache_path = cache_root / cls_name / (stem + ".npz")

        if not cache_path.exists():
            continue

        # 이미지 로드
        pil_img = Image.open(row["filepath"]).convert("RGB")
        inp     = img_transform(pil_img).unsqueeze(0)
        img_u8  = (inp.squeeze(0).permute(1,2,0).numpy() * 255).astype(np.uint8)
        fg_mask = get_fg_mask(img_u8)

        # IG 맵 로드
        loaded  = np.load(cache_path)
        ig_raw  = loaded["ig_map"].astype(float)
        p99     = np.percentile(ig_raw, 99)
        ig_map  = np.clip(ig_raw / (p99 + 1e-8), 0, 1)

        # 기준 성능 (마스킹 없음)
        pred_full, conf_full = predict(inp)
        correct_full = int(pred_full == true_label)

        base_rec = {
            "filename": stem, "class": cls_name,
            "true_label": true_label,
            "condition": "full",
            "mask_ratio": 0.0,
            "correct": correct_full,
            "confidence": round(conf_full, 4),
        }
        records.append(base_rec)

        for ratio in CFG["mask_ratios"]:

            # A. IG 기반 마스킹
            ig_mask = make_ig_mask(ig_map, ratio, fg_mask)
            inp_ig  = apply_mask(inp, ig_mask, CFG["mask_value"])
            pred_ig, conf_ig = predict(inp_ig)
            records.append({
                "filename": stem, "class": cls_name,
                "true_label": true_label,
                "condition": "ig_mask",
                "mask_ratio": ratio,
                "correct": int(pred_ig == true_label),
                "confidence": round(conf_ig, 4),
            })

            # B. 랜덤 마스킹 (n_random_trials 회 평균)
            rand_corrects, rand_confs = [], []
            for trial in range(CFG["n_random_trials"]):
                rand_mask = make_random_mask(
                    ig_map, ratio, fg_mask, seed=CFG["seed"]+trial)
                inp_rand  = apply_mask(inp, rand_mask, CFG["mask_value"])
                pred_r, conf_r = predict(inp_rand)
                rand_corrects.append(int(pred_r == true_label))
                rand_confs.append(conf_r)
            records.append({
                "filename": stem, "class": cls_name,
                "true_label": true_label,
                "condition": "random_mask",
                "mask_ratio": ratio,
                "correct": round(np.mean(rand_corrects), 4),
                "confidence": round(np.mean(rand_confs), 4),
            })

            # C-1. FFT_LowPass 기반 마스킹
            fft_mask = get_filter_mask(img_u8, "FFT_LowPass", ratio)
            inp_fft  = apply_mask(inp, fft_mask, CFG["mask_value"])
            pred_fft, conf_fft = predict(inp_fft)
            records.append({
                "filename": stem, "class": cls_name,
                "true_label": true_label,
                "condition": "fft_mask",
                "mask_ratio": ratio,
                "correct": int(pred_fft == true_label),
                "confidence": round(conf_fft, 4),
            })

            # C-2. Gabor_f0.3_t0° 기반 마스킹
            gabor_mask = get_filter_mask(img_u8, "Gabor", ratio)
            inp_gabor  = apply_mask(inp, gabor_mask, CFG["mask_value"])
            pred_gabor, conf_gabor = predict(inp_gabor)
            records.append({
                "filename": stem, "class": cls_name,
                "true_label": true_label,
                "condition": "gabor_mask",
                "mask_ratio": ratio,
                "correct": int(pred_gabor == true_label),
                "confidence": round(conf_gabor, 4),
            })

            # C-3. DistanceTransform 기반 마스킹
            dt_mask = get_filter_mask(img_u8, "DistanceTransform", ratio)
            inp_dt  = apply_mask(inp, dt_mask, CFG["mask_value"])
            pred_dt, conf_dt = predict(inp_dt)
            records.append({
                "filename": stem, "class": cls_name,
                "true_label": true_label,
                "condition": "dt_mask",
                "mask_ratio": ratio,
                "correct": int(pred_dt == true_label),
                "confidence": round(conf_dt, 4),
            })

    except Exception as e:
        tqdm.write(f"실패: {row['filepath']} — {e}")
        continue

df = pd.DataFrame(records)
df.to_csv(save_root / "occlusion_results.csv", index=False)

# ==========================================
# 집계
# ==========================================
conditions = ["full", "ig_mask", "random_mask",
              "fft_mask", "gabor_mask", "dt_mask"]
cond_labels = {
    "full":        "Full image",
    "ig_mask":     "IG top-k mask",
    "random_mask": "Random mask",
    "fft_mask":    "FFT_LowPass mask",
    "gabor_mask":  "Gabor_f0.3_t0° mask",
    "dt_mask":     "DistanceTransform mask",
}
colors = {
    "full":        "#2ECC71",
    "ig_mask":     "#E74C3C",
    "random_mask": "#95A5A6",
    "fft_mask":    "#3498DB",
    "gabor_mask":  "#F39C12",
    "dt_mask":     "#9B59B6",
}

print("\n" + "=" * 65)
print(" Occlusion Ablation 결과")
print("=" * 65)

summary_rows = []
for ratio in [0.0] + CFG["mask_ratios"]:
    for cond in conditions:
        if cond == "full" and ratio > 0:
            continue
        sub = df[(df["condition"] == cond) &
                 (df["mask_ratio"] == ratio)]
        if len(sub) == 0:
            continue
        acc  = sub["correct"].mean()
        conf = sub["confidence"].mean()
        summary_rows.append({
            "condition":  cond,
            "mask_ratio": ratio,
            "accuracy":   round(acc, 4),
            "confidence": round(conf, 4),
            "n":          len(sub),
        })
        label = cond_labels.get(cond, cond)
        print(f"  {label:<30s} ratio={ratio:.1f} | "
              f"Acc={acc:.4f}  Conf={conf:.4f}")

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(save_root / "occlusion_summary.csv", index=False)

# ==========================================
# 시각화 A: 마스킹 비율별 정확도 변화
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

full_acc = summary_df[summary_df["condition"] == "full"]["accuracy"].values[0]

for ax, metric in zip(axes, ["accuracy", "confidence"]):
    for cond in ["ig_mask", "random_mask",
                 "fft_mask", "gabor_mask", "dt_mask"]:
        sub = summary_df[summary_df["condition"] == cond]
        if len(sub) == 0:
            continue
        ax.plot(sub["mask_ratio"], sub[metric],
                marker='o', linewidth=2,
                color=colors[cond],
                label=cond_labels[cond])

    ax.axhline(full_acc if metric == "accuracy" else
               summary_df[summary_df["condition"]=="full"]["confidence"].values[0],
               color=colors["full"], linestyle='--', linewidth=1.5,
               label="Full image (baseline)")
    ax.set_xlabel("Masking ratio")
    ax.set_ylabel(metric.capitalize())
    ax.set_title(f"{metric.capitalize()} vs Masking ratio", fontsize=11)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_xlim(-0.01, 0.35)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.suptitle(
    "ConvNeXt-Small — Occlusion Ablation\n"
    "IG 기반 마스킹 vs 랜덤/필터 마스킹", fontsize=12)
plt.tight_layout()
plt.savefig(save_root / "occlusion_curve.png", dpi=200, bbox_inches='tight')
plt.close()

# ==========================================
# 시각화 B: 결과 요약 바차트
# ==========================================
ratios = CFG["mask_ratios"]
x      = np.arange(len(ratios))
width  = 0.15
fig, ax = plt.subplots(figsize=(10, 5))

for i, cond in enumerate(["ig_mask", "random_mask",
                           "fft_mask", "gabor_mask", "dt_mask"]):
    accs = []
    for ratio in ratios:
        sub = summary_df[(summary_df["condition"] == cond) &
                         (summary_df["mask_ratio"] == ratio)]
        accs.append(sub["accuracy"].values[0] if len(sub) > 0 else 0)
    ax.bar(x + i*width, accs, width,
           label=cond_labels[cond], color=colors[cond],
           edgecolor='white', alpha=0.85)

ax.axhline(full_acc, color=colors["full"], linestyle='--',
           linewidth=1.5, label=f"Full image ({full_acc:.4f})")
ax.set_xticks(x + width*2)
ax.set_xticklabels([f"Ratio={r}" for r in ratios])
ax.set_ylabel("Accuracy")
ax.set_ylim(0, 1.05)
ax.set_title("ConvNeXt-Small Occlusion Ablation — 마스킹 조건별 정확도",
             fontsize=11)
ax.legend(fontsize=8, loc='lower left')
ax.grid(alpha=0.3, axis='y')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(save_root / "occlusion_barchart.png", dpi=200, bbox_inches='tight')
plt.close()

# ==========================================
# 결과 JSON 저장
# ==========================================
result_json = {
    "full_accuracy": float(full_acc),
    "results": []
}
for _, r in summary_df[summary_df["condition"] != "full"].iterrows():
    result_json["results"].append({
        "condition":  r["condition"],
        "mask_ratio": float(r["mask_ratio"]),
        "accuracy":   float(r["accuracy"]),
        "delta_acc":  round(float(r["accuracy"]) - float(full_acc), 4),
    })
with open(save_root / "occlusion_summary.json", "w") as f:
    json.dump(result_json, f, indent=2)

print(f"\n완료. 저장 경로: {save_root}")
print("생성 파일: occlusion_results.csv, occlusion_summary.csv")
print("          occlusion_curve.png, occlusion_barchart.png")