# ==========================================
# Stage 5-2: 클래스 평균 맵 상관분석
# stage5_classwise.py
#
# - test.csv 이미지 기준
# - 클래스별 IG 맵 평균 → 클래스별 필터 맵 평균
# - 평균 맵 간 Spearman r 계산
# - 이미지 단위 결과와 비교
# ==========================================

import warnings
import numpy as np
import cv2
import pywt
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from scipy.stats import spearmanr
from scipy.ndimage import distance_transform_edt
from skimage.feature import local_binary_pattern
from skimage.filters import gabor
from torchvision import transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import json

warnings.filterwarnings('ignore')

CFG = {
    "split_dir":      "/data/JSM/convnext/0_stage1",
    "ig_cache_dir":   "/data/JSM/convnext/2_ig_cache/ig_zero",
    "save_dir":       "/data/JSM/convnext/4_correlation",
    "img_size":       224,
    "bg_threshold":   10,
    "gabor_frequencies": [0.1, 0.2, 0.3],
    "gabor_thetas":      [0, 45, 90, 135],
    "wavelet":           "db2",
    "wavelet_level":     2,
    "fft_low_pass_ratio": 0.3,
    "lbp_radius":     3,
    "lbp_n_points":   24,
    "fd_n_components": 20,
}

save_root  = Path(CFG["save_dir"])
cache_root = Path(CFG["ig_cache_dir"])
save_root.mkdir(parents=True, exist_ok=True)

load_transform = transforms.Compose([
    transforms.Resize((CFG["img_size"], CFG["img_size"])),
    transforms.ToTensor(),
])

test_df = pd.read_csv(Path(CFG["split_dir"]) / "test.csv")
with open(Path(CFG["split_dir"]) / "split_config.json") as f:
    split_cfg = json.load(f)
classes   = split_cfg["classes"]
short_cls = [c.replace("_crop", "") for c in classes]
H = W     = CFG["img_size"]

FILTER_CATEGORY = {
    "LBP": "Texture",
    **{f"Gabor_f{fr:.1f}_t{t}°": "Texture"
       for fr in CFG["gabor_frequencies"]
       for t in CFG["gabor_thetas"]},
    "FFT_LowPass": "Frequency", "FFT_HighPass": "Frequency",
    **{f"Wavelet_L{l}_{n}": "Frequency"
       for l in range(1, CFG["wavelet_level"]+1)
       for n in ["H","V","D"]},
    "Edge_Sobel": "Edge_Shape", "Curvature_Laplacian": "Edge_Shape",
    "FourierDescriptor": "Edge_Shape", "DistanceTransform": "Edge_Shape",
    "Brightness": "Color", "Saturation_HSV": "Color",
    "LAB_L": "Color", "LAB_Chroma": "Color",
}

# ==========================================
# 헬퍼 함수 (stage5_correlation.py와 동일)
# ==========================================
def normalize_map(m):
    mn, mx = m.min(), m.max()
    if mx - mn < 1e-8:
        return np.zeros_like(m)
    return (m - mn) / (mx - mn)

def get_fg_mask(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, CFG["bg_threshold"], 255, cv2.THRESH_BINARY)
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask    = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask    = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    return mask

def compute_filter_maps(img_np, fg_mask):
    img_u8  = (img_np * 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR)
    gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(float) / 255.0
    fg      = fg_mask > 0
    fmaps   = {}

    lbp = local_binary_pattern(gray, P=CFG["lbp_n_points"],
                                R=CFG["lbp_radius"], method='uniform').astype(float)
    fmaps["LBP"] = normalize_map(lbp)

    for freq in CFG["gabor_frequencies"]:
        for theta_deg in CFG["gabor_thetas"]:
            fr, fi = gabor(gray, frequency=freq, theta=np.radians(theta_deg))
            fmaps[f"Gabor_f{freq:.1f}_t{theta_deg}°"] = normalize_map(
                np.sqrt(fr**2 + fi**2))

    gray_fg   = gray * fg.astype(float)
    fft_shift = np.fft.fftshift(np.fft.fft2(gray_fg))
    cy, cx    = H // 2, W // 2
    r_circ    = int(min(H, W) * CFG["fft_low_pass_ratio"] / 2)
    ys, xs    = np.ogrid[:H, :W]
    circ_mask = ((ys-cy)**2 + (xs-cx)**2) <= r_circ**2
    lp_mask   = np.zeros((H, W), dtype=complex)
    lp_mask[circ_mask] = fft_shift[circ_mask]
    fmaps["FFT_LowPass"] = normalize_map(
        np.abs(np.fft.ifft2(np.fft.ifftshift(lp_mask))))
    hp_mask = fft_shift.copy(); hp_mask[circ_mask] = 0
    fmaps["FFT_HighPass"] = normalize_map(
        np.abs(np.fft.ifft2(np.fft.ifftshift(hp_mask))))

    coeffs = pywt.wavedec2(gray_fg, wavelet=CFG["wavelet"],
                            level=CFG["wavelet_level"])
    for level_idx in range(1, CFG["wavelet_level"] + 1):
        cH, cV, cD = coeffs[level_idx]
        for name, coeff in zip(["H","V","D"], [cH, cV, cD]):
            energy_up = cv2.resize(coeff**2, (W, H),
                                   interpolation=cv2.INTER_LINEAR)
            fmaps[f"Wavelet_L{level_idx}_{name}"] = normalize_map(energy_up)

    sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    fmaps["Edge_Sobel"]          = normalize_map(np.sqrt(sx**2 + sy**2))
    fmaps["Curvature_Laplacian"] = normalize_map(
        np.abs(cv2.Laplacian(gray, cv2.CV_64F)))

    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_NONE)
    fd_map = np.zeros((H, W), dtype=float)
    if contours:
        cnt      = max(contours, key=cv2.contourArea)
        cx_pts   = cnt[:,0,0].astype(float) + 1j*cnt[:,0,1].astype(float)
        fft_cnt  = np.fft.fft(cx_pts)
        n_keep   = CFG["fd_n_components"]
        fft_filt = np.zeros_like(fft_cnt)
        fft_filt[:n_keep] = fft_cnt[:n_keep]
        fft_filt[-n_keep:] = fft_cnt[-n_keep:]
        cnt_recon = np.fft.ifft(fft_filt)
        rx = np.clip(cnt_recon.real.astype(int), 0, W-1)
        ry = np.clip(cnt_recon.imag.astype(int), 0, H-1)
        pts = np.stack([rx, ry], axis=1).reshape(-1,1,2).astype(np.int32)
        recon_mask = np.zeros((H,W), dtype=np.uint8)
        cv2.drawContours(recon_mask, [pts], -1, 1, thickness=1)
        fd_map = normalize_map(
            1.0 / (distance_transform_edt(
                1 - np.clip(recon_mask,0,1)) + 1.0))
    fmaps["FourierDescriptor"] = fd_map
    fmaps["DistanceTransform"] = normalize_map(
        distance_transform_edt(fg_mask > 0))

    fmaps["Brightness"]     = normalize_map(gray.copy())
    hsv = cv2.cvtColor(img_u8, cv2.COLOR_RGB2HSV)
    fmaps["Saturation_HSV"] = normalize_map(hsv[:,:,1].astype(float))
    img_lab = cv2.cvtColor(img_u8, cv2.COLOR_RGB2LAB).astype(float)
    fmaps["LAB_L"]      = normalize_map(img_lab[:,:,0] / 100.0)
    fmaps["LAB_Chroma"] = normalize_map(
        np.sqrt(img_lab[:,:,1]**2 + img_lab[:,:,2]**2))

    for k in fmaps:
        fmaps[k] = fmaps[k] * fg.astype(float)
    return fmaps

# ==========================================
# 클래스별 평균 맵 누적
# ==========================================
print("클래스별 평균 맵 누적 중...")

avg_ig   = {cls: np.zeros((H, W), dtype=float) for cls in classes}
avg_feat = {cls: {} for cls in classes}
fg_accum = {cls: np.zeros((H, W), dtype=float) for cls in classes}
count    = {cls: 0 for cls in classes}

for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="누적"):
    try:
        cls_name   = row["class"]
        stem       = Path(row["filepath"]).stem
        cache_path = cache_root / cls_name / (stem + ".npz")
        if not cache_path.exists():
            continue

        loaded  = np.load(cache_path)
        ig_raw  = loaded["ig_map"].astype(float)
        p99     = np.percentile(ig_raw, 99)
        ig_map  = np.clip(ig_raw / (p99 + 1e-8), 0, 1)

        pil_img = Image.open(row["filepath"]).convert("RGB")
        img_np  = load_transform(pil_img).permute(1,2,0).numpy()
        img_u8  = (img_np * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR)
        fg_mask = get_fg_mask(img_bgr)

        avg_ig[cls_name]   += ig_map
        fg_accum[cls_name] += (fg_mask > 0).astype(float)
        count[cls_name]    += 1

        fmaps = compute_filter_maps(img_np, fg_mask)
        for feat, fmap in fmaps.items():
            if feat not in avg_feat[cls_name]:
                avg_feat[cls_name][feat] = np.zeros((H, W), dtype=float)
            avg_feat[cls_name][feat] += fmap

    except Exception as e:
        tqdm.write(f"실패: {row['filepath']} — {e}")

# 평균 계산
for cls in classes:
    n = count[cls]
    if n == 0: continue
    avg_ig[cls] /= n
    for feat in avg_feat[cls]:
        avg_feat[cls][feat] /= n

print(f"클래스별 이미지 수: "
      + " | ".join(f"{c.replace('_crop','')}={count[c]}" for c in classes))

# ==========================================
# 클래스 평균 맵 상관분석
# ==========================================
feat_names = sorted(avg_feat[classes[0]].keys())
rows = []

for cls in classes:
    fg_bool = fg_accum[cls] > (count[cls] * 0.3)
    ig_flat = avg_ig[cls][fg_bool].flatten()

    for feat in feat_names:
        feat_flat = avg_feat[cls][feat][fg_bool].flatten()
        if len(ig_flat) < 20:
            continue
        r, _ = spearmanr(ig_flat, feat_flat)
        rows.append({
            "class":       cls,
            "class_short": cls.replace("_crop", ""),
            "feature":     feat,
            "category":    FILTER_CATEGORY.get(feat, "Other"),
            "spearman_r":  round(float(r), 4),
        })

df_cw = pd.DataFrame(rows)
df_cw.to_csv(save_root / "classwise_corr.csv", index=False)

# ==========================================
# 결과 출력
# ==========================================
print("\n" + "=" * 60)
print(" 클래스 평균 맵 상관분석 결과")
print("=" * 60)
pivot = df_cw.pivot(index="feature", columns="class_short",
                    values="spearman_r").astype(float)
print(pivot.round(2).to_string())

# ==========================================
# 이미지 단위 vs 클래스 평균 비교 로드
# ==========================================
img_corr = pd.read_csv(save_root / "corr_mean.csv", index_col=0)
img_corr.columns = [c.replace("_crop","") for c in img_corr.columns]

# ==========================================
# 시각화 A: 클래스 평균 히트맵
# ==========================================
categories = ["Texture", "Frequency", "Edge_Shape", "Color"]

for cat in categories:
    feats = [f for f in feat_names
             if FILTER_CATEGORY.get(f,"") == cat]
    if not feats: continue
    sub = pivot.loc[feats].astype(float)

    fig, ax = plt.subplots(
        figsize=(len(classes)*2.2, len(feats)*0.6 + 1.5))
    sns.heatmap(sub, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, ax=ax,
                linewidths=0.3, cbar_kws={"label": "Spearman r"})
    ax.set_title(
        f"ConvNeXt-Small — [클래스 평균 맵] IG vs {cat}\n(Spearman r)",
        fontsize=11)
    ax.set_xlabel("Class"); ax.set_ylabel("Feature")
    plt.tight_layout()
    plt.savefig(save_root / f"classwise_heatmap_{cat.lower()}.png",
                dpi=200, bbox_inches='tight')
    plt.close()

# 전체 요약
ordered = []
for cat in categories:
    ordered += [f for f in feat_names if FILTER_CATEGORY.get(f,"") == cat]

fig, ax = plt.subplots(figsize=(len(classes)*2.5, len(ordered)*0.5+2))
sns.heatmap(pivot.loc[ordered].astype(float), annot=True, fmt=".2f",
            cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=ax,
            linewidths=0.3, cbar_kws={"label": "Spearman r"})
ax.set_title(
    "ConvNeXt-Small — 클래스 평균 맵 IG vs 전체 필터 (Spearman r)",
    fontsize=11)
ax.set_xlabel("Class"); ax.set_ylabel("Feature")
plt.tight_layout()
plt.savefig(save_root / "classwise_heatmap_summary.png",
            dpi=200, bbox_inches='tight')
plt.close()
print("\n클래스 평균 히트맵 저장 완료")

# ==========================================
# 시각화 B: 이미지 단위 vs 클래스 평균 비교
# ==========================================
# 대표 특징만 선택 (|r| > 0.2인 것)
key_feats = df_cw.groupby("feature")["spearman_r"].mean()
key_feats = key_feats[key_feats.abs() > 0.2].sort_values(ascending=False).index.tolist()

if key_feats:
    fig, axes = plt.subplots(1, 2, figsize=(14, max(4, len(key_feats)*0.4+2)))

    # 클래스 평균
    sub_cw = pivot.loc[key_feats].astype(float)
    sns.heatmap(sub_cw, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, ax=axes[0],
                linewidths=0.3, cbar_kws={"label": "r"})
    axes[0].set_title("클래스 평균 맵\n(노이즈 제거 후)", fontsize=11)
    axes[0].set_xlabel("Class"); axes[0].set_ylabel("Feature")

    # 이미지 단위
    common_feats = [f for f in key_feats if f in img_corr.index]
    if common_feats:
        sub_img = img_corr.loc[common_feats].astype(float)
        sns.heatmap(sub_img, annot=True, fmt=".2f", cmap="RdBu_r",
                    center=0, vmin=-1, vmax=1, ax=axes[1],
                    linewidths=0.3, cbar_kws={"label": "r"})
        axes[1].set_title("이미지 단위\n(개별 이미지 mean)", fontsize=11)
        axes[1].set_xlabel("Class"); axes[1].set_ylabel("")

    plt.suptitle(
        "ConvNeXt-Small — 이미지 단위 vs 클래스 평균 맵 상관계수 비교\n"
        "(|r| > 0.2 특징만)", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_root / "comparison_imgwise_vs_classwise.png",
                dpi=200, bbox_inches='tight')
    plt.close()
    print("비교 그래프 저장 완료")

# ==========================================
# 평균 IG 맵 시각화
# ==========================================
fig, axes = plt.subplots(1, len(classes),
                          figsize=(len(classes)*3.5, 4))
for ax, cls, short in zip(axes, classes, short_cls):
    im = ax.imshow(avg_ig[cls], cmap='jet', vmin=0, vmax=1)
    ax.set_title(f"{short}\n(n={count[cls]})", fontsize=10)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.suptitle("ConvNeXt-Small — 클래스별 평균 IG 맵", fontsize=12)
plt.tight_layout()
plt.savefig(save_root / "avg_ig_maps.png", dpi=200, bbox_inches='tight')
plt.close()
print("평균 IG 맵 저장 완료")

# 평균 IG 맵 저장 (후속 분석용)
np.save(save_root / "avg_ig_maps.npy",
        np.stack([avg_ig[cls] for cls in classes]))

print(f"\n완료. 저장 경로: {save_root}")
print("생성 파일: classwise_corr.csv, classwise_heatmap_*.png")
print("          classwise_heatmap_summary.png")
print("          comparison_imgwise_vs_classwise.png")
print("          avg_ig_maps.png, avg_ig_maps.npy")