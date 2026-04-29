# ==========================================
# Stage 5: 필터 맵 상관분석
# stage5_correlation.py
#
# - test.csv 이미지 (225장) 기준
# - IG 맵(zero baseline) vs 기하학적 필터 맵
# - 이미지 단위 Spearman r → 클래스별 mean ± std
# - t-test + FDR(BH) 보정
# - Permutation test (1000회)
# - 필터 카테고리: Texture / Frequency / Edge·Shape / Color
# ==========================================

import warnings
import numpy as np
import cv2
import pywt
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from scipy.stats import spearmanr, ttest_1samp
from scipy.ndimage import distance_transform_edt
from statsmodels.stats.multitest import multipletests
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
    # Gabor
    "gabor_frequencies": [0.1, 0.2, 0.3],
    "gabor_thetas":      [0, 45, 90, 135],
    # Wavelet
    "wavelet":           "db2",
    "wavelet_level":     2,
    # FFT
    "fft_low_pass_ratio": 0.3,
    # LBP
    "lbp_radius":     3,
    "lbp_n_points":   24,
    # Fourier descriptor
    "fd_n_components": 20,
    # Permutation test
    "n_permutations": 1000,
    "seed":           42,
}

np.random.seed(CFG["seed"])
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
print(f"Test set: {len(test_df)}장 | 클래스: {classes}")

# ==========================================
# 필터 카테고리 정의
# ==========================================
FILTER_CATEGORY = {
    # Texture
    "LBP":                   "Texture",
    "Gabor_f0.1_t0°":        "Texture",
    "Gabor_f0.1_t45°":       "Texture",
    "Gabor_f0.1_t90°":       "Texture",
    "Gabor_f0.1_t135°":      "Texture",
    "Gabor_f0.2_t0°":        "Texture",
    "Gabor_f0.2_t45°":       "Texture",
    "Gabor_f0.2_t90°":       "Texture",
    "Gabor_f0.2_t135°":      "Texture",
    "Gabor_f0.3_t0°":        "Texture",
    "Gabor_f0.3_t45°":       "Texture",
    "Gabor_f0.3_t90°":       "Texture",
    "Gabor_f0.3_t135°":      "Texture",
    # Frequency
    "FFT_LowPass":           "Frequency",
    "FFT_HighPass":          "Frequency",
    "Wavelet_L1_H":          "Frequency",
    "Wavelet_L1_V":          "Frequency",
    "Wavelet_L1_D":          "Frequency",
    "Wavelet_L2_H":          "Frequency",
    "Wavelet_L2_V":          "Frequency",
    "Wavelet_L2_D":          "Frequency",
    # Edge / Shape
    "Edge_Sobel":            "Edge_Shape",
    "Curvature_Laplacian":   "Edge_Shape",
    "FourierDescriptor":     "Edge_Shape",
    "DistanceTransform":     "Edge_Shape",
    # Color
    "Brightness":            "Color",
    "Saturation_HSV":        "Color",
    "LAB_L":                 "Color",
    "LAB_Chroma":            "Color",
}

# ==========================================
# 헬퍼 함수
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
    H, W    = img_np.shape[:2]
    img_u8  = (img_np * 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR)
    gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(float) / 255.0
    fg      = fg_mask > 0
    fmaps   = {}

    # --- Texture ---
    lbp = local_binary_pattern(gray, P=CFG["lbp_n_points"],
                                R=CFG["lbp_radius"], method='uniform').astype(float)
    fmaps["LBP"] = normalize_map(lbp)

    for freq in CFG["gabor_frequencies"]:
        for theta_deg in CFG["gabor_thetas"]:
            fr, fi = gabor(gray, frequency=freq,
                           theta=np.radians(theta_deg))
            fmaps[f"Gabor_f{freq:.1f}_t{theta_deg}°"] = normalize_map(
                np.sqrt(fr**2 + fi**2))

    # --- Frequency ---
    gray_fg   = gray * fg.astype(float)
    fft_shift = np.fft.fftshift(np.fft.fft2(gray_fg))
    cy, cx    = H // 2, W // 2
    r_circ    = int(min(H, W) * CFG["fft_low_pass_ratio"] / 2)
    ys, xs    = np.ogrid[:H, :W]
    circ_mask = ((ys-cy)**2 + (xs-cx)**2) <= r_circ**2

    lp_mask = np.zeros((H, W), dtype=complex)
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
        for name, coeff in zip(["H", "V", "D"], [cH, cV, cD]):
            energy_up = cv2.resize(coeff**2, (W, H),
                                   interpolation=cv2.INTER_LINEAR)
            fmaps[f"Wavelet_L{level_idx}_{name}"] = normalize_map(energy_up)

    # --- Edge / Shape ---
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
        cx_pts   = cnt[:,0,0].astype(float) + 1j * cnt[:,0,1].astype(float)
        fft_cnt  = np.fft.fft(cx_pts)
        n_keep   = CFG["fd_n_components"]
        fft_filt = np.zeros_like(fft_cnt)
        fft_filt[:n_keep] = fft_cnt[:n_keep]
        fft_filt[-n_keep:] = fft_cnt[-n_keep:]
        cnt_recon = np.fft.ifft(fft_filt)
        rx = np.clip(cnt_recon.real.astype(int), 0, W-1)
        ry = np.clip(cnt_recon.imag.astype(int), 0, H-1)
        pts = np.stack([rx, ry], axis=1).reshape(-1,1,2).astype(np.int32)
        recon_mask = np.zeros((H, W), dtype=np.uint8)
        cv2.drawContours(recon_mask, [pts], -1, 1, thickness=1)
        fd_map = normalize_map(
            1.0 / (distance_transform_edt(1 - np.clip(recon_mask,0,1)) + 1.0))
    fmaps["FourierDescriptor"] = fd_map
    fmaps["DistanceTransform"] = normalize_map(
        distance_transform_edt(fg_mask > 0))

    # --- Color ---
    fmaps["Brightness"]    = normalize_map(gray.copy())
    hsv = cv2.cvtColor(img_u8, cv2.COLOR_RGB2HSV)
    fmaps["Saturation_HSV"] = normalize_map(hsv[:,:,1].astype(float))
    img_lab = cv2.cvtColor(img_u8, cv2.COLOR_RGB2LAB).astype(float)
    fmaps["LAB_L"]      = normalize_map(img_lab[:,:,0] / 100.0)
    fmaps["LAB_Chroma"] = normalize_map(
        np.sqrt(img_lab[:,:,1]**2 + img_lab[:,:,2]**2))

    # 전경 마스크 적용
    for k in fmaps:
        fmaps[k] = fmaps[k] * fg.astype(float)

    return fmaps

# ==========================================
# 메인 루프 — 이미지 단위 상관분석
# ==========================================
print("\n이미지 단위 상관분석 시작...")
records = {cls: {} for cls in classes}
missing = 0

for _, row in tqdm(test_df.iterrows(), total=len(test_df),
                   desc="Correlation", unit="img"):
    try:
        cls_name   = row["class"]
        stem       = Path(row["filepath"]).stem
        cache_path = cache_root / cls_name / (stem + ".npz")

        if not cache_path.exists():
            missing += 1
            continue

        loaded  = np.load(cache_path)
        ig_raw  = loaded["ig_map"].astype(float)
        # 99th percentile clipping (시각화 일관성)
        p99     = np.percentile(ig_raw, 99)
        ig_map  = np.clip(ig_raw / (p99 + 1e-8), 0, 1)

        pil_img = Image.open(row["filepath"]).convert("RGB")
        img_np  = load_transform(pil_img).permute(1,2,0).numpy()
        img_u8  = (img_np * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR)
        fg_mask = get_fg_mask(img_bgr)
        fg_bool = fg_mask > 0

        ig_flat = ig_map[fg_bool].flatten()
        if len(ig_flat) < 20:
            continue

        fmaps = compute_filter_maps(img_np, fg_mask)
        for feat, fmap in fmaps.items():
            feat_flat = fmap[fg_bool].flatten()
            if len(feat_flat) < 20:
                continue
            r, _ = spearmanr(ig_flat, feat_flat)
            if np.isnan(r):
                continue
            if feat not in records[cls_name]:
                records[cls_name][feat] = []
            records[cls_name][feat].append(float(r))

    except Exception as e:
        tqdm.write(f"실패: {row['filepath']} — {e}")

if missing:
    print(f"⚠ 캐시 없음: {missing}장")

# ==========================================
# 집계 — mean ± std
# ==========================================
feat_names = sorted({f for cls_d in records.values() for f in cls_d})
corr_mean  = pd.DataFrame(index=feat_names, columns=classes, dtype=float)
corr_std   = pd.DataFrame(index=feat_names, columns=classes, dtype=float)

for cls in classes:
    for feat in feat_names:
        vals = records[cls].get(feat, [])
        corr_mean.loc[feat, cls] = np.mean(vals) if vals else np.nan
        corr_std.loc[feat, cls]  = np.std(vals)  if vals else np.nan

corr_mean.to_csv(save_root / "corr_mean.csv")
corr_std.to_csv(save_root  / "corr_std.csv")

# ==========================================
# 통계 검정 + FDR 보정
# ==========================================
stat_rows = []
for cls in classes:
    for feat in feat_names:
        vals = records[cls].get(feat, [])
        if len(vals) < 3:
            continue
        t_stat, p_val = ttest_1samp(vals, popmean=0)
        stat_rows.append({
            "class":      cls,
            "feature":    feat,
            "category":   FILTER_CATEGORY.get(feat, "Other"),
            "mean_r":     round(np.mean(vals), 4),
            "std_r":      round(np.std(vals),  4),
            "n":          len(vals),
            "t_stat":     round(t_stat, 4),
            "p_value":    float(p_val),
        })

stat_df = pd.DataFrame(stat_rows)
if len(stat_df) > 0:
    reject, p_fdr, _, _ = multipletests(
        stat_df["p_value"].values, alpha=0.05, method="fdr_bh")
    stat_df["p_fdr"]     = p_fdr
    stat_df["sig_fdr"]   = stat_df["p_fdr"].apply(
        lambda p: "***" if p < 0.001 else "**" if p < 0.01
                  else "*" if p < 0.05 else "")
stat_df.to_csv(save_root / "corr_stats.csv", index=False)

# ==========================================
# Permutation Test
# ==========================================
print("\nPermutation test 실행 중...")
perm_rows = []
rng = np.random.RandomState(CFG["seed"])  # 루프 밖에서 한 번만 초기화

for cls in classes:
    for feat in feat_names:
        vals = records[cls].get(feat, [])
        if len(vals) < 3:
            continue
        r_obs  = np.mean(vals)
        r_perm = []
        for _ in range(CFG["n_permutations"]):
            r_perm.append(np.mean(rng.permutation(vals)))
        r_perm = np.array(r_perm)
        p_perm = (r_perm >= r_obs).sum() / CFG["n_permutations"]
        perm_rows.append({
            "class":    cls,
            "feature":  feat,
            "category": FILTER_CATEGORY.get(feat, "Other"),
            "r_obs":    round(r_obs, 4),
            "p_perm":   round(p_perm, 4),
            "sig_perm": "***" if p_perm < 0.001 else "**" if p_perm < 0.01
                        else "*" if p_perm < 0.05 else "",
            "null_mean": round(r_perm.mean(), 4),
            "null_std":  round(r_perm.std(),  4),
        })

perm_df = pd.DataFrame(perm_rows)
perm_df.to_csv(save_root / "permutation_results.csv", index=False)

print(f"  유의한 상관 수 (p_perm < 0.05): "
      f"{(perm_df['p_perm'] < 0.05).sum()} / {len(perm_df)}")

# ==========================================
# 시각화 A: 카테고리별 히트맵
# ==========================================
categories = ["Texture", "Frequency", "Edge_Shape", "Color"]
cat_colors  = {"Texture": "#0D9488", "Frequency": "#F59E0B",
               "Edge_Shape": "#8B5CF6", "Color": "#EF4444"}

for cat in categories:
    feats = [f for f in feat_names
             if FILTER_CATEGORY.get(f, "") == cat]
    if not feats:
        continue
    sub = corr_mean.loc[feats].astype(float)
    sub.columns = short_cls

    fig, ax = plt.subplots(
        figsize=(len(classes) * 2.2, len(feats) * 0.6 + 1.5))
    sns.heatmap(sub, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, vmin=-0.5, vmax=0.5, ax=ax,
                linewidths=0.3, cbar_kws={"label": "Spearman r"})
    ax.set_title(
        f"ConvNeXt-Small — IG vs {cat} 필터 (Spearman r, 이미지 단위)",
        fontsize=11)
    ax.set_xlabel("Class"); ax.set_ylabel("Feature")
    plt.tight_layout()
    plt.savefig(save_root / f"heatmap_{cat.lower()}.png",
                dpi=200, bbox_inches='tight')
    plt.close()

print("히트맵 저장 완료")

# ==========================================
# 시각화 B: 카테고리별 Boxplot
# ==========================================
import matplotlib.patches as mpatches

fig, axes = plt.subplots(1, len(categories),
                          figsize=(len(categories) * 4.5, 5))

for ax, cat in zip(axes, categories):
    feats = [f for f in feat_names
             if FILTER_CATEGORY.get(f, "") == cat]
    if not feats:
        ax.axis('off'); continue

    plot_data, plot_labels = [], []
    for feat in feats:
        for cls in classes:
            vals = records[cls].get(feat, [])
            if vals:
                plot_data.append(vals)
                plot_labels.append(feat.replace("Gabor_", "G_")
                                       .replace("Wavelet_", "W_")
                                       .replace("_crop", ""))

    if not plot_data:
        ax.axis('off'); continue

    bp = ax.boxplot(plot_data, patch_artist=True, vert=True,
                    medianprops=dict(color='black', linewidth=1.5))
    for patch in bp['boxes']:
        patch.set_facecolor(cat_colors[cat])
        patch.set_alpha(0.7)

    ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
    ax.set_title(f"{cat}", fontsize=11, fontweight='bold',
                 color=cat_colors[cat])
    ax.set_ylabel("Spearman r" if cat == categories[0] else "")
    ax.set_xticks(range(1, len(plot_data) + 1))
    ax.set_xticklabels(plot_labels, rotation=45, ha='right', fontsize=7)
    ax.grid(alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.suptitle("ConvNeXt-Small IG vs 필터 카테고리별 상관계수 분포",
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(save_root / "boxplot_by_category.png",
            dpi=200, bbox_inches='tight')
plt.close()
print("Boxplot 저장 완료")

# ==========================================
# 시각화 C: 전체 요약 히트맵 (카테고리 구분)
# ==========================================
# 카테고리 순서대로 정렬
ordered_feats = []
for cat in categories:
    ordered_feats += [f for f in feat_names
                      if FILTER_CATEGORY.get(f, "") == cat]

sub_all = corr_mean.loc[ordered_feats].astype(float)
sub_all.columns = short_cls

fig, ax = plt.subplots(
    figsize=(len(classes) * 2.5, len(ordered_feats) * 0.5 + 2))
sns.heatmap(sub_all, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, vmin=-0.5, vmax=0.5, ax=ax,
            linewidths=0.3, cbar_kws={"label": "Spearman r"})

# 카테고리 구분선
cat_sizes = [len([f for f in feat_names
                  if FILTER_CATEGORY.get(f,"") == cat])
             for cat in categories]
boundary = 0
for i, (cat, size) in enumerate(zip(categories, cat_sizes)):
    ax.axhline(boundary, color='white', linewidth=2)
    ax.text(-0.5, boundary + size/2, cat,
            va='center', ha='right', fontsize=8,
            color=list(cat_colors.values())[i], fontweight='bold')
    boundary += size

ax.set_title(
    "ConvNeXt-Small — IG vs 전체 필터 상관계수 요약\n"
    "(Spearman r, 이미지 단위 mean)", fontsize=11)
ax.set_xlabel("Class"); ax.set_ylabel("Feature")
plt.tight_layout()
plt.savefig(save_root / "heatmap_summary.png",
            dpi=200, bbox_inches='tight')
plt.close()
print("전체 요약 히트맵 저장 완료")

# ==========================================
# 결과 출력
# ==========================================
print("\n" + "=" * 60)
print(" Stage 5 결과 요약")
print("=" * 60)

for cat in categories:
    feats = [f for f in feat_names if FILTER_CATEGORY.get(f,"") == cat]
    if not feats:
        continue
    sub = corr_mean.loc[feats].astype(float)
    print(f"\n[{cat}]")
    for feat in feats:
        vals_all = []
        for cls in classes:
            vals_all += records[cls].get(feat, [])
        if vals_all:
            print(f"  {feat:<30s}: "
                  f"r={np.mean(vals_all):.4f} ± {np.std(vals_all):.4f}")

print(f"\n완료. 저장 경로: {save_root}")
print("생성 파일: corr_mean/std/stats.csv, permutation_results.csv")
print("          heatmap_*.png, boxplot_by_category.png, heatmap_summary.png")