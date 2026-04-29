# ==========================================
# Stage 7: Domain Interpretation
# stage7_domain.py
#
# - 종별 attention 전략 프로파일 생성
# - 상관분석 + occlusion 결과 통합
# - 클래스별 핵심 특징 시각화
# - 평균 IG 맵 + 대표 필터 맵 나란히 비교
# ==========================================

import warnings
import numpy as np
import cv2
import pywt
import pandas as pd
from pathlib import Path
from PIL import Image
from scipy.ndimage import distance_transform_edt
from skimage.filters import gabor
from torchvision import transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import json

warnings.filterwarnings('ignore')

CFG = {
    "split_dir":      "/data/JSM/convnext/0_stage1",
    "ig_cache_dir":   "/data/JSM/convnext/2_ig_cache/ig_zero",
    "corr_dir":       "/data/JSM/convnext/4_correlation",
    "occ_dir":        "/data/JSM/convnext/5_occlusion",
    "save_dir":       "/data/JSM/convnext/6_domain",
    "img_size":       224,
    "bg_threshold":   10,
    "gabor_freq":     0.3,
    "gabor_theta":    0,
    "fft_low_pass_ratio": 0.3,
    "wavelet":        "db2",
    "wavelet_level":  2,
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

# 종별 도메인 설명 (식물형태학 기반)
SPECIES_INFO = {
    "ARSE_crop": {
        "korean":   "살구나무 (Armeniaca vulgaris)",
        "shape":    "타원형, 편평한 표면, 약한 능선",
        "texture":  "표면 매끄러움, 줄무늬 약함",
        "color":    "황갈색, 채도 낮음",
        "key_feat": "밝기 분포, 전체 형태",
    },
    "ARSS_crop": {
        "korean":   "시베리아살구나무 (Armeniaca sibirica)",
        "shape":    "난형, 표면 거침",
        "texture":  "줄무늬 명확, 능선 발달",
        "color":    "갈색, 채도 중간",
        "key_feat": "수평 텍스처, 밝기 분포",
    },
    "PJNA_crop": {
        "korean":   "욱리인 (Prunus japonica)",
        "shape":    "구형에 가까움, 소형",
        "texture":  "표면 균일, 줄무늬 미약",
        "color":    "황갈색~적갈색, 색도 높음",
        "key_feat": "색도(LAB_Chroma), 밝기 분포",
    },
    "PRDA_crop": {
        "korean":   "산복숭아 (Prunus davidiana)",
        "shape":    "편구형, 능선 선명",
        "texture":  "표면 거침, 능선 패턴",
        "color":    "갈색~적갈색, 색도 높음",
        "key_feat": "색도, 수평 텍스처, 공간 구조",
    },
    "PRPE_crop": {
        "korean":   "복숭아 (Prunus persica)",
        "shape":    "난형, 대형, 깊은 봉합선",
        "texture":  "표면 거침, 뚜렷한 줄무늬",
        "color":    "황갈색, 채도 매우 높음",
        "key_feat": "색도(LAB_Chroma), 채도(Saturation), 수평 텍스처",
    },
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
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask    = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask    = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    return mask

def compute_key_filters(img_np, fg_mask):
    """핵심 필터 맵만 계산 (도메인 해석용)"""
    img_u8  = (img_np * 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR)
    gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(float) / 255.0
    fg      = fg_mask > 0
    gray_fg = gray * fg.astype(float)
    fmaps   = {}

    # FFT_LowPass
    fft_shift = np.fft.fftshift(np.fft.fft2(gray_fg))
    cy, cx    = H//2, W//2
    r_c       = int(min(H,W) * CFG["fft_low_pass_ratio"] / 2)
    ys, xs    = np.ogrid[:H, :W]
    cm        = ((ys-cy)**2 + (xs-cx)**2) <= r_c**2
    lp        = np.zeros((H,W), dtype=complex)
    lp[cm]    = fft_shift[cm]
    fmaps["FFT_LowPass"] = normalize_map(
        np.abs(np.fft.ifft2(np.fft.ifftshift(lp)))) * fg.astype(float)

    # Gabor_f0.3_t0°
    fr, fi = gabor(gray, frequency=CFG["gabor_freq"],
                   theta=np.radians(CFG["gabor_theta"]))
    fmaps["Gabor_f0.3_t0°"] = normalize_map(
        np.sqrt(fr**2 + fi**2)) * fg.astype(float)

    # LAB_Chroma
    img_lab = cv2.cvtColor(img_u8, cv2.COLOR_RGB2LAB).astype(float)
    fmaps["LAB_Chroma"] = normalize_map(
        np.sqrt(img_lab[:,:,1]**2 + img_lab[:,:,2]**2)) * fg.astype(float)

    # DistanceTransform
    fmaps["DistanceTransform"] = normalize_map(
        distance_transform_edt(fg_mask > 0)) * fg.astype(float)

    return fmaps

# ==========================================
# 클래스별 평균 IG + 필터 맵 누적
# ==========================================
print("클래스별 평균 맵 계산 중...")
avg_ig    = {cls: np.zeros((H,W)) for cls in classes}
avg_filt  = {cls: {} for cls in classes}
avg_orig  = {cls: np.zeros((H,W,3)) for cls in classes}
count     = {cls: 0 for cls in classes}

for _, row in test_df.iterrows():
    try:
        cls_name   = row["class"]
        stem       = Path(row["filepath"]).stem
        cache_path = cache_root / cls_name / (stem + ".npz")
        if not cache_path.exists():
            continue

        loaded = np.load(cache_path)
        ig_raw = loaded["ig_map"].astype(float)
        p99    = np.percentile(ig_raw, 99)
        ig_map = np.clip(ig_raw / (p99 + 1e-8), 0, 1)

        pil_img = Image.open(row["filepath"]).convert("RGB")
        img_np  = load_transform(pil_img).permute(1,2,0).numpy()
        img_u8  = (img_np * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR)
        fg_mask = get_fg_mask(img_bgr)

        avg_ig[cls_name]   += ig_map
        avg_orig[cls_name] += img_np
        count[cls_name]    += 1

        fmaps = compute_key_filters(img_np, fg_mask)
        for k, v in fmaps.items():
            if k not in avg_filt[cls_name]:
                avg_filt[cls_name][k] = np.zeros((H,W))
            avg_filt[cls_name][k] += v

    except Exception as e:
        print(f"실패: {e}")

for cls in classes:
    n = count[cls]
    if n == 0: continue
    avg_ig[cls]   /= n
    avg_orig[cls] /= n
    for k in avg_filt[cls]:
        avg_filt[cls][k] /= n

print("완료:", {c.replace("_crop",""): count[c] for c in classes})

# ==========================================
# Figure 1: 클래스별 평균 IG + 핵심 필터 비교
# ==========================================
filter_names = ["FFT_LowPass", "Gabor_f0.3_t0°",
                "LAB_Chroma", "DistanceTransform"]
filter_labels = ["FFT LowPass\n(밝기 구조)", "Gabor f0.3 0°\n(수평 텍스처)",
                 "LAB Chroma\n(색도)", "Distance Transform\n(공간 구조)"]
cmaps_f = ["hot", "hot", "YlOrRd", "Blues"]

n_cols = 2 + len(filter_names)  # 원본 + IG + 필터들
fig, axes = plt.subplots(len(classes), n_cols,
                          figsize=(n_cols * 2.8, len(classes) * 2.8))

col_titles = ["원본 이미지", "IG Attribution"] + filter_labels

for j, title in enumerate(col_titles):
    axes[0, j].set_title(title, fontsize=9, fontweight='bold', pad=4)

for i, (cls, short) in enumerate(zip(classes, short_cls)):
    # 행 라벨
    axes[i, 0].set_ylabel(
        f"{short}\n({SPECIES_INFO[cls]['korean'].split('(')[0].strip()})",
        fontsize=8, fontweight='bold', rotation=90, labelpad=4)

    # 원본
    axes[i, 0].imshow(np.clip(avg_orig[cls], 0, 1))
    axes[i, 0].axis('off')

    # IG 맵
    axes[i, 1].imshow(avg_ig[cls], cmap='jet', vmin=0, vmax=1)
    axes[i, 1].axis('off')

    # 필터 맵
    for j, (fname, cmap) in enumerate(zip(filter_names, cmaps_f)):
        fmap = avg_filt[cls].get(fname, np.zeros((H,W)))
        axes[i, j+2].imshow(fmap, cmap=cmap, vmin=0, vmax=1)
        axes[i, j+2].axis('off')

plt.suptitle(
    "ConvNeXt-Small — 클래스별 평균 IG 맵 vs 핵심 기하학적 특징",
    fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(save_root / "domain_ig_filter_comparison.png",
            dpi=200, bbox_inches='tight')
plt.close()
print("Figure 1 저장: domain_ig_filter_comparison.png")

# ==========================================
# Figure 2: 종별 attention 전략 프로파일
# ==========================================
cw_df  = pd.read_csv(Path(CFG["corr_dir"]) / "classwise_corr.csv")
pivot  = cw_df.pivot(index="feature", columns="class_short",
                     values="spearman_r").astype(float)

# 핵심 특징만
key_feats = [
    "FFT_LowPass", "LAB_L", "LAB_Chroma",
    "Gabor_f0.3_t0°", "Gabor_f0.2_t0°",
    "DistanceTransform", "FourierDescriptor",
    "Wavelet_L2_D", "Saturation_HSV", "LBP",
]
key_feats = [f for f in key_feats if f in pivot.index]
sub = pivot.loc[key_feats, short_cls]

fig, ax = plt.subplots(figsize=(9, 6))
sns.heatmap(sub, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, vmin=-1, vmax=1, ax=ax,
            linewidths=0.5, linecolor='white',
            cbar_kws={"label": "Spearman r", "shrink": 0.8})
ax.set_title(
    "종별 Attention 전략 프로파일\n(클래스 평균 IG vs 기하학적 특징, Spearman r)",
    fontsize=12, fontweight='bold')
ax.set_xlabel("Species", fontsize=11)
ax.set_ylabel("Feature", fontsize=11)
plt.tight_layout()
plt.savefig(save_root / "domain_attention_profile.png",
            dpi=200, bbox_inches='tight')
plt.close()
print("Figure 2 저장: domain_attention_profile.png")

# ==========================================
# Figure 3: 종별 특화 전략 radar chart
# ==========================================
from matplotlib.patches import FancyArrowPatch

categories_radar = ["밝기\n(LAB_L)", "색도\n(LAB_Chroma)",
                    "수평텍스처\n(Gabor0°)", "공간구조\n(DT)",
                    "채도\n(Sat_HSV)"]
feat_map_radar = {
    "밝기\n(LAB_L)":     "LAB_L",
    "색도\n(LAB_Chroma)": "LAB_Chroma",
    "수평텍스처\n(Gabor0°)": "Gabor_f0.3_t0°",
    "공간구조\n(DT)":    "DistanceTransform",
    "채도\n(Sat_HSV)":   "Saturation_HSV",
}

N     = len(categories_radar)
angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
angles += angles[:1]

colors_radar = ["#0D9488", "#F59E0B", "#10B981", "#8B5CF6", "#EF4444"]

fig, ax = plt.subplots(figsize=(8, 8),
                        subplot_kw=dict(polar=True))

for cls, short, color in zip(classes, short_cls, colors_radar):
    vals = []
    for cat in categories_radar:
        feat = feat_map_radar[cat]
        r    = pivot.loc[feat, short] if feat in pivot.index else 0
        vals.append(max(float(r), 0))  # 음수는 0으로
    vals += vals[:1]
    ax.plot(angles, vals, linewidth=2, color=color, label=short)
    ax.fill(angles, vals, alpha=0.1, color=color)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories_radar, fontsize=10)
ax.set_ylim(0, 1)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(["0.2","0.4","0.6","0.8","1.0"], fontsize=7)
ax.grid(color='gray', linestyle='--', alpha=0.4)
ax.set_title("종별 Attention 전략 레이더 차트\n(클래스 평균 Spearman r)",
             fontsize=12, fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=10)
plt.tight_layout()
plt.savefig(save_root / "domain_radar_chart.png",
            dpi=200, bbox_inches='tight')
plt.close()
print("Figure 3 저장: domain_radar_chart.png")

# ==========================================
# Figure 4: Occlusion + 상관분석 통합 요약
# ==========================================
occ_df = pd.read_csv(Path(CFG["occ_dir"]) / "occlusion_summary.csv")
full_acc = occ_df[occ_df["condition"]=="full"]["accuracy"].values[0]

conditions_plot = {
    "ig_mask":     ("IG top-k", "#E74C3C"),
    "random_mask": ("Random",   "#95A5A6"),
    "fft_mask":    ("FFT_LowPass", "#3498DB"),
    "gabor_mask":  ("Gabor_f0.3_t0°", "#F39C12"),
    "dt_mask":     ("DistanceTransform", "#9B59B6"),
}
ratios = [0.1, 0.2, 0.3]

fig, ax = plt.subplots(figsize=(8, 5))
ax.axhline(full_acc, color="#2ECC71", linestyle='--',
           linewidth=2, label=f"Full ({full_acc:.4f})")

for cond, (label, color) in conditions_plot.items():
    accs = []
    for ratio in ratios:
        sub_o = occ_df[(occ_df["condition"]==cond) &
                       (occ_df["mask_ratio"]==ratio)]
        accs.append(sub_o["accuracy"].values[0] if len(sub_o)>0 else np.nan)
    ax.plot(ratios, accs, marker='o', linewidth=2.5,
            color=color, label=label)
    for r, a in zip(ratios, accs):
        if not np.isnan(a):
            ax.annotate(f"{a:.2f}", (r, a),
                        textcoords="offset points",
                        xytext=(0, 8), ha='center', fontsize=8,
                        color=color)

ax.set_xlabel("Masking Ratio", fontsize=11)
ax.set_ylabel("Accuracy", fontsize=11)
ax.set_title("Occlusion Ablation — 마스킹 조건별 정확도 변화",
             fontsize=12, fontweight='bold')
ax.legend(fontsize=9, loc='lower left')
ax.set_xlim(0.05, 0.35)
ax.set_ylim(0.1, 1.05)
ax.grid(alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(save_root / "domain_occlusion_summary.png",
            dpi=200, bbox_inches='tight')
plt.close()
print("Figure 4 저장: domain_occlusion_summary.png")

# ==========================================
# 종별 프로파일 텍스트 출력
# ==========================================
print("\n" + "=" * 65)
print(" 종별 Attention 전략 프로파일")
print("=" * 65)

for cls in classes:
    short = cls.replace("_crop","")
    info  = SPECIES_INFO[cls]
    print(f"\n[{short}] {info['korean']}")
    print(f"  형태:    {info['shape']}")
    print(f"  질감:    {info['texture']}")
    print(f"  색상:    {info['color']}")
    print(f"  핵심특징: {info['key_feat']}")

    # 상관분석 결과
    print("  상관분석 (클래스 평균 r):")
    top_feats = cw_df[cw_df["class_short"]==short].nlargest(5,"spearman_r")
    for _, r in top_feats.iterrows():
        print(f"    {r['feature']:<30s}: r={r['spearman_r']:.3f}")

print(f"\n완료. 저장 경로: {save_root}")
print("생성 파일:")
print("  domain_ig_filter_comparison.png  (클래스별 IG + 필터 비교)")
print("  domain_attention_profile.png     (attention 전략 히트맵)")
print("  domain_radar_chart.png           (레이더 차트)")
print("  domain_occlusion_summary.png     (occlusion 요약)")