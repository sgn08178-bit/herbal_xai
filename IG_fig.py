# ==========================================
# Figure: 클래스별 IG 히트맵 + 필터 시각화
# figure_samples.py
#
# 각 클래스에서 이미지 1장씩 선택
# 원본 / IG / 필터맵 4종 나란히 시각화
# ==========================================

import warnings
import numpy as np
import cv2
import pywt
import pandas as pd
from pathlib import Path
from PIL import Image
from scipy.ndimage import distance_transform_edt, gaussian_filter
from skimage.feature import local_binary_pattern
from skimage.filters import gabor
from torchvision import transforms
import torch
import timm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import json

warnings.filterwarnings('ignore')

CFG = {
    "data_dir":    "/data/JSM/06_2",
    "model_path":  "/data/JSM/convnext/1_train/best_model.pth",
    "split_dir":   "/data/JSM/convnext/0_stage1",
    "ig_cache_dir":"/data/JSM/convnext/2_ig_cache/ig_zero",
    "save_dir":    "/data/JSM/convnext/figures",
    "model_name":  "convnext_small",
    "num_classes": 5,
    "img_size":    224,
    "ig_steps":    100,
    "ig_chunk":    10,
    # 필터 설정
    "gabor_freq":        0.3,
    "gabor_theta":       0,
    "fft_low_pass_ratio": 0.3,
    "wavelet":           "db2",
    "lbp_radius":        3,
    "lbp_n_points":      24,
    "fd_n_components":   20,
    "blur_sigma":        20,
    "seed":              42,
}

np.random.seed(CFG["seed"])
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

_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1,3,1,1)
_std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1,3,1,1)

def normalize_batch(t):
    return (t - _mean) / _std

img_transform = transforms.Compose([
    transforms.Resize((CFG["img_size"], CFG["img_size"])),
    transforms.ToTensor(),
])

# ==========================================
# 클래스 정보
# ==========================================
with open(Path(CFG["split_dir"]) / "split_config.json") as f:
    split_cfg = json.load(f)
classes   = split_cfg["classes"]
short_cls = [c.replace("_crop", "") for c in classes]
test_df   = pd.read_csv(Path(CFG["split_dir"]) / "test.csv")

# ==========================================
# IG 계산 함수
# ==========================================
def compute_ig(inp, pred_class):
    baseline  = torch.zeros_like(inp)
    diff      = inp - baseline
    alphas    = torch.linspace(0, 1, CFG["ig_steps"] + 1, device=device)
    grads_sum = torch.zeros_like(inp.squeeze(0))

    for start in range(0, CFG["ig_steps"] + 1, CFG["ig_chunk"]):
        end   = min(start + CFG["ig_chunk"], CFG["ig_steps"] + 1)
        a_sub = alphas[start:end].view(-1,1,1,1)
        scaled  = (baseline + a_sub * diff).detach().requires_grad_(True)
        outputs = model(normalize_batch(scaled))
        outputs[:, pred_class].sum().backward()
        grads   = scaled.grad.detach()
        weights = torch.ones(grads.shape[0], device=device)
        if start == 0:       weights[0]  = 0.5
        if end == CFG["ig_steps"] + 1: weights[-1] = 0.5
        weights = weights.view(-1,1,1,1)
        grads_sum += (grads * weights).sum(dim=0)
        del scaled, outputs, grads, weights
        torch.cuda.empty_cache()

    avg_grads = grads_sum / CFG["ig_steps"]
    ig_signed = diff.squeeze(0).detach() * avg_grads
    ig_map    = ig_signed.abs().sum(dim=0).cpu().numpy().astype(np.float32)
    del avg_grads, ig_signed, grads_sum, diff, baseline
    torch.cuda.empty_cache()
    return ig_map

# ==========================================
# 필터 맵 계산 함수
# ==========================================
def normalize_map(m):
    mn, mx = m.min(), m.max()
    if mx - mn < 1e-8: return np.zeros_like(m)
    return (m - mn) / (mx - mn)

def get_fg_mask(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)
    return mask

def compute_all_filters(img_np, fg_mask):
    H, W    = img_np.shape[:2]
    img_u8  = (img_np * 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR)
    gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(float) / 255.0
    fg      = fg_mask > 0
    gray_fg = gray * fg.astype(float)
    fmaps   = {}

    # --- Texture ---
    lbp = local_binary_pattern(gray, P=CFG["lbp_n_points"],
                                R=CFG["lbp_radius"], method='uniform').astype(float)
    fmaps["LBP"] = normalize_map(lbp) * fg.astype(float)

    fr, fi = gabor(gray, frequency=CFG["gabor_freq"],
                   theta=np.radians(CFG["gabor_theta"]))
    fmaps[f"Gabor_f{CFG['gabor_freq']:.1f}_t{CFG['gabor_theta']}°"] = \
        normalize_map(np.sqrt(fr**2 + fi**2)) * fg.astype(float)

    # --- Frequency ---
    fft_shift = np.fft.fftshift(np.fft.fft2(gray_fg))
    cy, cx    = H//2, W//2
    r_c       = int(min(H,W) * CFG["fft_low_pass_ratio"] / 2)
    ys, xs    = np.ogrid[:H, :W]
    cm        = ((ys-cy)**2 + (xs-cx)**2) <= r_c**2
    lp        = np.zeros((H,W), dtype=complex); lp[cm] = fft_shift[cm]
    fmaps["FFT_LowPass"] = normalize_map(
        np.abs(np.fft.ifft2(np.fft.ifftshift(lp)))) * fg.astype(float)
    hp = fft_shift.copy(); hp[cm] = 0
    fmaps["FFT_HighPass"] = normalize_map(
        np.abs(np.fft.ifft2(np.fft.ifftshift(hp)))) * fg.astype(float)

    coeffs = pywt.wavedec2(gray_fg, wavelet=CFG["wavelet"], level=2)
    for lv in range(1, 3):
        cH, cV, cD = coeffs[lv]
        for name, coeff in zip(["H","V","D"], [cH, cV, cD]):
            eu = cv2.resize(coeff**2, (W,H), interpolation=cv2.INTER_LINEAR)
            fmaps[f"Wavelet_L{lv}_{name}"] = normalize_map(eu) * fg.astype(float)

    # --- Edge / Shape ---
    sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    fmaps["Edge_Sobel"] = normalize_map(np.sqrt(sx**2+sy**2)) * fg.astype(float)
    fmaps["Curvature_Laplacian"] = normalize_map(
        np.abs(cv2.Laplacian(gray, cv2.CV_64F))) * fg.astype(float)

    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    fd_map = np.zeros((H,W), dtype=float)
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        cx_pts = cnt[:,0,0].astype(float) + 1j*cnt[:,0,1].astype(float)
        fft_cnt = np.fft.fft(cx_pts)
        n_k = CFG["fd_n_components"]
        fft_f = np.zeros_like(fft_cnt)
        fft_f[:n_k] = fft_cnt[:n_k]; fft_f[-n_k:] = fft_cnt[-n_k:]
        rec = np.fft.ifft(fft_f)
        rx = np.clip(rec.real.astype(int), 0, W-1)
        ry = np.clip(rec.imag.astype(int), 0, H-1)
        pts = np.stack([rx, ry], axis=1).reshape(-1,1,2).astype(np.int32)
        rm = np.zeros((H,W), dtype=np.uint8)
        cv2.drawContours(rm, [pts], -1, 1, thickness=1)
        fd_map = normalize_map(1.0 / (distance_transform_edt(
            1-np.clip(rm,0,1)) + 1.0))
    fmaps["FourierDescriptor"] = fd_map * fg.astype(float)
    fmaps["DistanceTransform"] = normalize_map(
        distance_transform_edt(fg_mask > 0)) * fg.astype(float)

    # --- Color ---
    fmaps["Brightness"]     = normalize_map(gray.copy()) * fg.astype(float)
    hsv = cv2.cvtColor(img_u8, cv2.COLOR_RGB2HSV)
    fmaps["Saturation_HSV"] = normalize_map(
        hsv[:,:,1].astype(float)) * fg.astype(float)
    lab = cv2.cvtColor(img_u8, cv2.COLOR_RGB2LAB).astype(float)
    fmaps["LAB_L"]      = normalize_map(lab[:,:,0]/100.0) * fg.astype(float)
    fmaps["LAB_Chroma"] = normalize_map(
        np.sqrt(lab[:,:,1]**2 + lab[:,:,2]**2)) * fg.astype(float)

    return fmaps

# ==========================================
# Figure 1: 클래스별 샘플 — 원본 + IG + 필터 전체
# ==========================================
# 각 클래스에서 test set의 첫 번째 이미지 선택
selected = []
for cls in classes:
    sub = test_df[test_df["class"] == cls].reset_index(drop=True)
    selected.append(sub.iloc[0])

# 그릴 필터 목록 (카테고리 순)
filter_list = [
    ("Gabor_f0.3_t0°",  "Gabor\nf=0.3, θ=0°", "hot"),
    ("LBP",             "LBP",                  "hot"),
    ("FFT_LowPass",     "FFT\nLowPass",          "hot"),
    ("FFT_HighPass",    "FFT\nHighPass",          "hot"),
    ("Wavelet_L1_V",    "Wavelet\nL1 Vertical",  "hot"),
    ("Wavelet_L2_D",    "Wavelet\nL2 Diagonal",  "hot"),
    ("Edge_Sobel",      "Edge\nSobel",            "hot"),
    ("DistanceTransform","Distance\nTransform",   "Blues"),
    ("FourierDescriptor","Fourier\nDescriptor",   "hot"),
    ("Brightness",      "Brightness",             "hot"),
    ("LAB_L",           "LAB_L",                  "hot"),
    ("LAB_Chroma",      "LAB_Chroma",             "YlOrRd"),
    ("Saturation_HSV",  "Saturation\nHSV",        "YlOrRd"),
]

n_cols = 2 + len(filter_list)  # 원본 + IG + 필터들
n_rows = len(classes)

print("Figure 1 생성 중 (원본 + IG + 전체 필터)...")
fig, axes = plt.subplots(n_rows, n_cols,
                          figsize=(n_cols * 2.2, n_rows * 2.5))

# 컬럼 헤더
col_headers = ["Original", "IG\nAttribution"] + [f[1] for f in filter_list]
for j, title in enumerate(col_headers):
    axes[0, j].set_title(title, fontsize=7, fontweight='bold', pad=3)

cache_root = Path(CFG["ig_cache_dir"])

for i, (row, cls, short) in enumerate(zip(selected, classes, short_cls)):
    # 이미지 로드
    pil_img  = Image.open(row["filepath"]).convert("RGB")
    inp      = img_transform(pil_img).unsqueeze(0).to(device)
    img_np   = inp.squeeze(0).permute(1,2,0).cpu().numpy()
    img_u8   = (img_np * 255).astype(np.uint8)
    img_bgr  = cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR)
    fg_mask  = get_fg_mask(img_bgr)

    # IG 로드 or 계산
    stem       = Path(row["filepath"]).stem
    cache_path = cache_root / cls / (stem + ".npz")
    if cache_path.exists():
        loaded     = np.load(cache_path)
        ig_raw     = loaded["ig_map"].astype(float)
        pred_class = int(loaded["pred_class"])
    else:
        with torch.no_grad():
            pred_class = model(normalize_batch(inp)).argmax(1).item()
        ig_raw = compute_ig(inp, pred_class)

    p99    = np.percentile(ig_raw, 99)
    ig_map = np.clip(ig_raw / (p99 + 1e-8), 0, 1)

    # 필터 맵 계산
    fmaps = compute_all_filters(img_np, fg_mask)

    # 행 라벨
    axes[i, 0].set_ylabel(short, fontsize=9, fontweight='bold', rotation=90)

    # 원본
    axes[i, 0].imshow(np.clip(img_np, 0, 1))
    axes[i, 0].axis('off')

    # IG
    axes[i, 1].imshow(ig_map, cmap='jet', vmin=0, vmax=1)
    axes[i, 1].axis('off')

    # 필터들
    for j, (fname, _, cmap) in enumerate(filter_list):
        fmap = fmaps.get(fname, np.zeros_like(ig_map))
        axes[i, j+2].imshow(fmap, cmap=cmap, vmin=0, vmax=1)
        axes[i, j+2].axis('off')

    print(f"  {short} 완료")

plt.suptitle("ConvNeXt-Small — Class-wise IG Attribution & Geometric Filter Maps",
             fontsize=11, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(save_root / "fig1_ig_filters_all.png",
            dpi=200, bbox_inches='tight')
plt.close()
print("Figure 1 저장: fig1_ig_filters_all.png")

# ==========================================
# Figure 2: 논문용 핵심 필터만 (깔끔한 버전)
# ==========================================
key_filters = [
    ("FFT_LowPass",      "FFT LowPass\n(밝기 구조)",   "hot"),
    ("Gabor_f0.3_t0°",   "Gabor f=0.3, θ=0°\n(수평 텍스처)", "hot"),
    ("LAB_Chroma",       "LAB Chroma\n(색도)",         "YlOrRd"),
    ("DistanceTransform","Distance Transform\n(공간 구조)", "Blues"),
    ("FourierDescriptor","Fourier Descriptor\n(윤곽선)", "RdBu_r"),
]

n_cols2 = 2 + len(key_filters)
print("\nFigure 2 생성 중 (논문용 핵심 필터)...")
fig2, axes2 = plt.subplots(n_rows, n_cols2,
                            figsize=(n_cols2 * 2.8, n_rows * 3.0))

col_headers2 = ["Original", "IG Attribution"] + [f[1] for f in key_filters]
for j, title in enumerate(col_headers2):
    axes2[0, j].set_title(title, fontsize=9, fontweight='bold', pad=4)

for i, (row, cls, short) in enumerate(zip(selected, classes, short_cls)):
    pil_img  = Image.open(row["filepath"]).convert("RGB")
    inp      = img_transform(pil_img).unsqueeze(0).to(device)
    img_np   = inp.squeeze(0).permute(1,2,0).cpu().numpy()
    img_u8   = (img_np * 255).astype(np.uint8)
    img_bgr  = cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR)
    fg_mask  = get_fg_mask(img_bgr)

    stem       = Path(row["filepath"]).stem
    cache_path = cache_root / cls / (stem + ".npz")
    if cache_path.exists():
        loaded  = np.load(cache_path)
        ig_raw  = loaded["ig_map"].astype(float)
    else:
        with torch.no_grad():
            pred_class = model(normalize_batch(inp)).argmax(1).item()
        ig_raw = compute_ig(inp, pred_class)

    p99    = np.percentile(ig_raw, 99)
    ig_map = np.clip(ig_raw / (p99 + 1e-8), 0, 1)
    fmaps  = compute_all_filters(img_np, fg_mask)

    axes2[i, 0].set_ylabel(short, fontsize=10, fontweight='bold', rotation=90)

    # IG overlay (원본 위에 IG 오버레이)
    axes2[i, 0].imshow(np.clip(img_np, 0, 1))
    axes2[i, 0].axis('off')

    # IG 맵
    axes2[i, 1].imshow(ig_map, cmap='jet', vmin=0, vmax=1)
    axes2[i, 1].axis('off')

    # 핵심 필터
    for j, (fname, _, cmap) in enumerate(key_filters):
        fmap = fmaps.get(fname, np.zeros_like(ig_map))
        axes2[i, j+2].imshow(fmap, cmap=cmap, vmin=0, vmax=1)
        axes2[i, j+2].axis('off')

plt.suptitle(
    "ConvNeXt-Small — Class-wise IG Attribution vs Key Geometric Features\n"
    "(Class average: one representative image per species)",
    fontsize=11, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(save_root / "fig2_ig_key_filters.png",
            dpi=200, bbox_inches='tight')
plt.close()
print("Figure 2 저장: fig2_ig_key_filters.png")

# ==========================================
# Figure 3: IG overlay on original image
# ==========================================
print("\nFigure 3 생성 중 (IG overlay)...")
fig3, axes3 = plt.subplots(2, len(classes),
                            figsize=(len(classes) * 3, 7))

for i, (row, cls, short) in enumerate(zip(selected, classes, short_cls)):
    pil_img = Image.open(row["filepath"]).convert("RGB")
    inp     = img_transform(pil_img).unsqueeze(0).to(device)
    img_np  = inp.squeeze(0).permute(1,2,0).cpu().numpy()

    stem       = Path(row["filepath"]).stem
    cache_path = cache_root / cls / (stem + ".npz")
    if cache_path.exists():
        ig_raw = np.load(cache_path)["ig_map"].astype(float)
    else:
        with torch.no_grad():
            pc = model(normalize_batch(inp)).argmax(1).item()
        ig_raw = compute_ig(inp, pc)

    p99    = np.percentile(ig_raw, 99)
    ig_map = np.clip(ig_raw / (p99 + 1e-8), 0, 1)

    # 원본
    axes3[0, i].imshow(np.clip(img_np, 0, 1))
    axes3[0, i].set_title(short, fontsize=11, fontweight='bold')
    axes3[0, i].axis('off')

    # IG overlay
    import matplotlib.cm as cm
    ig_colored = cm.jet(ig_map)[:,:,:3]
    overlay    = np.clip(img_np * 0.4 + ig_colored * 0.6, 0, 1)
    axes3[1, i].imshow(overlay)
    axes3[1, i].axis('off')

axes3[0, 0].set_ylabel("Original", fontsize=10, fontweight='bold', rotation=90)
axes3[1, 0].set_ylabel("IG Overlay", fontsize=10, fontweight='bold', rotation=90)

plt.suptitle("ConvNeXt-Small — IG Attribution Overlay per Species",
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(save_root / "fig3_ig_overlay.png", dpi=200, bbox_inches='tight')
plt.close()
print("Figure 3 저장: fig3_ig_overlay.png")

print(f"\n모든 Figure 저장 완료: {save_root}")
print("생성 파일:")
print("  fig1_ig_filters_all.png  — 원본 + IG + 전체 필터 (13종)")
print("  fig2_ig_key_filters.png  — 원본 + IG + 핵심 필터 5종 (논문용)")
print("  fig3_ig_overlay.png      — IG 오버레이")