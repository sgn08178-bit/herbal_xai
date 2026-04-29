# ==========================================
# Stage 1: QC + Stratified Split (8:2)
# stage1.py
# ==========================================

import cv2
import numpy as np
import pandas as pd
import json
from pathlib import Path
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

CFG = {
    "data_dir":   "/data/JSM/06_2",
    "save_dir":   "/data/JSM/convnext/0_stage1",
    "seed":       42,
    "test_size":  0.2,         # 8:2 split
    "img_size":   224,
    "bg_threshold":      10,
    "min_fg_ratio":      0.05,
    "max_bg_residue":    0.02,
    "edge_cut_margin":   5,
}

save_root = Path(CFG["save_dir"])
save_root.mkdir(parents=True, exist_ok=True)
data_path = Path(CFG["data_dir"])

from torchvision import transforms
load_transform = transforms.Compose([
    transforms.Resize((CFG["img_size"], CFG["img_size"])),
    transforms.ToTensor(),
])

valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
classes   = sorted([d.name for d in data_path.iterdir()
                    if d.is_dir() and not d.name.startswith('.')])
print(f"클래스: {classes}")

# ==========================================
# 헬퍼 함수
# ==========================================
def get_fg_mask(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, CFG["bg_threshold"], 255, cv2.THRESH_BINARY)
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask    = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask    = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    return mask

def check_edge_cut(mask, margin=5):
    h, w = mask.shape
    return (mask[:margin, :].sum() > 0 or
            mask[h-margin:, :].sum() > 0 or
            mask[:, :margin].sum() > 0 or
            mask[:, w-margin:].sum() > 0)

def check_bg_residue(mask, img_gray):
    bg_region = (mask == 0)
    bg_pixels = img_gray[bg_region]
    residue   = (bg_pixels > CFG["bg_threshold"]).sum()
    return residue / (bg_region.sum() + 1e-8)

# ==========================================
# Stage 1-1: 이미지 수집 + QC
# ==========================================
image_files, labels, qc_records = [], [], []

for cls_idx, cls in enumerate(classes):
    imgs = sorted([p for p in (data_path / cls).iterdir()
                   if p.suffix.lower() in valid_ext])
    for img_path in tqdm(imgs, desc=f"QC {cls}"):
        try:
            pil_img  = Image.open(img_path).convert("RGB")
            img_np   = load_transform(pil_img).permute(1, 2, 0).numpy()
            img_u8   = (img_np * 255).astype(np.uint8)
            img_bgr  = cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR)
            img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            mask     = get_fg_mask(img_bgr)

            fg_ratio   = (mask > 0).sum() / mask.size
            edge_cut   = check_edge_cut(mask, CFG["edge_cut_margin"])
            bg_residue = check_bg_residue(mask, img_gray)

            qc_records.append({
                "class":          cls,
                "filename":       img_path.name,
                "filepath":       str(img_path),
                "label":          cls_idx,
                "fg_ratio":       round(fg_ratio, 4),
                "bg_residue":     round(bg_residue, 4),
                "edge_cut":       edge_cut,
                "small_fg":       fg_ratio < CFG["min_fg_ratio"],
                "bg_residue_flag": bg_residue > CFG["max_bg_residue"],
            })
            image_files.append(img_path)
            labels.append(cls_idx)

        except Exception as e:
            print(f"실패: {img_path.name} — {e}")

df_qc = pd.DataFrame(qc_records)
df_qc.to_csv(save_root / "qc_results.csv", index=False)
labels_arr = np.array(labels)

# ==========================================
# Stage 1-2: Class balance 확인
# ==========================================
print("\n" + "=" * 55)
print(" Stage 1-2: Class Balance")
print("=" * 55)

counts = []
for cls_idx, cls in enumerate(classes):
    n = (labels_arr == cls_idx).sum()
    counts.append(n)
    print(f"  {cls}: {n}장")
print(f"  합계: {len(image_files)}장")

imbalance = max(counts) / min(counts)
print(f"\n  최대/최소 비율: {imbalance:.2f}")
if imbalance > 1.5:
    print("  ⚠ 불균형 → weighted cross-entropy 권장")
else:
    print("  ✅ 균형 양호")

# ==========================================
# Stage 1-3: Stratified Split (8:2)
# ==========================================
print("\n" + "=" * 55)
print(" Stage 1-3: Stratified Split (8:2)")
print("=" * 55)

sss = StratifiedShuffleSplit(
    n_splits=1, test_size=CFG["test_size"], random_state=CFG["seed"])
train_idx, test_idx = next(sss.split(image_files, labels_arr))

print(f"  Train: {len(train_idx)}장 ({len(train_idx)/len(image_files)*100:.1f}%)")
print(f"  Test:  {len(test_idx)}장 ({len(test_idx)/len(image_files)*100:.1f}%)")

print("\n  [클래스별 분포]")
for split_name, idx_arr in [("Train", train_idx), ("Test", test_idx)]:
    row = f"  {split_name}: "
    for cls_idx, cls in enumerate(classes):
        n = (labels_arr[idx_arr] == cls_idx).sum()
        row += f"{cls.replace('_crop','')}={n}  "
    print(row)

# CSV 저장
def save_csv(idx_arr, split_name):
    rows = [{"filepath": str(image_files[i]),
             "class":    classes[labels_arr[i]],
             "label":    int(labels_arr[i]),
             "split":    split_name}
            for i in idx_arr]
    df = pd.DataFrame(rows)
    df.to_csv(save_root / f"{split_name}.csv", index=False)
    return df

train_df = save_csv(train_idx, "train")
test_df  = save_csv(test_idx,  "test")
pd.concat([train_df, test_df]).to_csv(save_root / "all_splits.csv", index=False)

# split config 저장
config = {
    "seed": CFG["seed"], "split": "8:2",
    "train_n": int(len(train_idx)), "test_n": int(len(test_idx)),
    "total_n": int(len(image_files)), "classes": classes,
    "class_counts": {k: int(v) for k, v in zip(classes, counts)},
}
with open(save_root / "split_config.json", "w") as f:
    json.dump(config, f, indent=2, ensure_ascii=False)

# ==========================================
# QC 요약 출력
# ==========================================
print("\n" + "=" * 55)
print(" Stage 1-1: QC 요약")
print("=" * 55)
edge_n    = df_qc["edge_cut"].sum()
residue_n = df_qc["bg_residue_flag"].sum()
small_n   = df_qc["small_fg"].sum()
any_prob  = (df_qc["edge_cut"] | df_qc["bg_residue_flag"] | df_qc["small_fg"]).sum()
print(f"  가장자리 잘림:  {edge_n}장 ({edge_n/len(df_qc)*100:.1f}%)")
print(f"  배경 잔여물:    {residue_n}장 ({residue_n/len(df_qc)*100:.1f}%)")
print(f"  전경 너무 작음: {small_n}장 ({small_n/len(df_qc)*100:.1f}%)")
print(f"  문제 합계:      {any_prob}장 ({any_prob/len(df_qc)*100:.1f}%)")

# ==========================================
# 시각화
# ==========================================
short_cls = [c.replace("_crop", "") for c in classes]
colors    = ["#0D9488", "#F59E0B", "#10B981", "#8B5CF6", "#EF4444"]

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# A: 클래스별 전체 이미지 수
axes[0].bar(short_cls, counts, color=colors, edgecolor='white')
for i, cnt in enumerate(counts):
    axes[0].text(i, cnt + 0.5, str(cnt), ha='center', fontsize=10, fontweight='bold')
axes[0].set_title("전체 클래스별 이미지 수", fontsize=11)
axes[0].set_ylim(0, max(counts) * 1.15)
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)

# B: Split 분포
for ax, (split_name, idx_arr) in zip(axes[1:], [
        ("Train (80%)", train_idx), ("Test (20%)", test_idx)]):
    cnts = [(labels_arr[idx_arr] == i).sum() for i in range(len(classes))]
    ax.bar(short_cls, cnts, color=colors, edgecolor='white')
    for i, cnt in enumerate(cnts):
        ax.text(i, cnt + 0.3, str(cnt), ha='center', fontsize=9)
    ax.set_title(f"{split_name}\n(n={len(idx_arr)})", fontsize=11)
    ax.set_ylim(0, max(cnts) * 1.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.suptitle("Stage 1: Class Balance & Stratified Split (8:2)", fontsize=12)
plt.tight_layout()
plt.savefig(save_root / "stage1_summary.png", dpi=150, bbox_inches='tight')
plt.close()

print(f"\n완료. 저장 경로: {save_root}")
print("생성 파일: qc_results.csv, train.csv, test.csv, split_config.json, stage1_summary.png")
print(f"\n⚠  IG 분석은 test.csv 이미지만 사용 (n={len(test_idx)})")