# ==========================================
# Stage 3: Integrated Gradients 추출
# stage3_ig.py
#
# - test.csv 이미지만 사용 (225장)
# - Baseline: zero image + Gaussian blur (σ=20)
# - Steps 수렴 검증: 50, 100, 200
# - Completeness error: signed sum 기준
# - 최종 캐시: ig_zero/, ig_blur/ 각각 저장
# ==========================================

import gc
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.stats import spearmanr
from tqdm import tqdm

import torch
import timm
from torchvision import transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

CFG = {
    "split_dir":   "/data/JSM/convnext/0_stage1",
    "model_path":  "/data/JSM/convnext/1_train/best_model.pth",
    "save_dir":    "/data/JSM/convnext/2_ig_cache",
    "model_name":  "convnext_small",
    "num_classes": 5,
    "img_size":    224,
    "ig_steps":    100,
    "ig_chunk":    10,
    "blur_sigma":  20,
    "conv_steps":  [50, 100, 200],
    "conv_n_imgs": 10,
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"device: {device}")

save_root = Path(CFG["save_dir"])
(save_root / "ig_zero").mkdir(parents=True, exist_ok=True)
(save_root / "ig_blur").mkdir(parents=True, exist_ok=True)
(save_root / "convergence").mkdir(parents=True, exist_ok=True)

# ==========================================
# 모델 로드
# ==========================================
model = timm.create_model(CFG["model_name"], pretrained=False,
                           num_classes=CFG["num_classes"])
model.load_state_dict(torch.load(CFG["model_path"], map_location=device))
model = model.to(device)
model.eval()
print("모델 로드 완료")

_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
_std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

def normalize_batch(t):
    return (t - _mean) / _std

load_transform = transforms.Compose([
    transforms.Resize((CFG["img_size"], CFG["img_size"])),
    transforms.ToTensor(),
])

# ==========================================
# 데이터 로드
# ==========================================
test_df = pd.read_csv(Path(CFG["split_dir"]) / "test.csv")
with open(Path(CFG["split_dir"]) / "split_config.json") as f:
    split_cfg = json.load(f)
classes = split_cfg["classes"]
print(f"Test set: {len(test_df)}장")

# ==========================================
# IG 계산 함수
# ==========================================
def compute_ig(input_tensor, baseline_tensor, target_class,
               steps=100, chunk_size=10):
    """
    반환:
      ig_map   : (H, W) float32 — abs().sum(dim=0), 시각화용
      ig_signed_sum : float — signed sum, completeness 검증용
          signed sum = sum(IG) ≈ f(x) - f(baseline)
    """
    diff      = input_tensor - baseline_tensor
    alphas    = torch.linspace(0, 1, steps + 1, device=device)
    grads_sum = torch.zeros_like(input_tensor.squeeze(0))  # (3,H,W)

    for start in range(0, steps + 1, chunk_size):
        end   = min(start + chunk_size, steps + 1)
        a_sub = alphas[start:end].view(-1, 1, 1, 1)

        scaled  = (baseline_tensor + a_sub * diff).detach().requires_grad_(True)
        normed  = normalize_batch(scaled)
        outputs = model(normed)
        outputs[:, target_class].sum().backward()

        grads   = scaled.grad.detach()

        weights = torch.ones(grads.shape[0], device=device)
        if start == 0:
            weights[0] = 0.5
        if end == steps + 1:
            weights[-1] = 0.5
        weights = weights.view(-1, 1, 1, 1)

        grads_sum = grads_sum + (grads * weights).sum(dim=0)

        del scaled, normed, outputs, a_sub, grads, weights
        torch.cuda.empty_cache()

    avg_grads = grads_sum / steps
    ig_signed = diff.squeeze(0).detach() * avg_grads      # (3,H,W) — 부호 있음

    # completeness 검증용: signed sum (abs 없이)
    ig_signed_sum = ig_signed.sum().item()

    # 시각화용: abs().sum(dim=0) → (H,W)
    ig_map = ig_signed.abs().sum(dim=0).cpu().numpy().astype(np.float32)

    del avg_grads, ig_signed, grads_sum, diff
    torch.cuda.empty_cache()

    return ig_map, ig_signed_sum


def completeness_error(input_tensor, baseline_tensor,
                       target_class, ig_signed_sum):
    """
    rel_error = |ig_signed_sum - delta_f| / |delta_f|
    delta_f   = f(x) - f(baseline)
    """
    with torch.no_grad():
        f_x    = model(normalize_batch(input_tensor))[:, target_class].item()
        f_base = model(normalize_batch(baseline_tensor))[:, target_class].item()
    delta_f = f_x - f_base
    error   = abs(ig_signed_sum - delta_f)
    rel_err = error / (abs(delta_f) + 1e-8)
    return rel_err, delta_f

# ==========================================
# Stage 3-2: Steps 수렴 검증
# ==========================================
print("\n" + "=" * 60)
print(" Stage 3-2: Steps 수렴 검증 (Completeness Axiom)")
print("=" * 60)

sample_rows = test_df.sample(n=CFG["conv_n_imgs"],
                              random_state=42).reset_index(drop=True)
conv_records = []

for _, row in tqdm(sample_rows.iterrows(), total=len(sample_rows),
                   desc="수렴 검증"):
    pil_img   = Image.open(row["filepath"]).convert("RGB")
    inp       = load_transform(pil_img).unsqueeze(0).to(device)
    base_zero = torch.zeros_like(inp)

    with torch.no_grad():
        pred_class = model(normalize_batch(inp)).argmax(dim=1).item()

    for steps in CFG["conv_steps"]:
        ig_map, ig_signed_sum = compute_ig(
            inp, base_zero, pred_class, steps, CFG["ig_chunk"])
        rel_err, delta_f = completeness_error(
            inp, base_zero, pred_class, ig_signed_sum)

        conv_records.append({
            "filename":   Path(row["filepath"]).name,
            "steps":      steps,
            "delta_f":    round(float(delta_f), 5),
            "ig_signed_sum": round(float(ig_signed_sum), 5),
            "rel_error":  round(float(rel_err), 5),
        })

conv_df = pd.DataFrame(conv_records)
conv_df.to_csv(save_root / "convergence" / "steps_convergence.csv", index=False)

print("\n[Steps별 평균 Completeness Error]")
for steps in CFG["conv_steps"]:
    sub = conv_df[conv_df["steps"] == steps]
    print(f"  steps={steps:3d}: mean rel_error = {sub.rel_error.mean():.5f}"
          f"  (< 0.05 수렴 기준)")

# 시각화
fig, ax = plt.subplots(figsize=(7, 4))
for fname in conv_df["filename"].unique():
    sub = conv_df[conv_df["filename"] == fname]
    ax.plot(sub["steps"], sub["rel_error"],
            color="gray", alpha=0.4, linewidth=1)
mean_by = conv_df.groupby("steps")["rel_error"].mean()
ax.plot(mean_by.index, mean_by.values,
        color="#EF4444", linewidth=2.5, marker='o', label="Mean")
ax.axhline(0.05, color='blue', linestyle='--',
           alpha=0.7, label="0.05 threshold")
ax.set_xlabel("IG Steps"); ax.set_ylabel("Relative Completeness Error")
ax.set_title("IG Steps 수렴 검증 (Completeness Axiom)", fontsize=11)
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(save_root / "convergence" / "steps_convergence.png",
            dpi=150, bbox_inches='tight')
plt.close()
print(f"수렴 그래프 저장 완료")

# ==========================================
# Stage 3-3: IG 캐시 생성 (test set 전체)
# ==========================================
print("\n" + "=" * 60)
print(f" Stage 3-3: IG 캐시 생성 (steps={CFG['ig_steps']})")
print(f" Zero baseline + Gaussian blur (σ={CFG['blur_sigma']})")
print("=" * 60)

computed_zero = computed_blur = skipped = failed = 0

for _, row in tqdm(test_df.iterrows(), total=len(test_df),
                   desc="IG Cache", unit="img"):
    try:
        cls_name = row["class"]
        stem     = Path(row["filepath"]).stem

        save_zero = save_root / "ig_zero" / cls_name / (stem + ".npz")
        save_blur = save_root / "ig_blur" / cls_name / (stem + ".npz")
        (save_root / "ig_zero" / cls_name).mkdir(parents=True, exist_ok=True)
        (save_root / "ig_blur" / cls_name).mkdir(parents=True, exist_ok=True)

        if save_zero.exists() and save_blur.exists():
            skipped += 1
            continue

        pil_img = Image.open(row["filepath"]).convert("RGB")
        inp     = load_transform(pil_img).unsqueeze(0).to(device)

        with torch.no_grad():
            logits     = model(normalize_batch(inp))
            pred_class = logits.argmax(dim=1).item()

        true_class = classes.index(cls_name)
        is_correct = (pred_class == true_class)

        # --- Zero baseline ---
        if not save_zero.exists():
            base_zero = torch.zeros_like(inp)
            ig_map_z, ig_signed_z = compute_ig(
                inp, base_zero, pred_class, CFG["ig_steps"], CFG["ig_chunk"])
            rel_err_z, delta_f_z = completeness_error(
                inp, base_zero, pred_class, ig_signed_z)

            np.savez_compressed(
                save_zero,
                ig_map=ig_map_z,
                pred_class=np.int32(pred_class),
                true_class=np.int32(true_class),
                is_correct=np.bool_(is_correct),
                completeness_error=np.float32(rel_err_z),
                ig_signed_sum=np.float32(ig_signed_z),
                delta_f=np.float32(delta_f_z),
            )
            del base_zero, ig_map_z
            computed_zero += 1

        # --- Gaussian blur baseline ---
        if not save_blur.exists():
            img_np  = np.array(pil_img.resize(
                (CFG["img_size"], CFG["img_size"]))) / 255.0
            blurred = gaussian_filter(img_np,
                                       sigma=[CFG["blur_sigma"],
                                              CFG["blur_sigma"], 0])
            base_blur = torch.tensor(
                blurred.transpose(2, 0, 1),
                dtype=torch.float32).unsqueeze(0).to(device)

            ig_map_b, ig_signed_b = compute_ig(
                inp, base_blur, pred_class, CFG["ig_steps"], CFG["ig_chunk"])
            rel_err_b, delta_f_b = completeness_error(
                inp, base_blur, pred_class, ig_signed_b)

            np.savez_compressed(
                save_blur,
                ig_map=ig_map_b,
                pred_class=np.int32(pred_class),
                true_class=np.int32(true_class),
                is_correct=np.bool_(is_correct),
                completeness_error=np.float32(rel_err_b),
                ig_signed_sum=np.float32(ig_signed_b),
                delta_f=np.float32(delta_f_b),
            )
            del base_blur, ig_map_b, blurred, img_np
            computed_blur += 1

        torch.cuda.empty_cache()

    except Exception as e:
        tqdm.write(f"실패: {Path(row['filepath']).name} — {e}")
        failed += 1
        continue

    finally:
        if 'inp' in dir(): del inp
        if 'pil_img' in dir(): del pil_img
        gc.collect()
        torch.cuda.empty_cache()

print(f"\n완료")
print(f"  Zero: {computed_zero}장 | Blur: {computed_blur}장 | "
      f"스킵: {skipped}장 | 실패: {failed}장")

# ==========================================
# Zero vs Blur 상관 확인 (10장 샘플)
# ==========================================
print("\n[Zero vs Blur IG 상관 확인 (10장 샘플)]")
sample_rows2 = test_df.sample(n=10, random_state=42).reset_index(drop=True)
corrs = []
for _, row in sample_rows2.iterrows():
    cls_name = row["class"]
    stem     = Path(row["filepath"]).stem
    z = np.load(save_root / "ig_zero" / cls_name / (stem + ".npz"))["ig_map"]
    b = np.load(save_root / "ig_blur" / cls_name / (stem + ".npz"))["ig_map"]
    r, _ = spearmanr(z.flatten(), b.flatten())
    corrs.append(r)
    print(f"  {stem}: r={r:.4f}")

print(f"\n  평균 상관: {np.mean(corrs):.4f} ± {np.std(corrs):.4f}")
if np.mean(corrs) > 0.7:
    print("  ✅ baseline 간 일관성 확인 → Robustness 근거")
else:
    print("  ⚠ baseline 간 차이 있음 → Discussion 포인트")

# 결과 저장
summary = {
    "ig_steps":            CFG["ig_steps"],
    "blur_sigma":          CFG["blur_sigma"],
    "test_n":              len(test_df),
    "computed_zero":       computed_zero,
    "computed_blur":       computed_blur,
    "failed":              failed,
    "zero_blur_corr_mean": round(float(np.mean(corrs)), 4),
    "zero_blur_corr_std":  round(float(np.std(corrs)), 4),
}
with open(save_root / "ig_cache_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n완료. 저장 경로: {save_root}")
print("생성 파일: ig_zero/**, ig_blur/**, convergence/**, ig_cache_summary.json")