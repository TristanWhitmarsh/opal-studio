# ============================================================
# InstanSeg inference notebook — run cells top to bottom
# ============================================================

# ── Cell 1: Paths — edit these ───────────────────────────────
from pathlib import Path
import sys

INSTANSEG_ROOT = Path("/storage/scratch.space/users/tristan/TrainModels/InstanSeg/instanseg")
MODEL_NAME     = "my_instanseg_cells"
IMAGE_PATH     = Path("img.tiff")
PIXEL_SIZE     = 1.0          # microns/pixel used during training
TARGET         = "C"          # "C" = cells, "N" = nuclei, "NC" = both

sys.path.insert(0, str(INSTANSEG_ROOT))
MODEL_DIR = INSTANSEG_ROOT / "instanseg" / "models" / MODEL_NAME

print("Model dir:", MODEL_DIR)
print("Files in model dir:", [f.name for f in MODEL_DIR.iterdir()])


# ── Cell 2 (fixed): Export using trace instead of script ─────
import torch
from instanseg.utils.model_loader import load_model

result = load_model(str(MODEL_DIR))
model  = result[0] if isinstance(result, (tuple, list)) else result
model.eval()

state = torch.load(MODEL_DIR / "model_weights.pth", map_location="cpu")
if isinstance(state, dict) and "model_state_dict" in state:
    state = state["model_state_dict"]
first_conv_key = next(k for k in state if "weight" in k and state[k].ndim == 4)
# Replace the n_in detection block with this:
# Use the actual first Conv2d in the model, not a heuristic key search
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Conv2d):
        n_in = module.in_channels
        print(f"First Conv2d '{name}': in_channels={n_in}")
        break

dummy = torch.zeros(1, n_in, 256, 256)

with torch.no_grad():
    traced_backbone = torch.jit.trace(model, dummy)

class WrappedInstanSeg(torch.nn.Module):
    pixel_size:       float
    cells_and_nuclei: bool

    def __init__(self, backbone: torch.nn.Module, px: float, cells_and_nuclei: bool):
        super().__init__()
        self.backbone         = backbone
        self.pixel_size       = px
        self.cells_and_nuclei = cells_and_nuclei

    def forward(self, x: torch.Tensor, target_segmentation: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

wrapped  = WrappedInstanSeg(traced_backbone, float(PIXEL_SIZE), TARGET == "NC")
scripted = torch.jit.script(wrapped)
scripted.save(str(PT_PATH))
print(f"Saved — pixel_size={scripted.pixel_size}, cells_and_nuclei={scripted.cells_and_nuclei}")


# ── Cell 3: Load image ────────────────────────────────────────
import numpy as np
import tifffile

raw = tifffile.imread(IMAGE_PATH)
print("Raw image shape:", raw.shape, "  dtype:", raw.dtype)

# Expand grayscale to n_in channels (model was trained expecting 3)
if raw.ndim == 2 and n_in > 1:
    raw_input = np.stack([raw] * n_in, axis=0)   # (n_in, H, W)
    print(f"Expanded to {n_in} channels: {raw_input.shape}")
elif raw.ndim == 2:
    raw_input = raw[None]                          # (1, H, W)
else:
    raw_input = raw
    
# ── Cell 4: Run InstanSeg inference ──────────────────────────
from instanseg.utils.loss.instanseg_loss import InstanSeg as InstanSegPostProcessor
from instanseg.utils.model_loader import load_model
from instanseg.utils.utils import percentile_normalize
import torch

result   = load_model(str(MODEL_DIR))
backbone = result[0] if isinstance(result, (tuple, list)) else result
backbone.eval()

tensor = torch.from_numpy(raw_input).unsqueeze(0).float()
tensor = torch.stack([percentile_normalize(tensor[0])])

with torch.no_grad():
    embeddings = backbone(tensor)
print("Embeddings:", embeddings.shape)

postprocessor = InstanSegPostProcessor(
    n_sigma=2, dim_coords=2, dim_seeds=1,
    cells_and_nuclei=False, device="cpu",
)
postprocessor.initialize_pixel_classifier(backbone)
postprocessor.pixel_classifier = postprocessor.pixel_classifier.to("cpu")

with torch.no_grad():
    instance_map = postprocessor.postprocessing(
        embeddings[0], device="cpu", classifier=postprocessor.pixel_classifier,
    )

cell_mask = instance_map[0].cpu().numpy()
print("Cell mask shape:", cell_mask.shape)
print("Instances found:", int(cell_mask.max()))


# ── Cell 5: Plot ──────────────────────────────────────────────

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_dilation

display_img = make_display_image(raw)
cell_rgb    = label_to_rgb(cell_mask.astype(int))

fig, axes = plt.subplots(1, 3, figsize=(16, 6), facecolor="#111")
for ax in axes:
    ax.axis("off")
    ax.set_facecolor("#111")

axes[0].imshow(display_img, cmap="gray")
axes[0].set_title("Input image", color="white", fontsize=13)

axes[1].imshow(cell_rgb)
axes[1].set_title(f"Instance labels  (n={int(cell_mask.max())})", color="white", fontsize=13)

boundary = np.zeros_like(cell_mask, dtype=bool)
for cell_id in np.unique(cell_mask):
    if cell_id == 0:
        continue
    mask = cell_mask == cell_id
    boundary |= binary_dilation(mask) ^ mask

overlay = np.stack([display_img]*3, axis=-1).copy()
overlay[boundary] = [1, 0.2, 0.1]

axes[2].imshow(overlay)
axes[2].set_title("Overlay", color="white", fontsize=13)

fig.suptitle(IMAGE_PATH.name, color="white", fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(str(MODEL_DIR / "inference_result.png"), dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.show()
print("Plot saved to:", MODEL_DIR / "inference_result.png")

# ── Cell 6: Save mask as TIFF ─────────────────────────────────
out_tiff = IMAGE_PATH.parent / (IMAGE_PATH.stem + "_cell_mask.tif")
tifffile.imwrite(str(out_tiff), cell_mask.astype(np.int32))
print("Mask saved to:", out_tiff)