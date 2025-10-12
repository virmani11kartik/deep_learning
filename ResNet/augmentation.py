# This script creates a synthetic RGB image (so no internet is needed) and applies
# the 12 augmentations requested. It then displays a grid of results and saves the
# figure to /mnt/data for download.

from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms.functional as F
import torch
import math
import matplotlib.pyplot as plt

# --- Create a synthetic demo image (256x256) ---
W, H = 256, 256
img = Image.new("RGB", (W, H), (240, 240, 240))
draw = ImageDraw.Draw(img)

# Background gradients / shapes
for y in range(H):
    c = int(120 + 120 * (y / H))
    draw.line([(0, y), (W, y)], fill=(c, 100, 160))

# A few shapes to make augmentations visible
draw.rectangle([20, 20, 120, 90], outline=(255, 255, 255), width=3)
draw.ellipse([140, 30, 230, 120], outline=(255, 255, 255), width=3)
draw.polygon([(40, 200), (100, 140), (160, 200)], outline=(255, 255, 255), width=3)

# Text (fallback if no default font)
try:
    font = ImageFont.truetype("DejaVuSans.ttf", 24)
except:
    font = ImageFont.load_default()
draw.text((70, 110), "Demo Img", font=font, fill=(255, 255, 255))

# --- Define augmentations (reasonable parameters) ---
# We follow torchvision's functional API similar to RandAugment ops naming.
augs = []

# (a) ShearX: shear along x-axis in degrees
augs.append(("ShearX (+20°)", lambda im: F.affine(im, angle=0, translate=(0,0), scale=1.0, shear=(20.0, 0.0))))

# (b) ShearY
augs.append(("ShearY (-20°)", lambda im: F.affine(im, angle=0, translate=(0,0), scale=1.0, shear=(0.0, -20.0))))

# (c) TranslateX (pixels). Using 0.2*W as a reasonable shift.
tx = int(0.2 * W)
augs.append((f"TranslateX ({tx}px)", lambda im: F.affine(im, angle=0, translate=(tx, 0), scale=1.0, shear=(0.0, 0.0))))

# (d) TranslateY (pixels)
ty = int(0.2 * H)
augs.append((f"TranslateY ({ty}px)", lambda im: F.affine(im, angle=0, translate=(0, ty), scale=1.0, shear=(0.0, 0.0))))

# (e) Rotate
augs.append(("Rotate (30°)", lambda im: F.rotate(im, angle=30, expand=False)))

# (f) Brightness
augs.append(("Brightness (×1.5)", lambda im: F.adjust_brightness(im, 1.5)))

# (g) Color (saturation)
augs.append(("Color/Saturation (×1.5)", lambda im: F.adjust_saturation(im, 1.5)))

# (h) Contrast
augs.append(("Contrast (×1.5)", lambda im: F.adjust_contrast(im, 1.5)))

# (i) Sharpness
augs.append(("Sharpness (×2.0)", lambda im: F.adjust_sharpness(im, 2.0)))

# (j) Posterize (bits)
augs.append(("Posterize (bits=3)", lambda im: F.posterize(im, bits=3)))

# (k) Solarize (threshold)
augs.append(("Solarize (thr=128)", lambda im: F.solarize(im, threshold=128)))

# (l) Equalize
augs.append(("Equalize", lambda im: F.equalize(im)))

# --- Build grid: Original + 12 augmented images ---
images = [("Original", img)] + [(name, fn(img)) for name, fn in augs]

cols = 4
rows = math.ceil(len(images) / cols)

fig_w = cols * 3
fig_h = rows * 3
fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h))
axes = axes.flatten()

for ax, (title, im) in zip(axes, images):
    ax.imshow(im)
    ax.set_title(title, fontsize=10)
    ax.axis('off')

# Turn off any extra axes
for k in range(len(images), len(axes)):
    axes[k].axis('off')

plt.tight_layout()
out_path = "augmentations_grid.png"
plt.savefig(out_path, bbox_inches="tight")
out_path
