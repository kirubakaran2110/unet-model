# scripts/generate_synthetic_data.py
"""
Generate a small synthetic dataset for binary segmentation.
Each image will have random colored shapes ("vehicles") on a background,
and the corresponding binary mask will mark those shapes.
"""

import os
import random
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFilter


# ============================
# CONFIG
# ============================
OUT_ROOT = Path("data")
OUT_IMAGES = OUT_ROOT / "images"
OUT_MASKS = OUT_ROOT / "masks"
IMAGE_SIZE = 256


# ============================
# FUNCTIONS
# ============================
def ensure_dirs():
    OUT_IMAGES.mkdir(parents=True, exist_ok=True)
    OUT_MASKS.mkdir(parents=True, exist_ok=True)


def random_vehicle(img, mask, bbox, color):
    """Draw a random 'vehicle' (rectangle or ellipse)"""
    x0, y0, x1, y1 = bbox
    shape_type = random.choice(["rect", "ellipse", "rotated"])

    if shape_type == "rect":
        ImageDraw.Draw(img).rectangle(bbox, fill=color)
        ImageDraw.Draw(mask).rectangle(bbox, fill=255)

    elif shape_type == "ellipse":
        ImageDraw.Draw(img).ellipse(bbox, fill=color)
        ImageDraw.Draw(mask).ellipse(bbox, fill=255)

    else:
        # Create rotated patch
        w, h = x1 - x0, y1 - y0
        patch = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        pdraw = ImageDraw.Draw(patch)
        pdraw.rectangle((0, 0, w, h), fill=color + (255,))

        angle = random.uniform(-30, 30)
        patch = patch.rotate(angle, expand=True)

        # Paste on main image
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        ox, oy = cx - patch.width // 2, cy - patch.height // 2
        img.paste(patch, (ox, oy), patch)

        # Mask version
        mask_patch = Image.new("L", (w, h), 0)
        mdraw = ImageDraw.Draw(mask_patch)
        mdraw.rectangle((0, 0, w, h), fill=255)
        mask_patch = mask_patch.rotate(angle, expand=True)
        mask.paste(mask_patch, (ox, oy), mask_patch)


def make_one(idx, size=IMAGE_SIZE):
    """Create one synthetic image-mask pair"""
    # Background
    bg_color = tuple([random.randint(60, 200) for _ in range(3)])
    img = Image.new("RGB", (size, size), bg_color)
    mask = Image.new("L", (size, size), 0)

    # Add random shapes
    for _ in range(random.randint(3, 8)):
        w, h = random.randint(15, 50), random.randint(10, 40)
        x0, y0 = random.randint(0, size - w - 1), random.randint(0, size - h - 1)
        x1, y1 = x0 + w, y0 + h
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        random_vehicle(img, mask, (x0, y0, x1, y1), color)

    # Optional blur/brightness noise
    if random.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0, 1.5)))

    # Save
    img.save(OUT_IMAGES / f"im_{idx:04d}.png")
    mask.save(OUT_MASKS / f"m_{idx:04d}.png")


def generate(n=500, size=IMAGE_SIZE):
    ensure_dirs()
    for i in range(n):
        make_one(i, size=size)
    print(f"✅ Generated {n} image-mask pairs in {OUT_IMAGES} and {OUT_MASKS}")


# ============================
# MAIN ENTRY
# ============================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--size", type=int, default=256)
    args = parser.parse_args()

    generate(n=args.n, size=args.size)
