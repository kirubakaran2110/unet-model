import os
import argparse
import numpy as np
from glob import glob
from PIL import Image
import torch
from torch import nn
import matplotlib.pyplot as plt
from torchvision import transforms

# ---------------- UNet (same as train.py) ---------------- #

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super(UNet, self).__init__()
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        self.up1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv1 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv2 = DoubleConv(256, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv3 = DoubleConv(128, 64)
        self.outc = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4)
        x = torch.cat([x, x3], dim=1)
        x = self.conv1(x)
        x = self.up2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv2(x)
        x = self.up3(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv3(x)
        logits = self.outc(x)
        return logits


# ---------------- PREDICTION FUNCTION ---------------- #

def predict_image(model, image_path, device):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((256, 256))
    image_np = np.array(image, dtype=np.float32) / 255.0
    image_tensor = torch.tensor(image_np.transpose(2, 0, 1)).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.sigmoid(logits)[0, 0].cpu().numpy()

    mask = (probs > 0.5).astype(np.uint8) * 255
    return image_np, mask


# ---------------- MAIN ---------------- #

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")

    model = UNet().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    image_paths = sorted(glob(os.path.join(args.images, "*.png")))[:5]
    print(f">> Running inference on {len(image_paths)} images...")

    for i, path in enumerate(image_paths):
        img, pred_mask = predict_image(model, path, device)

        # Save predicted mask
        out_path = os.path.join(output_dir, f"mask_pred_{i+1}.png")
        Image.fromarray(pred_mask).save(out_path)

        # Show side-by-side
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.title("Original Image")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(pred_mask, cmap='gray')
        plt.title("Predicted Mask")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

        print(f"✅ Saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=str, default="data/images", help="Path to input images")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/unet_final.pth", help="Path to trained model")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run inference")
    args = parser.parse_args()
    main(args)
