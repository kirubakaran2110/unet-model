import os
import argparse
import numpy as np
from tqdm import tqdm
from glob import glob

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ---------------- U-Net MODEL ---------------- #

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


# ---------------- DATASET ---------------- #

class SegDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.imgs = sorted(glob(os.path.join(img_dir, "*.png")))
        self.masks = sorted(glob(os.path.join(mask_dir, "*.png")))
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = np.array(Image.open(self.imgs[idx]).convert("RGB"), dtype=np.float32) / 255.0
        mask = np.array(Image.open(self.masks[idx]).convert("L"), dtype=np.float32) / 255.0

        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']

        # Ensure correct types
        img = img.float()
        mask = mask.float()

        return img, mask


# ---------------- LOSSES ---------------- #

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, target):
        probs = torch.sigmoid(logits)
        num = 2 * (probs * target).sum() + self.smooth
        den = probs.sum() + target.sum() + self.smooth
        return 1 - num / den


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5):
        super(BCEDiceLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight

    def forward(self, logits, target):
        return self.bce_weight * self.bce(logits, target) + (1 - self.bce_weight) * self.dice(logits, target)


# ---------------- TRAINING ---------------- #

def get_transforms():
    return A.Compose([
        A.Resize(256, 256),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        ToTensorV2(),
    ])

def train_loop(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, total_dice = 0, 0
    for imgs, masks in tqdm(loader, desc="Training", leave=False):
        imgs = imgs.to(device)
        masks = masks.unsqueeze(1).to(device)  # ✅ fix added here

        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            preds = (torch.sigmoid(logits) > 0.5).float()
            dice = (2 * (preds * masks).sum()) / ((preds + masks).sum() + 1e-6)

        total_loss += loss.item()
        total_dice += dice.item()

    return total_loss / len(loader), total_dice / len(loader)


def val_loop(model, loader, criterion, device):
    model.eval()
    total_loss, total_dice = 0, 0
    with torch.no_grad():
        for imgs, masks in tqdm(loader, desc="Validation", leave=False):
            imgs = imgs.to(device)
            masks = masks.unsqueeze(1).to(device)  # ✅ fix added here too
            logits = model(imgs)
            loss = criterion(logits, masks)
            preds = (torch.sigmoid(logits) > 0.5).float()
            dice = (2 * (preds * masks).sum()) / ((preds + masks).sum() + 1e-6)
            total_loss += loss.item()
            total_dice += dice.item()

    return total_loss / len(loader), total_dice / len(loader)


# ---------------- MAIN ---------------- #

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f">> Using device: {device}")

    all_imgs = sorted(glob(os.path.join(args.images, "*.png")))
    n_val = int(0.1 * len(all_imgs))
    train_imgs = all_imgs[:-n_val]
    val_imgs = all_imgs[-n_val:]

    os.makedirs("checkpoints", exist_ok=True)

    train_dataset = SegDataset(args.images, args.masks, transform=get_transforms())
    val_dataset = SegDataset(args.images, args.masks, transform=get_transforms())
    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch, shuffle=False)

    print(f"Train images: {len(train_dataset)}   Val images: {len(val_dataset)}")

    model = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = BCEDiceLoss()

    for epoch in range(1, args.epochs + 1):
        train_loss, train_dice = train_loop(model, train_loader, optimizer, criterion, device)
        val_loss, val_dice = val_loop(model, val_loader, criterion, device)
        print(f"Epoch [{epoch}/{args.epochs}] | Train Loss: {train_loss:.4f}, Dice: {train_dice:.4f} | "
              f"Val Loss: {val_loss:.4f}, Dice: {val_dice:.4f}")

    torch.save(model.state_dict(), "checkpoints/unet_final.pth")
    print("✅ Training completed. Model saved to checkpoints/unet_final.pth")


# ---------------- RUN ---------------- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=str, default="data/images")
    parser.add_argument("--masks", type=str, default="data/masks")
    parser.add_argument("--epochs", type=int, default=3)     # ✅ changed to 3
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    main(args)
