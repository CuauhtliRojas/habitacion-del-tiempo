import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# Residual Block
# =========================
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        shortcut = self.shortcut(x)
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.bn(out)
        return out + shortcut


class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)

    def forward(self, x, skip):
        x = self.up(x)
        return torch.cat((x, skip), dim=1)


# =========================
# MODEL
# =========================
class DualSegmentationModel(nn.Module):
    def __init__(self, img_channels=3, output_channels=1):
        super().__init__()

        self.encoder1 = ResidualBlock(img_channels, 64)
        self.encoder2 = ResidualBlock(64, 128)
        self.encoder3 = ResidualBlock(128, 256)
        self.encoder4 = ResidualBlock(256, 512)
        self.encoder5 = ResidualBlock(512, 1024)

        self.pool = nn.MaxPool2d(2, 2)

        self.decoder_instr1 = UpConv(1024, 512)
        self.decoder_instr2 = UpConv(1024, 256)
        self.decoder_instr3 = UpConv(512, 128)
        self.decoder_instr4 = UpConv(256, 64)
        self.final_instr = nn.Conv2d(128, output_channels, 1)

        self.decoder_org1 = UpConv(1024, 512)
        self.decoder_org2 = UpConv(1024, 256)
        self.decoder_org3 = UpConv(512, 128)
        self.decoder_org4 = UpConv(256, 64)
        self.final_org = nn.Conv2d(128, output_channels, 1)

    def decoder_branch(self, conv1, conv2, conv3, conv4, conv5, decoders, final):
        up1 = decoders[0](conv5, conv4)
        up2 = decoders[1](up1, conv3)
        up3 = decoders[2](up2, conv2)
        up4 = decoders[3](up3, conv1)
        return torch.sigmoid(final(up4))

    def forward(self, x):
        c1 = self.encoder1(x)
        c2 = self.encoder2(self.pool(c1))
        c3 = self.encoder3(self.pool(c2))
        c4 = self.encoder4(self.pool(c3))
        c5 = self.encoder5(self.pool(c4))

        out_i = self.decoder_branch(c1,c2,c3,c4,c5,
            [self.decoder_instr1,self.decoder_instr2,self.decoder_instr3,self.decoder_instr4],
            self.final_instr)

        out_o = self.decoder_branch(c1,c2,c3,c4,c5,
            [self.decoder_org1,self.decoder_org2,self.decoder_org3,self.decoder_org4],
            self.final_org)

        return out_i, out_o


# =========================
# DATASET
# =========================
class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_instr_dir, mask_org_dir, transform=None):
        self.image_dir = image_dir
        self.mask_instr_dir = mask_instr_dir
        self.mask_org_dir = mask_org_dir
        self.transform = transform
        self.images = [f for f in os.listdir(image_dir)
                       if f.lower().endswith(('.jpg','.png','.jpeg'))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        name = self.images[idx]
        img = Image.open(os.path.join(self.image_dir,name)).convert("RGB")

        mask_i = Image.open(os.path.join(self.mask_instr_dir,name)).convert("L")
        mask_o = Image.open(os.path.join(self.mask_org_dir,name)).convert("L")

        if self.transform:
            img = self.transform(img)
            mask_i = self.transform(mask_i)
            mask_o = self.transform(mask_o)

        return img, mask_i, mask_o


# =========================
# Loss Functions
# =========================
def dice_loss(pred, target, smooth=1.0):
    intersection = (pred * target).sum()
    return 1 - (2*intersection + smooth) / (pred.sum() + target.sum() + smooth)


def iou_loss(pred, target):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum(dim=(2,3))
    union = pred.sum(dim=(2,3)) + target.sum(dim=(2,3)) - intersection
    return 1 - ((intersection + 1e-6)/(union + 1e-6)).mean()