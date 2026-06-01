from __future__ import annotations

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = self.shortcut(x)
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.bn(out)
        return out + shortcut


class UpConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        return torch.cat((x, skip), dim=1)


class DualSegmentationModel(nn.Module):
    def __init__(self, img_channels: int = 3, output_channels: int = 1) -> None:
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

    def decoder_branch(
        self,
        conv1: torch.Tensor,
        conv2: torch.Tensor,
        conv3: torch.Tensor,
        conv4: torch.Tensor,
        conv5: torch.Tensor,
        decoders: list[nn.Module],
        final: nn.Module,
    ) -> torch.Tensor:
        up1 = decoders[0](conv5, conv4)
        up2 = decoders[1](up1, conv3)
        up3 = decoders[2](up2, conv2)
        up4 = decoders[3](up3, conv1)
        return torch.sigmoid(final(up4))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        c1 = self.encoder1(x)
        c2 = self.encoder2(self.pool(c1))
        c3 = self.encoder3(self.pool(c2))
        c4 = self.encoder4(self.pool(c3))
        c5 = self.encoder5(self.pool(c4))

        pred_fake = self.decoder_branch(
            c1,
            c2,
            c3,
            c4,
            c5,
            [
                self.decoder_instr1,
                self.decoder_instr2,
                self.decoder_instr3,
                self.decoder_instr4,
            ],
            self.final_instr,
        )

        pred_authentic = self.decoder_branch(
            c1,
            c2,
            c3,
            c4,
            c5,
            [
                self.decoder_org1,
                self.decoder_org2,
                self.decoder_org3,
                self.decoder_org4,
            ],
            self.final_org,
        )

        return pred_fake, pred_authentic
