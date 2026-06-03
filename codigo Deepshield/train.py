# -*- coding: utf-8 -*-
"""

Paper:
    "DeepShield: A Dual-Decoder Semantic Segmentation Software for Deepfake Face Detection and Digital Identity Protection"

Authors:
    Rodrigo Eduardo Arevalo-Ancona1 Manuel Cedillo-Hernandez, and Francisco Garcia-Ugalde

Institution:
    SEPI ESIME Culhuacán, Instituto Politécnico Nacional, 
    Facultad de Ingenierìa, Universidad Nacional Autónoma de México

Year:
    2026

Description:
    Training process from a dual Dual Decoder Network model that predicts
    both the original region mask and the manipulated (deepfake) region mask.

    
"""

# train.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm
from model import *

transform = transforms.Compose([
     transforms.Resize((128, 128)),
     transforms.ToTensor()
 ])
 
carpeta = 'Train_D/images/'
 
 
dataset = SegmentationDataset("Train_D/images/", "Train_D/original_mask/", "Train_D/fake_mask/", transform=transform)




train_size = int(0.1*len(dataset))
test_size = len(dataset)-train_size
train_ds, val_ds = random_split(dataset,[train_size,test_size])

train_loader = DataLoader(train_ds,batch_size=40,shuffle=True)
val_loader = DataLoader(val_ds,batch_size=32)


model = DualSegmentationModel().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([65]).to(device))



for epoch in range(1):
    
    model.train()
    total = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", unit="batch")

    for imgs, original_mask, deepfake_mask in pbar:

        imgs = imgs.to(device)
        original_mask = original_mask.to(device)
        deepfake_mask = deepfake_mask.to(device)

        predicted_o, predicted_d = model(imgs)

        loss_real = (
            bce(predicted_o, original_mask)
            + dice_loss(predicted_o, original_mask)
            + iou_loss(predicted_o, original_mask)
        )

        loss_deepfake = (
            bce(predicted_d, deepfake_mask)
            + dice_loss(predicted_d, deepfake_mask)
            + iou_loss(predicted_d, deepfake_mask)
        )

        loss = loss_real + loss_deepfake

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total += loss.item()

        # actualizar barra con loss actual
        pbar.set_postfix(loss=loss.item())

    print(f"Epoch {epoch+1} Loss {total/len(train_loader):.4f}")

    torch.save(model.state_dict(),"deep_sseg_dual2.pth")