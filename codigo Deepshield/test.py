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
    Testing process from a dual Dual Decoder Network model that predicts
    both the original region mask and the manipulated (deepfake) region mask.

    
"""


import torch
import numpy as np
import cv2
import time
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from torchvision import transforms
from model import *
from tqdm import tqdm

transform = transforms.Compose([
     transforms.Resize((128, 128)),
     transforms.ToTensor()
 ])
 
carpeta = 'Train_D/images/'
 
 
dataset = SegmentationDataset("Train_D/images/", "Train_D/original_mask/", "Train_D/fake_mask/", transform=transform)

loader = torch.utils.data.DataLoader(dataset,batch_size=1)

models = DualSegmentationModel().to(device)
models.load_state_dict(torch.load("deep_sseg_dual2.pth",map_location=device))
models.eval()

iou_scores = []



save_dir = "Predicted Mask"
os.makedirs(save_dir, exist_ok=True)

cont = 0  # para nombrar archivos

with torch.no_grad():
    for imgs, original_mask, fake_mask in tqdm(loader, desc="Testing", unit="batch"):

        imgs = imgs.to(device)

        pred_fake_logits, pred_original_logits = models(imgs)

        pred_fake = (pred_fake_logits.cpu().numpy() > 0.5).astype(np.uint8)
        pred_original = (pred_original_logits.cpu().numpy() > 0.5).astype(np.uint8)

        gt_fake = fake_mask.numpy().astype(np.uint8)
        gt_original = original_mask.numpy().astype(np.uint8)

        batch_size = pred_fake.shape[0]

        for b in range(batch_size):

            mask_fake = (pred_fake[b,0] * 255).astype(np.uint8)
            mask_original = (pred_original[b,0] * 255).astype(np.uint8)

            Image.fromarray(mask_fake).save(
                os.path.join(save_dir, f"{cont}_fake_mask.png")
            )

            Image.fromarray(mask_original).save(
                os.path.join(save_dir, f"{cont}_original_mask.png")
            )

            cont += 1
      


