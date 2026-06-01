import torch
import numpy as np
import cv2
import time
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from torchvision import transforms
from Modelo_U_Net_dual_decoder import *
from tqdm import tqdm

transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor()
])

carpeta = r'C:\ALURA ONE\PythonProject\tesis_dataset\Dataset_U-Net_dual_decoder_fs_li\local_inpainting\test\imagen_original'

dataset = SegmentationDataset(r"C:\ALURA ONE\PythonProject\tesis_dataset\Dataset_U-Net_dual_decoder_fs_li\local_inpainting\test\imagen_original",
                              r"C:\ALURA ONE\PythonProject\tesis_dataset\Dataset_U-Net_dual_decoder_fs_li\local_inpainting\test\mascara_fake",
                              r"C:\ALURA ONE\PythonProject\tesis_dataset\Dataset_U-Net_dual_decoder_fs_li\local_inpainting\test\mascara_autentica",
                              transform=transform)
loader = torch.utils.data.DataLoader(dataset ,batch_size=1)

models = DualSegmentationModel().to(device)
models.load_state_dict(torch.load("deep_sseg_dual2.pth" ,map_location=device))
models.eval()

iou_scores = []



save_dir = "Predicted Mask"
os.makedirs(save_dir, exist_ok=True)

cont = 0  # para nombrar archivos

with torch.no_grad():
    for imgs, fake_mask, original_mask in tqdm(loader, desc="Testing", unit="batch"):

        imgs = imgs.to(device)

        pred_fake_logits, pred_original_logits = models(imgs)

        pred_fake = (pred_fake_logits.cpu().numpy() > 0.5).astype(np.uint8)
        pred_original = (pred_original_logits.cpu().numpy() > 0.5).astype(np.uint8)

        gt_fake = fake_mask.numpy().astype(np.uint8)
        gt_original = original_mask.numpy().astype(np.uint8)

        batch_size = pred_fake.shape[0]

        for b in range(batch_size):

            mask_fake = (pred_fake[b ,0] * 255).astype(np.uint8)
            mask_original = (pred_original[b ,0] * 255).astype(np.uint8)

            Image.fromarray(mask_fake).save(
                os.path.join(save_dir, f"{cont}_fake_mask.png")
            )

            Image.fromarray(mask_original).save(
                os.path.join(save_dir, f"{cont}_original_mask.png")
            )

            cont += 1