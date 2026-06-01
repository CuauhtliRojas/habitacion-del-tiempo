import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm
from Modelo_U_Net_dual_decoder import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Dispositivo usado:", device)

transform = transforms.Compose([
    transforms.Resize((256, 256)), #128x128
    transforms.ToTensor()
])

epochs=15 #20
batch_size=10 #60
carpeta = r'C:\ALURA ONE\PythonProject\tesis_dataset\Dataset_U-Net_dual_decoder_fs_li\local_inpainting\train\imagen_original'

dataset = SegmentationDataset(r"C:\ALURA ONE\PythonProject\tesis_dataset\Dataset_U-Net_dual_decoder_fs_li\local_inpainting\train\imagen_original",
                              r"C:\ALURA ONE\PythonProject\tesis_dataset\Dataset_U-Net_dual_decoder_fs_li\local_inpainting\train\mascara_fake",
                              r"C:\ALURA ONE\PythonProject\tesis_dataset\Dataset_U-Net_dual_decoder_fs_li\local_inpainting\train\mascara_autentica",
                              transform=transform)

val_size = int(0.1 * len(dataset)) #10% para validacion
train_size = len(dataset) - val_size #90% para entrenamiento
train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=batch_size)

model = DualSegmentationModel().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([65]).to(device))

for epoch in range(epochs):

    model.train()
    total = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}", unit="batch")

    for imgs, deepfake_mask,original_mask in pbar:
        imgs = imgs.to(device)
        original_mask = original_mask.to(device)
        deepfake_mask = deepfake_mask.to(device)

        predicted_d,predicted_o = model(imgs)

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

    print(f"Epoch {epoch + 1} Loss {total / len(train_loader):.4f}")

    torch.save(model.state_dict(), "deep_sseg_dual2.pth")