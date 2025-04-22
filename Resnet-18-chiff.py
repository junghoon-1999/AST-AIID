import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
import torchaudio
import os
from os import listdir
from os.path import join
import numpy as np
import tqdm
from tqdm import tqdm
from torcheval.metrics import MulticlassAUROC, MulticlassAccuracy

# generate spectrograms from audio wav files


model = models.resnet18(pretrained=True)

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(trainable_params)

num_classes = 13
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
])

train_dataset = ImageFolder('./cnn-data/train', transform = transform)
test_dataset = ImageFolder('./cnn-data/test', transform = transform)

split = int(len(train_dataset)*0.8)
train_dataset, val_dataset = random_split(train_dataset, [split, len(train_dataset)-split])

train_loader = DataLoader(train_dataset, batch_size = 32, shuffle = True)
valid_loader = DataLoader(val_dataset, batch_size = 32, shuffle = False)
test_loader = DataLoader(test_dataset, batch_size = 32, shuffle = False)

device = torch.device('cuda')
loss = torch.nn.CrossEntropyLoss()
opt = torch.optim.AdamW(model.parameters(), lr = 1e-4)
num_epochs = 10
model.to(device)
min_val_loss = 1000000

for epoch in range(num_epochs):
    model.train()
    loss_val = 0
    for inputs, labels in tqdm(train_loader):
        s, t = inputs.to(device), labels.to(device)
        opt.zero_grad()
        preds = model(s)
        celoss = loss(preds, t)
        celoss.backward()
        opt.step()

        loss_val += celoss.item()

    avg_loss = loss_val/len(train_loader)
    model.eval()

# Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        val_loss_val = 0
        for inputs, labels in tqdm(valid_loader):
            s = inputs.to(device)
            t = labels.to(device)
            preds = model(s)
            celoss = loss(preds, t)
            val_loss_val += celoss.item()
    val_avg_loss = val_loss_val / len(valid_loader)
    if val_avg_loss < min_val_loss:
        min_val_loss = val_avg_loss
        best_model = model.state_dict()
    print(f'Epoch {epoch}, {avg_loss = }, {val_avg_loss = }')
    torch.cuda.empty_cache()

model.load_state_dict(best_model)
model.to(device)  # Ensure model is on the correct device
model.eval()

metric_1 = MulticlassAUROC(num_classes=13)
metric_2 = MulticlassAccuracy()

output = []

with torch.no_grad():  # Prevents unnecessary gradient tracking
    for inputs, labels in tqdm(test_loader):
        s = inputs.to(device)
        t = labels.to(device)

        with torch.cuda.amp.autocast():  # FP16 inference
            preds = model(s)

        preds = torch.logit(preds, eps = 1e-6)

        output.append(preds.cpu())  # Move to CPU to free GPU memory

        metric_1.update(preds, t)
        metric_2.update(preds, t)

        torch.cuda.empty_cache()  # Free GPU memory after each batch

# Compute metrics after all batches
test_loss_val_1 = metric_1.compute()
test_loss_val_2 = metric_2.compute()

print(f"Test AUROC: {test_loss_val_1:.4f}")
print(f"Test Accuracy: {test_loss_val_2:.4f}")