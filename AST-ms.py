# No CV, no extra layers froze the first 11 layers 

from transformers import AutoFeatureExtractor, ASTForAudioClassification, ASTFeatureExtractor
from datasets import load_dataset
import torch
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, random_split, Subset
import os
from os import listdir
from os.path import isfile, join
import transformers
import accelerate
import peft
import numpy as np
import torch
from torcheval.metrics import MulticlassAUROC, MulticlassAccuracy
###################################################################################################
# Reconstructing the Training set with 4 examples for each label 
###################################################################################################

dirpath = './dataset-inputs-tot/Train'
# Open pt file
files = [f for f in listdir(dirpath)]
train_limits = {}
for filename in files:
    if 'aug' not in filename:
        if int(filename[-5:-3]) not in train_limits.keys():
            train_limits[int(filename[-5:-3])] = 1
        else:
            train_limits[int(filename[-5:-3])] += 1
for i in train_limits.keys():
    train_limits[i] *= 1
cache = {}
training_temp = {'input_values': [], 'labels':[]}
for filename in files:
    if 'aug' not in filename:
        if filename[-5:-3] not in cache.keys():
            cache[filename[-5:-3]] = 1
            training_temp['input_values'].append(torch.load(os.path.join(dirpath, filename)))
            training_temp['labels'].append(filename[-5:-3])
        else:
            cache[filename[-5:-3]] += 1
            training_temp['input_values'].append(torch.load(os.path.join(dirpath, filename)))
            training_temp['labels'].append(filename[-5:-3])

train_seq = torch.stack(training_temp['input_values'])
train_tgt = torch.tensor([int(s) for s in training_temp['labels']])

shuffled_indices = np.random.permutation(len(train_seq))
seq = train_seq[shuffled_indices]
tgt = train_tgt[shuffled_indices]

train_size = int(len(train_seq)*0.8)

train_seq = seq[:train_size]
train_tgt = tgt[:train_size]

valid_seq = seq[train_size:]
valid_tgt = tgt[train_size:]

###################################################################################################
# Reconstructing the Testing set with 2 examples for each label 
###################################################################################################

dirpath = './dataset-inputs/Test'
# Open pt file

files = [f for f in listdir(dirpath)]
cache = {}
testing_temp = {'input_values': [], 'labels':[]}
for filename in files:
    if filename[-5:-3] not in cache.keys():
        cache[filename[-5:-3]] = 1
        testing_temp['input_values'].append(torch.load(os.path.join(dirpath, filename)))
        testing_temp['labels'].append(filename[-5:-3])
    else:
        cache[filename[-5:-3]] += 1
        testing_temp['input_values'].append(torch.load(os.path.join(dirpath, filename)))
        testing_temp['labels'].append(filename[-5:-3])

test_seq = torch.stack(testing_temp['input_values'])
test_tgt = torch.tensor([int(s) for s in testing_temp['labels']])
###################################################################################################

# Getting specific parameters required for feature extraction

# we define which pretrained model we want to use and instantiate a feature extractor
pretrained_model = "MIT/ast-finetuned-audioset-10-10-0.4593"

from transformers import ASTConfig, ASTForAudioClassification

# Load configuration from the pretrained model
config = ASTConfig.from_pretrained(pretrained_model)

### ADD CSV FILE TO THE MIX TO SMOOTH OUT NAMES AND LABEL2ID
names = ['PC1101','PC1102','PC1103','PC1104','PC1105','PC1106','PC1107','PC1108','PC1109','PC1110','PC1111','PC1112','PC1113',
'0212', '0312', '0712', '0811', '0911', '1011', '1211', '1511', '1611', '1711']
####

#### SMOOTH OUT
# Define label2id for the model
label2id = dict()
for i in range(23):
    label2id[names[i]] = i

# NEED TO EXPORT 'names' object
# Update configuration with the number of labels in our dataset
config.num_labels = 23 # SMOOTH OUT
config.label2id = label2id
config.id2label = {v: k for k, v in label2id.items()}

# Initialize the model with the updated configuration
model = ASTForAudioClassification.from_pretrained(pretrained_model, config=config, ignore_mismatched_sizes=True)


modules_to_freeze = [model.audio_spectrogram_transformer.encoder.layer[i] for i in range(11)]

for module in modules_to_freeze:
    for param in module.parameters():
        param.requires_grad = False

model.init_weights()

#for i in range(12):
#    layer = model.audio_spectrogram_transformer.encoder.layer[i]
#    print(all(not param.requires_grad for param in layer.attention.parameters()))
#    print(all(not param.requires_grad for param in layer.intermediate.parameters()))
#    print(all(not param.requires_grad for param in layer.output.parameters()))

model.gradient_checkpointing_enable()

################################################################


from torch.utils.data import DataLoader

num_epochs = 10
bs = 8
min_val_loss = 1000000

train_seq_dataloader = DataLoader(train_seq, batch_size=bs)
valid_seq_dataloader = DataLoader(valid_seq, batch_size=bs)
train_tgt_dataloader = DataLoader(train_tgt, batch_size = bs)
valid_tgt_dataloader = DataLoader(valid_tgt, batch_size = bs)

device = torch.device("cuda")
model.to(device)
loss = torch.nn.CrossEntropyLoss()
opt = torch.optim.AdamW(model.parameters(), lr = 1e-4)


for epoch in range(num_epochs):
    model.train()
    loss_val = 0
    for s, t in zip(train_seq_dataloader, train_tgt_dataloader):
        s = s.to(device)
        t = t.to(device)
        opt.zero_grad()
        preds = model(s)
        #print(epoch, preds.logits, t)
        celoss = loss(preds.logits, t)
        celoss.backward()
        opt.step()
        loss_val += celoss.item()
    avg_loss = loss_val/len(train_seq_dataloader)
    model.eval()


# Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        val_loss_val = 0
        for s, t in zip(valid_seq_dataloader, valid_tgt_dataloader):
            s = s.to(device)
            t = t.to(device)
            preds = model(s)
            #print(preds.logits.shape)
            #print(t, t.shape)
            celoss = loss(preds.logits, t)
            val_loss_val += celoss.item()
    val_avg_loss = val_loss_val / len(valid_seq_dataloader)
    if val_avg_loss < min_val_loss:
        min_val_loss = val_avg_loss
        best_model = model.state_dict()
    print(f'Epoch {epoch}, {avg_loss = }, {val_avg_loss = }')
    torch.cuda.empty_cache()

#torch.save(model.state_dict(), 'best_model_7.pt')

bs = 8
test_seq_dataloader = DataLoader(test_seq, batch_size=bs)
test_tgt_dataloader = DataLoader(test_tgt, batch_size=bs)

model.load_state_dict(best_model)
model.to(device)  # Ensure model is on the correct device
model.eval()

metric_1 = MulticlassAUROC(num_classes=23)
metric_2 = MulticlassAccuracy(average = 'none', num_classes = 23)

test_loss_val_1 = 0
test_loss_val_2 = 0
output = []

with torch.no_grad():  # Prevents unnecessary gradient tracking
    for s, t in zip(test_seq_dataloader, test_tgt_dataloader):
        s = s.to(device)
        t = t.to(device)

        with torch.cuda.amp.autocast():  # FP16 inference
            preds = model(s)

        output.append(preds.logits.cpu())  # Move to CPU to free GPU memory

        metric_1.update(preds.logits, t)
        metric_2.update(preds.logits, t)

        torch.cuda.empty_cache()  # Free GPU memory after each batch

# Compute metrics after all batches
test_loss_val_1 = metric_1.compute()
test_loss_val_2 = metric_2.compute()

print(f"Test AUROC: {test_loss_val_1:.4f}")
print(f"Test Accuracy: {test_loss_val_2}")