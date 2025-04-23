This repository contains codes to train and develop an Automatic Acoustic Individual Identification (AIID) for animals by conducting transfer learning on the Audio Spectrogram Transformer (AST). 
The data used to train this model can be found: https://zenodo.org/records/1413495 collected by Stowell et al. (2019) (https://doi.org/10.1111/2041-210X.13103).
First set up a venv and install the libraries in the requirements.txt file. 

# AST-Chiff
This AST model will be trained and tested on 13 individual Chiffchaffs from the dataset. In order to train the model, first download the data and organize the directory as:
```bash
Bird-audio/ 
    ├── chiffchaff-fg/ 
    │    │ 
    │    ├── cutted_day1_PC1101_0000.wav 
    │    │ 
    │    ├── cutted_day1_PC1101_0001.wav 
    │    │ 
    │    └── ... 
    └── csv/ 
         │ 
         ├── chiffchaff-acrossyear-fg-trn.csv 
         │ 
         ├── chiffchaff-withinyear-fg-trn.csv 
         │ 
         └── ... 
```


Make sure this Bird-audio is in the same folder as the codes.
Then set up a folder to store the spectrogram inputs from AST-data-init.py as follows:

```bash
dataset-inputs/ 
    ├── Train/  
    └── Test/ 
```
Then run the AST-data-init.py and run AST-chiff.py.

# AST-Pipits
This AST model will be trained and tested on 10 individual Pipits from the dataset. Using the same Bird-audio configuration create a new folder to store the spectrogram inputs from the pipit-data-init.py. 
```bash
dataset-input-pipit/ 
    ├── Train/  
    └── Test/ 
```

Run pipit-data-init.py and then run AST-pipit.py

# AST-mm

This configuration will be trained and tested on both Chiffchaff and Pipits recordings from the same year. Set up a directory in the following architecture. 

```bash
dataset-inputs-tot/ 
    ├── Train/  
    └── Test/ 
```

Then run mm-data-init.py and AST-mm.py.

# AST-ms

This confugration will train the AST on both Chiffchaff and Pipits but only test on Chiffchaffs. Run AST-ms.py directly. 

# resent-18-chiff

This file runs the codes to test how a CNN model would work under the same conditions for comparison on Chiffchaffs. 
Set up a new cnn-data folder to store the spectrogram files as as follows:

```bash
cnn-data/ 
    ├── Train/ 
    │    │ 
    │    ├── PC1101/ 
    │    │ 
    │    ├── PC1102/ 
    │    │ 
    │    └── ... 
    └── Test/ 
         │ 
         ├── PC1101/ 
         │ 
         ├── PC1102/ 
         │ 
         └── ... 
```

Run resnet18-chiff-data-init.py and then run resnet-18-chiff.py.

# resnet-18-pipit
This file runs the codes to test how a CNN model would work under the same conditions for comparison on Pipits. 
Set up a new cnn-data folder to store the spectrogram files as as follows:

```bash
cnn-data/ 
    ├── Train/ 
    │    │ 
    │    ├── 0212/ 
    │    │ 
    │    ├── 0312/ 
    │    │ 
    │    └── ... 
    └── Test/ 
         │ 
         ├── 0212/ 
         │ 
         ├── 0312/ 
         │ 
         └── ... 
```

Run resnet18-pipit-data-init.py and then run resnet-18-pipit.py.


