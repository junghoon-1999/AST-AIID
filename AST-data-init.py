import numpy as np
from datasets import Dataset, Audio, ClassLabel, Features
import pandas as pd
from os import listdir
from os.path import isfile, join
import torch
from transformers import ASTFeatureExtractor
# # Import records info from the training set and testing set 
train_records = pd.read_csv('./Bird-audio/csv/chiffchaff-withinyear-fg-trn.csv')
test_records = pd.read_csv('./Bird-audio/csv/chiffchaff-withinyear-fg-tst.csv')

# Obrain unique names given to individual birds
names = list(train_records.columns[1:])

# Attain dummy matrix of the records
def generate_dummy(df):

    # Remove first header 
    dummy_cols = df.columns[1:]

    # Impute the null values as 0 
    dummy_dat = df[dummy_cols].fillna(0)
    dummy_dat = dummy_dat.to_numpy()
    return dummy_dat

# Generate dummy matrix for training and testing dataset 
dummy_train = generate_dummy(train_records)
dummy_test = generate_dummy(test_records)

# Define classes
class_labels = ClassLabel(names = names)

# Define features with audio and label columns
features = Features({
    "audio": Audio(),  # Define the audio feature
    "labels": class_labels # Assign the class labels
})


# Constructing audio data
filepath = './Bird-audio/chiffchaff-fg/'

# filepath = os.path.join(DATA_DIR,"chiffchaff-fg")

audios_train = [filepath + f for f in listdir(filepath) if isfile(join(filepath, f)) and 'day1' in f and 'aug' not in f  and f.endswith('.wav')]
audios_test = [filepath + f for f in listdir(filepath) if isfile(join(filepath, f)) and 'day2' in f and 'aug' not in f and f.endswith('.wav')]

labels_train = []
# Setting up labels
for files in audios_train:
    for i in names:
        if i in files:
            labels_train.append(i)
            continue
labels_test = []
for files in audios_test:
    for i in names:
        if i in files:
            labels_test.append(i)
            continue

# Construct the dataset from a dictionary
dataset_train = Dataset.from_dict({
    "audio": audios_train,
    "labels": labels_train
}, features=features).cast_column('audio', Audio())

dataset_test = Dataset.from_dict({
    "audio": audios_test,
    "labels": labels_test
}, features=features).cast_column('audio', Audio())

dataset_train = dataset_train.rename_column("audio", "input_values")
dataset_test = dataset_test.rename_column('audio', 'input_values')

# we define which pretrained model we want to use and instantiate a feature extractor
pretrained_model = "MIT/ast-finetuned-audioset-10-10-0.4593"
feature_extractor = ASTFeatureExtractor.from_pretrained(pretrained_model)

# we save model input name and sampling rate for later use
model_input_name = feature_extractor.model_input_names[0]  # key -> 'input_values'
SAMPLING_RATE = feature_extractor.sampling_rate

def preprocess_audio(batch):
    wavs = [audio["array"] for audio in batch["input_values"]]
    # inputs are spectrograms as torch.tensors now
    inputs = feature_extractor(wavs, sampling_rate=SAMPLING_RATE, return_tensors="pt")

    output_batch = {model_input_name: inputs.get(model_input_name), "labels": list(batch["labels"])}
    return output_batch

# Create function to generalize to dataset_train, dataset_test
def spectrogram_generator(dataset):
    # calculate values for normalization
    feature_extractor.do_normalize = False  # we set normalization to False in order to calculate the mean + std of the dataset
    mean = []
    std = []

    # we use the transformation w/o augmentation on the training dataset to calculate the mean + std
    dataset.set_transform(preprocess_audio, output_all_columns=False)
    for i, (audio_input, labels) in enumerate(dataset):
        cur_mean = torch.mean(dataset[i][audio_input])
        cur_std = torch.std(dataset[i][audio_input])
        mean.append(cur_mean)
        std.append(cur_std)
    feature_extractor.mean = np.mean(mean)
    feature_extractor.std = np.mean(std)
    feature_extractor.do_normalize = True

    # Apply the transformation to the dataset  # rename audio column
    dataset.set_transform(preprocess_audio, output_all_columns=False)
    return dataset


# Generate spectrogram that has been normalized on the training dataset 
dataset_train = spectrogram_generator(dataset_train)

dataset_test = spectrogram_generator(dataset_test)

print('spectrogram complete')

# Seperating the pytroch array values from the training dataset 
start = 0
cache = dict()
for i in range(len(dataset_train)):
    if len(str(dataset_train[i]['labels'])) == 1:
        label_name = '0' + str(dataset_train[i]['labels'])
    else:
        label_name = str(dataset_train[i]['labels'])
    if dataset_train[i]['labels'] in cache.keys():
        cache[dataset_train[i]['labels']] += 1
        start = cache[dataset_train[i]['labels']]
        torch.save(dataset_train[i]['input_values'], f'./dataset-inputs/Train/audio_arrays_{start}_' + label_name + '.pt')
        start += 1
    else:
        cache[dataset_train[i]['labels']] = 0
        start = cache[dataset_train[i]['labels']]
        torch.save(dataset_train[i]['input_values'], f'./dataset-inputs/Train/audio_arrays_{start}_' + label_name + '.pt')

print('exported training spectrograms as pt files ')

# Seperating the pytroch array values from the training dataset 
start = 0
cache = dict()
for i in range(len(dataset_test)):
    if len(str(dataset_test[i]['labels'])) == 1:
        label_name = '0' + str(dataset_test[i]['labels'])
    else:
        label_name = str(dataset_test[i]['labels'])
    if dataset_test[i]['labels'] in cache.keys():
        cache[dataset_test[i]['labels']] += 1
        start = cache[dataset_test[i]['labels']]
        torch.save(dataset_test[i]['input_values'], f'./dataset-inputs/Test/audio_arrays_{start}_' + label_name + '.pt')
    else:
        cache[dataset_test[i]['labels']] = 0
        start = cache[dataset_test[i]['labels']]
        torch.save(dataset_test[i]['input_values'], f'./dataset-inputs/Test/audio_arrays_{start}_' + label_name + '.pt')