from data.preprocessors.sequence_preprocessing.sequence_preprocessors import RandomSequencePreprocessor
from models.long_short_term_memory.personal_trainer import LongShortTermMemoryPersonalTrainer
from models.long_short_term_memory.architecture import LSTMNetworkSG
from data.source_datasets.datasets import LBNL_FTP_PKTDatasetCombined
from torch.utils.data import DataLoader
from main.helper import load_model
import torch.nn as nn
import pickle
import torch
import os

"""
global variables for training purpose
"""
LOG_INTERVAL = 3
MODEL_SAVE_PATH = "LSTM_ftp.pt"
BACKBONE1_SAVE_PATH = "AE_ftp.pt"
BACKBONE2_SAVE_PATH = "SOM_AE_ftp.p"
NUM_EPOCHS = 2
DATA_LENGTH = 1024
BATCH_SIZE = 128
LEARNING_RATE = 0.005
ALPHABET_SIZE = 160

"""
get data
"""
# all the source datasets
source_dataset = LBNL_FTP_PKTDatasetCombined()
source_dataset.shuffle_dataset()

# split off validation and test datasets
training_dataset, test_dataset = source_dataset.split(0.75)
training_dataset, validation_dataset = training_dataset.split(0.66)

"""
create or load model
"""
model = load_model(MODEL_SAVE_PATH, LSTMNetworkSG)
backbone = []
if os.path.isfile(BACKBONE1_SAVE_PATH):
    backbone.append(torch.load(BACKBONE1_SAVE_PATH))
    print("loaded ", backbone[0])
with open(BACKBONE2_SAVE_PATH, 'rb') as infile:
    backbone.append(pickle.load(infile))
    print("loaded som")

# one preprocessor each
training_preprocessor = RandomSequencePreprocessor(training_dataset, ALPHABET_SIZE, DATA_LENGTH,
                                                   backbone[0], backbone[1], 16)
validation_preprocessor = RandomSequencePreprocessor(validation_dataset, ALPHABET_SIZE, DATA_LENGTH,
                                                     backbone[0], backbone[1], 16)
test_preprocessor = RandomSequencePreprocessor(test_dataset, ALPHABET_SIZE, DATA_LENGTH,
                                               backbone[0], backbone[1], 16)

# one dataloader each
training_dataloader = DataLoader(training_preprocessor, BATCH_SIZE, shuffle=True, drop_last=True)
validation_dataloader = DataLoader(validation_preprocessor, BATCH_SIZE, shuffle=True, drop_last=True)
test_dataloader = DataLoader(test_preprocessor, BATCH_SIZE, shuffle=True, drop_last=True)

"""
prepare teachers
"""
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.NLLLoss()

"""
run personal training
"""
lstmpt = LongShortTermMemoryPersonalTrainer(model, training_dataloader, validation_dataloader, LOG_INTERVAL,
                                            MODEL_SAVE_PATH, criterion, optimizer)

# lstmpt.run_training(num_epochs=NUM_EPOCHS)
lstmpt.set_testset(dataloader=test_dataloader)
lstmpt.finalize_test()
lstmpt.get_new_statements(num_classes=16, filename="sg_test_ftp")
