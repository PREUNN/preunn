from data.source_datasets.datasets import LBNL_FTP_PKTDataset1
from data.preprocessors.image_preprocessing.image_preprocessors import NormalImagePreprocessor
from torch.utils.data import DataLoader
from models.auto_encoder.architecture import AE
from models.auto_encoder.personal_trainer import AutoEncoderPersonalTrainer
from main.helper import load_model
import torch.nn as nn
import torch

"""
global variables for training purpose
"""
LOG_INTERVAL = 10
MODEL_SAVE_PATH = "AE_balanced_ftp.pt"
NUM_EPOCHS = 5
BATCH_SIZE = 128
DATA_LENGTH = 1024
LEARNING_RATE = 0.0005

"""
get data
"""
# all the source datasets
source_dataset = LBNL_FTP_PKTDataset1()
source_dataset.shuffle_dataset()
source_dataset.balance_dataset(class_limit=100)
source_preprocessor = NormalImagePreprocessor(source_dataset, DATA_LENGTH)

# one preprocessor each
train_preprocessor, test_preprocessor = source_preprocessor.split(0.75)
train_preprocessor, validation_preprocessor = train_preprocessor.split(0.66)

# one dataloader each
training_dataloader = DataLoader(train_preprocessor, BATCH_SIZE, shuffle=True, drop_last=True)
validation_dataloader = DataLoader(validation_preprocessor, BATCH_SIZE, shuffle=True, drop_last=True)
test_dataloader = DataLoader(test_preprocessor, BATCH_SIZE, shuffle=True, drop_last=True)

"""
create or load model
"""
model = load_model(MODEL_SAVE_PATH, AE())

"""
prepare teachers
"""
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.MSELoss()

"""
run personal training
"""
aept = AutoEncoderPersonalTrainer(model, training_dataloader, validation_dataloader, LOG_INTERVAL,
                                  MODEL_SAVE_PATH, criterion, optimizer)
aept.run_training(num_epochs=NUM_EPOCHS)
aept.set_testset(test_dataloader)
aept.finalize_test()
print("Hamming Distance Average: ", aept.get_hamming_metric(num_batches=8))
# Balanced Hamming: 41.28
