from models.convolutional_neural_network.personal_trainer import ConvolutionalNeuralNetworkPersonalTrainer
from models.convolutional_neural_network.architecture import CNN
from data.source_datasets.datasets import LBNL_FTP_PKTDataset1
from data.preprocessors.image_preprocessing.image_preprocessors import ScrambledImagePreprocessor
from torch.utils.data import DataLoader
from main.helper import load_model
import torch.nn as nn
import torch

"""
global variables for training purpose
"""
LOG_INTERVAL = 100
MODEL_SAVE_PATH = "CNN_ftp.pt"
NUM_EPOCHS = 10
BATCH_SIZE = 128
DATA_LENGTH = 1024
LEARNING_RATE = 0.005

"""
get data
"""
# all the source datasets
source_dataset = LBNL_FTP_PKTDataset1()
source_preprocessor = ScrambledImagePreprocessor(source_dataset, DATA_LENGTH)
source_preprocessor.shuffle_dataset()

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
model = load_model(MODEL_SAVE_PATH, CNN())

"""
prepare teachers
"""
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

"""
run personal training
"""
cnnpt = ConvolutionalNeuralNetworkPersonalTrainer(model, training_dataloader, validation_dataloader, LOG_INTERVAL,
                                                  MODEL_SAVE_PATH, criterion, optimizer)
cnnpt.run_training(num_epochs=NUM_EPOCHS)
cnnpt.set_testset(test_dataloader)
cnnpt.finalize_test()
cnnpt.visualize(num_samples=10, display=True)
