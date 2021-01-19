from models.long_short_term_memory.architecture import LSTMNetworkFRE
from models.long_short_term_memory.personal_trainer import LongShortTermMemoryPersonalTrainer
from data.source_datasets.datasets import LBNL_FTP_PKTDataset1
from data.preprocessors.sequence_preprocessing.sequence_preprocessors import RandomSequencePreprocessor
from torch.utils.data import DataLoader
from main.helper import load_model
import torch.nn as nn
import torch

"""
global variables for training purpose
"""
LOG_INTERVAL = 2
MODEL_SAVE_PATH = "LSTM_balanced_ftp.pt"
NUM_EPOCHS = 1
DATA_LENGTH = 1024
BATCH_SIZE = 128
LEARNING_RATE = 0.005
ALPHABET_SIZE = 130

"""
get data
"""
# all the source datasets
source_dataset = LBNL_FTP_PKTDataset1()
source_dataset.shuffle_dataset()
source_dataset.balance_dataset(class_limit=100)
source_preprocessor = RandomSequencePreprocessor(source_dataset, ALPHABET_SIZE, DATA_LENGTH)

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
model = load_model(MODEL_SAVE_PATH, LSTMNetworkFRE())

"""
prepare teachers
"""
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.NLLLoss()

"""
run personal training
"""
lstmpt = LongShortTermMemoryPersonalTrainer(model, training_dataloader, test_dataloader, LOG_INTERVAL,
                                            MODEL_SAVE_PATH, criterion, optimizer)

lstmpt.run_training(num_epochs=NUM_EPOCHS)
lstmpt.set_testset(dataloader=test_dataloader)
lstmpt.finalize_test()
lstmpt.get_new_statements(num_classes=1, filename="fre_test_ftp")
#  100% FTP
