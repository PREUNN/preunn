from data.preprocessors.sequence_preprocessing.sequence_preprocessors import RandomSequencePreprocessor
from models.long_short_term_memory.personal_trainer import LongShortTermMemoryPersonalTrainer
from models.long_short_term_memory.architecture import LSTMNetworkSG
from models.auto_encoder.architecture import AE
from data.source_datasets.datasets import LBNL_FTP_PKTDataset1
from torch.utils.data import DataLoader
from main.helper import load_model
import torch.nn as nn
import pickle
import torch


"""
global variables for training purpose
"""
LOG_INTERVAL = 1
MODEL_SAVE_PATH = "LSTM_balanced_ftp.pt"
BACKBONE1_SAVE_PATH = "AE_balanced_ftp.pt"
BACKBONE2_SAVE_PATH = "SOM_AE_balanced_ftp.p"
NUM_EPOCHS = 1
DATA_LENGTH = 1024
BATCH_SIZE = 128
LEARNING_RATE = 0.005


"""
get data
"""
# all the source datasets
source_dataset = LBNL_FTP_PKTDataset1()
source_dataset.shuffle_dataset()
source_dataset.balance_dataset(class_limit=100)

# split off validation and test datasets
training_dataset, test_dataset = source_dataset.split(0.75)
training_dataset, validation_dataset = training_dataset.split(0.66)

"""
create or load model
"""
backbone = []
if "AE" in BACKBONE1_SAVE_PATH: backbone.append(load_model(BACKBONE1_SAVE_PATH, AE()))
with open(BACKBONE2_SAVE_PATH, 'rb') as infile:
    backbone.append(pickle.load(infile))
    print("loaded som " + BACKBONE2_SAVE_PATH)

NUM_CLUSTERS = backbone[1].get_weights().shape[1]
ALPHABET_SIZE = 128 + 2 * NUM_CLUSTERS

model = load_model(MODEL_SAVE_PATH, LSTMNetworkSG(num_classes=NUM_CLUSTERS, num_hidden=100, num_layers=1))

# one preprocessor each
training_preprocessor = RandomSequencePreprocessor(training_dataset, ALPHABET_SIZE, DATA_LENGTH,
                                                   backbone[0], backbone[1], num_clusters=NUM_CLUSTERS)
validation_preprocessor = RandomSequencePreprocessor(validation_dataset, ALPHABET_SIZE, DATA_LENGTH,
                                                     backbone[0], backbone[1], num_clusters=NUM_CLUSTERS)
test_preprocessor = RandomSequencePreprocessor(test_dataset, ALPHABET_SIZE, DATA_LENGTH,
                                               backbone[0], backbone[1], num_clusters=NUM_CLUSTERS)

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
# lstmpt.finalize_test()
lstmpt.get_new_statements(num_classes=NUM_CLUSTERS, filename="sg_test_ftp")
