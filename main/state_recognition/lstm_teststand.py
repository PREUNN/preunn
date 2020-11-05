from data.source_datasets.datasets import IEEEHTTPDataset1, IEEEHTTPDataset2, \
    IEEEHTTPDataset3, NitrobaHTTPDataset
from data.preprocessors.image_preprocessing.image_preprocessors \
    import ClusteringPreprocessor
from models.long_short_term_memory.architecture import LSTMNetworkSR
from models.long_short_term_memory.personal_trainer \
    import LongShortTermMemoryPersonalTrainer
from torch.utils.data import DataLoader
from main.helper import load_model
import torch.nn as nn
import pickle
import torch
import os


"""
global variables for training purpose
"""
LOG_INTERVAL = 2
MODEL_SAVE_PATH = "LSTM_http.pt"
BACKBONE1_SAVE_PATH = "AEimage_http.pt"
BACKBONE2_SAVE_PATH = "SOMAE1_http.p"
NUM_EPOCHS = 100
BATCH_SIZE = 32
SEQ_LENGTH = 6
DATA_LENGTH = 1024

"""
get data
"""
# all the source datasets
# source_dataset = AllHTTPDatasetsCombined()
source_dataset = IEEEHTTPDataset1()
source_dataset.merge(IEEEHTTPDataset2())
source_dataset.merge(IEEEHTTPDataset3())
source_dataset.merge(NitrobaHTTPDataset())
# no shuffling!

train_dataset, test_dataset = source_dataset.split(split_value=0.75)
train_dataset, validation_dataset = train_dataset.split(split_value=0.66)


"""
create or load model and backbone
"""
model = load_model(MODEL_SAVE_PATH, LSTMNetworkSR)
backbone = []
if os.path.isfile(BACKBONE1_SAVE_PATH):
    backbone.append(torch.load(BACKBONE1_SAVE_PATH))
    print("loaded ", backbone[0])
with open(BACKBONE2_SAVE_PATH, 'rb') as infile:
    backbone.append(pickle.load(infile))
    print("loaded som")
train_preprocessor = ClusteringPreprocessor(train_dataset, DATA_LENGTH,
                                            backbone[0], backbone[1],
                                            SEQ_LENGTH)
validation_preprocessor = ClusteringPreprocessor(validation_dataset,
                                                 DATA_LENGTH, backbone[0],
                                                 backbone[1], SEQ_LENGTH)
test_preprocessor = ClusteringPreprocessor(test_dataset, DATA_LENGTH,
                                           backbone[0], backbone[1], SEQ_LENGTH)

# one dataloader each
training_dataloader = DataLoader(train_preprocessor, BATCH_SIZE, shuffle=False,
                                 drop_last=True)
validation_dataloader = DataLoader(validation_preprocessor, BATCH_SIZE,
                                   shuffle=False, drop_last=True)
test_dataloader = DataLoader(test_preprocessor, BATCH_SIZE, shuffle=False,
                             drop_last=True)

"""
prepare teachers
"""
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, betas=(0.3, 0.9))
criterion = nn.CrossEntropyLoss()

"""
run personal training
"""
lstmpt = LongShortTermMemoryPersonalTrainer(model, training_dataloader,
                                            validation_dataloader, LOG_INTERVAL,
                                            MODEL_SAVE_PATH, criterion,
                                            optimizer)
lstmpt.run_training(num_epochs=NUM_EPOCHS)
lstmpt.set_testset(dataloader=test_dataloader)
lstmpt.finalize_test()
lstmpt.get_cluster_prediction_metric()
