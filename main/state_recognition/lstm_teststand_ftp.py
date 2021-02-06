import torch.nn as nn
import pickle
import torch
import os
from data.source_datasets.datasets import LBNL_FTP_PKTDataset1, LBNL_FTP_PKTDataset2
from data.preprocessors.image_preprocessing.image_preprocessors \
    import ClusteringPreprocessor
from models.long_short_term_memory.architecture import LSTMNetworkSR
from models.long_short_term_memory.personal_trainer \
    import LongShortTermMemoryPersonalTrainer
from models.auto_encoder.architecture import AE
from torch.utils.data import DataLoader
from main.helper import load_model
from main.state_recognition import metrics
from numpy import genfromtxt


"""
global variables for training purpose
"""
PROTOCOL = "ftp"
LOG_INTERVAL = 2
MODEL_SAVE_PATH = "LSTM_"+PROTOCOL+".pt"
BACKBONE1_SAVE_PATH = "AE_balanced_"+PROTOCOL+".pt"
BACKBONE2_SAVE_PATH = "SOM_AE_balanced_"+PROTOCOL+".p"
NUM_EPOCHS = 5
BATCH_SIZE = 128
SEQ_LENGTH = 2
DATA_LENGTH = 1024


"""
get data
"""
# all the source datasets
source_dataset = LBNL_FTP_PKTDataset2()
# source_dataset.shuffle_dataset()
# source_dataset.balance_dataset(class_limit=100)
# no shuffling!

"""
create or load model and backbone
"""
backbone = []
if os.path.isfile(BACKBONE1_SAVE_PATH):
    backbone.append(load_model(BACKBONE1_SAVE_PATH, AE()))
with open(BACKBONE2_SAVE_PATH, 'rb') as infile:
    backbone.append(pickle.load(infile))
    print("loaded som " + BACKBONE2_SAVE_PATH)
model = load_model(MODEL_SAVE_PATH, LSTMNetworkSR(backbone[1].get_weights().shape[1], 50, 5))


source_preprocessor = ClusteringPreprocessor(source_dataset, DATA_LENGTH, feature_extractor=backbone[0],
                                             som=backbone[1], sequence_length=SEQ_LENGTH)

train_dataset, test_dataset = source_dataset.split(0.9)
train_dataset, validation_dataset = train_dataset.split(0.66)


train_preprocessor = ClusteringPreprocessor(train_dataset, DATA_LENGTH, feature_extractor=backbone[0],
                                            som=backbone[1], sequence_length=SEQ_LENGTH)
validation_preprocessor = ClusteringPreprocessor(validation_dataset, DATA_LENGTH, feature_extractor=backbone[0],
                                                 som=backbone[1], sequence_length=SEQ_LENGTH)
test_preprocessor = ClusteringPreprocessor(test_dataset, DATA_LENGTH, feature_extractor=backbone[0],
                                           som=backbone[1], sequence_length=SEQ_LENGTH)

# one dataloader each
training_dataloader = DataLoader(train_preprocessor, BATCH_SIZE, shuffle=False, drop_last=True)
validation_dataloader = DataLoader(validation_preprocessor, BATCH_SIZE, shuffle=False, drop_last=True)
test_dataloader = DataLoader(test_preprocessor, BATCH_SIZE, shuffle=False, drop_last=True)

"""
prepare teachers
"""
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, betas=(0.3, 0.9))
criterion = nn.MSELoss()

"""
run personal training
"""
lstmpt = LongShortTermMemoryPersonalTrainer(model, training_dataloader, validation_dataloader, LOG_INTERVAL,
                                            MODEL_SAVE_PATH, criterion, optimizer)
lstmpt.run_training(num_epochs=NUM_EPOCHS)
lstmpt.set_testset(dataloader=test_dataloader)
lstmpt.finalize_test()

"""
metrics
"""
# cluster prediction clusterwise
acc = genfromtxt('accuracy_matrix_AE_balanced_'+PROTOCOL+'.csv', delimiter=',')
cw_acc = metrics.get_cluster_prediction_clusterwise(lstmpt, acc)
print(cw_acc)
# cluster prediction typewise
