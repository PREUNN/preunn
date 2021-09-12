from data.source_datasets.datasets import LBNL_FTP_PKTDataset1
from data.preprocessors.image_preprocessing.image_preprocessors import NormalImagePreprocessor, ClusteringPreprocessor
from data.postprocessing.image_postprocessing.processing import image_tensor_to_string_list

from models.auto_encoder.architecture import AE
from models.long_short_term_memory.architecture import LSTMNetworkSR

from main.helper import load_model

import torch.nn as nn
import pickle
import torch
from torch.utils.data import DataLoader
import os


source_dataset = LBNL_FTP_PKTDataset1()
sd, _ = source_dataset.split(0.00001) # the first about 55 samples
print("####### original")
for each in sd:
    print(each)
print(10*'#')

AE_SAVE_PATH_CLST = "/main/clustering/AE_balanced_http.pt"
SOM_SAVE_PATH_CLST = "/main/clustering/SOM_AE_balanced_ftp.p"
AE_SAVE_PATH_SR = "/main/state_recognition/AE_balanced_ftp.pt"
SOM_SAVE_PATH_SR = "/main/state_recognition/SOM_AE_balanced_ftp.p"


ae = load_model(AE_SAVE_PATH_CLST, AE())
with open(SOM_SAVE_PATH_SR, 'rb') as infile:
    som = pickle.load(infile)

nip = NormalImagePreprocessor(sd, data_length=1024)
tdl_nip = DataLoader(nip, batch_size=1, shuffle=True, drop_last=True)

print("##### nip")
for (each, _) in nip:
    print(each)
    print(image_tensor_to_string_list(each))
    print(ae.create_output(each.cuda()))
print(10*'#')

cp = ClusteringPreprocessor(sd, data_length=1024, feature_extractor=ae, som=som, sequence_length=2)
tdl_cp = DataLoader(cp, batch_size=1, shuffle=False, drop_last=True)

for _, (a, _) in enumerate(tdl_cp):
    print(a)

for _, (b, _) in enumerate(tdl_nip):
    ae.cuda()
    new_data = ae.create_output(b.cuda()).squeeze(1)
    b = ae.create_output(b.cuda())
    print(som.winner(new_data.cpu().detach()))


