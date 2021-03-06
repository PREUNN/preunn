import pickle
import numpy as np
import main.clustering.metrics as metrics
from data.source_datasets.datasets import LBNL_FTP_PKTDataset1
from data.preprocessors.image_preprocessing.image_preprocessors import NormalImagePreprocessor
from models.convolutional_neural_network.architecture import CNN
from models.auto_encoder.architecture import AE
from models.self_organizing_map.personal_trainer import SelfOrganizingMapPersonalTrainer
from torch.utils.data import DataLoader
from minisom import MiniSom
from main.helper import load_model
from matplotlib import pyplot as plt

"""
global variables for training purpose
"""
BACKBONE = "AE_balanced"
PROTOCOL = "ftp"
INPUT_LENGTH = 1024
if "AE" in BACKBONE: INPUT_LENGTH = 128
if "CNN" in BACKBONE: INPUT_LENGTH = 240
LOG_INTERVAL = 100
MODEL_SAVE_PATH = "SOM_"+BACKBONE+"_"+PROTOCOL+".p"
NUM_EPOCHS = 10
BATCH_SIZE = 128

"""
get data
"""
# all the source datasets
source_dataset = LBNL_FTP_PKTDataset1()
source_dataset.shuffle_dataset()
source_dataset.balance_dataset(class_limit=100)
source_preprocessor = NormalImagePreprocessor(source_dataset, data_length=1024)

# one preprocessor each
train_preprocessor, test_preprocessor = source_preprocessor.split(0.975)

# one dataloader each
training_dataloader = DataLoader(train_preprocessor, BATCH_SIZE, shuffle=True, drop_last=True)
test_dataloader = DataLoader(test_preprocessor, BATCH_SIZE, shuffle=True, drop_last=True)

"""
create or load model
"""
backbone = None
if "AE" in BACKBONE: backbone = load_model(BACKBONE+"_"+PROTOCOL+".pt", AE())
if "CNN" in BACKBONE: backbone = load_model(BACKBONE+"_"+PROTOCOL+".pt", CNN())
try:
    with open(MODEL_SAVE_PATH, 'rb') as infile:
        model = pickle.load(infile)
        print("loaded " + MODEL_SAVE_PATH)
except:
    print("New model created")
    model = MiniSom(1, 64, input_len=INPUT_LENGTH, sigma=3, learning_rate=0.005)
NUM_CLUSTERS = model.get_weights().shape[1]
"""
run personal training
"""
sompt = SelfOrganizingMapPersonalTrainer(model, training_dataloader, test_dataloader, LOG_INTERVAL,
                                         MODEL_SAVE_PATH, backbone)
# sompt.run_training(num_epochs=NUM_EPOCHS)
# with open(MODEL_SAVE_PATH, 'wb') as outfile:
#     pickle.dump(model, outfile)

"""
metrics
"""
# accuracy matrix
accuracy_matrix = metrics.get_accuracy_matrix(sompt)
np.savetxt("accuracy_matrix_" + BACKBONE + "_"+PROTOCOL+".csv", accuracy_matrix, delimiter=",")
plt.matshow(accuracy_matrix)
plt.title("General Clustering")
plt.xlabel(PROTOCOL+" Types")
plt.ylabel("SOM Clusters")
plt.show()

# clusterwise share
clusterwise_matrix = metrics.get_clusterwise_share(accuracy_matrix)
np.savetxt("clusterwise_matrix_" + BACKBONE + "_"+PROTOCOL+".csv", clusterwise_matrix, delimiter=",")
plt.matshow(clusterwise_matrix)
plt.title("Clusterwise Share")
plt.xlabel(PROTOCOL+" Types")
plt.ylabel("SOM Clusters")
plt.show()

# typewise share
typewise_matrix = metrics.get_typewise_share(accuracy_matrix)
np.savetxt("typewise_matrix_" + BACKBONE + "_"+PROTOCOL+".csv", typewise_matrix, delimiter=",")
plt.matshow(typewise_matrix)
plt.title("Typewise Share")
plt.xlabel(PROTOCOL+" Types")
plt.ylabel("SOM Clusters")
plt.show()

acc = metrics.get_accuracy_metric(clusterwise_matrix)
relevant_acc = metrics.get_accuracy_metric(clusterwise_matrix, skip_zeros=True)

print("Accuracy: " + str(acc)) # CNN 29.6875% / AE_balanced 67.1875% # Baseline 60.9375%
print("Relevant Accuracy: " + str(relevant_acc)) # CNN 29.6875% / AE_balanced 86% # Baseline 72.22%
print("success")
