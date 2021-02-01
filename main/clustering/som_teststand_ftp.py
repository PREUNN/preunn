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
INPUT_LENGTH = 1024
if "AE" in BACKBONE: INPUT_LENGTH = 128
if "CNN" in BACKBONE: INPUT_LENGTH = 240
LOG_INTERVAL = 100
MODEL_SAVE_PATH = "SOM_"+BACKBONE+"_ftp.p"
NUM_EPOCHS = 3
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
if "AE" in BACKBONE: backbone = load_model(BACKBONE+"_http.pt", AE())
if "CNN" in BACKBONE: backbone = load_model(BACKBONE+"_http.pt", CNN())
try:
    with open(MODEL_SAVE_PATH, 'rb') as infile:
        model = pickle.load(infile)
        print("loaded " + MODEL_SAVE_PATH)
except:
    print("New model created")
    model = MiniSom(1, 64, input_len=INPUT_LENGTH, sigma=3, learning_rate=0.005)

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
plt.matshow(accuracy_matrix)
plt.title("General Clustering")
plt.xlabel("FTP Types")
plt.ylabel("SOM Clusters")
plt.show()

# clusterwise share
clusterwise_matrix = metrics.get_clusterwise_share(accuracy_matrix)
plt.matshow(clusterwise_matrix)
plt.title("Clusterwise Share")
plt.xlabel("FTP Types")
plt.ylabel("SOM Clusters")
plt.show()

# typewise share
typewise_matrix = metrics.get_typewise_share(accuracy_matrix)
plt.matshow(typewise_matrix)
plt.title("Typewise Share")
plt.xlabel("FTP Types")
plt.ylabel("SOM Clusters")
plt.show()

conf = metrics.get_confident_cluster_metric(clusterwise_matrix)
relevant_conf = metrics.get_confident_cluster_metric(clusterwise_matrix, skip_zeros=True)

print("Confidence: " + str(conf)) # CNN 29.6875% / AE_balanced 79.6875% # Baseline 60.9375%
print("Relevant confidence: " + str(relevant_conf)) # CNN 29.6875% / AE_balanced 86.44% # Baseline 72.22%
np.savetxt("accuracy_matrix_" + BACKBONE + "_ftp.csv", accuracy_matrix, delimiter=",")
np.savetxt("clusterwise_matrix_" + BACKBONE + "_ftp.csv", clusterwise_matrix, delimiter=",")
np.savetxt("typewise_matrix_" + BACKBONE + "_ftp.csv", typewise_matrix, delimiter=",")
print("success")
