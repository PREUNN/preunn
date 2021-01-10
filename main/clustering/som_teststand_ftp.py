import pickle
import main.clustering.metrics as metrics
from data.source_datasets.datasets import LBNL_FTP_PKTDataset1
from data.preprocessors.image_preprocessing.image_preprocessors import NormalImagePreprocessor
from models.auto_encoder.architecture import AE
from models.self_organizing_map.personal_trainer import SelfOrganizingMapPersonalTrainer
from torch.utils.data import DataLoader
from minisom import MiniSom
from main.helper import load_model
from matplotlib import pyplot as plt


"""
global variables for training purpose
"""
LOG_INTERVAL = 100
MODEL_SAVE_PATH = "SOMAE1_ftp.p"
NUM_EPOCHS = 5
BATCH_SIZE = 128

"""
get data
"""
# all the source datasets
source_dataset = LBNL_FTP_PKTDataset1()
source_preprocessor = NormalImagePreprocessor(source_dataset, data_length=1024)
source_preprocessor.shuffle_dataset()

# one preprocessor each
train_preprocessor, test_preprocessor = source_preprocessor.split(0.975)

# one dataloader each
training_dataloader = DataLoader(train_preprocessor, BATCH_SIZE, shuffle=True, drop_last=True)
test_dataloader = DataLoader(test_preprocessor, BATCH_SIZE, shuffle=True, drop_last=True)

"""
create or load model
"""
backbone = load_model("AEimage_ftp.pt", AE())
# backbone = None
try:
    with open(MODEL_SAVE_PATH, 'rb') as infile:
        model = pickle.load(infile)
        print("loaded som")
except:
    model = MiniSom(1, 64, input_len=128, sigma=3, learning_rate=0.005)

"""
run personal training
"""
sompt = SelfOrganizingMapPersonalTrainer(model, training_dataloader, test_dataloader, LOG_INTERVAL,
                                         MODEL_SAVE_PATH, backbone)
sompt.run_training(num_epochs=NUM_EPOCHS)
with open(MODEL_SAVE_PATH, 'wb') as outfile:
    pickle.dump(model, outfile)

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

print("Confidence: " + str(conf))
print("Relevant confidence: " + str(relevant_conf))
with open("accuracy_matrix_ftp.p", 'wb') as outfile:
    pickle.dump(accuracy_matrix, outfile)
with open("clusterwise_matrix_ftp.p", 'wb') as outfile:
    pickle.dump(clusterwise_matrix, outfile)
with open("typewise_matrix_ftp.p", 'wb') as outfile:
    pickle.dump(typewise_matrix, outfile)
print("success")
