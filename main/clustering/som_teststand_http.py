from models.convolutional_neural_network.architecture import *
from models.auto_encoder.architecture import *
from models.self_organizing_map import *
from torch.utils.data import DataLoader
from minisom import MiniSom
from main.helper import *
from data import *
import pickle


"""
global variables for training purpose
"""
LOG_INTERVAL = 300
MODEL_SAVE_PATH = "SOMAEkleinTest.p"
NUM_EPOCHS = 20
BATCH_SIZE = 128

"""
get data
"""
# all the source datasets
source_dataset = AllHTTPDatasetsCombined()
source_preprocessor = NormalImagePreprocessor(source_dataset=source_dataset, data_length=1024)
source_preprocessor.shuffle_dataset()

# one preprocessor each
train_preprocessor, test_preprocessor = source_preprocessor.split(split_value=0.9)

# one dataloader each
training_dataloader = DataLoader(dataset=train_preprocessor, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
test_dataloader = DataLoader(dataset=test_preprocessor, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

"""
create or load model
"""
backbone = load_model("AEimage_http.pt", AE)
# backbone = None
try:
    with open(MODEL_SAVE_PATH, 'rb') as infile:
        model = pickle.load(infile)
        print("loaded som")
except:
    model = MiniSom(1, 4, input_len=128, sigma=1, learning_rate=0.005)

"""
run personal training
"""
sompt = SelfOrganizingMapPersonalTrainer(model, training_dataloader, test_dataloader, LOG_INTERVAL,
                                         MODEL_SAVE_PATH, backbone)
sompt.run_training(num_epochs=NUM_EPOCHS)
with open(MODEL_SAVE_PATH, 'wb') as outfile:
    pickle.dump(model, outfile)

# metrics
accuracy_matrix = sompt.get_accuracy_matrix()
print("\n", accuracy_matrix)

copy = copy.deepcopy(accuracy_matrix)
# get clusterwise share
for i in range(len(accuracy_matrix)):
    accuracy_matrix[i] /= max(sum(accuracy_matrix[i]), 1)
accuracy_matrix = 1000 * accuracy_matrix
accuracy_matrix = np.rint(accuracy_matrix)
accuracy_matrix /= 10
print("\n", accuracy_matrix)

accuracy_matrix = copy
# get typewise share
for j in range(len(accuracy_matrix[0])):
    col_max = 0
    for i in range(len(accuracy_matrix)):
        col_max += accuracy_matrix[i][j]
    for i in range(len(accuracy_matrix)):
        accuracy_matrix[i][j] /= max(col_max, 1)
accuracy_matrix = 1000 * accuracy_matrix
accuracy_matrix = np.rint(accuracy_matrix)
accuracy_matrix /= 10
print("\n", accuracy_matrix)







