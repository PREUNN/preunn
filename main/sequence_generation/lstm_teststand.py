from models.long_short_term_memory import *
from torch.utils.data import DataLoader
from main.helper import *
from data import *
import pickle

"""
global variables for training purpose
"""
LOG_INTERVAL = 3
MODEL_SAVE_PATH = "LSTM.pt"
BACKBONE1_SAVE_PATH = "AEimage.pt"
BACKBONE2_SAVE_PATH = "SOMAE1.p"
NUM_EPOCHS = 2
DATA_LENGTH = 1024
BATCH_SIZE = 128
LEARNING_RATE = 0.005
ALPHABET_SIZE = 160

"""
get data
"""
# all the source datasets
source_dataset = AllHTTPDatasetsCombined()
source_dataset.shuffle_dataset()

# split off validation and test datasets
training_dataset, test_dataset = source_dataset.split(0.75)
training_dataset, validation_dataset = training_dataset.split(0.66)

"""
create or load model
"""
model = load_model(MODEL_SAVE_PATH, LSTMNetworkSG)
backbone = []
if os.path.isfile(BACKBONE1_SAVE_PATH):
    backbone.append(torch.load(BACKBONE1_SAVE_PATH))
    print("loaded ", backbone[0])
with open(BACKBONE2_SAVE_PATH, 'rb') as infile:
    backbone.append(pickle.load(infile))
    print("loaded som")

# one preprocessor each
training_preprocessor = RandomSequencePreprocessor(source_dataset=training_dataset, alphabet_size=ALPHABET_SIZE,
                                                   data_length=DATA_LENGTH, feature_extractor=backbone[0], som=backbone[1],
                                                   num_clusters=16)
validation_preprocessor = RandomSequencePreprocessor(source_dataset=validation_dataset, alphabet_size=ALPHABET_SIZE,
                                                     data_length=DATA_LENGTH, feature_extractor=backbone[0], som=backbone[1],
                                                     num_clusters=16)
test_preprocessor = RandomSequencePreprocessor(source_dataset=test_dataset, alphabet_size=ALPHABET_SIZE,
                                               data_length=DATA_LENGTH, feature_extractor=backbone[0], som=backbone[1],
                                               num_clusters=16)

# one dataloader each
training_dataloader = DataLoader(dataset=training_preprocessor, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
validation_dataloader = DataLoader(dataset=validation_preprocessor, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
test_dataloader = DataLoader(dataset=test_preprocessor, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

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
lstmpt.finalize_test()
# lstmpt.get_new_http_statements()
