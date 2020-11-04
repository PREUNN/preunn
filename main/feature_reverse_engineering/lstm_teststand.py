from models.long_short_term_memory import *
from torch.utils.data import DataLoader
from main.helper import *
from data import *

"""
global variables for training purpose
"""
LOG_INTERVAL = 2
MODEL_SAVE_PATH = "LSTM.pt"
NUM_EPOCHS = 1
DATA_LENGTH = 1024
BATCH_SIZE = 128
LEARNING_RATE = 0.005
ALPHABET_SIZE = 130

"""
get data
"""
# all the source datasets
source_dataset = AllHTTPDatasetsCombined()
source_preprocessor = RandomSequencePreprocessor(source_dataset=source_dataset, alphabet_size=ALPHABET_SIZE, data_length=1024)
source_preprocessor.shuffle_dataset()

# one preprocessor each
train_preprocessor, test_preprocessor = source_preprocessor.split(split_value=0.75)
train_preprocessor, validation_preprocessor = train_preprocessor.split(split_value=0.66)

# one dataloader each
training_dataloader = DataLoader(dataset=train_preprocessor, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
validation_dataloader = DataLoader(dataset=validation_preprocessor, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
test_dataloader = DataLoader(dataset=test_preprocessor, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

"""
create or load model
"""
model = load_model(MODEL_SAVE_PATH, LSTMNetworkFRE)

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

# lstmpt.run_training(num_epochs=NUM_EPOCHS)
lstmpt.set_testset(dataloader=test_dataloader)
# lstmpt.finalize_test()
lstmpt.get_new_http_statements()
