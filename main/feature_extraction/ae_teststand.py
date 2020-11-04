from torch.utils.data import DataLoader
from models.auto_encoder import *
from data import *
from main.helper import *

"""
global variables for training purpose
"""
LOG_INTERVAL = 25
MODEL_SAVE_PATH = "AEimage.pt"
NUM_EPOCHS = 10
BATCH_SIZE = 128
ALPHABET_SIZE = 130
DATA_LENGTH = 1024
LEARNING_RATE = 0.0005

"""
get data
"""
# all the source datasets
source_dataset = AllHTTPDatasetsCombined()
source_dataset.shuffle_dataset()
source_preprocessor = NormalImagePreprocessor(source_dataset=source_dataset, data_length=DATA_LENGTH)

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
model = load_model(MODEL_SAVE_PATH, AE)

"""
prepare teachers
"""
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.MSELoss()

"""
run personal training
"""
aept = AutoEncoderPersonalTrainer(model, training_dataloader, validation_dataloader, LOG_INTERVAL,
                                  MODEL_SAVE_PATH, criterion, optimizer)
aept.run_training(num_epochs=NUM_EPOCHS)
aept.set_testset(test_dataloader)
aept.finalize_test()
print("Hamming Distance Average: ", aept.get_hamming_metric(num_batches=8))

