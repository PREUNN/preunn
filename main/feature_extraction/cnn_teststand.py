from models.convolutional_neural_network import *
from torch.utils.data import DataLoader
from main.helper import *
from data import *

"""
global variables for training purpose
"""
LOG_INTERVAL = 10
MODEL_SAVE_PATH = "CNNfinal.pt"
NUM_EPOCHS = 10
BATCH_SIZE = 128
DATA_LENGTH = 1024

"""
get data
"""
# all the source datasets
source_dataset = AllHTTPDatasetsCombined()
source_preprocessor = ScrambledImagePreprocessor(source_dataset=source_dataset, data_length=DATA_LENGTH)
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
model = load_model(MODEL_SAVE_PATH, CNN)

"""
prepare teachers
"""
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
criterion = nn.CrossEntropyLoss()

"""
run personal training
"""
cnnpt = ConvolutionalNeuralNetworkPersonalTrainer(model, training_dataloader, validation_dataloader,
                                                  LOG_INTERVAL, MODEL_SAVE_PATH, criterion, optimizer)
cnnpt.run_training(num_epochs=NUM_EPOCHS)
cnnpt.set_testset(test_dataloader)
cnnpt.finalize_test()
cnnpt.visualize(num_samples=10, display=True)
