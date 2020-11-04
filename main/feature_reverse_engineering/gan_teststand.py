from data.postprocessing.image_postprocessing.processing import *
from models.generative_adversarial_nets import *
from torch.utils.data import DataLoader
from main.helper import *
from data import *


"""
global variables for training purpose
"""
LOG_INTERVAL = 25
G_MODEL_SAVE_PATH = "Generator.pt"
D_MODEL_SAVE_PATH = "Discriminator.pt"
NUM_EPOCHS = 10
BATCH_SIZE = 128
LEARNING_RATE = 0.0005
BETAS = (0.5, 0.99)

"""
get data
"""
# all the source datasets
source_dataset = AllHTTPDatasetsCombined()
source_preprocessor = FuzzyImagePreprocessor(source_dataset=source_dataset, data_length=1024)
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
g_model = load_model(G_MODEL_SAVE_PATH, Generator)
d_model = load_model(D_MODEL_SAVE_PATH, Discriminator)

"""
prepare teachers
"""
g_optimizer = torch.optim.Adam(g_model.parameters(), lr=LEARNING_RATE, betas=BETAS)
d_optimizer = torch.optim.Adam(d_model.parameters(), lr=LEARNING_RATE, betas=BETAS)
criterion = nn.BCELoss()

"""
run personal training
"""
ganpt = GenerativeAdversarialNetPersonalTrainer(g_model, d_model, training_dataloader, validation_dataloader, LOG_INTERVAL,
                                                G_MODEL_SAVE_PATH, D_MODEL_SAVE_PATH, criterion, g_optimizer, d_optimizer)
ganpt.run_training(num_epochs=NUM_EPOCHS)
ganpt.set_testset(test_dataloader)
ganpt.finalize_test()
shape = [1, 1, 1024]
for _ in range(10):
    print(100*"#")
    print(image_tensor_to_string_list(ganpt.create_output(shape))[0])
