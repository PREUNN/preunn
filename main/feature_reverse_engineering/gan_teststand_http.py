from data.postprocessing.image_postprocessing.processing \
    import image_tensor_to_string_list
from data.source_datasets.datasets import AllHTTPDatasetsCombined
from data.preprocessors.image_preprocessing.image_preprocessors \
    import FuzzyImagePreprocessor
from models.generative_adversarial_nets.architecture import Generator, \
    Discriminator
from models.generative_adversarial_nets.personal_trainer \
    import GenerativeAdversarialNetPersonalTrainer
from torch.utils.data import DataLoader
from main.helper import load_model
import torch.nn as nn
import torch


"""
global variables for training purpose
"""
LOG_INTERVAL = 25
G_MODEL_SAVE_PATH = "Generator_http.pt"
D_MODEL_SAVE_PATH = "Discriminator_http.pt"
NUM_EPOCHS = 10
BATCH_SIZE = 128
LEARNING_RATE = 0.0005
BETAS = (0.5, 0.99)
DATA_LENGTH = 1024

"""
get data
"""
# all the source datasets
source_dataset = AllHTTPDatasetsCombined()
source_preprocessor = FuzzyImagePreprocessor(source_dataset, DATA_LENGTH)
source_preprocessor.shuffle_dataset()

# one preprocessor each
train_preprocessor, test_preprocessor = source_preprocessor.split(0.75)
train_preprocessor, validation_preprocessor = train_preprocessor.split(0.66)

# one dataloader each
training_dataloader = DataLoader(train_preprocessor, BATCH_SIZE, shuffle=True,
                                 drop_last=True)
validation_dataloader = DataLoader(validation_preprocessor, BATCH_SIZE,
                                   shuffle=True, drop_last=True)
test_dataloader = DataLoader(test_preprocessor, BATCH_SIZE, shuffle=True,
                             drop_last=True)
"""
create or load model
"""
g_model = load_model(G_MODEL_SAVE_PATH, Generator)
d_model = load_model(D_MODEL_SAVE_PATH, Discriminator)

"""
prepare teachers
"""
g_optimizer = torch.optim.Adam(g_model.parameters(), lr=LEARNING_RATE,
                               betas=BETAS)
d_optimizer = torch.optim.Adam(d_model.parameters(), lr=LEARNING_RATE,
                               betas=BETAS)
criterion = nn.BCELoss()

"""
run personal training
"""
ganpt = GenerativeAdversarialNetPersonalTrainer(g_model, d_model,
                                                training_dataloader,
                                                validation_dataloader,
                                                LOG_INTERVAL, G_MODEL_SAVE_PATH,
                                                D_MODEL_SAVE_PATH, criterion,
                                                g_optimizer, d_optimizer)
ganpt.run_training(num_epochs=NUM_EPOCHS)
ganpt.set_testset(test_dataloader)
ganpt.finalize_test()
shape = [1, 1, 1024]
for _ in range(10):
    print(100*"#")
    print(image_tensor_to_string_list(ganpt.create_output(shape))[0])
