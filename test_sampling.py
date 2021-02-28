from main.helper import load_model
from data.source_datasets.datasets import AllHTTPDatasetsCombined
from data.source_datasets.datasets import LBNL_FTP_PKTDataset1
from data.preprocessors.sequence_preprocessing.sequence_preprocessors import NormalImagePreprocessor
from data.postprocessing.image_postprocessing.processing import image_tensor_to_string_list
from data.postprocessing.sequence_postprocessing.processing import sequence_tensor_to_string_list
from data.preprocessors.sequence_preprocessing.processing import one_hot_encode_string
from models.long_short_term_memory.architecture import LSTMNetworkSG
from models.long_short_term_memory.personal_trainer import LongShortTermMemoryPersonalTrainer
import torch.nn as nn
import torch.optim as op
from torch.utils.data import DataLoader
import torch


ftp_model_path = "D:\\Wissenschaft\\Projekte\\preunn\\main\\sequence_generation\\LSTM_balanced_ftp.pt"
http_model_path = "D:\\Wissenschaft\\Projekte\\preunn\\main\\sequence_generation\\LSTM_balanced_http.pt"
model_path = ftp_model_path
model_path = http_model_path

# all the source datasets
source_dataset = AllHTTPDatasetsCombined()
source_dataset, _ = source_dataset.split(0.01)
source_dataset.shuffle_dataset()
source_dataset.balance_dataset(class_limit=100)

# split off validation and test datasets
training_dataset, test_dataset = source_dataset.split(0.9)
training_dataset, validation_dataset = training_dataset.split(0.9)
tp = NormalImagePreprocessor(training_dataset, data_length=1024)


model = load_model(model_path, LSTMNetworkSG)
td = DataLoader(tp, 64, shuffle=True, drop_last=True)
lstmpt = LongShortTermMemoryPersonalTrainer(model, td, td, 1,
                                            model_path, nn.MSELoss(),
                                            op.Adam(model.parameters(), 0.005))

input_str = "Cookie: pcid=148515463850921542; NateMain=Loc=; SAVED_NATEID=%7C0; SSL_LOGIN=1; UD3=m5dcf6d0d9f67b1f0840e27f2b5ae03a; UA3=MTAwMDg0NjI4MjY=||; LOGIN=keeplogin=off&iplevel=2&loginid=; SVC=; Logout= "
input_tensor = one_hot_encode_string(input_str)
# input_tensor = [129, 129, 129, 129]
sample_sequence, _ = lstmpt.sample_statement(random_delimiter=3,
                                             length=380,
                                             data=torch.LongTensor(input_tensor).cuda().unsqueeze_(0))
print(sequence_tensor_to_string_list(sample_sequence))