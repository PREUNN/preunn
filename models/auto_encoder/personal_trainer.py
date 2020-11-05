from data.postprocessing.sequence_postprocessing.processing \
    import sequence_tensor_to_string_list
from data.postprocessing.image_postprocessing.processing \
    import image_tensor_to_string_list
from models.abstract_personaltrainer import AbstractPersonalTrainer
from torch.autograd import Variable
import torch

class AutoEncoderPersonalTrainer(AbstractPersonalTrainer):
    """
    Training and testing class for auto encoder
    """
    def __init__(self, model, training_data, test_data, log_interval,
                 model_save_path, criterion, optimizer):
        super().__init__(model, training_data, test_data, log_interval,
                         model_save_path, criterion, optimizer)
        return

    def train(self, epoch):
        """
        training subroutine for one epoch, prints out loss
        :param epoch: number of the current epoch
        :return: None
        """
        self.model.train()
        train_loss = 0

        # iterating over dataset
        for batch_id, (data, _) in enumerate(self.training_data):
            loss = 0
            data = Variable(data.to(self.device))

            # getting loss
            self.optimizer.zero_grad()
            output = self.model(data)
            ""
            # loss = self.criterion(output, data)
            for c in range(data.shape[1]):
                loss += self.criterion(output[:, c], data[:, c])
            loss /= data.shape[1]
            ""
            loss.backward()
            train_loss += loss.item()
            self.optimizer.step()

            # logging interval
            if batch_id % self.log_interval == 0 and batch_id != 0:
                train_loss /= self.log_interval
                self.print_training_loss(epoch=epoch, batch_id=batch_id,
                                         batch_size=len(data), loss=train_loss)
                print(image_tensor_to_string_list(data[0].unsqueeze(0)))
                print(image_tensor_to_string_list(output[0].unsqueeze(0)))
                self.model.store_model(self.model_save_path)
            if batch_id % self.log_interval == 0:
                train_loss = 0
        return

    def test(self, epoch):
        """
        testing subroutine, prints out loss
        :return: None
        """
        self.model.eval()
        test_loss = 0
        with torch.no_grad():

            # iterating over dataset
            for batch_id, (data, _) in enumerate(self.test_data):
                loss = 0
                data = Variable(data.to(self.device))

                # getting loss
                output = self.model(data)
                ""
                # loss = self.criterion(output, data)
                for c in range(data.shape[1]):
                    loss += self.criterion(output[:, c], data[:, c])
                loss /= data.shape[1]
                ""
                test_loss += loss.item()

                # logging interval
                if batch_id % self.log_interval == 0 and batch_id != 0:
                    test_loss /= self.log_interval
                    self.print_test_loss(epoch=epoch, batch_id=batch_id,
                                         batch_size=len(data), loss=test_loss)
                    test_loss = 0
        self.model.train()
        return

    def get_hamming_metric(self, num_batches: int):
        """
        Evaluation metric for auto encoder
        :param num_batches: Number of batches to check
        :return: Average hamming distance of the auto encoder
        """
        self.model.eval()
        sum_hamming_distance = 0
        with torch.no_grad():

            # iterating over testdata
            for batch_id, (item, _) in enumerate(self.test_data):
                item = item.to(self.device)
                output = self.model(item)
                output_string = image_tensor_to_string_list(output)
                item_string = image_tensor_to_string_list(item)

                # getting hamming distance for each data sample in string form
                for i in range(len(item_string)):
                    sum_hamming_distance += sum(c1 != c2 for c1, c2
                                                in zip(item_string[i],
                                                       output_string[i]))
                if batch_id == num_batches-1:
                    break
        self.model.train()

        return sum_hamming_distance / (len(item) * num_batches)

