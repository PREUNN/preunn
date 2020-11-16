from models.self_organizing_map.helper import classify_http_statement, \
    get_clustering_accuracy
from data.postprocessing.image_postprocessing.processing \
    import image_tensor_to_string_list
import numpy as np
import torch


class SelfOrganizingMapPersonalTrainer:
    """
    Training and testing class for auto encoder
    """
    def __init__(self, model, training_data, test_data, log_interval,
                 model_save_path, backbone):
        self.model = model
        self.training_data = training_data
        self.test_data = test_data
        self.log_interval = log_interval
        self.model_save_path = model_save_path
        self.backbone = backbone

    def run_training(self, num_epochs: int):
        """
        training process
        :return: None
        """
        for epoch in range(num_epochs):
            print("epoch: ", epoch)
            self.train(epoch)
            self.test(epoch)
            torch.save(self.model, self.model_save_path)

    def train(self, epoch):
        """
        training subroutine for one epoch, prints out loss
        :param epoch: number of the current epoch
        :return: None
        """
        for batch_id, (data, _) in enumerate(self.training_data):
            if self.backbone:
                self.backbone.cuda()
                new_data = self.backbone.create_output(data.cuda()).squeeze(1)
            else:
                new_data = data.cuda().squeeze(1)
            arr = new_data.cpu().detach().numpy()
            self.model.train(arr, 100)
            if batch_id % self.log_interval == 0 and batch_id != 0:
                print("Epoch ", epoch, ", Batch ", batch_id, " of ",
                      len(self.training_data))

    def test(self, epoch):
        """
        testing subroutine, prints out loss
        :return: None
        """

    def get_http_accuracy_matrix(self):
        """
        Evaluation method for http clustering
        :return: Matrix of clustering
        """
        winner_list = []
        for _, (item, _) in enumerate(self.test_data):
            if self.backbone:
                self.backbone.cuda()
                new_data = self.backbone.create_output(item.cuda()).cpu()
            else:
                new_data = item.cpu().squeeze(1)

            winners = np.array([self.model.winner(x)[1]
                                for x in new_data.detach()])
            http_strings = image_tensor_to_string_list(item)
            winner_list.extend([(classify_http_statement(http), win)
                                for http, win in zip(http_strings, winners)])

        accuracy_matrix = get_clustering_accuracy(winner_list)

        # rounding to 3 digits
        accuracy_matrix = 1000 * accuracy_matrix
        accuracy_matrix = np.rint(accuracy_matrix)
        accuracy_matrix /= 1000
        return accuracy_matrix


