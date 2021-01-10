import torch


class SelfOrganizingMapPersonalTrainer:
    """
    Training and testing class for auto encoder
    """
    def __init__(self, model, training_data, test_data, log_interval, model_save_path, backbone):
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
            new_data = self.get_new_data(data)
            arr = new_data.detach().numpy()
            self.model.train(arr, 100)
            if batch_id % self.log_interval == 0 and batch_id != 0:
                print("Epoch ", epoch, ", Batch ", batch_id, " of ", len(self.training_data))

    def test(self, epoch):
        pass

    def get_new_data(self, data):
        """
        Returns data after backbone preprocessing if needed.
        :param data: Input data.
        return: New data after backbone if given.
        """
        if self.backbone:
            self.backbone.cuda()
            new_data = self.backbone.create_output(data.cuda()).squeeze(1)
        else:
            new_data = data.cuda().squeeze(1)
        return new_data.cpu()
