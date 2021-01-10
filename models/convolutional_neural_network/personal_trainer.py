from data.postprocessing.image_postprocessing.processing import image_tensor_to_string_list
from models.abstract_personaltrainer import AbstractPersonalTrainer
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import torch


class ConvolutionalNeuralNetworkPersonalTrainer(AbstractPersonalTrainer):
    """
    Training and testing class for convolutional neural networks
    """
    def __init__(self, model, training_data, test_data, log_interval, model_save_path, criterion, optimizer):

        super().__init__(model, training_data, test_data, log_interval, model_save_path, criterion, optimizer)
        self.fig2, (self.ax3, self.ax4) = plt.subplots(2, sharey=True)
        self.ax3.set(xlabel="Iterations", ylabel="%")
        self.ax3.set_title("Training Accuracy")
        self.ax4.set(xlabel="Iterations", ylabel="%")
        self.ax4.set_title("Validation Accuracy")
        self.training_accuracies = []
        self.test_accuracies = []
        return

    def train(self, epoch):
        """
        training subroutine for one epoch, prints out loss
        :param epoch: number of the current epoch
        :return: None
        """
        self.model.train()
        train_logging_loss = 0
        correct = 0

        # iterating over dataset
        for batch_id, (data, label) in enumerate(self.training_data):
            data = Variable(data.to(self.device))
            label = Variable(label.to(self.device)).squeeze_()

            # getting loss
            self.optimizer.zero_grad()
            out = self.model(data)
            loss = self.criterion(out, label)
            train_logging_loss += loss.item()

            # getting predictions
            prediction = out.data.max(1, keepdim=True)[1]
            prediction = prediction.squeeze_()
            correct += prediction.eq(label).sum()
            loss.backward()
            self.optimizer.step()

            # logging interval
            if batch_id % self.log_interval == 0 and batch_id != 0:
                train_logging_loss /= self.log_interval
                self.print_training_loss(epoch=epoch, batch_id=batch_id, batch_size=len(data), loss=train_logging_loss)
                accuracy = (100. * correct / (self.log_interval * len(data))).item()
                print("Train Accuracy: ", accuracy)
                self.training_accuracies.append(accuracy)
                train_logging_loss = 0
                correct = 0

        # plotting accuracy
        self.ax3.plot(self.training_accuracies, "tab:blue")
        self.ax3.set_ylim(top=100, bottom=0)
        return

    def test(self, epoch):
        self.model.eval()
        """
        testing subroutine, prints out loss
        :return: None
        """
        with torch.no_grad():
            test_logging_loss = 0
            correct = 0

            # iterating over dataset
            for batch_id, (data, label) in enumerate(self.test_data):
                data = Variable(data.to(self.device))
                label = Variable(label.to(self.device)).squeeze_()

                # getting loss
                out = self.model(data)
                loss = self.criterion(out, label).data
                test_logging_loss += loss.item()

                # getting predictions
                prediction = out.data.max(1, keepdim=True)[1]
                prediction = prediction.squeeze_()
                correct += prediction.eq(label).sum()

                # logging interval
                if batch_id % self.log_interval == 0 and batch_id != 0:
                    test_logging_loss /= self.log_interval
                    self.print_test_loss(epoch=epoch, batch_id=batch_id, batch_size=len(data), loss=test_logging_loss)
                    accuracy = (100. * correct / (self.log_interval * len(data))).item()
                    print("Test Accuracy: ", accuracy)
                    self.test_accuracies.append(accuracy)
                    test_logging_loss = 0
                    correct = 0

            # plotting accuracy
            self.ax4.plot(self.test_accuracies, "tab:orange")
            self.ax4.set_ylim(top=100, bottom=0)
            self.fig2.show()

        self.model.train()
        return

    def visualize(self, num_samples: int, display=False):
        """
        Method for creating and displaying gradcam
        :return: None
        """
        activation = {}
        self.model.eval()

        def get_all_layers(net):
          for name, layer in net._modules.items():
            if isinstance(layer, torch.nn.Sequential):
              get_all_layers(layer)
            else:
              layer.register_forward_hook(hook_fn)

        def hook_fn(m, i, o):
            activation[m] = o.detach()

        import cv2
        import termcolor

        def get_color(value: float):
            if value < 1/6:
                return 'grey'
            if 1/6 <= value < 2/6:
                return 'blue'
            if 2/6 <= value < 3/6:
                return 'green'
            if 3/6 <= value:
                return 'yellow'

        count = 0
        # iterating over test data
        for batch_id, (data, label) in enumerate(self.test_data):
            data = Variable(data.to(self.device))
            label = Variable(label.to(self.device)).squeeze_()
            if label[0].item() != 5:
                continue
            get_all_layers(self.model)
            out = self.model(data)
            self.criterion(out, label).backward()

            # calculate gradcam heatmap
            keys = list(activation.keys())
            act = activation[keys[19]].squeeze()
            pooled_act = torch.mean(act, dim=[1])
            features = self.model.get_feature_maps((data.detach())).detach()
            for i in range(data.shape[1]):
                features[:, i, :] *= pooled_act[i]
            heatmap = torch.mean(features, dim=1).squeeze().cpu()
            heatmap = np.maximum(heatmap, 0)
            heatmap /= torch.max(heatmap)

            # display routine
            if display:
                heatmap = cv2.resize(heatmap.data[0, :].unsqueeze(0).numpy(), (data.size()[2], 100))
                fig, ax = plt.subplots(1, figsize=(15, 3))

                # set graphics
                ax.matshow(heatmap)
                fig.gca().xaxis.tick_bottom()
                title = image_tensor_to_string_list(data[0].unsqueeze(0))[0]
                colored_title = ""
                for i in range(len(title)):
                    colored_title += termcolor.colored(title[i], get_color(heatmap[0][i]))
                print("\n", colored_title)
                plt.yticks([])
                fig.show()
                count += 1
            if count == num_samples:
                break

        self.model.train()
        return
