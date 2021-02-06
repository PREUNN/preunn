import torch
import random
import re
from data.source_datasets.protocol_types import Protocol
from models.abstract_personaltrainer import AbstractPersonalTrainer
from scapy.all import wrpcap, Ether, IP, TCP
from data.postprocessing.sequence_postprocessing.processing import sequence_tensor_to_string_list


class LongShortTermMemoryPersonalTrainer(AbstractPersonalTrainer):
    """
    Training and testing class for LSTMs
    """
    def __init__(self, model, training_data, test_data, log_interval, model_save_path, criterion, optimizer):
        super().__init__(model, training_data, test_data, log_interval, model_save_path, criterion, optimizer)
        return

    def train(self, epoch):
        """
        training subroutine for one epoch, prints out loss
        :param epoch: number of the current epoch
        :return: None
        """
        self.model.train()
        loss_sum = 0

        # iterating over dataset
        for batch_id, (data, target) in enumerate(self.training_data):
            # lstm prep
            batch_size = data.shape[0]
            sequence_length = data.shape[1]
            h = self.model.init_hidden(batch_size=batch_size)
            h = tuple([each.data.to(self.device) for each in h])
            # change between the following 2 lines from fre/sg and sr experiments
            # target = target.to(self.device).long()
            target = target.to(self.device).float()
            data = data.to(self.device).long()
            self.optimizer.zero_grad()
            loss = 0

            # getting loss
            output, h = self.model(data, h)
            for c in range(sequence_length):
                loss += self.criterion(output[:, c], target[:, c])
            loss /= data.shape[1]
            loss_sum += loss.item()
            loss.backward(retain_graph=False)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
            self.optimizer.step()

            # logging interval
            if batch_id % self.log_interval == 0 and batch_id != 0:
                loss_sum /= self.log_interval
                self.print_training_loss(epoch=epoch, batch_id=batch_id, batch_size=batch_size, loss=loss_sum)
                self.model.store_model(self.model_save_path)
            if batch_id % self.log_interval == 0:
                loss_sum = 0

        return

    def test(self, epoch):
        self.model.eval()
        loss_sum = 0

        # iterating over dataset
        for batch_id, (data, target) in enumerate(self.test_data):
            batch_size = data.shape[0]
            sequence_length = data.shape[1]
            h = self.model.init_hidden(batch_size=batch_size)
            h = tuple([each.data.to(self.device) for each in h])
            target = target.to(self.device).squeeze(1).long()
            data = data.to(self.device).squeeze(1).long()

            # getting loss
            output, h = self.model(data, h)
            loss = 0
            for c in range(sequence_length):
                loss += self.criterion(output[:, c], target[:, c])
            loss /= data.shape[1]
            loss_sum += loss.item()

            # logging interval
            if batch_id % self.log_interval == 0 and batch_id != 0:
                loss_sum /= self.log_interval
                self.print_test_loss(epoch=epoch, batch_id=batch_id, batch_size=batch_size, loss=loss_sum)
            if batch_id % self.log_interval == 0:
                loss_sum = 0

        self.model.train()
        return

    def sample_statement(self, random_delimiter: int, length: int, data):
        """
        Subroutine to create a sample from the lstm
        :param random_delimiter: stepsize for when to sample by probability
        :param length: length of the sample to be created
        :param data: initialization data
        :return: tensor of the created sample
        """
        likelihood = 0
        with torch.no_grad():
            h = self.model.init_hidden(batch_size=data.shape[0])
            for k in range(length):
                h = tuple([each.data for each in h])
                letter, h = self.model(data, h)
                if k % random_delimiter == 0 and k != 0:
                    # Sample from the network as a multinomial distribution
                    letter_dist = letter[:, data.shape[1]-1].exp()
                    top_i = torch.multinomial(letter_dist, 1)
                else:
                    likelihood, top_i = letter[:, data.shape[1]-1].topk(1)
                if data.shape[1] == length:
                    data = data[:, 1:]
                data = torch.cat([data, top_i], dim=1)
        return data, likelihood

    def sample_states(self, length: int, data):
        """
        Subroutine to create state samples from lstm
        :param length: length of the sample to be created
        :param data: initialization data
        :return: tensor of the created sample
        """
        with torch.no_grad():
            h = self.model.init_hidden(batch_size=data.shape[0])
            for k in range(length):
                h = tuple([each.data for each in h])
                letter, h = self.model(data, h)
                letter = torch.round(letter).long()
                if data.shape[1] == length:
                    data = data[:, 1:]
                data = torch.cat([data, letter[:, data.shape[1]-1].unsqueeze(dim=1)], dim=1)
        return data

    def get_new_statements(self, num_classes: int, filename: str):
        """
        This method will create a new pcap file with 1000 newly generated
        samples of the same protocol type as the testset.
        :param num_classes: number of classes used for the protocol clustering.
        :param filename: name of the file to be saved.
        :return: None
        """
        if self.protocol == Protocol.HTTP:
            port = 80
        if self.protocol == Protocol.FTP:
            port = 21
        splitter = ''
        for x in range(num_classes):
            splitter += 'EOP°' + str(x) + '\n\n|'
            splitter += 'SOP°' + str(x) + '\n|'
        splitter = splitter[:-1]
        statement_list = []
        package_list = []

        # iterating over test data for initialization vectors
        while len(package_list) < 1000:
            for _, (item, _) in enumerate(self.test_data):
                sample_sequence, _ = self.sample_statement(random_delimiter=3, length=item.shape[1], data=item.to(
                    self.device))

                # evaluating each sample separately and create network packages
                for each in sample_sequence:
                    each = sequence_tensor_to_string_list(each.unsqueeze(0), num_classes=1)[0]
                    temp_list = re.split(splitter, each)
                    for each in temp_list[1:-1]:
                        if each != "":
                            print(each)
                            statement_list.append(each)
                            address = str(random.randint(1, 192)) + "." + str(random.randint(1, 192)) + "." + \
                                      str(random.randint(1, 192)) + "." + str(random.randint(1, 192))
                            package = Ether() / IP(dst=address) / TCP(dport=port, flags='S') / each
                            package_list.append(package)
                    if len(package_list) > 1000:
                        package_list = package_list[:1000]
                        wrpcap(filename + ".pcap", package_list)
                        return
        wrpcap(filename + ".pcap", package_list)
        return
