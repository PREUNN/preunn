from models.abstract_personaltrainer import AbstractPersonalTrainer
from models.generative_adversarial_nets.helper import create_noise
from torch.autograd import Variable
import torch


class GenerativeAdversarialNetPersonalTrainer(AbstractPersonalTrainer):
    """
    Training and testing class for auto encoder
    """
    def __init__(self, generator, discriminator: torch.nn.Module, training_data, test_data, log_interval,
                 generator_save_path, discrminator_save_path, criterion, g_optimizer, d_optimizer):

        super().__init__(generator, training_data, test_data, log_interval, generator_save_path, criterion, g_optimizer)
        self.generator = self.model.to(self.device)
        self.discriminator = discriminator.to(self.device)
        self.g_optimizer = self.optimizer
        self.d_optimizer = d_optimizer
        self.generator_save_path = self.model_save_path
        self.discriminator_save_path = discrminator_save_path
        return

    def train(self, epoch):
        self.model.train()

        def train_discriminator(real_data, target):
            """
            separate training method for the discriminator, one data element at
            the time
            :param real_data: example of the real dataset
            :param target: target of the real data example
            :return: both the error for real data and fake data prediction is
            returned
            """
            fake_data = self.generator(Variable(create_noise(real_data.shape).to(self.device))).detach()
            self.d_optimizer.zero_grad()
            prediction_fake = self.discriminator(fake_data)
            error_fake = self.criterion(prediction_fake, target=Variable(torch.zeros(real_data.shape[0], 1).to(self.device)))
            error_fake.backward()
            prediction_real = self.discriminator(real_data)
            error_real = self.criterion(prediction_real, target=target)
            error_real.backward()
            self.d_optimizer.step()
            return error_real, error_fake

        def train_generator(real_data, target):
            """
            separate training method for the generator, one data element at the
            time
            :param real_data: example of the real dataset
            :param target: target of the real data example
            :return: generation error is returned
            """
            self.g_optimizer.zero_grad()
            fake_data = self.generator(create_noise([real_data.shape[0], 1, real_data.shape[2]]).to(self.device))
            prediction = self.discriminator(fake_data)
            error = self.criterion(prediction, target=target)
            error.backward()
            self.g_optimizer.step()
            return error

        # training routine begins here
        self.discriminator.train()
        self.generator.train()
        d_error_real, d_error_fake, g_error = 1, 1, 1
        g_breaker = 0.75
        d_breaker = 0.75

        # iterating over dataset
        for batch_id, (real_data, target) in enumerate(self.training_data):
            real_data = Variable(real_data.to(self.device))
            target = Variable(target.to(self.device))

            # checking if discrmininator needs training
            if g_error < g_breaker or d_error_real >= d_breaker or d_error_fake >= d_breaker:
                d_error_real, d_error_fake = train_discriminator(real_data, target)

            # checking if generator needs training
            if d_error_real < d_breaker or d_error_fake < d_breaker or g_error >= g_breaker:
                g_error = train_generator(real_data, target)

            # logging interval
            if batch_id % self.log_interval == 0 and batch_id != 0:
                self.print_training_loss(epoch=epoch, batch_id=batch_id, batch_size=len(real_data), loss=g_error.item())
                print("\n#### avg. discrimination error real: {:.4f}, avg. "
                      "discrimination error fake: {:.4f} and avg. "
                      "generation error: {:.4f} ####".format(d_error_real, d_error_fake, g_error))

        # has to be saved here, since generator is automatically saved as the
        # model
        torch.save(self.discriminator, self.discriminator_save_path)
        return

    def test(self, epoch):
        self.model.eval()

        # iterating over dataset
        for batch_id, (real_data, target) in enumerate(self.test_data):
            real_data = Variable(real_data.to(self.device))
            target = Variable(target.to(self.device))

            # getting loss
            fake_data = self.generator(create_noise(real_data.shape).to(self.device))
            prediction = self.discriminator(fake_data)
            g_error = self.criterion(prediction, target=target)

            # logging interval
            if batch_id % self.log_interval == 0 and batch_id != 0:
                self.print_test_loss(epoch=epoch, batch_id=batch_id, batch_size=len(real_data), loss=g_error.item())

        self.model.train()
        return

    def create_output(self, dims: []):
        with torch.no_grad():
            return self.generator.create_output(Variable(create_noise(dims).to(self.device))).detach()
