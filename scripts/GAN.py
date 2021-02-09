import datetime
from matplotlib import pyplot as plt


plt.rcParams["figure.figsize"] = (10, 10)


class GAN:
    def __init__(self, latent_dim=100, content_shape=(128, 128, 1), data_generator=None, name='GAN'):
        self.name = name
        self.latent_dim = latent_dim
        self.content_shape = content_shape
        self.data_loader = data_generator
        self.generator = None
        self.discriminator = None
        self.combined = None
        self.d_loss = []
        self.g_loss = []
        self.d_acc = []

    def build_generator(self):
        raise NotImplemented

    def build_discriminator(self, lr=0.001, beta=0.999):
        raise NotImplemented

    def combined_model(self, lr=0.001, beta=0.999):
        raise NotImplemented

    def generate_sample(self, epoch):
        raise NotImplemented

    def generate_sample_to(self, path):
        raise NotImplemented

    def save_models(self, epoch):
        path = "../models/%s_%s_%s_epoch_%d.hdf5"
        self.generator.save(
            path % (datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"), 'generator', self.name, epoch))
        self.discriminator.save(
            path % (datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"), 'discriminator', self.name, epoch))

    def plot_progress(self, suffix=''):
        timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

        plt.plot(self.d_loss, c='red')
        plt.plot(self.g_loss, c='blue')
        plt.title("GAN Loss per Epoch")
        plt.legend(['Discriminator', 'Generator'])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig('../plots/%s_%s_GAN_Loss_per_Epoch_%s.png' % (timestamp, self.name, suffix), transparent=True)
        plt.close()

        plt.plot(self.d_acc)
        plt.title("GAN Discriminator Accuracy per Epoch")
        plt.legend(['Accuracy'])
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim((0, 1))
        plt.savefig('../plots/%s_%s_GAN_Accuracy_per_Epoch_%s.png' % (timestamp, self.name, suffix), transparent=True)
        plt.close()
