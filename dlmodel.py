import torch
import torch.nn as nn
from torch.utils.data import Dataset

class ConversionDataset(Dataset):
    def __init__(self, data, transforms, nx, ny) -> None:
        self.data = data
        self.transforms = transforms
        self.nx = nx
        self.ny = ny

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        d = self.data[index]
        # self.facies_map = d.reshape(1, self.nx, self.ny)
        self.facies_map = d.reshape(self.nx, self.ny)
        if self.transforms:
            self.facies_map = self.transforms(self.facies_map)
        else:
            self.facies_map = torch.FloatTensor(self.facies_map.reshape(1,self.nx, self.ny))
        return self.facies_map

class Autoencoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.fc_input_dim = args.fc_input_dim
        self.encoded_space_dim = args.encoded_space_dim
        self.Encoder = Encoder(args)
        self.Decoder = Decoder(args)

    def forward(self, x):
        z = self.Encoder(x)
        y = self.Decoder(z)
        return y


class VAE(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = args.device
        self.fc_input_dim = args.fc_input_dim
        self.encoded_space_dim = args.encoded_space_dim
        self.Encoder = VariationalInference(args)
        self.Decoder = Generator(args)

    def forward(self, x):
        z, mu, var = self.Encoder(x)
        # for avoiding negative value of standard deviation, we apply log transformation at standard deviation.
        x_hat = self.Decoder(z)
        return x_hat, mu, var


class Encoder(nn.Module):
    def __init__(self, args, encoded_space_dim=None):
        self.args = args
        self.fc_input_dim = args.fc_input_dim
        if not encoded_space_dim:
            self.encoded_space_dim = args.encoded_space_dim
        else:
            self.encoded_space_dim = encoded_space_dim
        super().__init__()

        # Convolutional encoder
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
        )

        self.flatten = nn.Flatten(start_dim=1)

        self.encoder_linear = nn.Sequential(
            nn.Linear(4 * 4 * 32, self.fc_input_dim),
            nn.ReLU(True),
            nn.Linear(self.fc_input_dim, self.encoded_space_dim)
        )

    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        z = self.encoder_linear(x)
        return z

class Decoder(nn.Module):
    def __init__(self, args, encoded_space_dim=None):
        self.args = args
        self.fc_input_dim = args.fc_input_dim
        if not encoded_space_dim:
            self.encoded_space_dim = args.encoded_space_dim
        else:
            self.encoded_space_dim = encoded_space_dim
        super().__init__()
        # Convolutional decoder
        self.decoder_linear = nn.Sequential(
            nn.Linear(self.encoded_space_dim, self.fc_input_dim),
            nn.ReLU(True),
            nn.Linear(self.fc_input_dim, 4 * 4 * 32)
        )
        self.unflatten = nn.Unflatten(
            dim=1,
            unflattened_size=(32, 4, 4)
        )
        self.decoder_cnn = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=0)
        )

    def forward(self, z):
        x = self.decoder_linear(z)
        x = self.unflatten(x)
        x = self.decoder_cnn(x)
        return x

class VariationalInference(nn.Module):
    def __init__(self, args, encoded_space_dim=None):
        self.args = args
        self.fc_input_dim = args.fc_input_dim
        self.device = args.device
        if not encoded_space_dim:
            self.encoded_space_dim = args.encoded_space_dim
        else:
            self.encoded_space_dim = encoded_space_dim
        super().__init__()

        # Convolutional encoder
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
        )

        self.flatten = nn.Flatten(start_dim=1)

        self.mu = nn.Sequential(
            nn.Linear(4 * 4 * 16, self.fc_input_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(self.fc_input_dim, self.encoded_space_dim)
        )
        self.var = nn.Sequential(
            nn.Linear(4 * 4 * 16, self.fc_input_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(self.fc_input_dim, self.encoded_space_dim)
        )
    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(self.device)
        z = mean + var * epsilon
        return z

    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        mu = self.mu(x)
        var = self.var(x)
        z = self.reparameterization(mu, torch.exp(0.5 * var))
        return z, mu, var

class Generator(nn.Module):
    def __init__(self, args, encoded_space_dim=None):
        self.args = args
        self.fc_input_dim = args.fc_input_dim
        if not encoded_space_dim:
            self.encoded_space_dim = args.encoded_space_dim
        else:
            self.encoded_space_dim = encoded_space_dim
        super().__init__()
        # Convolutional decoder
        self.decoder_linear = nn.Sequential(
            nn.Linear(self.encoded_space_dim, self.fc_input_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(self.fc_input_dim, 4 * 4 * 16)
        )

        self.unflatten = nn.Unflatten(
            dim=1,
            unflattened_size=(16, 4, 4)
        )
        self.decoder_cnn = nn.Sequential(
            nn.ConvTranspose2d(16, 32, 3, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=0),
        )

    def forward(self, z):
        x = self.decoder_linear(z)
        x = self.unflatten(x)
        x = self.decoder_cnn(x)
        x_hat = torch.sigmoid(x)
        return x_hat

# class VGGEncoder(nn.Module):
#     # VGG11 with BN
#     def __init__(self, args):
#         self.args = args
#         self.fc_input_dim = args.fc_input_dim
#         self.device = args.device
#         if not encoded_space_dim:
#             self.encoded_space_dim = args.encoded_space_dim
#         else:
#             self.encoded_space_dim = encoded_space_dim
#
#         super().__init__()
#         self.conv = nn.Sequential(
#             #3 224 128
#             nn.Conv2d(1, 64, 3, padding=1),
#             nn.BatchNorm2d(64, eps=1e-05, momentum=0.1),
#             nn.LeakyReLU(0.2),
#             nn.MaxPool2d(2, 2),
#
#             #64 112 64
#             nn.Conv2d(64, 128, 3, padding=1),
#             nn.LeakyReLU(0.2),
#             nn.BatchNorm2d(128, eps=1e-05, momentum=0.1),
#             nn.MaxPool2d(2, 2),
#
#             #128 56 32
#             nn.Conv2d(128, 256, 3, padding=1)
#             nn.BatchNorm2d(256, eps=1e-05, momentum=0.1),
#             nn.LeakyReLU(0.2),
#             nn.MaxPool2d(2, 2),
#
#             #256 28 16
#             nn.Conv2d(256, 512, 3, padding=1),
#             nn.BatchNorm2d(512, eps=1e-05, momntum=0.1),
#             nn.LeakyReLU(0.2),
#             nn.MaxPool2d(2, 2),
#
#             #512 14 8
#             nn.Conv2d(512, 512, 3, padding=1),
#             nn.BatchNorm2d(512, eps=1e-05, momentum=0.1),
#             nn.LeakyReLU(0.2),
#             nn.MaxPool2d(2, 2),
#
#             nn.Conv2d(512, 512, 3, padding=1),
#             nn.BatchNorm2d(512, eps=1e-05, momentum=0.1),
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(512, 512, 3, padding=1),
#             nn.BatchNorm2d(512, eps=1e-05, momentum=0.1),
#             nn.LeakyReLU(0.2),
#             nn.MaxPool2d(2, 2),
#             nn.AdaptiveAvgPool2d(output_size=(7,7))
#         )
#
#         self.flatten = nn.Flatten(start_dim=1)
#
#         self.encoder_linear = nn.Sequential(
#             nn.Linear(4 * 4 * 32, self.fc_input_dim),
#             nn.ReLU(True),
#             nn.Linear(self.fc_input_dim, self.encoded_space_dim)
#         )

