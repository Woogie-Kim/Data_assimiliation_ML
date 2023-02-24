import numpy as np
import os
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler

from dlmodel import ConversionDataset, Autoencoder, VAE

class GenerativeModel:
    def __init__(self,
                 args,
                 model_name='Autoencoder'):
        self.model_name = model_name
        if model_name == 'Autoencoder':
            self.model = Autoencoder(args=args)
        elif model_name == 'VAE':
            self.model = VAE(args=args)
        # elif model_name == 'GAN':
        #     self.model = GAN(args=args)

        self.args = args
        self.nx = args.nx
        self.ny = args.ny
        self.k_sand = args.k_sand
        self.k_clay = args.k_clay
        self.k_mean = (args.k_sand + args.k_clay) / 2
        self.model_name = model_name
        self.batch_size = args.batch_size
        self.device = args.device
        self.num_epoch = args.num_epoch


    def preprocess(self, data):
        # 수정 필요
        if isinstance(data, list): data_input = np.array([e.perm.to_numpy() for e in data]).reshape(-1, 1)
        else:data_input = np.array([e.perm.to_numpy() for e in [data]]).reshape(-1, 1)
        scaler = MinMaxScaler()
        scaler.fit(data_input)
        scaled_input = scaler.transform(data_input)
        self.scaler = scaler
        return scaled_input.reshape(-1, self.nx * self.ny)


    def make_dataloader(self, data, train_ratio, valid_ratio):
        args = self.args
        tf = transforms.Compose([transforms.ToTensor(), transforms.ConvertImageDtype(dtype=torch.float)])
        scaled_data = self.preprocess(data)
        dataset = ConversionDataset(scaled_data, tf, args.nx, args.ny)
        sequence = [round(len(dataset) * ratio) for ratio in [train_ratio, valid_ratio]]
        sequence.append(len(dataset) - sum(sequence))
        train_dataset, validation_dataset, test_dataset = random_split(dataset, sequence)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        valid_dataloader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

        return train_dataloader, valid_dataloader, test_dataloader

    def train_model(self, data, train_ratio=0.7, valid_ratio=0.15, saved_dir='./model', saved_name='best_model'):
        train_dataloader, valid_dataloader, test_dataloader = self.make_dataloader(data, train_ratio, valid_ratio)
        self.model = self.train(self.model, train_dataloader, valid_dataloader, test_dataloader, saved_dir, saved_name)
        return self.model

    def train(self, model, train_dataloader, valid_dataloader, test_dataloader, saved_dir, saved_name):
        model.to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-05)
        loss_fn = nn.MSELoss().to(self.device)

        self.train_log = []
        self.valid_log = []
        self.test_log = []
        min_valid_loss = np.inf

        iter_bar = tqdm(range(self.num_epoch))
        for epoch in iter_bar:
            train_loss = 0
            valid_loss = 0
            model.train()
            for batch in train_dataloader:
                batch = batch.to(self.device)
                b_size = len(batch)
                optimizer.zero_grad()
                if self.model_name == 'VAE':
                    x_hat, mu, var = model(batch)
                    BCE, KLD = self.loss_function(batch, x_hat, mu, var)
                    loss = BCE + KLD
                else:
                    x_hat = model(batch)
                    loss = loss_fn(batch, x_hat)

                loss.backward()
                optimizer.step()
                loss /= b_size
                train_loss += loss.item()

            model.eval().to(self.device)
            for batch in valid_dataloader:
                batch = batch.to(self.device)
                b_size = len(batch)
                if self.model_name == 'VAE':
                    x_hat, mu, var = model(batch)
                    BCE, KLD = self.loss_function(batch, x_hat, mu, var)
                    loss = BCE + KLD
                else:
                    x_hat = model(batch)
                    loss = loss_fn(batch, x_hat)
                loss /= b_size
                valid_loss += loss.item()
            t_loss = np.mean(train_loss / len(train_dataloader))
            v_loss = np.mean(valid_loss/ len(valid_dataloader))
            self.train_log.append(t_loss)
            self.valid_log.append(v_loss)

            if (epoch+1) % 50 == 0:
                print(
                    f'Epoch {epoch + 1} \t\t Training Loss: {t_loss:.5f} \t\t '
                    f'Validation Loss: {v_loss:.5f}')

            if min_valid_loss > self.valid_log[-1]:
                min_valid_loss = self.valid_log[-1]
                iter_bar.set_description(f'Best : {min_valid_loss:.5f}')
                if not os.path.exists(saved_dir):
                    os.mkdir(saved_dir)
                torch.save(model.state_dict(), f'{saved_dir}/{saved_name}.pth')


        model.load_state_dict(torch.load(f'{saved_dir}/{saved_name}.pth'))
        self.verify(model, test_dataloader)
        return model


    def loss_function(self, x, x_hat, mu, var):
        reproduction_loss = F.binary_cross_entropy(x_hat, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + var - mu.pow(2) - var.exp())
        return reproduction_loss, KLD

    def verify(self, model, dataloader):
        model.eval()
        self.predictions = []
        self.reals = []
        self.latent_vector = []
        if not self.test_log:
            self.test_log = []

        test_loss = 0
        loss_fn = nn.MSELoss().to(self.device)
        for batch in dataloader:
            batch = batch.to(self.device)
            b_size = len(batch)
            if self.model_name == 'VAE':
                # x_hat, mu, var = model(batch)
                z, mu, var = model.Encoder(batch)
                x_hat = model.Decoder(z)
                BCE, KLD = self.loss_function(batch, x_hat, mu, var)
                loss = BCE + KLD
            else:
                x_hat = model(batch)
                loss = loss_fn(batch, x_hat)
            prediction = self.scaler.inverse_transform(x_hat.view(b_size, -1).detach().cpu().numpy()).reshape(b_size, 1, self.nx, self.ny)
            prediction = self._adjust_scaler(prediction)
            real = self.scaler.inverse_transform(batch.view(b_size, -1).cpu().numpy()).reshape(b_size, 1, self.nx, self.ny)
            real = self._adjust_scaler(real)
            loss /= b_size
            test_loss += loss.item()
            self.predictions.extend(prediction)
            self.latent_vector.extend(z)
            self.reals.extend(real)
        self.test_log.append(np.mean(test_loss / len(dataloader)))
        print(f"Test Loss : {self.test_log[-1]:.5f}")
        return torch.cat(self.latent_vector).reshape(len(self.predictions), -1)

    def _adjust_scaler(self, rst):
        rst[rst > self.k_mean] = self.k_sand
        rst[rst < self.k_mean] = self.k_clay
        return rst







