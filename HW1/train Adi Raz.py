## Scikit-learn built-in dataset generators
from sklearn.datasets import make_moons, make_circles, make_blobs

## Progress bar
import tqdm
import os
import math
import time
import numpy as np
import copy

## PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import Adam
from torch.distributions.multivariate_normal import MultivariateNormal

# Traning will be done on CPU for this homework.
# For K=4, N=1500, epochs=1000 takes < 3 mins.
device = torch.device("cpu")
print("Using device", device)


def sample_2d_datasets(dist_type, num_samples=1000, seed=0, num_gaussians=5):
    """
  function samples from simple pre-defined distributions in 2D.
  Inputs:
    - dist_type: str specifying the distribution to be chosen from:
      {'Circles', 'Moons', 'GaussiansGrid', 'GaussiansRot'}
    - num_samples: Number of samples to draw from dist_type (int).
    - seed: Random seed integer.
    - num_gaussians: Number of rotated gaussians if dist_type='GaussiansRot'.
      (relevant only for dist_type='GaussiansRot', should be a keyword argument)
  Outputs:
    - data (np.array): array of num_samplesx2 samples from dist_type
  """
    np.random.seed(seed)
    if dist_type == 'Circles':
        data = make_circles(num_samples, noise=.1, factor=.8, random_state=seed, shuffle=True)[0]
    elif dist_type == 'Moons':
        data = make_moons(num_samples, noise=.1, random_state=seed, shuffle=True)[0]
    elif dist_type == 'GaussiansGrid':
        centers = np.array([[0, 0], [0, 2], [2, 0], [2, 2]])
        data = make_blobs(num_samples, centers=centers, cluster_std=.5, random_state=seed, shuffle=True)[0]
    elif dist_type == 'GaussiansRot':
        angles = np.linspace(0, 2 * np.pi, num_gaussians, endpoint=False)
        centers = np.stack([2.5 * np.array([np.cos(angle), np.sin(angle)]) for angle in angles])
        data = make_blobs(num_samples, centers=centers, cluster_std=np.sqrt(.1), random_state=seed, shuffle=True)[0]
    else:
        raise NotImplementedError
    return data


class ToyDataset(Dataset):
    def __init__(self, dist_type, num_samples=1000, seed=0, num_gaussians=5):
        """
      Wrapper around the function "sample_2d_datasets" to allow iterating
      batches using a datalaoder when training our normalizing flow model.
      """
        self.data = sample_2d_datasets(dist_type, num_samples, seed, num_gaussians)
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        return torch.from_numpy(self.data[index]).type(torch.FloatTensor)


class CouplingLayer(nn.Module):
    def __init__(self, mask):
        super(CouplingLayer, self).__init__()

        # mask for splitting (fixed not learnable)
        self.mask = nn.Parameter(mask, requires_grad=False)

        # scaling function and stabilizing scale_factor init. to 0
        self.s_func = nn.Sequential(nn.Linear(in_features=2, out_features=32),
                                    nn.LeakyReLU(),
                                    nn.Linear(in_features=32, out_features=32),
                                    nn.LeakyReLU(),
                                    nn.Linear(in_features=32, out_features=2))
        self.scale_factor = nn.Parameter(torch.Tensor(2).fill_(0.0))

        # shifting function
        self.t_func = nn.Sequential(nn.Linear(in_features=2, out_features=32),
                                    nn.LeakyReLU(),
                                    nn.Linear(in_features=32, out_features=32),
                                    nn.LeakyReLU(),
                                    nn.Linear(in_features=32, out_features=2))

    # def forward(self, x):
    #   """
    #   TODO: replace y and log_det_jac with your code.
    #   """
    #   x_part_1 = x*self.mask
    #   inv_mask = 1-self.mask
    #   x_part_2 = x*inv_mask
    #   s = self.s_func(x_part_1)
    #   t = self.t_func(x_part_1)
    #   s = s*inv_mask
    #   exp_scale = torch.exp(s*self.scale_factor)
    #   t = t*inv_mask
    #   #y_part_1 = x_part_1
    #   y = exp_scale*x + t
    #   #y = torch.concat((y_part_1[:,torch.where(self.mask==1)],y_part_2[:,torch.where(inv_mask==1)]),dim=1)
    #   log_det_jac = torch.sum(s,dim=1)
    #   return y, log_det_jac

    def forward(self, x):
        """
    TODO: replace y and log_det_jac with your code.
    """
        x_part_1 = torch.mul(x, self.mask)
        inv_mask = 1 - self.mask
        x_part_2 = torch.mul(x, inv_mask)
        s = self.s_func(x_part_1)
        t = self.t_func(x_part_1)
        exp_scale = torch.exp(torch.mul(s, self.scale_factor))
        y_part_1 = x_part_1
        y_part_2 = torch.mul(torch.mul(exp_scale, x_part_2) + t, inv_mask)
        y = y_part_1 + y_part_2
        log_det_jac = torch.sum(s)
        return y, log_det_jac

    def inverse(self, y):
        """
    TODO: replace x and inv_log_det_jac with your code.
    """
        y_part_1 = torch.mul(y, self.mask)
        inv_mask = 1 - self.mask
        y_part_2 = torch.mul(y, inv_mask)
        s = self.s_func(y_part_1)
        t = self.t_func(y_part_1)
        exp_scale = torch.exp(-torch.mul(s, self.scale_factor))
        x_part_2 = torch.mul((y_part_2 - t), exp_scale)
        x_part_1 = y_part_1

        x = torch.concat((x_part_1[:, torch.where(self.mask == 1)], x_part_2[:, torch.where(inv_mask == 1)]), dim=1)
        inv_log_det_jac = -torch.sum(s, dim=1)
        return x, inv_log_det_jac


class CouplingFlow(nn.Module):
    def __init__(self, num_layers):
        super(CouplingFlow, self).__init__()

        # concatenate coupling layers with alternating masks
        masks = F.one_hot(torch.tensor([i % 2 for i in range(num_layers)])).float()
        self.layers = nn.ModuleList([CouplingLayer(mask) for mask in masks])

        # define prior distribution to be z~N(0,I)
        self.prior = MultivariateNormal(torch.zeros(2), torch.eye(2))

    def log_probability(self, x):
        """
    TODO: replace log_prob with your code.
    """
        log_det_jac = 0
        for i, _ in enumerate(self.layers):
            x, log_det_jac_i = self.layers[i](x)
            log_det_jac += log_det_jac_i
        y = x  # switch to y for convinience
        log_p_y = self.prior.log_prob(y)
        log_prob = log_p_y + log_det_jac
        return log_prob

    def sample_x(self, num_samples):
        """
    TODO: replace x and log_prob with your code.
    """
        log_det_jac = 0
        y = self.prior.sample([num_samples])
        num_of_layers = len(self.layers)
        inv_log_det_jac = 0
        for i, _ in enumerate(self.layers):
            y, inv_log_det_jac_i = self.layers[num_of_layers - i - 1].inverse(y)
            inv_log_det_jac += inv_log_det_jac_i
        x = y
        log_p_y = self.prior.log_prob(y)
        log_prob = log_p_y + inv_log_det_jac  # +inv_log_det_jac_i
        return x, log_prob

    def sample_x_each_step(self, num_samples):
        """
    TODO: replace samples with your code.
    """
        log_det_jac = 0
        samples = []
        z = self.prior.sample([num_samples])
        for layer in self.layers:
            z, log_det_jac_i = layer(z)
            samples.append(z)
        return samples


# detach tensor and transfer to numpy
def to_np(x):
    return x.detach().numpy()


# Simple training function
def train(model, data, epochs=100, batch_size=64):
    # move model into the device
    model = model.to(device)

    # split into training and validation, and create the loaders
    lengths = [int(len(data) * 0.9), len(data) - int(len(data) * 0.9)]
    train_set, valid_set = random_split(data, lengths)
    train_loader = DataLoader(train_set, batch_size=batch_size)
    valid_loader = DataLoader(valid_set, batch_size=batch_size)

    # define the optimizer and scheduler
    optimizer = Adam(model.parameters(), lr=1e-3)

    # train the model
    train_losses, valid_losses, min_valid_loss = [], [], np.Inf
    with tqdm.tqdm(range(epochs), unit=' Epoch') as tepoch:
        for epoch in tepoch:

            # training loop
            epoch_loss = 0
            model.train(True)
            for batch_index, training_sample in enumerate(train_loader):
                log_prob = model.log_probability(training_sample)
                loss = - log_prob.mean(0)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss
            epoch_loss /= len(train_loader)
            train_losses.append(np.copy(to_np(epoch_loss)))

            # validation loop
            epoch_loss_valid = 0
            model.train(False)
            for batch_index, valid_sample in enumerate(valid_loader):
                log_prob = model.log_probability(valid_sample)
                loss_valid = - log_prob.mean(0)
                epoch_loss_valid += loss_valid

            epoch_loss_valid /= len(valid_loader)
            valid_losses.append(np.copy(to_np(epoch_loss_valid)))

            # save best model based off validation loss
            if epoch_loss_valid < min_valid_loss:
                model_best = copy.deepcopy(model)
                min_valid_loss = epoch_loss_valid
                epoch_min = epoch

            # report progress with tqdm pbar
            tepoch.set_postfix(train_loss=to_np(epoch_loss), valid_loss=to_np(epoch_loss_valid))

    # report best model on val.
    print('\n Best Model achieved {:.4f} validation loss at epoch {} \n'.
          format(min_valid_loss, epoch_min))

    # if the number of samples is too low take the final weights regardless of
    # valdiation loss due to weak statistics (overfitting avoided by early stopping)
    if lengths[1] < 500:
        model_best = model

    return model_best, train_losses, valid_losses


# seeds to ensure reproducibility
torch.manual_seed(8)
np.random.seed(0)

# dataset
num_samples = 1500
data = ToyDataset('Moons', num_samples=num_samples)

# learning hyper-parameters
K = 4
nepochs = 1000

# instantiate model and optimize the parameters
Flow_model = CouplingFlow(num_layers=K)
moon_model, train_loss, valid_loss = train(Flow_model, data, epochs=nepochs)
