# -*- coding: utf-8 -*-
"""ml_project/utils.py

Author -- Moritz Riedl
Contact -- K12119148@students.jku.at
Date -- 06.07.2022

###############################################################################

The following copyright statement applies to all code within this file.

Copyright statement:
This material, no matter whether in printed or electronic form, may be used for
personal and non-commercial educational use only. Any reproduction of this
manuscript, no matter whether as a whole or in parts, no matter whether in
printed or in electronic form, requires explicit prior acceptance of the
authors.

###############################################################################

Utils file of example project.
"""

import numpy as np
import torch

from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


def create_arrays(image_array, offset, spacing):
    try:
        int(offset[0])
        int(offset[1])

        int(spacing[0])
        int(spacing[1])
    except ValueError:
        raise ValueError

    if type(image_array).__module__ != np.__name__:
        raise TypeError
    elif image_array.ndim != 3:
        raise NotImplementedError
    elif image_array.shape[2] != 3:
        raise NotImplementedError
    elif offset[0] < 0 or offset[0] > 32:
        raise ValueError
    elif offset[1] < 0 or offset[1] > 32:
        raise ValueError
    elif spacing[0] < 2 or spacing[0] > 8:
        raise ValueError
    elif spacing[1] < 2 or spacing[1] > 8:
        raise ValueError

    # Create input_array:
    np_zeros = np.zeros_like(image_array)
    np_zeros[offset[1]::spacing[1], offset[0]::spacing[0], ::] = image_array[offset[1]::spacing[1],
                                                                 offset[0]::spacing[0], ::]
    input_array = np_zeros.copy()
    input_array = np.transpose(input_array, (2, 0, 1))

    # Create known_array:
    known_array = np.zeros_like(image_array)
    known_array[offset[1]::spacing[1], offset[0]::spacing[0], ::] = 1
    known_array = np.transpose(known_array, (2, 0, 1))

    if int(np.count_nonzero(known_array) / 3) < 144:
        raise ValueError

    # Create target_array:
    target_array = image_array.copy()
    target_array = np.transpose(target_array, (2, 0, 1))
    boolean_mask = known_array < 1
    target_array = target_array[boolean_mask]

    return input_array, known_array, target_array


def dataloader_sample(dataloader, samples):

    for sample in range(0, samples):
        train_features, train_labels = next(iter(dataloader))

        print(f"Feature batch shape: {train_features.size()}")
        print(f"Labels batch shape: {train_labels.size()}")

        img = np.transpose(train_features[0], (2, 1, 0))
        label = np.transpose(train_labels[0], (1, 2, 0))
        print(f"Input tensor dtype: {img.dtype}")
        print(f"Label tensor dtype: {label.dtype}")

        print(f"Input tensor shape: {img.shape}")
        print(f"Label tensor shape: {label.shape}")

        fig = plt.figure(figsize=(10, 7))

        fig.add_subplot(samples, 2, 1)
        plt.imshow(img)
        plt.axis("off")
        plt.title("Input")

        fig.add_subplot(samples, 2, 2)
        plt.imshow(label)
        plt.axis("off")
        plt.title("Label")

    plt.show()


def train_network(cnn: torch.nn.Module, data_loader: torch.utils.data.DataLoader,
                  optimizer: torch.optim.Optimizer, device: torch.device = r'cpu') -> None:
    """
    Train specified network for one epoch on specified data loader.

    :param cnn: network to train
    :param data_loader: data loader to be trained on
    :param optimizer: optimizer used to train network
    :param device: device on which to train network
    """

    cnn.train()

    # Define loss function:
    loss_function = torch.nn.MSELoss()

    for inputs, labels in tqdm(data_loader):
        # Send inputs and labels to CUDA:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Reset gradients:
        optimizer.zero_grad()

        # Compute model output:
        output = cnn(inputs)

        # Compute loss:
        loss = loss_function(output, labels)

        # Compute gradients:
        loss.backward()

        # Perform update step:
        optimizer.step()


def test_network(cnn: torch.nn.Module, data_loader: torch.utils.data.DataLoader,
                 device: torch.device = r'cpu'):
    """
    Test specified network on specified data loader.

    :param cnn: network to test on
    :param data_loader: data loader to be tested on
    :param device: device on which to test network
    :return: cross-entropy loss as well as accuracy
    """
    cnn.eval()

    loss = 0.0

    # Define loss function:
    loss_function = torch.nn.MSELoss()

    with torch.no_grad():
        for inputs, labels in data_loader:
            # Send inputs and labels to CUDA:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Compute model output:
            output = cnn(inputs)

            # Compute loss:
            loss = loss_function(output, labels)

    return loss


def train_and_evaluate(cnn: torch.nn.Module, optimizer: torch.optim.Optimizer,
                       device: torch.device, num_epochs: int,
                       loader_train: torch.utils.data.DataLoader,
                       loader_test: torch.utils.data.DataLoader) -> None:
    """
    Auxiliary function for training and evaluating a corresponding model.

    :param cnn: model instance to train and evaluate
    :param optimizer: optimizer to use for model training
    :param device: device to use for model training and evaluation
    :param num_epochs: amount of epochs for model training
    :param loader_train: data loader supplying the training samples
    :param loader_test: data loader supplying the test samples
    """
    for epoch in tqdm(range(num_epochs)):
        # Train model instance for one epoch.
        train_network(
            cnn=cnn,
            data_loader=loader_train,
            device=device,
            optimizer=optimizer
        )

        # Evaluate current model instance.
        performance = test_network(
            cnn=cnn,
            data_loader=loader_train,
            device=device
        )

        # Print result of current epoch to standard out.
        print(f'Epoch: {str(epoch + 1).zfill(len(str(num_epochs)))} ' +
              f'/ Train loss: {performance:.4f}')

    # Evaluate final model on test data set.
    performance = test_network(cnn=cnn, data_loader=loader_test, device=device)
    print(f'\nTest loss: {performance:.4f}')
