# -*- coding: utf-8 -*-
"""ML-Project_Image-Inpainting/main.py

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

Main file of ML-Project_Image-Inpainting.
"""

import torch.utils.data

from architectures import CNN
from datasets import dataloader_train, dataloader_test
from utils import train_and_evaluate

# Check if CUDA is available:
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device for training: {device}")

# Create CNN instance:
cnn = CNN(n_input_channels=3, n_hidden_layers=7, n_hidden_kernels=64, n_output_channels=3).to(device)

# Define optimizer:
optimizer = torch.optim.Adam(cnn.parameters(), lr=1e-3)

# Train and evaluate CNN instance:
train_and_evaluate(
    cnn=cnn,
    optimizer=optimizer,
    device=device,
    num_epochs=100,
    loader_train=dataloader_train,
    loader_test=dataloader_test
)

# Save trained model:
torch.save(cnn, "path_to_model")
