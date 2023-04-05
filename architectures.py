# -*- coding: utf-8 -*-
"""ML-Project_Image-Inpainting/architectures.py

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

Architecture file of ML-Project_Image-Inpainting.
"""

import torch


class CNN(torch.nn.Module):
    def __init__(self, n_input_channels: int, n_hidden_layers: int, n_hidden_kernels: int, n_output_channels: int):
        super().__init__()

        # Add the input layer:
        self.input_layer = torch.nn.Conv2d(in_channels=n_input_channels, out_channels=n_hidden_kernels, kernel_size=3, padding=9)

        # Add the hidden layers:
        hidden_layers = []
        for _ in range(n_hidden_layers):
            # Add a CNN layer:
            layer = torch.nn.Conv2d(in_channels=n_hidden_kernels, out_channels=n_hidden_kernels, kernel_size=3)
            hidden_layers.append(layer)

            # Add relu activation module to list of modules:
            hidden_layers.append(torch.nn.ReLU())
            n_input_channels = n_hidden_kernels

        self.hidden_layers = torch.nn.Sequential(*hidden_layers)

        # Add the output layer:
        self.output_layer = torch.nn.Conv2d(in_channels=n_hidden_kernels, out_channels=n_output_channels, kernel_size=3)

    def forward(self, x: torch.Tensor):
        x = self.input_layer(x)
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        return x
