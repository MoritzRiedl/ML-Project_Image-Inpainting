# -*- coding: utf-8 -*-
"""ML-Project_Image-Inpainting/datasets.py

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

Datasets file of ML-Project_Image-Inpainting.
"""

import cv2
import glob
import numpy as np
import os
import PIL.Image
import torch
import torchvision.transforms as transforms

from sample_project.path import input_path_train, input_path_validation, input_path_test
from torch.utils.data import Dataset, DataLoader
from utils import create_arrays

# Set seed for reproducibility:
np.random.seed(0)
torch.manual_seed(0)

# Create a custom PyTorch dataset:
class ImageDataset(Dataset):
    def __init__(self, input_dir: str, scoring: bool = False):
        """ Initializes the ImageDataset class with the input directory and scoring flag. 
        
        Also initializes  the instance variables for input_array, known_array, and target_array. Uses 
        the glob module to get a list of file paths from the input directory that end with ".jpg".

        Args:
            input_dir: Str - The path to the input directory containing the JPG files to be processed.
            scoring: Str- A flag indicating whether or not to perform scoring. Defaults to False.

        """

        # Store the input directory and scoring flag as instance variables:
        self.input_dir = input_dir
        self.scoring = scoring
        
        # Initialize the input, known, and target arrays to None:
        self.input_array = None
        self.known_array = None
        self.target_array = None

        # Get the file paths:
        self.file_paths = [file_path for file_path in 
                           sorted(glob.glob(os.path.join(input_dir, "**"), recursive=True)) 
                           if os.path.isfile(file_path) and file_path.endswith(".jpg")]
    
    def __len__(self):
        """ Returns the number of file paths in the file_paths list.
        """
        return len(self.file_paths)

    def __getitem__(self, idx):
        """ Loads an image from the file paths at the given index and applies some image preprocessing. 
        
        Creates the input, known, and target arrays from the processed image. If scoring is False, 
        the function returns the input_array and image_array, and if scoring is True, the function returns 
        also the offset, and spacing.
       
        Args:
            idx (int): The index of the file path to load and process.
        
        Returns:
            Case 1, Scoring == False: A tuple of input_array and image_array.
            Case 2, Scoring == True: A tuple of input_array, image_array, offset, and spacing.
        """
        
        # Load images and convert them to numpy array:
        image = PIL.Image.open(self.file_paths[idx])

        # Resize the image to shape 100:
        im_shape = 100
        resize_transforms = transforms.Compose([
            transforms.Resize(size=im_shape),
            transforms.CenterCrop(size=(im_shape, im_shape))
        ])

        image = resize_transforms(image)

        # Create an array from the image:
        image_array = np.array(image)

        # Create a random offset:
        offset_n = np.random.randint(low=0, high=9, size=1)
        offset_m = np.random.randint(low=0, high=9, size=1)
        offset = (offset_n[0], offset_m[0])

        # Create a random spacing:
        spacing_n = np.random.randint(low=2, high=7, size=1)
        spacing_m = np.random.randint(low=2, high=7, size=1)
        spacing = (spacing_n[0], spacing_m[0])

        # Create input, known and target array:
        input_array, known_array, target_array = create_arrays(image_array=image_array, offset=offset, spacing=spacing)

        # Preprocess the input_array:
        known_array = np.transpose(known_array, (1, 2, 0))
        known_array = np.where((known_array == 0) | (known_array == 1), known_array ^ 1, known_array)
        known_array = cv2.split(known_array)
        known_array = known_array[0]

        cv2_array = np.transpose(input_array, (1, 2, 0))

        cv2_array = cv2.inpaint(cv2_array, known_array, 3, cv2.INPAINT_TELEA)

        cv2_array = np.transpose(cv2_array, (2, 0, 1))

        # Change the shape to match with the input_array:
        image_array = np.transpose(image_array, (2, 0, 1))

        known_array = np.where((known_array == 0) | (known_array == 1), known_array ^ 1, known_array)
        known_array = known_array[None][:][:]

        input_array = np.concatenate((input_array, known_array, cv2_array), axis=0)

        self.input_array = input_array
        self.known_array = known_array
        self.target_array = target_array
        self.image_array = image_array
        self.offset = offset
        self.spacing = spacing

        # Change the data type of the tensor to match with Conv2d:
        self.input_array = torch.from_numpy(self.input_array).to(dtype=torch.float32) / 255
        self.image_array = torch.from_numpy(self.image_array).to(dtype=torch.float32) / 255

        if not self.scoring:
            return self.input_array, self.image_array

        elif self.scoring:
            return self.input_array, self.image_array, self.offset, self.spacing


# Create the Dataset's:
dataset_train = ImageDataset(input_dir=input_path_train)
dataset_val = ImageDataset(input_dir=input_path_validation)
dataset_test = ImageDataset(input_dir=input_path_test)

# Create the DataLoader's:
dataloader_train = DataLoader(dataset_train, batch_size=64, shuffle=True)
dataloader_val = DataLoader(dataset_val, batch_size=64, shuffle=True)
dataloader_test = DataLoader(dataset_test, batch_size=64, shuffle=False)
