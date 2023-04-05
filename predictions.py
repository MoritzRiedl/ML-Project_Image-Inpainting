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

Predictions file of ML-Project_Image-Inpainting.
"""

import bz2
import gzip
import lzma
import pickle
import zipfile
import cv2
import dill as pkl
import numpy as np
import torch.jit

from tqdm.notebook import tqdm
from utils import create_arrays

# Load the trained model:
cnn = torch.load("path_to_model", map_location=torch.device("cpu"))

# Load and open the pickl file:
TEST_DATA_PATH = r"/content/drive/MyDrive/ML Challenge/inputs.pkl.zip"

# Load the compressed file:
def load_data(file: str):
    if file.endswith(".zip"):
        def zip_open(file_, mode):
            with zipfile.ZipFile(file_, "r") as myzip:
                return myzip.open(myzip.namelist()[0])
        open_fn = zip_open
    
    elif file.endswith(".bz2"):
        open_fn = bz2.open

    elif file.endswith(".xz"):
        open_fn = lzma.open

    elif file.endswith(".gz"):
        open_fn = gzip.open

    else:
        open_fn = open

    with open_fn(file, "rb") as pfh:
        return pkl.load(pfh)

# Load the data for the ML Challenge from the .pkl file:
test_data_dict = load_data(TEST_DATA_PATH)

test_data_input_arrays = test_data_dict["input_arrays"]
test_data_known_arrays = test_data_dict["known_arrays"]
test_data_offsets = test_data_dict["offsets"]
test_data_spacings = test_data_dict["spacings"]

# Make predictions:
target_arrays = []
for array_idx in tqdm(range(0, len(test_data_input_arrays))):

    input_array = test_data_input_arrays[array_idx]
    input_2 = input_array
    known_array = test_data_known_arrays[array_idx]
    offset = test_data_offsets[array_idx]
    spacing = test_data_spacings[array_idx]

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

    # Change the data type of the tensor to match with Conv2d:
    input_array = torch.from_numpy(input_array).to(dtype=torch.float32) / 255

    # Make prediction:
    with torch.no_grad():
        output = cnn(input_array)

    output = output.numpy()
    output = np.transpose(output, (1, 2, 0))

    # Create target array:
    target_array = create_arrays(image_array=output, offset=offset, spacing=spacing)
    target_array = target_array[2]

    # Convert target_array from float32 to uint8:
    target_array = (target_array * 255).astype(np.uint8)
    target_arrays.append(target_array)

# Save target_arrays as .pkl file:
file_name = "predictions.pkl"

open_file = open(file_name, "wb")
pickle.dump(target_arrays, open_file)
open_file.close()
