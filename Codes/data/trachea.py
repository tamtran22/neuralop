import logging
import os
from pathlib import Path
from typing import Union, List

from torch.utils.data import DataLoader

from neuralop.data.datasets.pt_dataset import PTDataset
# from .web_utils import download_from_zenodo_record

from neuralop.utils import get_project_root

from PIL import Image
import numpy as np
import torch


def load_trachea_slice_dataset(root_dir, dataset_name, n_train, n_test, resolution, batch_size, test_batch_size):

    ## Check root
    if not os.path.isdir(root_dir):
        raise ValueError('Data directory does not exist.')
    if not os.path.isfile(f'{root_dir}/{dataset_name}_train_{resolution}.pt'):
        ##
        file_names = sorted(os.listdir(root_dir))
        file_names = [f'{root_dir}/{file_name}' for file_name in file_names]
        images = []
        for file_name in file_names:
            image = Image.open(file_name)
            images.append(np.expand_dims(np.asarray(image, dtype='uint8')[:,:,0], axis=0))
        data = torch.tensor(np.concatenate(images, axis=0), dtype=torch.float32)
        ##
        number_of_input_timesteps = 10
        x = []
        y = []
        for i in range(number_of_input_timesteps, data.size(0)):
            x.append(data[i-number_of_input_timesteps:i].unsqueeze(0))
            y.append(data[i].unsqueeze(0).unsqueeze(0))
        x = torch.concatenate(x, dim=0)
        y = torch.concatenate(y, dim=0)
        print(x.size(), y.size())
    
        # Save dataset file
        x_train = x[0:n_train]
        y_train = y[0:n_train]
        data_train = {'x' : x_train, 'y' : y_train}
        torch.save(data_train, f'{root_dir}/{dataset_name}_train_{resolution}.pt')
        x_test = x[n_train:n_train+n_test]
        y_test = y[n_train:n_train+n_test]
        data_test = {'x' : x_test, 'y' : y_test}
        torch.save(data_test, f'{root_dir}/{dataset_name}_test_{resolution}.pt')

    # Load dataset
    dataset = PTDataset(
        root_dir=root_dir,
        dataset_name = dataset_name,
        n_train = n_train,
        n_tests = [n_test],
        batch_size = batch_size,
        test_batch_sizes = [test_batch_size],
        train_resolution = resolution,
        test_resolutions = [resolution],
        encode_input = False,
        encode_output = False,
        channel_dim=1,
        channels_squeezed = False
    )
    return dataset


    
    