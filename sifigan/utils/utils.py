# -*- coding: utf-8 -*-

# Copyright 2020 Yi-Chiao Wu (Nagoya University)
# based on a Parallel WaveGAN script by Tomoki Hayashi (Nagoya University)
# (https://github.com/kan-bayashi/ParallelWaveGAN)
#  MIT License (https://opensource.org/licenses/MIT)

"""Utility functions."""

import os
import sys
from logging import getLogger

import h5py
import numpy as np

# A logger for this file
logger = getLogger(__name__)


def read_hdf5(hdf5_name, hdf5_path):
    """Read hdf5 dataset.

    Args:
        hdf5_name (str): Filename of hdf5 file.
        hdf5_path (str): Dataset name in hdf5 file.

    Return:
        any: Dataset values.

    """
    if not os.path.exists(hdf5_name):
        logger.error(f"There is no such a hdf5 file ({hdf5_name}).")
        sys.exit(1)

    hdf5_file = h5py.File(hdf5_name, "r")

    if hdf5_path not in hdf5_file:
        logger.error(f"There is no such a data in hdf5 file. ({hdf5_path})")
        sys.exit(1)

    hdf5_data = hdf5_file[hdf5_path][()]
    hdf5_file.close()

    return hdf5_data


def write_hdf5(hdf5_name, hdf5_path, write_data, is_overwrite=True):
    """Write dataset to hdf5.

    Args:
        hdf5_name (str): Hdf5 dataset filename.
        hdf5_path (str): Dataset path in hdf5.
        write_data (ndarray): Data to write.
        is_overwrite (bool): Whether to overwrite dataset.

    """
    # convert to numpy array
    write_data = np.array(write_data)

    # check folder existence
    folder_name, _ = os.path.split(hdf5_name)
    if not os.path.exists(folder_name) and len(folder_name) != 0:
        os.makedirs(folder_name)

    # check hdf5 existence
    if os.path.exists(hdf5_name):
        # if already exists, open with r+ mode
        hdf5_file = h5py.File(hdf5_name, "r+")
        # check dataset existence
        if hdf5_path in hdf5_file:
            if is_overwrite:
                hdf5_file.__delitem__(hdf5_path)
            else:
                logger.error(
                    "Dataset in hdf5 file already exists. "
                    "if you want to overwrite, please set is_overwrite = True."
                )
                hdf5_file.close()
                sys.exit(1)
    else:
        # if not exists, open with w mode
        hdf5_file = h5py.File(hdf5_name, "w")

    # write data to hdf5
    hdf5_file.create_dataset(hdf5_path, data=write_data)
    hdf5_file.flush()
    hdf5_file.close()


def check_hdf5(hdf5_name, hdf5_path):
    """Check hdf5 file existence

    Args:
        hdf5_name (str): filename of hdf5 file
        hdf5_path (str): dataset name in hdf5 file

    Return:
        (bool): dataset exists then return true

    """
    if not os.path.exists(hdf5_name):
        return False
    else:
        with h5py.File(hdf5_name, "r") as f:
            if hdf5_path in f:
                return True
            else:
                return False


def read_txt(file_list):
    """Read .txt file list

    Arg:
        file_list (str): txt file filename

    Return:
        (list): list of read lines

    """
    with open(file_list, "r") as f:
        filenames = f.readlines()
    return [filename.replace("\n", "") for filename in filenames]


def check_filename(list1, list2):
    """Check the filenames of two list are matched

    Arg:
        list1 (list): file list 1
        list2 (list): file list 2

    Return:
        (bool): matched (True) or not (False)

    """

    def _filename(x):
        return os.path.basename(x).split(".")[0]

    list1 = list(map(_filename, list1))
    list2 = list(map(_filename, list2))
    return list1 == list2


def validate_length(xs, ys=None, hop_size=None):
    """Validate length

    Args:
        xs (ndarray): numpy array of features
        ys (ndarray): numpy array of audios
        hop_size (int): upsampling factor

    Returns:
        (ndarray): length adjusted features

    """
    min_len_x = min([x.shape[0] for x in xs])
    if ys is not None:
        min_len_y = min([y.shape[0] for y in ys])
        if min_len_y < min_len_x * hop_size:
            min_len_x = min_len_y // hop_size
        if min_len_y > min_len_x * hop_size:
            min_len_y = min_len_x * hop_size
        ys = [y[:min_len_y] for y in ys]
    xs = [x[:min_len_x] for x in xs]

    return xs + ys if ys is not None else xs

def create_scp_for_aug(audio_path, f0_conv_num):
    new_audio_path = audio_path.replace(".", "_aug.")
    
    with open(audio_path, "r") as f_read, open(new_audio_path, "w") as f_write:
        for line in f_read:
            f_write.write(line * (f0_conv_num+1))
    
    logger.info(f"Augmented data written to: {new_audio_path}")
    return new_audio_path