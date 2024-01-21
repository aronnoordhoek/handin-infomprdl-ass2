import logging
import os
import h5py
import zipfile
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.signal import resample

logger = logging.getLogger('scripts.utils')

sensor_mapping_df = pd.read_csv(os.path.join('..', 'assets', 'sensor_mapping.csv'))
sensor_indices = sensor_mapping_df['sensor'].values
row_indices = sensor_mapping_df['row'].values
col_indices = sensor_mapping_df['column'].values
num_rows = row_indices.max() + 1
num_cols = col_indices.max() + 1


def array_to_mesh(arr_1d):
    output = np.zeros((num_rows, num_cols))
    output[row_indices, col_indices] = arr_1d[sensor_indices]
    return output


def extract_zip_file():
    data_dir = os.path.join('..', 'data')
    test_dir = os.path.join(data_dir, 'Intra')

    if not os.path.exists(test_dir):
        zip_file_path = os.path.join(data_dir, 'final_project_data.zip')

        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            for member in tqdm(zip_ref.namelist()):
                if member.startswith('Final Project data/Cross/') or member.startswith('Final Project data/Intra/'):
                    target_path = os.path.join(data_dir, member.split('/')[1].lower(), *member.split('/')[2:])
                    os.makedirs(os.path.dirname(target_path), exist_ok=True)

                    if not member.endswith('/'):
                        with zip_ref.open(member) as source, open(target_path, 'wb') as target:
                            shutil.copyfileobj(source, target)
    else:
        logger.info(f'Files already unzipped')


def get_dataset_by_name(dataset_name: str, dataset_dir: str) -> np.ndarray:
    file_parts = sorted([f for f in os.listdir(dataset_dir) if f.startswith(dataset_name)])
    logger.info(f'Processing dataset {os.path.join(dataset_dir, dataset_name)}')
    logger.info(f'{len(file_parts)} files found: {file_parts}')

    all_data = []
    shapes = []
    for file_part in tqdm(file_parts):
        filename_path = os.path.join(dataset_dir, file_part)
        with h5py.File(filename_path, 'r') as f:
            data = f[dataset_name][()]
            shapes.append(data.shape)
            all_data.append(data)

    full_np_array = np.concatenate(all_data, axis=1)
    logger.info(f'Retrieved shapes:  {shapes}')
    logger.info(f'Resulting shape:  {full_np_array.shape}')
    return full_np_array


def z_score_norm(ndarray: np.ndarray) -> np.ndarray:
    norm_array = ndarray.copy()
    norm_array -= np.mean(norm_array, axis=0)
    norm_array /= np.std(norm_array, axis=0)
    return norm_array


def resample_features(input_array: np.ndarray, downsampling_factor) -> np.ndarray:
    new_num_time_steps = input_array.shape[1] // downsampling_factor
    resampled_ndarray = resample(input_array, num=new_num_time_steps, axis=1)
    logger.info(f'Shape change {input_array.shape} -> {resampled_ndarray.shape}')
    return resampled_ndarray


# https://stackoverflow.com/questions/45730504/how-do-i-create-a-sliding-window-with-a-50-overlap-with-a-numpy-array
def window(arr, window_size = 4, o = 2, copy = False):
    sh = (arr.size - window_size + 1, window_size)
    st = arr.strides * 2
    view = np.lib.stride_tricks.as_strided(arr, strides = st, shape = sh)[0::o]
    if copy:
        return view.copy()
    else:
        return view


def repeat_and_one_hot_encode(value, n, num_classes):
    repeated_values = np.repeat(value, n)
    one_hot_encoded = np.eye(num_classes)[repeated_values]
    return one_hot_encoded
