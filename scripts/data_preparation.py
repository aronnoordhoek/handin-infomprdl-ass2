import os
import logging
import h5py
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scripts.utils import (get_dataset_by_name, z_score_norm, resample_features, extract_zip_file, array_to_mesh,
                           repeat_and_one_hot_encode)


def prepare_dataset(type, splits, subject_ids, tasks):
    type_dir = os.path.join(data_dir, type)
    hdf5_file_path = os.path.join(processed_data_dir, f'{type}.h5')

    logging.info(f'Processing {type}')
    with h5py.File(hdf5_file_path, 'w') as hdf5_file:
        for split in splits:
            split_ndarrays = []
            split_ndarrays_meshes = []
            labels = []

            logging.info(f'\tProcessing split: {split}')
            split_dir = os.path.join(type_dir, split)
            for task, label in tasks:
                logging.info(f'\t\tProcessing task: {task}')
                for subject_id in subject_ids:
                    logging.info(f'\t\t\tProcessing subject: {subject_id}')
                    dataset_name = f'{task}_{subject_id}'

                    if not any(dataset_name in filename for filename in os.listdir(split_dir)):
                        logging.warning(f'{dataset_name} not in {split_dir}')
                        continue

                    dataset = get_dataset_by_name(dataset_name=dataset_name, dataset_dir=split_dir)
                    dataset = z_score_norm(dataset)
                    dataset = resample_features(dataset, downsampling_factor)
                    logging.info(f'Final shape: {dataset.shape}')

                    # Slicing
                    dataset_slices = sliding_window_view(dataset, window_shape)[:, ::sample_window_skip].squeeze(axis=0)
                    logging.info(f'Final sliced shape: {dataset_slices.shape}')
                    split_ndarrays.append(dataset_slices)

                    # Creating meshes out of sensor arrays
                    dataset_slices_meshes = np.apply_along_axis(array_to_mesh, axis=1, arr=dataset_slices)
                    logging.info(f'Final sliced meshes shape: {dataset_slices_meshes.shape}')
                    split_ndarrays_meshes.append(dataset_slices_meshes)

                    labels_slices = repeat_and_one_hot_encode(label, dataset_slices.shape[0], 4)
                    labels.append(labels_slices)

            # Concatenate lists
            if split_ndarrays:
                split_ndarrays_cc = np.concatenate(split_ndarrays, axis=0)
                split_ndarrays_meshes_cc = np.concatenate(split_ndarrays_meshes, axis=0)
                labels_cc = np.concatenate(labels, axis=0)

                # H5: Save data files in h5 group:
                group = hdf5_file.create_group(split)

                group.create_dataset('data_1d', data=split_ndarrays_cc)
                group.create_dataset('meshes', data=split_ndarrays_meshes_cc)
                group.create_dataset('labels', data=labels_cc)


def test_h5_result(hdf5_file_path):
    logging.info(f"Dataset: {type}")
    with h5py.File(hdf5_file_path, 'r') as file:
        logging.info(f"Groups in HDF5 file: {list(file.keys())}")

        for group in file.keys():
            logging.info(f'{group.upper()}:')
            data_1d = file[f'{group}/data_1d'][:]
            logging.info(f'\t1D data shape: {data_1d.shape}')
            mesh_data = file[f'{group}/meshes'][:]
            logging.info(f'\tMesh data shape: {mesh_data.shape}')
            label_data = file[f'{group}/labels'][:]
            logging.info(f'\tLabel data shape: {label_data.shape}')


if __name__ == "__main__":
    # Config ===============================================================================================================
    logging.basicConfig(level=logging.INFO)
    logging.getLogger('scripts.utils').setLevel(logging.INFO) # Set logging of util functions here, 'CRITICAL' for minimal
    data_dir = os.path.join('..', 'data')
    # 50 samples per second, see exploration.ipynb for visual test of reasonable factor
    downsampling_factor = 2034 // 50
    sample_window_size = 32
    sample_window_overlap_factor = 2/3

    # Extracting data from data files ======================================================================================
    # Has internal check if already done
    extract_zip_file()

    # General
    tasks = [('rest', 0), ('task_motor', 1), ('task_story_math', 2), ('task_working_memory', 3)]
    sample_window_skip = int(sample_window_size * (1-sample_window_overlap_factor))
    window_shape = (248, sample_window_size)
    processed_data_dir = os.path.join(data_dir, 'processed')
    os.makedirs(processed_data_dir, exist_ok=True)

    # Intra: 1 subject -> 80% - 20% test split over each task.
    subject_ids_intra = [105923]
    prepare_dataset(type='intra', splits=['train', 'val', 'test'], subject_ids=subject_ids_intra, tasks=tasks)

    # Cross: 2 train subjects, 3 test subjects.
    subject_ids_cross_train = [113922, 164636]
    subject_ids_cross_test = [162935, 707749, 725751, 735148]
    subject_ids_cross = subject_ids_cross_train + subject_ids_cross_test
    prepare_dataset(type='cross', splits=['train', 'test1', 'test2', 'test3'], subject_ids=subject_ids_cross, tasks=tasks)

    # Result test
    for type in ['intra', 'cross']:
        hdf5_file_path = os.path.join(processed_data_dir, f'{type}.h5')
        test_h5_result(hdf5_file_path)
