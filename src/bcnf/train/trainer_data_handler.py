import os
import pickle

import numpy as np
import torch
from torch._C import dtype as torch_dtype
from torch.utils.data import DataLoader, RandomSampler, Subset, TensorDataset

from bcnf.simulation.sampling import generate_data
# from bcnf.train.training_data_converter import RenderConverter
from bcnf.utils import ParameterIndexMapping


class TrainerDataHandler:
    def __init__(self) -> None:
        pass

    def get_data_for_training(
            self,
            data_config: dict,
            parameter_index_mapping: ParameterIndexMapping,
            dtype: torch_dtype,
            verbose: bool = False) -> TensorDataset:
        """
        Gts data for training the model

        Parameters
        ----------
        config : dict
            A dictionary containing the configuration parameters for the data

        Returns
        -------
        dataset : TensorDataset
            A PyTorch TensorDataset containing the data for training the model
        """
        if not os.path.exists(data_config['path']):
            if verbose:
                print(f'No data found at {data_config["path"]}. Generating data...')
            data = generate_data(
                n=data_config['n_samples'],
                output_type=data_config['output_type'],
                dt=data_config['dt'],
                T=data_config['T'],
                config_file=data_config['config_file'],
                verbose=data_config['verbose'],
                break_on_impact=data_config['break_on_impact'],
                do_filter=data_config['do_filter'])

            with open(data_config['path'], 'wb') as f:
                pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        else:
            if verbose:
                print(f'Loading data from {data_config["path"]}...')
            with open(data_config['path'], 'rb') as f:
                data = pickle.load(f)

        if data_config['output_type'] == 'videos':
            X = np.array(data['videos'])
            print(X.shape)
        elif data_config['output_type'] == 'trajectory':
            X = np.array(data['trajectory'])
        else:
            raise ValueError(f'Unknown output type: {data_config["output_type"]}')

        y = parameter_index_mapping.vectorize(data)

        # Make the correct type for the data
        X = torch.Tensor(X).to(dtype)
        y = torch.Tensor(y).to(dtype)

        if verbose:
            print(f'Using {data_config["output_type"]} data for training. Shapes:')
            print(f'X shape: {X.shape}')
            print(f'y shape: {y.shape}')

        # Matches pairs of lables and data, so dataset[0] returns tuple of the first entry in X and y
        dataset = TensorDataset(X, y)

        return dataset

    def _generate_data_for_training(self, config: dict) -> dict[str, list]:
        data = generate_data(
            name=config["name"],
            overwrite=config["overwrite"],
            config_file=config["data_generation_config_file"],
            n=config["n_samples"],
            type=config["data_type"],
            SPF=config["dt"],
            T=config["T"],
            ratio=config["ratio"],
            fov_horizontal=config["fov_horizontal"],
            print_acc_rej=config["print_acc_rej"],
            num_cams=config["num_cams"],
            break_on_impact=config["break_on_impact"],
            verbose=config["verbose"])

        return data

    def make_data_loader(
            self,
            dataset: TensorDataset,
            batch_size: int,
            pin_memory: bool,
            num_workers: int = 2) -> DataLoader:
        """
        Create a DataLoader for the given dataset

        Parameters
        ----------
        dataset : torch.tensor
            The dataset to use for training
        batch_size : int
            The batch size to use for training
        num_workers : int
            The number of workers to use for loading the data

        Returns
        -------
        loader : torch.utils.data.DataLoader
            The DataLoader for the given dataset
        """
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=pin_memory,
            num_workers=num_workers)

    def show_data_summary(
            self,
            data: DataLoader) -> None:
        """
        Verify the data

        Parameters
        ----------
        data : torch.utils.data.DataLoader
            The data to verify

        Returns
        -------
        None
        """
        print()
        print("Feature network input shape:", data[0][0].shape)
        print("Feature network device:", data[0][0].device)
        print("NF network input shape:", data[0][1].shape)
        print("NF network device:", data[0][1].device)

    def split_dataset(
            self,
            dataset: DataLoader,
            split_ratio: float) -> tuple[Subset, Subset]:
        """
        Split the data into training and validation sets

        Parameters
        ----------
        data : torch.utils.data.DataLoader
            The data to split
        split_ratio : float
            The ratio to split the data

        Returns
        -------
        train_data : torch.utils.data.Subset
            The training data
        val_data : torch.utils.data.Subset
            The validation data
        """
        train_size = int(1 - split_ratio * len(dataset))

        # Create a random sampler to shuffle the indices
        indices = list(range(len(dataset)))
        RandomSampler(indices)

        # Split indices into training and validation sets
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]

        # Create Subset datasets
        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)

        return train_dataset, val_dataset
