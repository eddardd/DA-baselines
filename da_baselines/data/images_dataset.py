import os
import torch
import numpy as np

from PIL import Image
from da_baselines.data import all_domains
from da_baselines.data import all_datasets
from da_baselines.data import all_definitions


class ImagesDataset(torch.utils.data.Dataset):
    r"""Dataset of Images. This class implements different object
    recognition domain adaptation benchmarks. Implemented datasets:
    Office31 of [1], Office-Home of [2], Adaptiope of [3],
    DomainNet of [4].

    Parameters
    ----------
    root : str
        Path towards the root of dataset.
    dataset_name : str, optional (default='office31')
        Name of selected dataset.
    domains : list of str
        List of domains in the dataset.
    transform : torchvision transforms
        List of transformations to apply to images
    train : bool, optional (default=True)
        Include training samples in the dataset
    test : bool, optional (default=True)
        Include test samples in the dataset
    multi_source : bool, optional (default=True)
        If True, considers that the dataset consists of multiple source
        domains. This implies that batches are separated according the domain.
    """

    def __init__(self,
                 root,
                 dataset_name='office31',
                 domains=None,
                 transform=None,
                 train=True,
                 test=True,
                 multi_source=True):
        assert dataset_name in all_datasets, (f"Dataset {dataset_name} not"
                                              f" implemented.")

        self.dataset_name = dataset_name
        self.name2cat = all_definitions[dataset_name]
        self.all_domains = all_domains[dataset_name]

        self.root = root
        self.folds_path = os.path.join(root, 'folds')
        if multi_source:
            default = self.all_domains[:-1]
        else:
            default = [self.all_domains[0]]
        self.domains = domains if domains is not None else default
        self.transform = transform
        self.train = train
        self.test = test
        self.multi_source = multi_source
        self.num_classes = len(self.name2cat)

        if multi_source:
            self.filepaths = {}
            self.labels = {}

            for domain in self.domains:
                (filepaths,
                 labels) = self.__get_filenames_and_labels(domain,
                                                           train=train,
                                                           test=test)

                self.filepaths[domain] = filepaths
                self.labels[domain] = labels
        else:
            self.filepaths = []
            self.labels = []

            for domain in self.domains:
                (filepaths,
                 labels) = self.__get_filenames_and_labels(domain,
                                                           train=train,
                                                           test=test)

                self.filepaths.append(filepaths)
                self.labels.append(labels)
            self.filepaths = np.concatenate(self.filepaths)
            self.labels = np.concatenate(self.labels)

    def __str__(self):
        r"""Returns a string representing the dataset."""
        return (f"Dataset {self.dataset_name} with domains {self.domains}"
                f"  and {len(self)} samples")

    def __repr__(self):
        r"""String representation for the dataset"""
        return str(self)

    def __get_filenames_and_labels(self, dom, train=True, test=True):
        r"""Get filenames and labels.

        Parameters
        ----------
        dom : str
            Domain for which filenames and labels will be acquired.
        train : bool, optional (default=True)
            If True, includes train samples in the filenames and labels.
        test : bool, optional (default=True)
            If True, includes test samples in the filenames and labels.
        """
        filepaths, labels = [], []

        class_and_filenames = []

        train_path = os.path.join(self.folds_path,
                                  f'{dom}_train_filenames.txt')
        with open(train_path, 'r') as file:
            train_filenames = file.read().split('\n')[:-1]

        if train:
            class_and_filenames += train_filenames

        test_path = os.path.join(self.folds_path,
                                 f'{dom}_test_filenames.txt')
        with open(test_path, 'r') as file:
            test_filenames = file.read().split('\n')[:-1]

        if test:
            class_and_filenames += test_filenames

        classes = [fname.split('/')[0] for fname in class_and_filenames]
        filenames = [fname.split('/')[1] for fname in class_and_filenames]

        for c, fname in zip(classes, filenames):
            filepaths.append(os.path.join(
                self.root, dom, c, fname))
            labels.append(self.name2cat[c])

        filepaths = np.array(filepaths)
        labels = torch.from_numpy(np.array(labels)).long()

        return filepaths, labels

    def __len__(self):
        r"""Returns the number of samples in the dataset"""
        if self.multi_source:
            return min([
                len(self.filepaths[domain]) for domain in self.filepaths])
        else:
            return len(self.filepaths)

    def __process_index_single_source(self, idx):
        r"""Returns an image and a label corresponding to the index idx."""
        x, y = Image.open(self.filepaths[idx]).convert('RGB'), self.labels[idx]

        if self.transform:
            x = self.transform(x)

        return x, y

    def __process_indices_single_source(self, idx):
        r"""Returns a batch of images and labels corresponding to the
        indices in idx."""
        x, y = [], []
        for i in idx:
            _x, _y = Image.open(self.filepaths[i]), self.labels[i]

            if self.transform:
                _x = self.transform(_x)
            x.append(_x)
            y.append(_y)
        return torch.stack(x), torch.stack(y)

    def __process_index_multi_source(self, idx):
        r"""Returns a list of images corresponding to the image and label
        idx of each domain."""
        x, y = [], []
        for domain in self.domains:
            _x = Image.open(self.filepaths[domain][idx])
            if self.transform:
                _x = self.transform(_x)
            _y = self.labels[domain][idx]
            x.append(_x)
            y.append(_y)
        return x, y

    def __process_indices_multi_source(self, idx):
        r"""Returns a list of batches of images and labels corresponding to
        the indices in idx."""
        x, y = [], []
        for domain, inds in zip(self.domains, idx):
            _x = Image.open(self.filepaths[domain][inds])
            if self.transform:
                _x = self.transform(_x)
            _y = self.labels[domain][inds]

            x.append(_x)
            y.append(_y)
        return x, y

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if isinstance(idx, list):
            if self.multi_source:
                return self.__process_indices_multi_source(idx)
            else:
                return self.__process_indices_single_source(idx)
        else:
            if self.multi_source:
                return self.__process_index_multi_source(idx)
            else:
                return self.__process_index_single_source(idx)
