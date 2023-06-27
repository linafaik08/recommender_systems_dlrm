import pandas as pd
import numpy as np
import time

from torchrec import *
from torchrec.models import dlrm

import torch

from torchrec.datasets.utils import Batch
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

class RecBatch:
    def __init__(
            self, data, cols_sparse,
            cols_dense, col_label, batch_size,
            num_generated_batches, seed, device
    ):
        """
        :param data: The data used for generating batches.
        :param cols_sparse: The list of column names representing sparse features.
        :param cols_dense: The list of column names representing dense features.
        :param col_label: The column name representing the label.
        :param batch_size: The size of each batch.
        :param num_generated_batches: The number of batches to be generated.
        :param seed: The seed used for random number generation.
        :param device: The device on which the tensors will be allocated (CPU or GPU).
        """

        self.data = data
        self.cols_sparse = cols_sparse
        self.cols_dense = cols_dense
        self.col_label = col_label

        self.batch_size = batch_size
        self.num_generated_batches = num_generated_batches
        self.index = self.get_index(seed)
        self.device = device

        self.batches = [self._generate_batch(i) for i in range(self.num_generated_batches)]
        self.batch_index = 0

    def get_index(self, seed):
        """
        Generates the indices for creating batches.
        Depending on the value of num_generated_batches, it performs different operations:
        - If num_generated_batches is None, it calculates the number of batches based on batch_size and the size of the data.
        It then divides the data indices into sublists of size batch_size and returns the indices as a list.
        - If num_generated_batches has a specific value, it uses NumPy's random number generator to sample indices from the data.
        The sampling is done with or without replacement, depending on the size of the data.
        :param seed: for random number generation.
        :return: lists of indices of size batch_size.
        """

        if self.num_generated_batches is None:
            n = self.data.shape[0] // self.batch_size
            r = self.data.shape[0] % self.batch_size
            index = list(np.array(self.data.index[:n * self.batch_size]).reshape(-1, self.data.shape[0] // n))
            if r > 0:
                index.append(self.data.index[n * self.batch_size: n * self.batch_size + r])
            self.num_generated_batches = len(index)
        else:
            np.random.seed(seed=seed)
            replace = True if self.data.shape[0] < self.batch_size * self.num_generated_batches else False
            print('replace', replace)
            index = np.random.choice(
                self.data.index, self.batch_size * self.num_generated_batches, replace=replace) \
                .reshape(-1, self.batch_size)
            index = list(index)

        return index

    def _generate_batch(self, i):
        """
        Generate a single batch based on a given index.
        :param i: index.
        :return: batch.
        """
        sample = self.data.iloc[self.index[i]].copy().reset_index(drop=True)

        values = sample[self.cols_sparse].sum(axis=0).sum(axis=0)
        values = torch.tensor([k for k in values if k != '']).to(self.device)

        lengths = torch.tensor(
            pd.concat([sample[feat].apply(lambda x: len([k for k in x if k != ""])) for feat in self.cols_sparse],
                      axis=0).values,
            dtype=torch.int32
        ).to(self.device)

        dense_features = torch.tensor(sample[self.cols_dense].values, dtype=torch.float32).to(self.device)
        labels = torch.tensor(sample[self.col_label].values, dtype=torch.int32).to(self.device)

        return values, lengths, dense_features, labels

def build_batch(batched_iterator, keys):
    """
    Builds a batch object from the given batched iterator.

    Args:
        batched_iterator (tuple): A tuple containing values, lengths, dense_features, and labels.
        keys (list): A list of keys representing the sparse feature names.

    Returns:
        Batch: A Batch object containing the batched data.
    """
    values, lengths, dense_features, labels = batched_iterator

    # Construct KeyedJaggedTensor for sparse features
    sparse_features = KeyedJaggedTensor.from_lengths_sync(
        keys=[k + '_enc' for k in keys],
        values=values,
        lengths=lengths
    )

    # Create a Batch object with the batched data
    batch = Batch(
        dense_features=dense_features,
        sparse_features=sparse_features,
        labels=labels,
    )
    return batch