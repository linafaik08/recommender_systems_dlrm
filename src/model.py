import pandas as pd
import numpy as np
import time

from batch import *

from torchrec import *

import torch

from torchrec.models.dlrm import DLRM, DLRM_DCN, DLRM_Projection, DLRMTrain
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec import EmbeddingBagCollection
from torchrec.optim.keyed import CombinedOptimizer, KeyedOptimizerWrapper
from torchrec.optim.optimizers import in_backward_optimizer_filter
from torchrec.optim.apply_optimizer_in_backward import apply_optimizer_in_backward

from tqdm import tqdm
from sklearn.metrics import roc_auc_score

import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
pio.templates.default = "plotly_white"

class DLRMCustom:
    def __init__(
            self,
            cols_dense, cols_sparse,
            embedding_dim, num_embeddings_per_feature,
            dense_arch_layer_sizes, over_arch_layer_sizes,
            adagrad, learning_rate, eps,
            device
    ):

        self.cols_dense = cols_dense
        self.cols_sparse = cols_sparse
        self.embedding_dim = embedding_dim
        self.num_embeddings_per_feature = num_embeddings_per_feature
        self.dense_arch_layer_sizes = dense_arch_layer_sizes
        self.over_arch_layer_sizes = over_arch_layer_sizes
        self.adagrad = adagrad
        self.learning_rate = learning_rate
        self.eps = eps
        self.device = device

        self.build_model()

    def build_model(self):
        """
        Build a model and initializing related components.
        :return: None
        """
        eb_configs = [
            EmbeddingBagConfig(
                name=f"t_{feature_name}",
                embedding_dim=self.embedding_dim,
                num_embeddings=self.num_embeddings_per_feature[feature_name + '_enc'],
                feature_names=[feature_name + '_enc'],
                # pooling=torchrec.PoolingType.SUM,
            )
            for feature_idx, feature_name in enumerate(self.cols_sparse)
        ]

        dlrm_model = DLRM(
            embedding_bag_collection=EmbeddingBagCollection(
                tables=eb_configs, device=self.device
            ),
            dense_in_features=len(self.cols_dense),
            dense_arch_layer_sizes=self.dense_arch_layer_sizes,
            over_arch_layer_sizes=self.over_arch_layer_sizes,
            dense_device=self.device,
        )

        self.train_model = DLRMTrain(dlrm_model).to(self.device)

        embedding_optimizer = torch.optim.Adagrad if self.adagrad else torch.optim.SGD

        optimizer_kwargs = {"lr": self.learning_rate}
        if self.adagrad:
            optimizer_kwargs["eps"] = self.eps

        apply_optimizer_in_backward(
            optimizer_class=embedding_optimizer,
            params=self.train_model.model.sparse_arch.parameters(),
            optimizer_kwargs=optimizer_kwargs,
        )

        dense_optimizer = KeyedOptimizerWrapper(
            dict(in_backward_optimizer_filter(self.train_model.named_parameters())),
            optimizer_with_params(self.adagrad, self.learning_rate, self.eps),
        )

        self.optimizer = CombinedOptimizer([dense_optimizer])

    def evaluate(self, eval_data):
        """
        Evaluates the trained model on the given evaluation data.
        :param eval_data: evaluation data used for evaluating the model.
        :return: a tuple containing the following:
            - l_labels: The ground truth labels for the evaluation data.
            - l_preds: The predicted probabilities for the evaluation data.
            - loss_mean: The mean loss calculated over the evaluation data.
            - auc: The area under the ROC curve (AUC) score.
        """
        self.train_model.eval()

        losses, l_labels, l_preds = [], [], []

        # Iterate over batches in the evaluation data
        for elements in eval_data.batches:
            # Build a batch object from the elements and move it to the device
            batch = build_batch(elements, self.cols_sparse).to(self.device)

            # Forward pass through the model and calculate the loss
            loss, (_, logits, labels) = self.train_model(batch)

            # Store the loss, labels, and predictions
            losses.append(loss)
            l_labels.append(labels)
            l_preds.append(torch.sigmoid(logits))

        # Convert losses, labels, and predictions to numpy arrays
        losses = torch.stack(losses).detach().cpu().numpy()
        l_labels = torch.cat(l_labels).detach().cpu().numpy()
        l_preds = torch.cat(l_preds).detach().cpu().numpy()

        # Calculate the AUC score
        auc = roc_auc_score(l_labels, l_preds)

        return l_labels, l_preds, np.mean(losses), auc

    def train(self, train_data, nb_batches):
        """
        Trains the model using the provided training data.
        :param train_data: training data used for training the model.
        :param nb_batches (int or None): The number of batches to train on.
                If None, train on all batches.
                If an integer is provided, randomly select the specified number of batches.
        :return:A tuple containing the following:
            - l_labels: The ground truth labels for the training data.
            - l_preds: The predicted probabilities for the training data.
            - loss_mean: The mean loss calculated over the training data.
            - auc: The area under the ROC curve (AUC) score.
        """

        self.train_model.train()
        self.optimizer.zero_grad()

        losses, l_labels, l_preds = [], [], []

        # Determine the batch index to train on
        if nb_batches is None or (nb_batches is not None and nb_batches < len(train_data.batches)):
            batch_index = np.arange(len(train_data.batches))
        else:
            batch_index = np.random.randint(0, len(train_data.batches), nb_batches, replace=False)

        # Iterate over selected batch indices and perform training
        for i in batch_index:
            # Build a batch object from the selected batch and move it to the device
            batch = build_batch(train_data.batches[i], self.cols_sparse).to(self.device)
            # Forward pass through the model and calculate the loss
            loss, (_, logits, labels) = self.train_model(batch)
            # Store the loss, labels, and predictions
            losses.append(loss)
            l_labels.append(labels)
            l_preds.append(torch.sigmoid(logits))
            # Perform backpropagation and optimization step
            loss.backward()
            self.optimizer.step()

        # Convert losses, labels, and predictions to numpy arrays
        losses = torch.stack(losses).detach().cpu().numpy()
        l_labels = torch.cat(l_labels).detach().cpu().numpy()
        l_preds = torch.cat(l_preds).detach().cpu().numpy()

        # Calculate the AUC score
        auc = roc_auc_score(l_labels, l_preds)

        return l_labels, l_preds, np.mean(losses), auc

    def train_test(self, train_data, test_data, n_epochs, e_patience, nb_batches = None):
        """
        Trains and evaluates the model on the provided training and validation data for multiple epochs.
        :param train_data: training data used for training the model.
        :param test_data: test data used for evaluating the model during training.
        :param n_epochs (int): number of epochs to train the model.
        :param e_patience (int): maximum number of epochs to tolerate without improvement in validation AUC before early stopping.
        :param nb_batches (int or None): The number of batches to train on. If None, train on all batches.
                                  If an integer is provided, randomly select the specified number of batches.
        :return: scores: a dictionary containing the evaluation scores for each epoch:
            {epoch_number: {
                    'loss_train': training loss for the epoch,
                    'loss_test': test loss for the epoch,
                    'auc_train': training AUC score for the epoch,
                    'auc_test': validation AUC score for the epoch
                },
        """

        pbar = tqdm(range(n_epochs))
        k = 0
        scores = {}

        try:
            # Iterate over the specified number of epochs
            for epoch in pbar:
                _, _, losses_train, auc_train = self.train(train_data, nb_batches)
                _, _, losses_val, auc_val = self.evaluate(test_data)

                # Update the progress bar with current epoch and evaluation scores
                pbar.set_postfix({
                    'epoch': epoch + 1,
                    'loss_train': round(losses_train, 4),
                    'losses_test': round(losses_val, 4),
                    'auc_train': round(auc_train, 4),
                    'auc_test': round(auc_val, 4)
                })

                # Store the evaluation scores for the current epoch
                scores[epoch] = {
                    'loss_train': losses_train,
                    'loss_test': losses_val,
                    'auc_train': round(auc_train, 4),
                    'auc_test': round(auc_val, 4)
                }
                # Check for early stopping based on the change in validation AUC
                if epoch > 0 and abs(scores[epoch]['auc_test'] - scores[epoch - 1]['auc_test']) < 0.001:
                    k += 1
                if k > e_patience:
                    print('Early stopping')
                    break

        except KeyboardInterrupt:
            print('Interrupted')
            return scores

        return scores


def plot_results(results):
    """
    Plots the training and validation results.
    :param results: a dictionary containing the evaluation scores for each epoch.
    :return: fig: the plotly figure object containing the plotted results.

    """
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    # Add training and validation loss traces to the figure
    fig.add_trace(go.Scatter(x=results.epoch, y=results.loss_train, name="loss_train"), secondary_y=False)
    fig.add_trace(go.Scatter(x=results.epoch, y=results.loss_test, name="loss_test"), secondary_y=False)
    # Add training and validation AUC traces to the figure
    fig.add_trace(go.Scatter(x=results.epoch, y=results.auc_train, name="auc_train"), secondary_y=True)
    fig.add_trace(go.Scatter(x=results.epoch, y=results.auc_test, name="auc_test"), secondary_y=True)
    # Update x-axis and y-axes titles
    fig.update_xaxes(title_text="epoch")
    fig.update_yaxes(title_text="loss", secondary_y=False)
    fig.update_yaxes(title_text="auc", secondary_y=True)

    return fig

def optimizer_with_params(adagrad, learning_rate, eps=None):
    """
    Returns an optimizer function with specified parameters.
    :param adagrad: flag indicating whether to use Adagrad optimizer.
    :param learning_rate: learning rate for the optimizer.
    :param eps: epsilon value for Adagrad optimizer (optional).
    :return: an optimizer function that takes model parameters as input and returns an optimizer.

    """
    if adagrad:
        # Return Adagrad optimizer with specified learning rate and epsilon
        return lambda params: torch.optim.Adagrad(
            params, lr=learning_rate, eps=eps
        )
    else:
        # Return SGD optimizer with specified learning rate
        return lambda params: torch.optim.SGD(params, lr=learning_rate)
