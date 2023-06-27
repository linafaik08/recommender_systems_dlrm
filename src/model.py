import pandas as pd
import numpy as np
import time

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
        self.train_model.eval()

        losses, l_labels, l_preds = [], [], []

        for elements in eval_data.batches:
            batch = build_batch(elements, self.cols_sparse).to(self.device)
            loss, (_, logits, labels) = self.train_model(batch)

            losses.append(loss)

            l_labels.append(labels)
            l_preds.append(torch.sigmoid(logits))

        losses = torch.stack(losses).detach().cpu().numpy()
        l_labels = torch.cat(l_labels).detach().cpu().numpy()
        l_preds = torch.cat(l_preds).detach().cpu().numpy()

        auc = roc_auc_score(l_labels, l_preds)

        return l_labels, l_preds, np.mean(losses), auc

    def train(self, train_data, nb_batches=None):

        self.train_model.train()
        self.optimizer.zero_grad()

        losses, l_labels, l_preds = [], [], []

        if nb_batches is None or (nb_batches is not None and nb_batches < len(train_data.batches)):
            batch_index = np.arange(len(train_data.batches))
        else:
            batch_index = np.random.randint(0, len(train_data.batches), nb_batches, replace=False)

        for i in batch_index:
            batch = build_batch(train_data.batches[i], self.cols_sparse).to(self.device)
            loss, (_, logits, labels) = self.train_model(batch)

            losses.append(loss)

            l_labels.append(labels)
            l_preds.append(torch.sigmoid(logits))

            loss.backward()
            self.optimizer.step()

        losses = torch.stack(losses).detach().cpu().numpy()
        l_labels = torch.cat(l_labels).detach().cpu().numpy()
        l_preds = torch.cat(l_preds).detach().cpu().numpy()

        auc = roc_auc_score(l_labels, l_preds)

        return l_labels, l_preds, np.mean(losses), auc

    def train_test(self, train_data, val_data, n_epochs, e_patience, nb_batches):
        pbar = tqdm(range(n_epochs))
        k = 0
        scores = {}

        try:
            for epoch in pbar:
                _, _, losses_train, auc_train = self.train(train_data, nb_batches)
                _, _, losses_val, auc_val = self.evaluate(val_data)

                pbar.set_postfix({
                    'epoch': epoch + 1,
                    'loss_train': round(losses_train, 4),
                    'losses_val': round(losses_val, 4),
                    'auc_train': round(auc_train, 4),
                    'auc_val': round(auc_val, 4)
                })

                scores[epoch] = {
                    'loss_train': losses_train,
                    'loss_val': losses_val,
                    'auc_train': round(auc_train, 4),
                    'auc_val': round(auc_val, 4)
                }
                if epoch > 0 and abs(scores[epoch]['auc_val'] - scores[epoch - 1]['auc_val']) < 0.001:
                    k += 1
                if k > e_patience:
                    print('Early stopping')
                    break

        except KeyboardInterrupt:
            print('Interrupted')
            return scores

        return scores


def cross_validation(
        train_df, cols_dense, cols_sparse,
        embedding_dim, num_embeddings_per_feature,
        dense_arch_layer_sizes, over_arch_layer_sizes,
        adagrad, learning_rate, eps,
        batch_size, num_generated_batches_train, num_generated_batches_val,
        kfolds,
        n_epochs, e_patience,
        device,
        seed=123

):
    scores = {}

    for i, (train_index, val_index) in enumerate(kfolds.split(train_df, train_df.purchased)):
        print(f"Fold {i + 1}/{kfolds.n_splits}")

        train_df_kf = train_df.iloc[train_index].reset_index(drop=True)
        val_df_kf = train_df.iloc[val_index].reset_index(drop=True)

        print('   Generate train data ...')

        train_data = RecBatch(
            data=train_df_kf,
            cols_sparse=[c + '_enc' for c in cols_sparse],
            cols_dense=cols_dense,
            col_label="purchased",
            batch_size=batch_size,
            num_generated_batches=num_generated_batches_train,  # 110000,
            seed=seed,
            device=device
        )

        print('   Generate val data ...')

        val_data = RecBatch(
            data=val_df_kf,
            cols_sparse=[c + '_enc' for c in cols_sparse],
            cols_dense=cols_dense,
            col_label="purchased",
            batch_size=batch_size,
            num_generated_batches=num_generated_batches_val,
            seed=seed,
            device=device

        )

        print('   Generate model ...')

        eb_configs = [
            EmbeddingBagConfig(
                name=f"t_{feature_name}",
                embedding_dim=embedding_dim,
                num_embeddings=num_embeddings_per_feature[feature_name + '_enc'],
                feature_names=[feature_name + '_enc'],
                # pooling=torchrec.PoolingType.SUM,
            )
            for feature_idx, feature_name in enumerate(cols_sparse)
        ]

        dlrm_model = DLRM(
            embedding_bag_collection=EmbeddingBagCollection(
                tables=eb_configs, device=device
            ),
            dense_in_features=len(cols_dense),
            dense_arch_layer_sizes=dense_arch_layer_sizes,
            over_arch_layer_sizes=over_arch_layer_sizes,
            dense_device=device,
        )

        train_model = DLRMTrain(dlrm_model).to(device)

        embedding_optimizer = torch.optim.Adagrad if adagrad else torch.optim.SGD

        optimizer_kwargs = {"lr": learning_rate}
        if self.adagrad:
            optimizer_kwargs["eps"] = eps

        apply_optimizer_in_backward(
            optimizer_class=embedding_optimizer,
            params=train_model.model.sparse_arch.parameters(),
            optimizer_kwargs=optimizer_kwargs,
        )

        dense_optimizer = KeyedOptimizerWrapper(
            dict(in_backward_optimizer_filter(train_model.named_parameters())),
            optimizer_with_params(adagrad, learning_rate, eps),
        )

        optimizer = CombinedOptimizer([dense_optimizer])

        print('   Train model ...')
        scores = train_test(train_data, val_data, cols_sparse, train_model, optimizer, n_epochs, e_patience, device)
        scores = pd.DataFrame(scores).T.reset_index().rename(columns={'index': 'epoch'})
        scores['kfold'] = i

        results = scores if i == 0 else pd.concat([results, scores], axis=0)

    return train_model, results


def plot_results(results):
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Scatter(x=results.epoch, y=results.loss_train, name="loss_train"), secondary_y=False)
    fig.add_trace(go.Scatter(x=results.epoch, y=results.loss_val, name="loss_val"), secondary_y=False)

    fig.add_trace(go.Scatter(x=results.epoch, y=results.auc_train, name="auc_train"), secondary_y=True)
    fig.add_trace(go.Scatter(x=results.epoch, y=results.auc_val, name="auc_val"), secondary_y=True)

    fig.update_xaxes(title_text="epoch")
    fig.update_yaxes(title_text="loss", secondary_y=False)
    fig.update_yaxes(title_text="auc", secondary_y=True)

    return fig

def optimizer_with_params(adagrad, learning_rate, eps=None):
    if adagrad:
        return lambda params: torch.optim.Adagrad(
            params, lr=learning_rate, eps=eps
        )
    else:
        return lambda params: torch.optim.SGD(params, lr=learning_rate)
