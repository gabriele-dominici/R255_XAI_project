import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import networkx as nx

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

from sklearn.cluster import KMeans, MeanShift, DBSCAN
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances_argmin_min

from torch_geometric.nn import GCNConv

from sklearn import tree, linear_model

import scipy.cluster.hierarchy as hierarchy
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestCentroid
#import umap

from torch_geometric.nn import GNNExplainer

from utilities import *
from activation_classifier import *
import random
from torch import nn
import models
import wandb
import yaml

set_rc_params()

# ensure reproducibility
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)


def main():
    # Set up your default hyperparameters
    with open('./config/random_sweep.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    with open('./config/random.yaml') as file:
        config_param = yaml.load(file, Loader=yaml.FullLoader)
    config = {**config_param, **config}
    tags = config['mode']['values']
    run = wandb.init(tags=tags, config=config)

    # general parameters
    dataset_name = wandb.config.dataset

    model_type = Model_random_proto

    # hyperparameters
    k = wandb.config.k
    seed = wandb.config.seed
    loss_factor = wandb.config.loss_factor

    # other parameters
    train_test_split = 0.8
    num_hidden_units = wandb.config.hidden_dim['values'][dataset_name]
    num_classes = wandb.config.num_classes['values'][dataset_name]
    epochs = wandb.config.epochs['values'][dataset_name]
    lr =wandb.config.lr['values'][dataset_name]
    num_layers = wandb.config.num_layers['values'][dataset_name]
    lr_decay = wandb.config.lr_decay['values']
    early_stopping = wandb.config.early_stopping['values']

    paths = prepare_output_paths(dataset_name, k, seed)

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    G, labels = load_syn_data(dataset_name)
    if dataset_name in ['Cora']:
        data = prepare_real_data(G, train_test_split)
    else:
        data = prepare_syn_data(G, labels, train_test_split)

    prot_indexes = list(anchor_index_for_classes(data['x'], data['y'], k,
                                                 num_classes, data['train_mask']))
    model = model_type(data["x"].shape[1], num_hidden_units,
                       num_classes, num_layers, prot_indexes,
                       data['train_mask'], dataset_name)

    model.apply(weights_init)
    if loss_factor == 0:
        proto_loss = False
    else:
        proto_loss = True
    train(model, data, epochs, lr, lr_decay, early_stopping, paths['base'], proto_loss=proto_loss)

main()


