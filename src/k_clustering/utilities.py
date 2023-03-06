import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import networkx as nx
import sys

from sklearn import tree, linear_model
from sklearn.cluster import KMeans, MeanShift, DBSCAN, AgglomerativeClustering
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances_argmin_min, pairwise_distances
from sklearn.neighbors import NearestCentroid
import scipy.cluster.hierarchy as hierarchy
import random
import torch_geometric
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader, DenseDataLoader
from torch_geometric.utils import train_test_split_edges
from torch_geometric.data import Data
from torch_geometric.datasets import Twitch, Planetoid

from models import *

def load_syn_data(dataset_str):
    if dataset_str == "BA_Shapes":
        G = nx.readwrite.read_gpickle("../../data/BA_Houses/graph_ba_300_80.gpickel")
        role_ids = np.load("../../data/BA_Houses/role_ids_ba_300_80.npy")

    elif dataset_str == "BA_Grid":
        G = nx.readwrite.read_gpickle("../../data/BA_Grid/graph_ba_300_80.gpickel")
        role_ids = np.load("../../data/BA_Grid/role_ids_ba_300_80.npy")

    elif dataset_str == "BA_Community":
        G = nx.readwrite.read_gpickle("../../data/BA_Community/graph_ba_350_100_2comm.gpickel")
        role_ids = np.load("../../data/BA_Community/role_ids_ba_350_100_2comm.npy")

    elif dataset_str == "Tree_Cycle":
        G = nx.readwrite.read_gpickle("../../data/Tree_Cycle/graph_tree_8_60.gpickel")
        role_ids = np.load("../../data/Tree_Cycle/role_ids_tree_8_60.npy")

    elif dataset_str == "Tree_Grid":
        G = nx.readwrite.read_gpickle("../../data/Tree_Grid/graph_tree_8_80.gpickel")
        role_ids = np.load("../../data/Tree_Grid/role_ids_tree_8_80.npy")

    elif dataset_str == 'Twitch':
        G = Twitch('../data/Twitch', 'EN')[0]
        role_ids = G.y
    elif dataset_str == 'Cora':
        G = Planetoid('../data/Cora', 'cora')[0]
        role_ids = G.y

    else:
        raise Exception("Invalid Syn Dataset Name")

    return G, role_ids

def prepare_real_data(G, train_split):
    train_mask = np.random.rand(len(G.x)) < train_split
    test_mask = ~train_mask
    edge_list = torch.transpose(G.edge_index, 1, 0)
    data = {"x": G.x, "y": G.y, "edges": G.edge_index, "edge_list": edge_list, "train_mask": train_mask,
         "test_mask": test_mask}
    return data

def load_real_data(dataset_str):
    if dataset_str == "Mutagenicity":
        graphs = TUDataset(root='.', name='Mutagenicity')

    elif dataset_str == "Reddit_Binary":
        graphs = TUDataset(root='.', name='REDDIT-BINARY', transform=torch_geometric.transforms.Constant())
    else:
        raise Exception("Invalid Real Dataset Name")

    print()
    print(f'Dataset: {graphs}:')
    print('====================')
    print(f'Number of graphs: {len(graphs)}')
    print(f'Number of features: {graphs.num_features}')
    print(f'Number of classes: {graphs.num_classes}')

    data = graphs[0]  # Get the first graph object.

    print()
    print(data)
    print('=============================================================')

    # Gather some statistics about the first graph.
    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')
    print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
    print(f'Contains isolated nodes: {data.contains_isolated_nodes()}')
    print(f'Contains self-loops: {data.contains_self_loops()}')
    print(f'Is undirected: {data.is_undirected()}')

    return graphs


def prepare_syn_data(G, labels, train_split, if_adj=False):
    existing_node = list(G.nodes)[-1]
    feat_dim = G.nodes[existing_node]["feat"].size
    features = np.zeros((G.number_of_nodes(), feat_dim), dtype=float)
    for i, u in enumerate(G.nodes()):
        features[i, :] = G.nodes[u]["feat"]

    features = torch.from_numpy(features).float()
    labels = torch.from_numpy(labels)

    edge_list = torch.from_numpy(np.array(G.edges))

    edges = edge_list.transpose(0, 1).long()
    if if_adj:
        edges = torch.tensor(nx.to_numpy_matrix(G), requires_grad=True, dtype=torch.float)

    train_mask = np.random.rand(len(features)) < train_split
    test_mask = ~train_mask

    print("Task: Node Classification")
    print("Number of features: ", len(features))
    print("Number of labels: ", len(labels))
    print("Number of classes: ", len(set(labels)))
    print("Number of edges: ", len(edges))

    return {"x": features, "y": labels, "edges": edges, "edge_list": edge_list, "train_mask": train_mask, "test_mask": test_mask}


def set_rc_params():
    small = 14
    medium = 20
    large = 28

    plt.rc('figure', autolayout=True, figsize=(10, 6))
    plt.rc('font', size=medium)
    plt.rc('axes', titlesize=medium, labelsize=small, grid=True)
    plt.rc('xtick', labelsize=small)
    plt.rc('ytick', labelsize=small)
    plt.rc('legend', fontsize=small)
    plt.rc('figure', titlesize=large, facecolor='white')
    plt.rc('legend', loc='upper left')


def plot_activation_space(data, labels, activation_type, layer_num, path, note="", naming_help=""):
    rows = len(data)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title(f"{activation_type} Activations of Layer {layer_num} {note}")

    scatter = ax.scatter(data[:,0], data[:,1], c=labels, cmap='rainbow')
    ax.legend(handles=scatter.legend_elements()[0], labels=list(np.unique(labels)), bbox_to_anchor=(1.05, 1))

    plt.savefig(os.path.join(path, f"{layer_num}_layer{naming_help}.png"))
    plt.show()

def get_subgraph(idx, y, edges, num_expansions):
    graphs = []
    color_maps = []
    labels = []
    node_labels = []

    df = pd.DataFrame(edges)

    # get neighbours
    neighbours = list()
    neighbours.append(idx)

    for i in range(0, num_expansions):
        new_neighbours = list()
        for e in edges:
            if (e[0] in neighbours) or (e[1] in neighbours):
                new_neighbours.append(e[0])
                new_neighbours.append(e[1])

        neighbours = neighbours + new_neighbours
        neighbours = list(set(neighbours))

    new_G = nx.Graph()
    df_neighbours = df[(df[0].isin(neighbours)) & (df[1].isin(neighbours))]
    remaining_edges = df_neighbours.to_numpy()
    new_G.add_edges_from(remaining_edges)

    color_map = []
    node_label = {}
    for node in new_G:
        if node == idx:
            color_map.append('red')
        else:
            color_map.append('green')
    return new_G, color_map, y[idx], node_label

def get_node_distances(clustering_model, data):
    if isinstance(clustering_model, AgglomerativeClustering) or isinstance(clustering_model, DBSCAN):
        x, y_predict = data
        clf = NearestCentroid()
        clf.fit(x, y_predict)
        centroids = clf.centroids_
        res = pairwise_distances(centroids, x)
        res_sorted = np.argsort(res, axis=-1)
    elif isinstance(clustering_model, KMeans):
        res_sorted = clustering_model.transform(data)

    return res_sorted

def plot_samples(indices, y, edges, num_expansions, sim):

    idx_sim = sorted(list(zip(indices[1:], sim)), key=lambda x:x[1])
    top_idx_sim = idx_sim[-5:] + idx_sim[:5]
    top_idx_sim = list(zip(*top_idx_sim))
    indices = [indices[0]] + list(top_idx_sim[0])
    sim = list(top_idx_sim[1])

    sim = [1.0] + sim

    fig, axes = plt.subplots(1, len(indices), figsize=(18, 3 * 1 + 2))
    fig.suptitle(f'Nearest Instances to Cluster Centroid for  Activations of Layer', y=1.005)

    ax_list = axes
    for index, ax in enumerate(axes):
        tg, cm, labels, node_labels = get_subgraph(indices[index], y, edges, num_expansions)

        nx.draw(tg, node_color=cm, with_labels=True, ax=ax)
        ax.set_title(f"label {labels}, sim {sim[index]:.3}", fontsize=14)

    plt.show()

def anchor_index(size, k):
    anchors = np.random.choice(size, size=k, replace=False)
    return anchors

def anchor_index_for_classes(dataset, labels, k, n_classes, train_mask):
    anchors = []
    for c in range(n_classes):
        index_tmp = np.array(range(dataset.shape[0]))[train_mask][labels[train_mask] == c]
        anchors += list(anchor_index(index_tmp, k))
    return anchors

def prepare_output_paths(dataset_name, k, seed):
    path = f"output/{dataset_name}_{k}_seed{seed}/"
    path_tsne = os.path.join(path, "TSNE")
    path_pca = os.path.join(path, "PCA")
    path_umap = os.path.join(path, "UMAP")
    path_kmeans = os.path.join(path, f"{k}_KMeans")
    path_hc = os.path.join(path, f"HC")
    path_ward = os.path.join(path, f"WARD")
    path_dbscan = os.path.join(path, f"DBSCAN")
    path_edges = os.path.join(path, f"edges")
    os.makedirs(path, exist_ok=True)
    os.makedirs(path_tsne, exist_ok=True)
    os.makedirs(path_pca, exist_ok=True)
    os.makedirs(path_umap, exist_ok=True)
    os.makedirs(path_kmeans, exist_ok=True)
    os.makedirs(path_hc, exist_ok=True)
    os.makedirs(path_ward, exist_ok=True)
    os.makedirs(path_dbscan, exist_ok=True)
    os.makedirs(path_edges, exist_ok=True)

    return {"base": path, "TSNE": path_tsne, "PCA": path_pca, "UMAP": path_umap, "KMeans": path_kmeans, "HC": path_hc,
            "Ward": path_ward, "DBSCAN": path_dbscan, "edges": path_edges}
