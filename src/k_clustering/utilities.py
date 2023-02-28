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
        train_mask = np.random.rand(len(G.x)) < 0.8
        test_mask = ~train_mask
        edge_list = torch.transpose(G.edge_index, 1, 0)
        G = {"x": G.x, "y": G.y, "edges": G.edge_index, "edge_list": edge_list, "train_mask": train_mask, "test_mask": test_mask}
    elif dataset_str == 'cora':
        G = Planetoid('../data/cora', 'cora')[0]
        role_ids = G.y
        train_mask = np.random.rand(len(G.x)) < 0.8
        test_mask = ~train_mask
        edge_list = torch.transpose(G.edge_index, 1, 0)
        G = {"x": G.x, "y": G.y, "edges": G.edge_index, "edge_list": edge_list, "train_mask": train_mask,
             "test_mask": test_mask}

    else:
        raise Exception("Invalid Syn Dataset Name")

    return G, role_ids


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

def prepare_syn_data_edges(G, labels, train_split, if_adj=False):
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

    data = Data(x=features, edge_index=edges)
    torch.manual_seed(0)
    data = train_test_split_edges(data, val_ratio=train_split*0.2, test_ratio=1-train_split)

    print("Task: Node Classification")
    print("Number of features: ", len(features))
    print("Number of labels: ", len(labels))
    print("Number of classes: ", len(set(labels)))
    print("Number of edges: ", len(edges))

    # complete_list_neg_edges = []
    # for index_i, i in enumerate(data['train_neg_adj_mask']):
    #     for index_j, j in enumerate(i):
    #         if i[index_j]:
    #             complete_list_neg_edges += [[index_i, index_j]]

    # tmp_test_neg = data['test_neg_edge_index'].transpose(0, 1).long().tolist()
    # tmp_test_pos = data['test_pos_edge_index'].transpose(0, 1).long().tolist()
    #
    # complete_list_neg_edges = [i for i in complete_list_neg_edges if i not in tmp_test_neg and i not in tmp_test_pos]

    # complete_list_neg_edges = np.array(complete_list_neg_edges)
    # subset_train_neg_edges = complete_list_neg_edges[np.random.choice(complete_list_neg_edges.shape[0],
    #                                                                   int(data['train_pos_edge_index'].shape[1]),
    #                                                                   replace=False)]

    # return {"x": features, "y": labels, "edges": edges, "edge_list": edge_list,
    #         "train_pos_edge_index": data['train_pos_edge_index'],
    #         "train_neg_edge_index": torch.from_numpy(subset_train_neg_edges).transpose(0, 1).long(),
    #         "test_neg_edge_index": data['test_neg_edge_index'],
    #         "test_pos_edge_index": data['test_pos_edge_index']}
    return {"x": features, "y": labels, "edges": data['train_pos_edge_index'],
            "edge_list": data['train_pos_edge_index'].transpose(0, 1).long(),
            "train_pos_edge_index": data['val_pos_edge_index'],
            "train_neg_edge_index": data['val_neg_edge_index'],
            "test_neg_edge_index": data['test_neg_edge_index'],
            "test_pos_edge_index": data['test_pos_edge_index']}

def prepare_syn_data_edge_classification(G, labels, train_split, if_adj=False):
    existing_node = list(G.nodes)[-1]
    feat_dim = G.nodes[existing_node]["feat"].size
    features = np.zeros((G.number_of_nodes(), feat_dim), dtype=float)
    for i, u in enumerate(G.nodes()):
        features[i, :] = G.nodes[u]["feat"]

    features = torch.from_numpy(features).float()
    node_labels = torch.from_numpy(labels)

    edge_list = torch.from_numpy(np.array(G.edges))

    edges = edge_list.transpose(0, 1).long()
    if if_adj:
        edges = torch.tensor(nx.to_numpy_matrix(G), requires_grad=True, dtype=torch.float)

    labels = []
    for index in range(edges.shape[1]):
        a = edges[1][index]
        b = edges[0][index]

        if node_labels[a] == 1 and node_labels[b] == 1:
            labels += [1]
        elif node_labels[a] == 2 and node_labels[b] == 2:
            labels += [2]
        elif (node_labels[a] == 1 and node_labels[b] == 3) or (node_labels[a] == 3 and node_labels[b] == 1):
            labels += [3]
        elif (node_labels[a] == 1 and node_labels[b] == 2) or (node_labels[a] == 2 and node_labels[b] == 1):
            labels += [4]
        else:
            labels += [0]

    labels = torch.tensor(labels).long()
    train_mask = np.random.rand(edges.shape[1]) < train_split
    test_mask = ~train_mask

    print("Task: Node Classification")
    print("Number of features: ", len(features))
    print("Number of labels: ", len(labels))
    print("Number of classes: ", len(set(labels)))
    print("Number of edges: ", len(edges))

    return {"x": features, "y": labels, "edges": edges, "edge_list": edge_list,
            "train_mask": train_mask, "test_mask": test_mask,
            "node_labels": node_labels}

def prepare_syn_data_edges_multiclass(G, labels, train_split, num_classes, if_adj=False):
    existing_node = list(G.nodes)[-1]
    feat_dim = G.nodes[existing_node]["feat"].size
    features = np.zeros((G.number_of_nodes(), feat_dim), dtype=float)
    for i, u in enumerate(G.nodes()):
        features[i, :] = G.nodes[u]["feat"]

    features = torch.from_numpy(features).float()
    node_labels = torch.from_numpy(labels)

    edge_list = torch.from_numpy(np.array(G.edges))

    edges = edge_list.transpose(0, 1).long()
    if if_adj:
        edges = torch.tensor(nx.to_numpy_matrix(G), requires_grad=True, dtype=torch.float)

    data = Data(x=features, edge_index=edges)
    torch.manual_seed(0)
    data = train_test_split_edges(data, val_ratio=train_split*0.2, test_ratio=1-train_split)

    print("Task: Node Classification")
    print("Number of features: ", len(features))
    print("Number of labels: ", len(node_labels))
    print("Number of classes: ", len(set(node_labels)))
    print("Number of edges: ", len(edges))

    train_labels = []
    test_labels = []
    to_add = [[],[]]
    final_train_edges = [[],[]]
    final_test_edges = [[],[]]
    train_pos = data['val_pos_edge_index']
    train_neg = data['val_neg_edge_index']
    train_neg = train_neg[:, :int(train_neg.shape[1]/num_classes)]
    test_pos = data['test_pos_edge_index']
    test_neg = data['test_neg_edge_index']
    test_neg = test_neg[:, :int(test_neg.shape[1]/num_classes)]

    for index in range(train_pos.shape[1]):
        a = train_pos[0][index]
        b = train_pos[1][index]

        if node_labels[a] == 1 and node_labels[b] == 1:
            train_labels += [2]
            final_train_edges[0] += [a]
            final_train_edges[1] += [b]
        elif node_labels[a] == 2 and node_labels[b] == 2:
            train_labels += [3]
            final_train_edges[0] += [a]
            final_train_edges[1] += [b]
        elif (node_labels[a] == 1 and node_labels[b] == 3) or (node_labels[a] == 3 and node_labels[b] == 1):
            train_labels += [4]
            final_train_edges[0] += [a]
            final_train_edges[1] += [b]
        elif (node_labels[a] == 1 and node_labels[b] == 2) or (node_labels[a] == 2 and node_labels[b] == 1):
            train_labels += [5]
            final_train_edges[0] += [a]
            final_train_edges[1] += [b]
        elif random.random() > 0.8:
            train_labels += [1]
            final_train_edges[0] += [a]
            final_train_edges[1] += [b]
        else:
            to_add[0] += [a]
            to_add[1] += [b]

    train_labels = torch.tensor(train_labels).long()
    train_labels = torch.cat([train_labels, torch.zeros(train_neg.shape[1]).long()])
    final_train_edges = torch.tensor(final_train_edges).long()

    for index in range(test_pos.shape[1]):
        a = test_pos[0][index]
        b = test_pos[1][index]

        if node_labels[a] == 1 and node_labels[b] == 1:
            test_labels += [2]
            final_test_edges[0] += [a]
            final_test_edges[1] += [b]
        elif node_labels[a] == 2 and node_labels[b] == 2:
            test_labels += [3]
            final_test_edges[0] += [a]
            final_test_edges[1] += [b]
        elif (node_labels[a] == 1 and node_labels[b] == 3) or (node_labels[a] == 3 and node_labels[b] == 1):
            test_labels += [4]
            final_test_edges[0] += [a]
            final_test_edges[1] += [b]
        elif (node_labels[a] == 1 and node_labels[b] == 2) or (node_labels[a] == 2 and node_labels[b] == 1):
            test_labels += [5]
            final_test_edges[0] += [a]
            final_test_edges[1] += [b]
        elif random.random() > 0.8:
            test_labels += [1]
            final_test_edges[0] += [a]
            final_test_edges[1] += [b]
        else:
            to_add[0] += [a]
            to_add[1] += [b]

    test_labels = torch.tensor(test_labels).long()
    test_labels = torch.cat([test_labels, torch.zeros(test_neg.shape[1]).long()])
    final_test_edges = torch.tensor(final_test_edges).long()

    to_add = torch.tensor(to_add).long()

    edges = torch.cat([data['train_pos_edge_index'], to_add], dim=1)

    return {"x": features, "y_train": train_labels, "y_test": test_labels, "edges": edges,
            "edge_list": edges.transpose(0, 1).long(),
            "train_pos_edge_index": final_train_edges,
            "train_neg_edge_index": train_neg,
            "test_pos_edge_index": final_test_edges,
            "test_neg_edge_index": test_neg,
            }

def prepare_real_data(graphs, train_split, batch_size, dataset_str):
    graphs = graphs.shuffle()

    train_idx = int(len(graphs) * train_split)
    train_set = graphs[:train_idx]
    test_set = graphs[train_idx:]

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

    if dataset_str == "Mutagenicity":
        full_loader = DataLoader(test_set, batch_size=int(len(test_set)), shuffle=True)
        small_loader = DataLoader(test_set, batch_size=int(len(test_set) * 0.1))

    elif dataset_str == "Reddit_Binary":
        full_loader = DataLoader(test_set, batch_size=int(len(test_set) * 0.1), shuffle=True)
        small_loader = DataLoader(test_set, batch_size=int(len(test_set) * 0.005))

    train_zeros = 0
    train_ones = 0
    for data in train_set:
        train_ones += np.sum(data.y.detach().numpy())
        train_zeros += len(data.y.detach().numpy()) - np.sum(data.y.detach().numpy())

    test_zeros = 0
    test_ones = 0
    for data in test_set:
        test_ones += np.sum(data.y.detach().numpy())
        test_zeros += len(data.y.detach().numpy()) - np.sum(data.y.detach().numpy())

    print()
    print(f"Class split - Training 0: {train_zeros} 1:{train_ones}, Test 0: {test_zeros} 1: {test_ones}")


    return train_loader, test_loader, full_loader, small_loader

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


# def plot_clusters(data, labels, clustering_type, k, layer_num, path, data_type, reduction_type="", note=""):
#     fig, ax = plt.subplots(figsize=(10, 6))
#     fig.suptitle(f'{clustering_type} Clustered {data_type} Activations of Layer {layer_num} {note}')
#
#     for i in range(k):
#         scatter = ax.scatter(data[labels == i,0], data[labels == i,1], label=f'Cluster {i}')
#
#     ax.legend(bbox_to_anchor=(1.05, 1))
#     plt.savefig(os.path.join(path, f"{layer_num}layer_{data_type}{reduction_type}.png"))
#     plt.show()

def get_top_subgraphs_with_edges(top_indices, y, edges_list, edges, num_expansions, graph_data=None, graph_name=None):
    graphs = []
    color_maps = []
    labels = []
    node_labels = []

    df = pd.DataFrame(edges_list)
    for idx in top_indices:
        # get neighbours
        neighbours = list()
        n1, n2 = edges[0][idx], edges[1][idx]
        neighbours.append(n1)
        neighbours.append(n2)

        for i in range(0, num_expansions):
            new_neighbours = list()
            for e in edges_list:
                if (e[0] in neighbours) or (e[1] in neighbours):
                    new_neighbours.append(e[0])
                    new_neighbours.append(e[1])

            neighbours = neighbours + new_neighbours
            neighbours = list(set(neighbours))

        new_G = nx.Graph()
        df_neighbours = df[(df[0].isin(neighbours)) & (df[1].isin(neighbours))]
        remaining_edges = df_neighbours.to_numpy()
        new_G.add_edges_from(remaining_edges)
        # if y[idx] == 1:
        #     new_G.add_edge(n1, n2)

        color_map = []
        node_label = {}
        if graph_data is None:
            for node in new_G:
                if node in [n1, n2]:
                    color_map.append('green')
                else:
                    color_map.append('pink')
        else:
            if graph_name == "Mutagenicity":
                ids = ["C", "O", "Cl", "H", "N", "F", "Br", "S", "P", "I", "Na", "K", "Li", "Ca"]
            elif graph_name == "REDDIT-BINARY":
                ids = []

            for node in zip(new_G):
                node = node[0]
                color_idx = graph_data[node]
                color_map.append(color_idx)
                node_label[node] = f"{ids[color_idx]}"

        color_maps.append(color_map)
        graphs.append(new_G)
        labels.append(y[idx])
        node_labels.append(node_label)

    return graphs, color_maps, labels, node_labels

def plot_samples_edges(clustering_model, data, y, layer_num, k, clustering_type, reduction_type, num_nodes_view, edges_list, edges,
                       num_expansions, path, graph_data=None, graph_name=None):

    res_sorted = get_node_distances(clustering_model, data)

    if isinstance(num_nodes_view, int):
        num_nodes_view = [num_nodes_view]
    col = sum([abs(number) for number in num_nodes_view])

    fig, axes = plt.subplots(k, col, figsize=(18, 3 * k + 2))
    fig.suptitle(f'Nearest Instances to {clustering_type} Cluster Centroid for {reduction_type} Activations of Layer {layer_num}', y=1.005)

    if graph_data is not None:
        fig2, axes2 = plt.subplots(k, col, figsize=(18, 3 * k + 2))
        fig2.suptitle(f'Nearest Instances to {clustering_type} Cluster Centroid for {reduction_type} Activations of Layer {layer_num} (by node index)', y=1.005)

    l = list(range(0, k))
    sample_graphs = []
    sample_feat = []

    for i, ax_list in zip(l, axes):
        if isinstance(clustering_model, AgglomerativeClustering) or isinstance(clustering_model, DBSCAN):
            distances = res_sorted[i]
        elif isinstance(clustering_model, KMeans):
            distances = res_sorted[:, i]

        top_graphs, color_maps = [], []
        for view in num_nodes_view:
            if view < 0:
                top_indices = np.argsort(distances)[::][view:]
            else:
                top_indices = np.argsort(distances)[::][:view]

            tg, cm, labels, node_labels = get_top_subgraphs_with_edges(top_indices, y, edges_list, edges, num_expansions, graph_data, graph_name)
            top_graphs = top_graphs + tg
            color_maps = color_maps + cm

        if graph_data is None:
            for ax, new_G, color_map, g_label in zip(ax_list, top_graphs, color_maps, labels):
                nx.draw(new_G, node_color=color_map, with_labels=True, ax=ax)
                ax.set_title(f"label {g_label}", fontsize=14)
        else:
            for ax, new_G, color_map, g_label, n_labels in zip(ax_list, top_graphs, color_maps, labels, node_labels):
                nx.draw(new_G, node_color=color_map, with_labels=True, ax=ax, labels=n_labels)
                ax.set_title(f"label {g_label}", fontsize=14)

            for ax, new_G, color_map, g_label, n_labels in zip(axes2[i], top_graphs, color_maps, labels, node_labels):
                nx.draw(new_G, node_color=color_map, with_labels=True, ax=ax)
                ax.set_title(f"label {g_label}", fontsize=14)

        sample_graphs.append((top_graphs[0], top_indices[0]))
        sample_feat.append(color_maps[0])

    views = ''.join((str(i) + "_") for i in num_nodes_view)
    if isinstance(clustering_model, AgglomerativeClustering):
        fig.savefig(os.path.join(path, f"{k}h_{layer_num}layer_{clustering_type}_{views}view.png"))
    else:
        fig.savefig(os.path.join(path, f"{layer_num}layer_{clustering_type}_{reduction_type}_{views}view.png"))

    if graph_data is not None:
        if isinstance(clustering_model, AgglomerativeClustering):
            fig2.savefig(os.path.join(path, f"{k}h_{layer_num}layer_{clustering_type}_{views}view_by_node.png"))
        else:
            fig2.savefig(os.path.join(path, f"{layer_num}layer_{clustering_type}_{reduction_type}_{views}view_by_node.png"))

    plt.show()

    return sample_graphs, sample_feat

def get_top_subgraphs(top_indices, y, edges, num_expansions, graph_data=None, graph_name=None):
    graphs = []
    color_maps = []
    labels = []
    node_labels = []

    df = pd.DataFrame(edges)

    for idx in top_indices:
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
        if graph_data is None:
            for node in new_G:
                if node in top_indices:
                    color_map.append('green')
                else:
                    color_map.append('pink')
        else:
            if graph_name == "Mutagenicity":
                ids = ["C", "O", "Cl", "H", "N", "F", "Br", "S", "P", "I", "Na", "K", "Li", "Ca"]
            elif graph_name == "REDDIT-BINARY":
                ids = []

            for node in zip(new_G):
                node = node[0]
                color_idx = graph_data[node]
                color_map.append(color_idx)
                node_label[node] = f"{ids[color_idx]}"

        color_maps.append(color_map)
        graphs.append(new_G)
        labels.append(y[idx])
        node_labels.append(node_label)


    return graphs, color_maps, labels, node_labels

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


def plot_clusters(data, labels, clustering_type, k, layer_num, path, data_type, reduction_type="", note=""):
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.suptitle(f'{clustering_type} Clustered {data_type} Activations of Layer {layer_num} {note}')

    for i in range(k):
        scatter = ax.scatter(data[labels == i,0], data[labels == i,1], label=f'Cluster {i}')

    ncol = 1
    if k > 20:
        ncol = int(k / 20) + 1
    ax.legend(bbox_to_anchor=(1.05, 1), ncol=ncol)
    plt.savefig(os.path.join(path, f"{layer_num}layer_{data_type}{reduction_type}.png"))
    plt.show()


def plot_samples(clustering_model, data, y, layer_num, k, clustering_type, reduction_type, num_nodes_view, edges, num_expansions, path, graph_data=None, graph_name=None):
    res_sorted = get_node_distances(clustering_model, data)

    if isinstance(num_nodes_view, int):
        num_nodes_view = [num_nodes_view]
    col = sum([abs(number) for number in num_nodes_view])

    fig, axes = plt.subplots(k, col, figsize=(18, 3 * k + 2))
    fig.suptitle(f'Nearest Instances to {clustering_type} Cluster Centroid for {reduction_type} Activations of Layer {layer_num}', y=1.005)

    if graph_data is not None:
        fig2, axes2 = plt.subplots(k, col, figsize=(18, 3 * k + 2))
        fig2.suptitle(f'Nearest Instances to {clustering_type} Cluster Centroid for {reduction_type} Activations of Layer {layer_num} (by node index)', y=1.005)

    l = list(range(0, k))
    sample_graphs = []
    sample_feat = []

    for i, ax_list in zip(l, axes):
        if isinstance(clustering_model, AgglomerativeClustering) or isinstance(clustering_model, DBSCAN):
            distances = res_sorted[i]
        elif isinstance(clustering_model, KMeans):
            distances = res_sorted[:, i]

        top_graphs, color_maps = [], []
        for view in num_nodes_view:
            if view < 0:
                top_indices = np.argsort(distances)[::][view:]
            else:
                top_indices = np.argsort(distances)[::][:view]

            tg, cm, labels, node_labels = get_top_subgraphs(top_indices, y, edges, num_expansions, graph_data, graph_name)
            top_graphs = top_graphs + tg
            color_maps = color_maps + cm

        if graph_data is None:
            for ax, new_G, color_map, g_label in zip(ax_list, top_graphs, color_maps, labels):
                nx.draw(new_G, node_color=color_map, with_labels=True, ax=ax)
                ax.set_title(f"label {g_label}", fontsize=14)
        else:
            for ax, new_G, color_map, g_label, n_labels in zip(ax_list, top_graphs, color_maps, labels, node_labels):
                nx.draw(new_G, node_color=color_map, with_labels=True, ax=ax, labels=n_labels)
                ax.set_title(f"label {g_label}", fontsize=14)

            for ax, new_G, color_map, g_label, n_labels in zip(axes2[i], top_graphs, color_maps, labels, node_labels):
                nx.draw(new_G, node_color=color_map, with_labels=True, ax=ax)
                ax.set_title(f"label {g_label}", fontsize=14)

        sample_graphs.append((top_graphs[0], top_indices[0]))
        sample_feat.append(color_maps[0])

    views = ''.join((str(i) + "_") for i in num_nodes_view)
    if isinstance(clustering_model, AgglomerativeClustering):
        fig.savefig(os.path.join(path, f"{k}h_{layer_num}layer_{clustering_type}_{views}view.png"))
    else:
        fig.savefig(os.path.join(path, f"{layer_num}layer_{clustering_type}_{reduction_type}_{views}view.png"))

    if graph_data is not None:
        if isinstance(clustering_model, AgglomerativeClustering):
            fig2.savefig(os.path.join(path, f"{k}h_{layer_num}layer_{clustering_type}_{views}view_by_node.png"))
        else:
            fig2.savefig(os.path.join(path, f"{layer_num}layer_{clustering_type}_{reduction_type}_{views}view_by_node.png"))

    plt.show()

    return sample_graphs, sample_feat





# def plot_samples(clustering_model, data, y, layer_num, k, clustering_type, reduction_type, num_nodes_view, edges, num_expansions, path, graph_data=None):
#     res_sorted = get_node_distances(clustering_model, data)
#
#     if isinstance(num_nodes_view, int):
#         num_nodes_view = [num_nodes_view]
#     col = sum([abs(number) for number in num_nodes_view])
#
#     fig, axes = plt.subplots(k, col, figsize=(18, 3 * k))
#     fig.suptitle(f'Nearest Instances to {clustering_type} Cluster Centroid for {reduction_type} Activations of Layer {layer_num}')
#
#     l = list(range(0, k))
#     sample_graphs = []
#     sample_feat = []
#
#     for i, ax_list in zip(l, axes):
#         if isinstance(clustering_model, AgglomerativeClustering) or isinstance(clustering_model, DBSCAN):
#             distances = res_sorted[i]
#         elif isinstance(clustering_model, KMeans):
#             distances = res_sorted[:, i]
#
#         top_graphs, color_maps = [], []
#         for view in num_nodes_view:
#             if view < 0:
#                 top_indices = np.argsort(distances)[::][view:]
#             else:
#                 top_indices = np.argsort(distances)[::][:view]
#
#             tg, cm, labels = get_top_subgraphs(top_indices, y, edges, num_expansions, graph_data)
#             top_graphs = top_graphs + tg
#             color_maps = color_maps + cm
#
#         for ax, new_G, color_map, g_label in zip(ax_list, top_graphs, color_maps, labels):
#             nx.draw(new_G, node_color=color_map, with_labels=True, ax=ax)
#             ax.set_title(f"label {g_label}", fontsize=14)
#
#         sample_graphs.append((top_graphs[0], top_indices[0]))
#         sample_feat.append(color_maps[0])
#
#     views = ''.join((str(i) + "_") for i in num_nodes_view)
#     if isinstance(clustering_model, AgglomerativeClustering):
#         plt.savefig(os.path.join(path, f"{k}h_{layer_num}layer_{clustering_type}_{views}view.png"))
#     else:
#         plt.savefig(os.path.join(path, f"{layer_num}layer_{clustering_type}_{views}view.png"))
#     plt.show()
#
#     return sample_graphs, sample_feat


def plot_dendrogram(data, reduction_type, layer_num, path):
    """Learned from: https://medium.com/@sametgirgin/hierarchical-clustering-model-in-5-steps-with-python-6c45087d4318 """
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(f'HC Dendrogram of {reduction_type} Activation Space of Layer {layer_num}')

    dendrogram = hierarchy.dendrogram(hierarchy.linkage(data, method='average'), truncate_mode="lastp", ax=ax, leaf_rotation=90.0, leaf_font_size=14)
    ax.set_xlabel("Nodes")
    ax.set_ylabel("Euclidean Distances")

    plt.savefig(os.path.join(path, f"hc_dendrograms_{reduction_type}.png"))
    plt.show()


def plot_completeness_table(model_type, calc_type, data, path):
    fig, ax = plt.subplots(figsize=(10, 2 * len(data)))
    headings = ["Data", "Layer", "Completeness Score"]

    ax.set_title(f"Completeness Score (Task Accuracy) for {model_type} Models using {calc_type}")
    ax.axis('off')
    ax.table(cellText=data, colLabels=headings, loc="center", rowLoc="center", cellLoc="center", colLoc="center", fontsize=18)

    calc_type = calc_type.replace(" ", "")
    plt.savefig(os.path.join(path, f"{model_type}_{calc_type}_completeness.png"))
    plt.show()


def calc_graph_similarity(top_graphs, max_nodes, num_nodes_view):
    top_G = top_graphs[0]

    print("Nodes ", top_G.number_of_nodes(), " Graphs ", len(top_graphs))

    if top_G.number_of_nodes() > max_nodes:
        return "skipping (too many nodes)"

    if_iso = True

    for G in top_graphs[1:]:
        if not nx.is_isomorphic(top_G, G):
            if_iso = False
            break

    if if_iso:
        return 0

    total_score = 0
    for G in top_graphs[1:]:

        if G.number_of_nodes() > max_nodes:
            return "skipping (too many nodes)"

        total_score += min(list(nx.optimize_graph_edit_distance(top_G, G)))

    return total_score / (len(top_graphs) - 1)


def plot_graph_similarity_table(model_type, data, path):
    fig, ax = plt.subplots(figsize=(10, 0.25 * len(data)))
    headings = ["Model", "Data", "Layer", "Concept/Cluster", "Graph Similarity Score"]

    ax.set_title(f"Graph Similarity for Concepts extracted using {model_type}")
    ax.axis('off')
    ax.table(cellText=data, colLabels=headings, loc="center", rowLoc="center", cellLoc="center", colLoc="center", fontsize=18)

    plt.savefig(os.path.join(path, f"{model_type}_graph_similarity.png"))
    plt.show()


def prepare_output_paths(dataset_name, k):
    path = f"output/{dataset_name}/"
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
