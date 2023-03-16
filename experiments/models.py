import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import wandb
import networkx as nx

from torch_geometric.nn import MessagePassing, GCNConv, DenseGCNConv, GINConv, GraphConv
from torch_geometric.utils import add_self_loops, degree, to_dense_adj, convert, to_undirected
from torch_geometric.nn import global_mean_pool, GlobalAttention, global_max_pool

#from torchmetrics.classification import BinaryAccuracy, MulticlassAccuracy, MulticlassConfusionMatrix

class Model_random_proto(nn.Module):
    def __init__(self, num_in_features, num_hidden_features, num_classes,
                 num_layers, prot_index, train_mask, name, top_p=0, cos_threshold=0, first_step_epochs=0):
        super(Model_random_proto, self).__init__()

        self.name = name
        self.prot = prot_index
        self.num_layers = num_layers
        self.train_mask = train_mask
        self.first_step_epochs = first_step_epochs

        self.layers = [GCNConv(num_in_features, num_hidden_features)]
        self.layers += [GCNConv(num_hidden_features, num_hidden_features) for _ in
                        range(num_layers - 1)]
        self.layers = nn.ModuleList(self.layers)

        # linear layers
        self.linear = nn.Sequential(nn.Linear(num_hidden_features, num_hidden_features),
                                    nn.ReLU(),
                                    nn.Linear(num_hidden_features, num_classes))


    def relative_rep(self, x, anchor_indexes):
        anchors = x[anchor_indexes]
        result = torch.Tensor()
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        for t in anchors:
            tmp = (cos(x, t).unsqueeze(dim=-1) - (-1)) / (1 - (-1))
            # tmp = cos(x, t).unsqueeze(dim=-1)
            result = torch.cat((result, tmp), dim=-1)
        # result = F.softmax(result, dim=-1)
        return result

    def y_composition(self, x, out_proto):
        y = F.log_softmax(torch.mm(x, out_proto), dim=-1)

        return y

    def forward(self, x, edge_index, epoch):
        for i in range(self.num_layers):
            x = self.layers[i](x, edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)

        qn = torch.norm(x, p=2, dim=1, keepdim=True)
        x = x.div(qn.expand_as(x))

        x_rel = self.relative_rep(x, self.prot)

        out_proto = F.log_softmax(self.linear(x[self.prot]), dim=-1)

        out = self.y_composition(x_rel, out_proto)

        return out, x_rel, out_proto

class Model_sequential(nn.Module):
    def __init__(self, num_in_features, num_hidden_features, num_classes,
                 num_layers, prot_index, train_mask,
                 name, top_p=0, cos_threshold=0, first_step_epochs=0):
        super(Model_sequential, self).__init__()

        self.name = name
        self.prot = prot_index
        self.num_layers = num_layers
        self.train_mask = train_mask
        self.p = torch.nn.Parameter(torch.Tensor([top_p]))
        self.t = torch.nn.Parameter(torch.Tensor([cos_threshold]))
        self.first_step_epochs = first_step_epochs

        self.layers = [GCNConv(num_in_features, num_hidden_features)]
        self.layers += [GCNConv(num_hidden_features, num_hidden_features) for _ in
                        range(num_layers - 1)]
        self.layers = nn.ModuleList(self.layers)

        # linear layers
        self.linear = nn.Sequential(nn.Linear(num_hidden_features, num_hidden_features),
                                    nn.ReLU(),
                                    nn.Linear(num_hidden_features, num_classes))

    def choose_prototype(self, pred, x):
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)

        train_index = np.transpose(np.argsort(pred[self.train_mask].detach().numpy(), axis=0))[:, -int(x.shape[0]*self.p):]
        true_indexes = np.array(range(pred.shape[0]))[self.train_mask][np.flip(train_index, axis=-1)]

        proto = set()
        top = set()
        for i in range(true_indexes.shape[0]):
            tmp_set = set(true_indexes[i])
            top = top.union(tmp_set)
            for idx1 in true_indexes[i]:
                if torch.argmax(pred[idx1]) == i:
                    if idx1 in tmp_set:
                        for idx2 in true_indexes[i]:
                            if idx2 in tmp_set and idx1 != idx2:
                                tmp = (cos(x[idx1].unsqueeze(0), x[idx2].unsqueeze(0)) + 1) / 2
                                if tmp > self.t:
                                    tmp_set.discard(idx2)
                else:
                    tmp_set.discard(idx1)
            if tmp_set == set():
                tmp_set = set([true_indexes[i][0]])
            proto = proto.union(tmp_set)

        return list(proto), list(top)
    def relative_rep(self, x, anchor_indexes):
        anchors = x[anchor_indexes]
        result = torch.Tensor()
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        for t in anchors:
            # tmp = (cos(x, t).unsqueeze(dim=-1) - (-1)) / (1 - (-1))
            tmp = cos(x, t).unsqueeze(dim=-1)
            result = torch.cat((result, tmp), dim=-1)
        # result = F.softmax(result, dim=-1)
        return result

    def y_composition(self, x, out_proto):
        y = F.log_softmax(torch.mm(x, out_proto), dim=-1)

        return y

    def forward(self, x, edge_index, epoch):
        if epoch >= self.first_step_epochs:
            with torch.no_grad():
                for i in range(self.num_layers):
                    x = self.layers[i](x, edge_index)
                    if i != self.num_layers - 1:
                        x = F.relu(x)
        else:
            for i in range(self.num_layers):
                # if epoch >= self.first_step_epochs:
                #     self.layers[i].weight.requires_grad = False
                x = self.layers[i](x, edge_index)
                if i != self.num_layers - 1:
                    x = F.relu(x)

        qn = torch.norm(x, p=2, dim=1, keepdim=True)
        x = x.div(qn.expand_as(x))

        if epoch < self.first_step_epochs:
            out = F.log_softmax(self.linear(x), dim=-1)
            self.prot, self.top = self.choose_prototype(F.log_softmax(self.linear(x), dim=-1), x)
            x_rel = self.relative_rep(x, self.prot)
            out_proto = F.log_softmax(self.linear(x[self.prot]), dim=-1)
            return out, x_rel, out_proto
        else:
            x_rel = self.relative_rep(x, self.prot)

            out_proto = F.log_softmax(self.linear(x[self.prot]), dim=-1)

            out = self.y_composition(x_rel, out_proto)

        return out, x_rel, out_proto

class Model_joint(nn.Module):
    def __init__(self, num_in_features, num_hidden_features, num_classes,
                 num_layers, prot_index, train_mask,
                 name, top_p=0, cos_threshold=0, first_step_epochs=0):
        super(Model_joint, self).__init__()

        self.name = name
        self.prot = prot_index
        self.num_layers = num_layers
        self.train_mask = train_mask
        self.p = torch.nn.Parameter(torch.Tensor([top_p]))
        self.t = torch.nn.Parameter(torch.Tensor([cos_threshold]))
        self.first_step_epochs = first_step_epochs

        self.layers = [GCNConv(num_in_features, num_hidden_features)]
        self.layers += [GCNConv(num_hidden_features, num_hidden_features) for _ in
                        range(num_layers - 1)]
        self.layers = nn.ModuleList(self.layers)

        # linear layers
        self.linear = nn.Sequential(nn.Linear(num_hidden_features, num_hidden_features),
                                    nn.ReLU(),
                                    nn.Linear(num_hidden_features, num_classes))

    def choose_prototype(self, pred, x):
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)

        train_index = np.transpose(np.argsort(pred[self.train_mask].detach().numpy(), axis=0))[:, -int(x.shape[0]*self.p):]
        true_indexes = np.array(range(pred.shape[0]))[self.train_mask][np.flip(train_index, axis=-1)]

        proto = set()
        top = set()
        for i in range(true_indexes.shape[0]):
            tmp_set = set(true_indexes[i])
            top = top.union(tmp_set)
            for idx1 in true_indexes[i]:
                if torch.argmax(pred[idx1]) == i:
                    if idx1 in tmp_set:
                        for idx2 in true_indexes[i]:
                            if idx2 in tmp_set and idx1 != idx2:
                                tmp = (cos(x[idx1].unsqueeze(0), x[idx2].unsqueeze(0)) + 1) / 2
                                if tmp > self.t:
                                    tmp_set.discard(idx2)
                else:
                    tmp_set.discard(idx1)
            if tmp_set == set():
                tmp_set = set([true_indexes[i][0]])
            proto = proto.union(tmp_set)

        return list(proto), list(top)
    def relative_rep(self, x, anchor_indexes):
        anchors = x[anchor_indexes]
        result = torch.Tensor()
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        for t in anchors:
            # tmp = (cos(x, t).unsqueeze(dim=-1) - (-1)) / (1 - (-1))
            tmp = cos(x, t).unsqueeze(dim=-1)
            result = torch.cat((result, tmp), dim=-1)
        # result = F.softmax(result, dim=-1)
        return result

    def y_composition(self, x, out_proto):
        y = F.log_softmax(torch.mm(x, out_proto), dim=-1)

        return y

    def forward(self, x, edge_index, epoch):
        for i in range(self.num_layers):
            # if epoch >= self.first_step_epochs:
            #     self.layers[i].weight.requires_grad = False
            x = self.layers[i](x, edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)

        qn = torch.norm(x, p=2, dim=1, keepdim=True)
        x = x.div(qn.expand_as(x))

        out = F.log_softmax(self.linear(x), dim=-1)
        self.prot, self.top = self.choose_prototype(F.log_softmax(self.linear(x), dim=-1), x)

        x_rel = self.relative_rep(x, self.prot)

        out_proto = F.log_softmax(self.linear(x[self.prot]), dim=-1)

        out = self.y_composition(x_rel, out_proto)

        return out, x_rel, out_proto

class Model_baseline(nn.Module):
    def __init__(self, num_in_features, num_hidden_features, num_classes,
                 num_layers, prot_index, train_mask, name, top_p=0, cos_threshold=0, first_step_epochs=0):
        super(Model_baseline, self).__init__()

        self.name = name
        self.prot = prot_index
        self.num_layers = num_layers
        self.train_mask = train_mask
        self.first_step_epochs = first_step_epochs

        self.layers = [GCNConv(num_in_features, num_hidden_features)]
        self.layers += [GCNConv(num_hidden_features, num_hidden_features) for _ in
                        range(num_layers - 1)]
        self.layers = nn.ModuleList(self.layers)

        # linear layers
        self.linear = nn.Sequential(nn.Linear(num_hidden_features, num_hidden_features),
                                    nn.ReLU(),
                                    nn.Linear(num_hidden_features, num_classes))


    def forward(self, x, edge_index, epoch):
        for i in range(self.num_layers):
            x = self.layers[i](x, edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)

        out = F.log_softmax(self.linear(x), dim=-1)

        return out, x, out

global activation_list
activation_list = {}


def get_activation(idx):
    '''Learned from: https://discuss.pytorch.org/t/how-can-l-load-my-best-model-as-a-feature-extractor-evaluator/17254/6'''
    def hook(model, input, output):
        activation_list[idx] = output.detach()
    return hook


def register_hooks(model):
    # register hooks to extract activations
    for name, m in model.named_modules():
        print(name, m)
        if isinstance(m, GCNConv) or isinstance(m, DenseGCNConv):
            m.register_forward_hook(get_activation(f"{name}"))
        if isinstance(m, nn.Linear):
            m.register_forward_hook(get_activation(f"{name}"))

    return model


def weights_init(m):
    if isinstance(m, GCNConv):
        torch.nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain("relu"))
        torch.nn.init.uniform_(m.bias.data)


def test(model, node_data_x, node_data_y, edge_list, mask, epoch):
    # enter evaluation mode
    model.eval()

    correct = 0
    pred, _, _ = model(node_data_x, edge_list, epoch)
    pred = pred.max(dim=1)[1]

    correct += pred[mask].eq(node_data_y[mask]).sum().item()
    return correct / (len(node_data_y[mask]))

def test_precision(model, node_data_x, node_data_y, edge_list, mask, epoch):
    model.eval()

    correct = 0
    _, rel_x, _ = model(node_data_x, edge_list, epoch)
    top_proto = torch.argsort(rel_x, dim=-1)[:, -1]
    pred = node_data_y[torch.Tensor(model.prot)[top_proto].long()]

    correct += pred[mask].eq(node_data_y[mask]).sum().item()
    return correct / (len(node_data_y[mask]))


# def additional_loss(x, labels, prot_labels, mode='equal'):
#     prot_labels = prot_labels.repeat(x.shape[0], 1)
#     labels = labels.unsqueeze(1)
#     if mode == 'equal':
#         labels_mask = prot_labels == labels
#         return (x[labels_mask]*(-1)).mean()
#     else:
#         labels_mask = prot_labels != labels
#         return x[labels_mask].mean()
#
def additional_loss(x, labels, prot_labels, mode='equal'):
    prot_labels = prot_labels.repeat(x.shape[0], 1)
    labels = labels.unsqueeze(1)
    if mode == 'equal':
        labels_mask = prot_labels == labels
        x_masked = x * labels_mask
        x_masked[x_masked == 0] = -2
        return torch.max(x_masked, dim=-1)[0].mean()*(-1)
    else:
        labels_mask = prot_labels != labels
        x_masked = x * labels_mask
        x_masked[x_masked == 0] = -2
        return torch.max(x_masked, dim=-1)[0].mean()


def train(model, data, epochs, lr, lr_decay, early_stopping, path, proto_loss=False, hard=False):
    # register hooks to track activation
    model = register_hooks(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1,
                                                  end_factor=lr_decay, total_iters=epochs)

    # list of accuracies
    train_accuracies, test_accuracies, train_losses, test_losses = list(), list(), list(), list()

    # get data
    x = data["x"]
    edges = data["edges"]
    y = data["y"]
    train_mask = data["train_mask"]
    test_mask = data["test_mask"]
    min_loss = 500000
    counter = 0
    # iterate for number of epochs
    for epoch in range(epochs):
        # set mode to training
        model.train()
        optimizer.zero_grad()
        if counter > early_stopping:
            break
        # input data
        out, x_rel, _ = model(x, edges, epoch)

        n_proto = len(model.prot)
        wandb.log({'Epoch': epoch, 'Number of prototype': n_proto})

        if not hard:
            equal_loss = additional_loss(x_rel[train_mask], y[train_mask], y[model.prot])
            divergence_loss = additional_loss(x_rel[train_mask], y[train_mask], y[model.prot], 'diverse')
            loss = F.nll_loss(out[train_mask], y[train_mask])
            wandb.log({'Train loss': loss.item(), 'Train_equal_loss': equal_loss,
                       'Train_diverse_loss': divergence_loss})
        else:
            loss = F.nll_loss(out[train_mask], y[train_mask])
            wandb.log({'Train loss': loss.item()})

        # calculate loss
        if proto_loss:
            final_loss = loss + equal_loss + divergence_loss
        else:
            final_loss = loss

        final_loss.backward()
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            test_loss = F.nll_loss(out[test_mask], y[test_mask])
            wandb.log({'Test loss': loss.item()})
            if epoch > model.first_step_epochs:
                if min_loss*1.05 >= test_loss:
                    min_loss = test_loss
                    counter = 0
                else:
                    counter += 1

            # get accuracy
            train_acc = test(model, x, y, edges, train_mask, epoch)
            test_acc = test(model, x, y, edges, test_mask, epoch)
            wandb.log({'Train accuracy': train_acc,
                       'Test accuracy': test_acc})

            if not isinstance(model, Model_baseline):
                train_p = test_precision(model, x, y, edges, train_mask, epoch)
                test_p = test_precision(model, x, y, edges, test_mask, epoch)
                wandb.log({'Train proto precision@1': train_p,
                    'Test proto precision@1': test_p})

        ## add to list and print
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        train_losses.append(loss.item())
        test_losses.append(test_loss.item())

        # print('Epoch: {:03d}, Loss: {:.5f}, Train Acc: {:.5f}, Test Acc: {:.5f}'.
        #       format(epoch, loss.item(), train_acc, test_acc), end="\r")

    # plut accuracy graph
    plt.plot(train_accuracies, label="Train Accuracy")
    plt.plot(test_accuracies, label="Testing Accuracy")
    plt.title(f"Accuracy of {model.name} Model during Training")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(path, f"model_accuracy_plot.png"))

    plt.plot(train_losses, label="Train Loss")
    plt.plot(test_losses, label="Testing Loss")
    plt.title(f"Loss of {model.name} Model during Training")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(path, f"model_loss_plot.png"))

    # save model
    torch.save(model.state_dict(), os.path.join(path, "model.pkl"))

    with open(os.path.join(path, "activations.txt"), 'wb') as file:
        pickle.dump(activation_list, file)

    return model

