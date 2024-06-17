from __future__ import division
from __future__ import print_function

import argparse
import time
import os

import numpy as np
import scipy.sparse as sp
import networkx as nx
import torch
from torch import optim

from model import GCNModelVAE
from optimizer import loss_function
from utils import load_data, load_data_2, mask_test_edges, preprocess_graph, get_roc_score

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='gcn_vae', help="models used")
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--hidden1', type=int, default=32, help='Number of units in hidden layer 1.')
parser.add_argument('--hidden2', type=int, default=16, help='Number of units in hidden layer 2.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--dropout', type=float, default=0., help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset-str', type=str, default='cora', help='type of dataset.')
parser.add_argument('--nhop', type=int, default=25, help='nhop to use')
args = parser.parse_args()


def gae_for(args):
    # print("Using {} dataset".format(args.dataset_str))
    # adj, features, labels_for_color = load_data_2(args.dataset_str)
    print("Using {} dataset".format('Amazon'), str(args.nhop)+'nhop')
    adj, features, labels_for_color = load_data_2(args.nhop)
    n_nodes, feat_dim = features.shape

    # Store original adjacency matrix (without diagonal entries) for later
    adj_orig = adj
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()

    adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
    adj = adj_train

    # Some preprocessing
    adj_norm = preprocess_graph(adj)
    adj_label = adj_train + sp.eye(adj_train.shape[0])
    # adj_label = sparse_to_tuple(adj_label)
    adj_label = torch.FloatTensor(adj_label.toarray())

    print(float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum())
    pos_weight = torch.FloatTensor([float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()])
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

    model = GCNModelVAE(feat_dim, args.hidden1, args.hidden2, args.dropout)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    hidden_emb = None
    for epoch in range(args.epochs):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        recovered, mu, logvar = model(features, adj_norm)
        loss = loss_function(preds=recovered, labels=adj_label,
                             mu=mu, logvar=logvar, n_nodes=n_nodes,
                             norm=norm, pos_weight=pos_weight)
        loss.backward()
        cur_loss = loss.item()
        optimizer.step()

        hidden_emb = mu.data.numpy()
        roc_curr, ap_curr = get_roc_score(hidden_emb, adj_orig, val_edges, val_edges_false)

        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(cur_loss),
              "val_ap=", "{:.5f}".format(ap_curr), "val_roc=", "{:.5f}".format(roc_curr),
              "time=", "{:.5f}".format(time.time() - t)
              )
        # print(recovered)
        # print(mu[:3,:])
        # print(logvar[:3,:])

    if not(os.path.isdir('./saves/nhop_'+str(args.nhop)+'-2')):
        os.mkdir('./saves/nhop_'+str(args.nhop)+'-2')
    torch.save(recovered.detach(), './saves/nhop_'+str(args.nhop)+'-2/Z.pt')
    torch.save(mu.detach(), './saves/nhop_'+str(args.nhop)+'-2/mean.pt')
    torch.save(logvar.detach(), './saves/nhop_'+str(args.nhop)+'-2/logvar.pt')

    print("Optimization Finished!")

    roc_score, ap_score = get_roc_score(hidden_emb, adj_orig, test_edges, test_edges_false)
    print('Test ROC score: ' + str(roc_score))
    print('Test AP score: ' + str(ap_score))

    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    tsne = TSNE()
    new_mu = tsne.fit_transform(mu.detach().numpy())
    torch.save(mu.detach(), './saves/nhop_'+str(args.nhop)+'-2/reduced_mean.pt')

    color_dict = {}
    increment = 1/(1517 + 10)
    value = 1
    for i in range(1517):
        color_dict[i+1] = value
        value -= increment 

    colors = []
    for node in labels_for_color:
        colors.append(color_dict[node.item()+1])

    adj_array = adj_orig.toarray()

    # mask = torch.ones(recovered.shape[0], recovered.shape[0])*(-0.8)
    # mask = recovered >= mask
    # recovered[mask] = 1
    # mask = torch.ones(recovered.shape[0], recovered.shape[0])*(-0.8)
    # mask = recovered < mask
    # recovered[mask] = 0
    # adj_array = recovered.detach().numpy()

    graph_obj = nx.from_numpy_array(adj_array)
    pos_dict = {}
    for node in graph_obj.nodes():
        pos_dict[node] = np.array([new_mu[node,0],new_mu[node,1]])

    print('graph thyme')
    nx.draw(graph_obj, pos_dict, node_color=colors, node_size=7, width=0.5)
    plt.show()

if __name__ == '__main__':
    gae_for(args)
