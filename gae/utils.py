import pickle as pkl

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
from sklearn.metrics import roc_auc_score, average_precision_score


def load_data(dataset):
    # load the data: x, tx, allx, graph
    names = ['x', 'tx', 'allx', 'graph']
    objects = []
    for i in range(len(names)):
        '''
        fix Pickle incompatibility of numpy arrays between Python 2 and 3
        https://stackoverflow.com/questions/11305790/pickle-incompatibility-of-numpy-arrays-between-python-2-and-3
        '''
        with open("data/ind.{}.{}".format(dataset, names[i]), 'rb') as rf:
            u = pkl._Unpickler(rf)
            u.encoding = 'latin1'
            cur_data = u.load()
            objects.append(cur_data)
        # objects.append(
        #     pkl.load(open("data/ind.{}.{}".format(dataset, names[i]), 'rb')))
    x, tx, allx, graph = tuple(objects)
    test_idx_reorder = parse_index_file(
        "data/ind.{}.test.index".format(dataset))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(
            min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    features = torch.FloatTensor(np.array(features.todense()))
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    return adj, features

def load_data_2(n):
    #get edges and nodes
    edges = np.genfromtxt("../data/rexampled_graph.txt", dtype="int32")
    edge = np.array(edges)
    nodes = list(set(edges[:,0]))
    nodes.sort()

    #get communites and create labels
    with open('../data/desampled_communities.txt', 'r') as f:
        lines = f.readlines()

    data = [list(map(int, line.strip().split())) for line in lines]
    communities = np.array(data, dtype=object)

    labels = []
    for node in nodes:
        for l, com in enumerate(communities):
            if node in com:
                labels.append(l)
                break

    labels_enumerated = np.array(labels)

    #create graph structure
    ids_ordered = {raw_id: order for order, raw_id in enumerate(nodes)}

    #need to reorgainze edges
    new_edges = list(map(ids_ordered.get, edges.flatten()))
    pairs = [(new_edges[i], new_edges[i + 1]) for i in range(0, len(new_edges), 2)]
    filtered_pairs = [pair for pair in pairs if None not in pair]
    new_edges = [item for pair in filtered_pairs for item in pair]

    new_shape = int((len(new_edges)/2))
    edges_ordered = np.array(new_edges,#list(map(ids_ordered.get, edges.flatten())),
                             dtype='int32').reshape(new_shape, 2)

    #adj matrix
    adj = sp.coo_matrix((np.ones(edges_ordered.shape[0]), (edges_ordered[:, 0], edges_ordered[:, 1])),
                            shape=(labels_enumerated.shape[0], labels_enumerated.shape[0]),
                            dtype=np.float32)
    # Make the adjacency matrix symmetric
    adj = adj + adj.T.multiply(adj.T > adj)
    # adj = normalize_adjacency(adj)

    #embedding time :D
    # features = torch.nn.Embedding(len(nodes), config.hidden_dim_size) 

    features = torch.zeros((len(labels), len(labels)), dtype=torch.float32)

    for i in range(0, len(new_edges), 2):
        node1 = new_edges[i]
        node2 = new_edges[i + 1]
        features[node1, node2] = 1
        features[node2, node1] = 1 

    # Initialize the final matrix
    final_matrix = torch.zeros_like(features)
    
    # Start with the original adjacency matrix
    current_hop_matrix = features.clone()
    
    for hop in range(1, n + 1):
        final_matrix[(final_matrix == 0) & (current_hop_matrix > 0)] = hop
        current_hop_matrix = current_hop_matrix @ features
    
    # Ensure direct connections remain as 1
    final_matrix[features == 1] = 1

    features = final_matrix

    # features = torch.FloatTensor()
    labels = torch.LongTensor(labels_enumerated)
    adj = nx.adjacency_matrix(nx.from_scipy_sparse_array(adj))
    # adj = convert_scipy_to_torch_sparse(adj)

    # if config.use_emb_weight == 'yes':
    #     features.weight = torch.load(config.emb_weight_path)

    print("Dataset loaded.")
    return adj, features, labels

def convert_scipy_to_torch_sparse(matrix):
    matrix_helper_coo = matrix.tocoo().astype('float32')
    data = torch.FloatTensor(matrix_helper_coo.data)
    rows = torch.LongTensor(matrix_helper_coo.row)
    cols = torch.LongTensor(matrix_helper_coo.col)
    indices = torch.vstack([rows, cols])

    shape = torch.Size(matrix_helper_coo.shape)
    matrix = torch.sparse.FloatTensor(indices, data, shape)
    return matrix


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def mask_test_edges(adj):
    # Function to build test set with 10% positive links
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
    # TODO: Clean up.

    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    edges_all = sparse_to_tuple(adj)[0]
    num_test = int(np.floor(edges.shape[0] / 10.))
    num_val = int(np.floor(edges.shape[0] / 20.))

    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)

    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])

    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], train_edges):
            continue
        if ismember([idx_j, idx_i], train_edges):
            continue
        if ismember([idx_i, idx_j], val_edges):
            continue
        if ismember([idx_j, idx_i], val_edges):
            continue
        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])

    assert ~ismember(test_edges_false, edges_all)
    assert ~ismember(val_edges_false, edges_all)
    assert ~ismember(val_edges, train_edges)
    assert ~ismember(test_edges, train_edges)
    assert ~ismember(val_edges, test_edges)

    data = np.ones(train_edges.shape[0])

    # Re-build adj matrix
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false


def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    # return sparse_to_tuple(adj_normalized)
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def get_roc_score(emb, adj_orig, edges_pos, edges_neg):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Predict on test set of edges
    adj_rec = np.dot(emb, emb.T)
    preds = []
    pos = []
    for e in edges_pos:
        preds.append(sigmoid(adj_rec[e[0], e[1]]))
        pos.append(adj_orig[e[0], e[1]])

    preds_neg = []
    neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
        neg.append(adj_orig[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score
