import torch
import torch.nn as nn
import torchvision
from torchvision.models.resnet import resnet18

import argparse
import itertools
import math
import random
import statistics
from dataset import OGBNDataset
from model import DeeperGCN, DeeperGCNDP
from args import ArgsInit
import time
import numpy as np
from ogb.nodeproppred import Evaluator
from utils.ckpt_util import save_ckpt
from utils.data_util import intersection, process_indexes
import logging

from torch_geometric.nn import DataParallel as PyGDataParallel
from torch_geometric.data import Data as PyGData

def get_linear_model():
    torch.manual_seed(128)
    return nn.Linear(32, 8)

def get_linear_inputs():
    n_gpus = torch.cuda.device_count()
    data, labels = [], []
    for i in range(n_gpus):
        torch.manual_seed(i)
        data.append(torch.randn(4, 32))
        torch.manual_seed(i)
        labels.append(torch.randint(0, 8, size=[4]))
    return data, labels

def get_resnet_model():
    torch.manual_seed(128)
    return resnet18(pretrained=True)

def get_resnet_inputs():
    n_gpus = torch.cuda.device_count()
    data, labels = [], []
    for i in range(n_gpus):
        torch.manual_seed(i)
        data.append(torch.randn(4, 3, 224, 224))
        torch.manual_seed(i)
        labels.append(torch.randint(0, 1000, size=[4]))
    return data, labels

def test_sequential(linear=True):

    if linear:
        m = get_linear_model()
        data, labels = get_linear_inputs()
    else:
        m = get_resnet_model()
        data, labels = get_resnet_inputs()

    m = m.cuda()

    # run sequence
    criterion = nn.CrossEntropyLoss().cuda()
    m.zero_grad()
    for d, l in zip(data, labels):
        outputs = m(d.cuda())
        loss = criterion(outputs, l.cuda())
        loss.backward()

    # collect gradients
    gradients = []
    for param in m.parameters():
        if not param.grad is None:
            gradients.append(param.grad.detach().cpu().numpy())
    return m, gradients

def test_dp(linear=True):

    if linear:
        m = get_linear_model()
        data, labels = get_linear_inputs()
    else:
        m = get_resnet_model()
        data, labels = get_resnet_inputs()

    m = nn.DataParallel(m).cuda()
    data = torch.cat(data, dim=0)
    labels = torch.cat(labels, dim=0)

    # run sequence
    criterion = nn.CrossEntropyLoss().cuda()
    m.zero_grad()
    outputs = m(data)
    loss = criterion(outputs, labels.cuda())
    loss.backward()

    # collect gradients
    gradients = []
    for param in m.module.parameters():
        if not param.grad is None:
            gradients.append(param.grad.detach().cpu().numpy())
    return m, gradients

def setup_deepgcn_dp():
    args = ArgsInit().args
    args.gcn_aggr = 'softmax'
    args.block = 'res+'
    args.conv_encode_edge = True
    args.use_one_hot_encoding = True
    args.learn_t = True
    args.t = 1.0
    args.dropout = 0.1
    args.num_layers = 3
    args.norm = 'layer'
    dataset = OGBNDataset(dataset_name=args.dataset)
    # extract initial node features
    nf_path = dataset.extract_node_features(args.aggr)

    args.num_tasks = dataset.num_tasks
    args.nf_path = nf_path

    torch.manual_seed(128)
    np.random.seed(128)
    random.seed(128)
    train_parts = dataset.random_partition_graph(dataset.total_no_of_nodes,
                                                    cluster_number=args.cluster_number)
    data = dataset.generate_sub_graphs(train_parts, cluster_number=args.cluster_number)
    return args, data, dataset

def test_sequential_deepgcn(args, data, dataset, clusters=None):
    torch.manual_seed(128)
    model = DeeperGCNDP(args).cuda()
    criterion = nn.BCEWithLogitsLoss().cuda()

    # process data
    PREDS = []
    n_gpus = torch.cuda.device_count()
    train_y = dataset.y[dataset.train_idx]
    model.zero_grad()
    sg_nodes, sg_edges, sg_edges_index, _ = data
    if clusters is None:
        clusters = range(n_gpus)
    for idx in clusters:
        x = dataset.x[sg_nodes[idx]].float().cuda()
        sg_nodes_idx = torch.LongTensor(sg_nodes[idx]).cuda()
        sg_edges_idx = sg_edges[idx].cuda()
        sg_edges_attr = dataset.edge_attr[sg_edges_index[idx]].cuda()

        mapper = {node: idx for idx, node in enumerate(sg_nodes[idx])}
        inter_idx = list(intersection(sg_nodes[idx], dataset.train_idx.tolist()))
        training_idx = [mapper[t_idx] for t_idx in inter_idx]

        data = PyGData(x=x, node_index=sg_nodes_idx, edge_index=sg_edges_idx, edge_attr=sg_edges_attr)
        pred = model(data)
        target = train_y[inter_idx].cuda()
        target = torch.ones_like(target, dtype=target.dtype).cuda()
        loss = criterion(pred[training_idx].to(torch.float32), target.to(torch.float32))
        loss.backward()

    # collect gradients
    gradients = []
    for m in model.parameters():
        if not m.grad is None:
            gradients.append(m.grad.detach().cpu().numpy())
    return model, gradients

def create_datalist(x, node_index, edge_index, edge_attr):
    data_list = []
    for i in range(len(x)):
        data_list.append(PyGData(x=x[i], node_index=node_index[i], edge_index=edge_index[i], edge_attr=edge_attr[i]))
    return data_list

def test_dp_deepgcn(args, data, dataset, clusters=None):
    torch.manual_seed(128)
    model = DeeperGCNDP(args)
    model = PyGDataParallel(model).cuda()
    criterion = nn.BCEWithLogitsLoss(reduction='none').cuda()

    # process data
    train_y = dataset.y[dataset.train_idx]
    model.zero_grad()
    sg_nodes, sg_edges, sg_edges_index, _ = data

    # stack 
    n_gpus = torch.cuda.device_count()
    if clusters is None:
        clusters = range(n_gpus)
    x = [dataset.x[sg_nodes[idx]].float() for idx in clusters]
    sg_nodes_idx = [torch.LongTensor(sg_nodes[idx]) for idx in clusters]
    sg_edges_ = [sg_edges[idx] for idx in clusters]
    sg_edges_attr = [dataset.edge_attr[sg_edges_index[idx]] for idx in clusters]
    data_list = create_datalist(x, sg_nodes_idx, sg_edges_, sg_edges_attr)

    # create training_idx, assuming cluster partitions are distinct
    all_nodes = np.concatenate([sg_nodes[idx] for idx in clusters], axis=0) # cat 4 lists in sorted order?
    mapper = {node: idx for idx, node in enumerate(all_nodes)} # duplicates? no: distinct partitions
    inter_idx = list(itertools.chain(*[intersection(sg_nodes[idx], dataset.train_idx.tolist()) for idx in clusters]))
    training_idx = [mapper[t_idx] for t_idx in inter_idx]

    # create weights
    indices = np.zeros(len(training_idx))
    sizes = [len(intersection(sg_nodes[idx], dataset.train_idx.tolist())) for idx in clusters]
    for i in range(len(sizes) - 1):
        indices[sum(sizes[:i+1])] = 1
    indices = np.cumsum(indices).astype(int)
    weights = np.array(sizes)[indices]
    weights = len(weights) / weights
    weights = torch.tensor(weights).cuda()

    # run FWD + BWD
    pred = model(data_list)
    target = train_y[inter_idx].cuda()
    target = torch.ones_like(target, dtype=target.dtype).cuda()

    loss = criterion(pred[training_idx].to(torch.float32), target.to(torch.float32))
    loss = (loss * weights.unsqueeze(1)).mean()
    # loss = loss.mean()
    loss.backward()

    # collect gradients
    gradients = []
    for m in model.module.parameters():
        if not m.grad is None:
            gradients.append(m.grad.detach().cpu().numpy())
    return model, gradients


if __name__ == '__main__':

    TEST_TYPE = 'gcn'

    if TEST_TYPE == 'gcn':
        args, data, dataset = setup_deepgcn_dp()

    ci, cj = 0, 1
    print("##### running sequential")
    if TEST_TYPE == 'linear':
        _, grads1 = test_sequential(linear=True)
    elif TEST_TYPE == 'resnet':
        _, grads1 = test_sequential(linear=False)
    elif TEST_TYPE == 'gcn':
        seq_model, grads1 = test_sequential_deepgcn(args, data, dataset, clusters=[ci,cj])
        # preds1 = test_sequential_deepgcn(args, data, dataset)
    print("#####")
    print("##### running DP")
    if TEST_TYPE == 'linear':
        _, grads2 = test_dp(linear=True)
    elif TEST_TYPE == 'resnet':
        _, grads2 = test_dp(linear=False)
    elif TEST_TYPE == 'gcn':
        dp_model, grads2 = test_dp_deepgcn(args, data, dataset, clusters=[ci,cj])
        # preds2 = test_dp_deepgcn(args, data, dataset)
    print("#####")
    print("##### grad ratio")
    for g1, g2 in zip(grads1, grads2):
        ratio = g1 / g2
        print("{:.4f}, {:.4f}, {:.4f}, {:.4f}, {}".format(ratio.mean(), ratio.min(), ratio.max(), ratio.std(), ratio.shape))

# if __name__ == '__main__':

#     args, data, dataset = setup_deepgcn_dp()

#     print("##### running sequential")
#     grads = []
#     for _ in range(4):
#         _, grads1 = test_sequential_deepgcn(args, data, dataset)
#         grads.append(grads1)
#     for i in range(1, 4):
#         print("##### comparing run 0 to run", i)
#         for g1, g2 in zip(grads[0], grads[i]):
#             ratio = g1 / g2
#             print("{:.4f}, {:.4f}, {}".format(ratio.mean(), ratio.std(), ratio.shape))

#     print("##### running DP")
#     grads = []
#     for _ in range(4):
#         _, grads2 = test_dp_deepgcn(args, data, dataset)
#         grads.append(grads2)
#     for i in range(1, 4):
#         print("##### comparing run 0 to run", i)
#         for g1, g2 in zip(grads[0], grads[i]):
#             ratio = g1 / g2
#             print("{:.4f}, {:.4f}, {}".format(ratio.mean(), ratio.std(), ratio.shape))