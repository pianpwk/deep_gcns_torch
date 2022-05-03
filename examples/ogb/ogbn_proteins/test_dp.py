import torch
import torch.nn as nn

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

def get_inputs():
    data, labels = [], []
    for i in range(4):
        torch.manual_seed(i)
        data.append(torch.randn(4, 32))
        torch.manual_seed(i)
        labels.append(torch.randint(0, 8, size=[4]))
    return data, labels

def test_sequential():

    # get model
    torch.manual_seed(128)
    m = nn.Linear(32, 8).cuda()

    # get inputs
    data, labels = get_inputs()

    # run sequence
    criterion = nn.CrossEntropyLoss().cuda()
    print(m.weight.grad)
    m.zero_grad()
    for d, l in zip(data, labels):
        outputs = m(d.cuda())
        loss = criterion(outputs, l.cuda())
        loss.backward()
        print(m.weight.grad)

    return m.weight.grad.detach().cpu().numpy()

def test_dp():

    # get model
    torch.manual_seed(128)
    m = nn.Linear(32, 8)
    m = nn.DataParallel(m).cuda()

    # get inputs
    data, labels = get_inputs()
    data = torch.cat(data, dim=0)
    labels = torch.cat(labels, dim=0)

    # run sequence
    criterion = nn.CrossEntropyLoss().cuda()
    print(m.module.weight.grad)
    m.zero_grad()
    outputs = m(data)
    loss = criterion(outputs, labels.cuda())
    loss.backward()
    print(m.module.weight.grad)

    return m.module.weight.grad.detach().cpu().numpy()

def setup_deepgcn_dp():
    args = ArgsInit().args
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

def test_sequential_deepgcn(args, data, dataset):
    torch.manual_seed(128)
    model = DeeperGCN(args).cuda()
    criterion = nn.BCEWithLogitsLoss().cuda()

    # process data
    train_y = dataset.y[dataset.train_idx]
    model.zero_grad()
    sg_nodes, sg_edges, sg_edges_index, _ = data
    for idx in range(4):
        x = dataset.x[sg_nodes[idx]].float().cuda()
        sg_nodes_idx = torch.LongTensor(sg_nodes[idx]).cuda()
        sg_edges_idx = sg_edges[idx].cuda()
        sg_edges_attr = dataset.edge_attr[idx].cuda()

        mapper = {node: idx for idx, node in enumerate(sg_nodes[idx])}
        inter_idx = intersection(sg_nodes[idx], dataset.train_idx.tolist())
        training_idx = [mapper[t_idx] for t_idx in inter_idx]

        pred = model(x, sg_nodes_idx, sg_edges_idx, sg_edges_attr)
        target = train_y[inter_idx].cuda()
        breakpoint()
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

def test_dp_deepgcn(args, data, dataset):
    torch.manual_seed(128)
    model = DeeperGCNDP(args)
    model = PyGDataParallel(model).cuda()
    criterion = nn.BCEWithLogitsLoss().cuda()

    # process data
    train_y = dataset.y[dataset.train_idx]
    model.zero_grad()
    sg_nodes, sg_edges, sg_edges_index, _ = data

    # stack 
    clusters = range(4)
    x = [dataset.x[sg_nodes[idx]].float() for idx in clusters]
    sg_nodes_idx = [torch.LongTensor(sg_nodes[idx]) for idx in clusters]
    sg_edges_ = [sg_edges[idx] for idx in clusters]
    sg_edges_attr = [dataset.edge_attr[sg_edges_index[idx]] for idx in clusters]
    data_list = create_datalist(x, sg_nodes_idx, sg_edges_, sg_edges_attr)

    # create training_idx, assuming cluster partitions are distinct
    all_nodes = np.concatenate([sg_nodes[idx] for idx in clusters], axis=0) # cat 4 lists in sorted order?
    mapper = {node: idx for idx, node in enumerate(all_nodes)} # duplicates? no: distinct partitions
    inter_idx = intersection(all_nodes, dataset.train_idx.tolist()) # what happens to order here?
    training_idx = [mapper[t_idx] for t_idx in inter_idx]

    # run FWD + BWD
    pred = model(data_list)
    target = train_y[inter_idx].cuda()
    breakpoint()
    target = torch.ones_like(target, dtype=target.dtype).cuda()
    loss = criterion(pred[training_idx].to(torch.float32), target.to(torch.float32))
    loss.backward()

    # collect gradients
    gradients = []
    for m in model.module.parameters():
        if not m.grad is None:
            gradients.append(m.grad.detach().cpu().numpy())
    return model, gradients




# if __name__ == '__main__':
#     print("##### running sequential")
#     g1 = test_sequential()
#     print("#####")
#     print("##### running DP")
#     g2 = test_dp()
#     print("#####")
#     print("##### grad ratio")
#     print( g1 / g2 )

if __name__ == '__main__':
    args, data, dataset = setup_deepgcn_dp()
    _, grads1 = test_sequential_deepgcn(args, data, dataset)
    _, grads2 = test_dp_deepgcn(args, data, dataset)
    for g1, g2 in zip(grads1, grads2):
        print(g1 / g2)

    
