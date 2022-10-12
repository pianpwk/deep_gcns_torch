import __init__

import logging
import numpy as np
import os
import pandas as pd
import pymetis
import scipy.sparse
import statistics
import time
import wandb

import torch
import torch.nn.functional as F
import torch_geometric as tg

from ogb.nodeproppred import Evaluator
from ogb.nodeproppred import PygNodePropPredDataset
from torch_sparse import SparseTensor
from torch_geometric.utils import add_self_loops

from args import ArgsInit
from model import DeeperGCN
from utils.ckpt_util import save_ckpt
from utils.data_util import intersection, process_indexes, random_partition_graph, generate_sub_graphs


def make_adjacency_list(nodes, edges):
    adjacency_list = []
    _edges = []
    E = edges.T
    for i, edge in enumerate(E):
        _edges.append(edge[1])
        if i == len(E) - 1 or E[i + 1][0] != edge[0]:
            adjacency_list.append(np.array(_edges))
            # fill in gaps
            next_node = len(nodes) - 1 if i == len(E) - 1 else E[i + 1][0]
            for i in range(next_node - edge[0] - 1):
                adjacency_list.append(np.array([]))
            _edges = []
    return adjacency_list


def metis_partition_graph(adj, num_nodes, cluster_number=100, uniform=False, metis_subparts=None):
    n_tries = 0
    while True:
        try:
            if metis_subparts is None:
                nodes = np.arange(num_nodes).astype(int)
                edges = tg.utils.from_scipy_sparse_matrix(adj)[0]
                adj_list = make_adjacency_list(nodes, edges.numpy())
                _, parts = pymetis.part_graph(int(cluster_number), adjacency=adj_list)
            else:
                parts = np.zeros(shape=[num_nodes]).astype(int)
                partition_1 = np.random.randint(metis_subparts, size=num_nodes)
                for i in range(metis_subparts):
                    nodes_1 = np.where(partition_1 == i)[0]
                    # assert len(nodes_1) > 10 and len(nodes_1) < 100000
                    edges_1 = tg.utils.from_scipy_sparse_matrix(adj[nodes_1][:, nodes_1])[0]
                    adj_list = make_adjacency_list(nodes_1, edges_1.numpy())
                    _, subparts = pymetis.part_graph(cluster_number, adjacency=adj_list)
                    subparts = np.array(subparts)
                    # for j in np.unique(subparts):
                    #     assert len(subparts[subparts == j]) > 0 and len(subparts[subparts == j]) < 20000
                    parts[nodes_1] = subparts + int(i * cluster_number)
            break
        except Exception as e:
            if n_tries > 10:
                raise e
            n_tries += 1
        print(n_tries)
                
    return parts


def sample_subclusters(adj, parts, n_clusters, number, idxs=None):

    if idxs is None:
        for _ in range(100):
            idxs = np.random.choice(np.arange(0, n_clusters).astype(int), size=[number], replace=False)
            nodes = np.unique(np.concatenate([np.where(parts == i)[0] for i in idxs], axis=0))
            if len(nodes) > 0:
                break
    else:
        nodes = np.unique(np.concatenate([np.where(parts == i)[0] for i in idxs], axis=0))

    edges = tg.utils.from_scipy_sparse_matrix(adj[nodes, :][:, nodes])[0]
    mapper = {nd_idx: nd_orig_idx for nd_idx, nd_orig_idx in enumerate(nodes)}

    print("nodes {}, edges {}".format(nodes.shape, edges.shape))

    return nodes, edges


@torch.no_grad()
def test(model, x, edge_index, y_true, split_idx, evaluator):
    # test on CPU
    model.eval()
    model.to('cpu')
    out = model(x, edge_index)

    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': y_true[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': y_true[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': y_true[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, valid_acc, test_acc


@torch.no_grad()
def test_subclusters(data, model, x, edge_index, y_true, split_idx, evaluator, device):
    # test on CPU
    model.eval()

    train_predict = []
    valid_predict = []
    test_predict = []

    train_target_idx = []
    valid_target_idx = []
    test_target_idx = []

    sg_nodes, sg_edges = data
    idx_clusters = np.arange(len(sg_nodes))
    for idx in idx_clusters:

        x_ = x[sg_nodes[idx]].to(device)
        sg_edges_ = sg_edges[idx].to(device)
        mapper = {node: idx for idx, node in enumerate(sg_nodes[idx])}

        inter_tr_idx = intersection(sg_nodes[idx], split_idx['train'].numpy())
        inter_v_idx = intersection(sg_nodes[idx], split_idx['valid'].numpy())
        inter_te_idx = intersection(sg_nodes[idx], split_idx['test'].numpy())

        train_idx = [mapper[t_idx] for t_idx in inter_tr_idx]
        valid_idx = [mapper[t_idx] for t_idx in inter_v_idx]
        test_idx = [mapper[t_idx] for t_idx in inter_te_idx]

        pred = model(x_, sg_edges_).cpu().detach()
        pred = pred.argmax(dim=-1, keepdim=True)

        train_predict.append(pred[train_idx])
        valid_predict.append(pred[valid_idx])
        test_predict.append(pred[test_idx])

        train_target_idx += inter_tr_idx
        valid_target_idx += inter_v_idx
        test_target_idx += inter_te_idx

    train_predict = torch.cat(train_predict, 0).numpy()
    valid_predict = torch.cat(valid_predict, 0).numpy()
    test_predict = torch.cat(test_predict, 0).numpy()

    train_pre = train_predict[process_indexes(train_target_idx)]
    valid_pre = valid_predict[process_indexes(valid_target_idx)]
    test_pre = test_predict[process_indexes(test_target_idx)]

    train_acc = evaluator.eval({
        'y_true': y_true[split_idx['train']],
        'y_pred': train_pre,
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': y_true[split_idx['valid']],
        'y_pred': valid_pre,
    })['acc']
    test_acc = evaluator.eval({
        'y_true': y_true[split_idx['test']],
        'y_pred': test_pre,
    })['acc']

    return train_acc, valid_acc, test_acc



def train_subclusters(num_iterations, train_parts, n_clusters, adj, dataset, model, x, y_true, train_idx, optimizer, device):
    loss_list = []
    model.train()

    train_y = y_true[train_idx].squeeze(1)

    for i in range(num_iterations):

        nodes, edges = sample_subclusters(adj, train_parts, n_clusters, 5)
        nodes_cpu = nodes

        _x = x[nodes].float().to(device)
        edges = edges.to(device)
        mapper = {node: idx for idx, node in enumerate(nodes_cpu)}

        inter_idx = intersection(nodes_cpu, train_idx)
        training_idx = [mapper[t_idx] for t_idx in inter_idx]
        print(x.shape, edges.shape)

        optimizer.zero_grad()
        pred = model(_x, edges)
        target = train_y[inter_idx].to(device)

        loss = F.nll_loss(pred[training_idx], target)
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())

        print(loss.item())

        wandb.log({'loss': loss.item()})

    return statistics.mean(loss_list)


def train(data, model, x, y_true, train_idx, optimizer, device):
    loss_list = []
    model.train()

    sg_nodes, sg_edges = data
    train_y = y_true[train_idx].squeeze(1)

    idx_clusters = np.arange(len(sg_nodes))
    np.random.shuffle(idx_clusters)

    for idx in idx_clusters:

        x_ = x[sg_nodes[idx]].to(device)
        sg_edges_ = sg_edges[idx].to(device)
        mapper = {node: idx for idx, node in enumerate(sg_nodes[idx])}

        inter_idx = intersection(sg_nodes[idx], train_idx)
        training_idx = [mapper[t_idx] for t_idx in inter_idx]

        optimizer.zero_grad()

        pred = model(x_, sg_edges_)
        target = train_y[inter_idx].to(device)

        loss = F.nll_loss(pred[training_idx], target)
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())

        wandb.log({'loss': loss.item()})

    return statistics.mean(loss_list)


def main():

    args = ArgsInit().save_exp()

    if args.use_gpu:
        device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    else:
        device = torch.device('cpu')

    dataset = PygNodePropPredDataset(name=args.dataset)
    graph = dataset[0]

    adj = SparseTensor(row=graph.edge_index[0],
                       col=graph.edge_index[1])

    if args.self_loop:
        adj = adj.set_diag()
        graph.edge_index = add_self_loops(edge_index=graph.edge_index,
                                          num_nodes=graph.num_nodes)[0]

    x, y, _ = adj.coo()
    adj_scipy = scipy.sparse.csr_matrix((np.ones([x.shape[0]]), (x.numpy(), y.numpy())), shape=(adj.size(0), adj.size(1)))

    split_idx = dataset.get_idx_split()
    train_idx = split_idx["train"].tolist()

    evaluator = Evaluator(args.dataset)

    valid_parts = random_partition_graph(graph.num_nodes, cluster_number=10)
    valid_data = generate_sub_graphs(adj_scipy, valid_parts, cluster_number=10)

    sub_dir = 'random-train_{}-full_batch_test'.format(args.cluster_number)
    logging.info(sub_dir)

    args.in_channels = graph.x.size(-1)
    args.num_tasks = dataset.num_classes

    logging.info('%s' % args)

    model = DeeperGCN(args).to(device)

    logging.info(model)
    wandb.init(project='deepgcn-ogbn-products')

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    results = {'highest_valid': 0,
               'final_train': 0,
               'final_test': 0,
               'highest_train': 0}

    start_time = time.time()

    for epoch in range(args.epochs):

        if args.metis and epoch % 10 == 0:
            if epoch < 100:
                parts = metis_partition_graph(adj_scipy, graph.num_nodes,
                                       cluster_number=args.cluster_number, uniform=False, metis_subparts=args.metis_subparts)
            else:
                parts = random_partition_graph(graph.num_nodes,
                                       cluster_number=10)
                data = generate_sub_graphs(adj_scipy, parts, cluster_number=10)
        else:
            parts = random_partition_graph(graph.num_nodes,
                                       cluster_number=args.cluster_number)
            data = generate_sub_graphs(adj_scipy, parts, cluster_number=args.cluster_number)

        if not args.metis or epoch > 100:
            epoch_loss = train(data, model, graph.x, graph.y, train_idx, optimizer, device)
        else:
            epoch_loss = train_subclusters(
                int(args.cluster_number / 5) if args.metis_subparts is None else int(args.cluster_number * args.metis_subparts / 5),
                parts,
                args.cluster_number if args.metis_subparts is None else int(args.cluster_number * args.metis_subparts),
                adj_scipy,
                dataset,
                model,
                graph.x,
                graph.y,
                train_idx,
                optimizer,
                device
            )

        logging.info('Epoch {}, training loss {:.4f}'.format(epoch, epoch_loss))
        model.print_params(epoch=epoch)

        if epoch % 5 == 0:

            result = test(model, graph.x, graph.edge_index, graph.y, split_idx, evaluator)
            # result = test_subclusters(valid_data, model, graph.x, graph.edge_index, graph.y, split_idx, evaluator, device)
            logging.info(result)

            train_accuracy, valid_accuracy, test_accuracy = result

            if train_accuracy > results['highest_train']:
                results['highest_train'] = train_accuracy

            if valid_accuracy > results['highest_valid']:
                results['highest_valid'] = valid_accuracy
                results['final_train'] = train_accuracy
                results['final_test'] = test_accuracy

                save_ckpt(model, optimizer,
                          round(epoch_loss, 4), epoch,
                          args.model_save_path,
                          sub_dir, name_post='valid_best')

        if epoch % 5 == 0:
            wandb.log({
                'loss': epoch_loss,
                'train': train_accuracy,
                'valid': valid_accuracy,
                'test': test_accuracy
            })
        else:
            wandb.log({'loss': epoch_loss})

    logging.info("%s" % results)

    end_time = time.time()
    total_time = end_time - start_time
    logging.info('Total time: {}'.format(time.strftime('%H:%M:%S', time.gmtime(total_time))))


if __name__ == "__main__":
    main()
