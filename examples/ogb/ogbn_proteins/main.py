import __init__
import math
import torch
import torch.nn as nn
import torch.optim as optim
import statistics
from dataset import OGBNDataset
from model import DeeperGCN
from args import ArgsInit
import time
import numpy as np
from ogb.nodeproppred import Evaluator
from utils.ckpt_util import save_ckpt
from utils.data_util import intersection, process_indexes
import logging
# import wandb

from torch_geometric.nn import DataParallel
from torch_geometric.data import Data


class GradualScheduler:
    '''
    Increases lr from 0 -> <lr> across first <warmup_steps> steps, then maintains value of <lr>
    '''
    def __init__(self, lr, warmup_steps):
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.current_step = 0
    def step(self):
        self.current_step += 1
        return self.lr * min(1.0, self.current_step / self.warmup_steps)


def pad_tensors(xs, node_indexs, edge_indexs, edge_attrs):
    num_nodes, num_edges = [x.size(0) for x in xs], [x.size(0) for x in edge_attrs]
    max_nodes, max_edges = max(num_nodes), max(num_edges)
    xs = torch.stack([x if x.size(0) == max_nodes else torch.cat([x, torch.zeros(max_nodes - x.size(0), 8, device=xs[0].device)], dim=0) for x in xs], dim=0)
    node_indexs = torch.stack([x if x.size(0) == max_nodes else torch.cat([x, torch.zeros(max_nodes - x.size(0), device=node_indexs[0].device).long()], dim=0) for x in node_indexs], dim=0) # not zero
    edge_indexs = torch.stack([x if x.size(1) == max_edges else torch.cat([x, torch.zeros(2, max_edges - x.size(1), device=edge_indexs[0].device).long()], dim=1) for x in edge_indexs], dim=0) # not zero
    edge_attrs = torch.stack([x if x.size(0) == max_edges else torch.cat([x, torch.zeros(max_edges - x.size(0), 8, device=edge_attrs[0].device)], dim=0) for x in edge_attrs], dim=0)
    return xs, node_indexs, edge_indexs, edge_attrs, torch.LongTensor(num_nodes), torch.LongTensor(num_edges)


def create_datalist(x, node_index, edge_index, edge_attr):
    data_list = []
    for i in range(len(x)):
        data_list.append(Data(x=x[i], node_index=node_index[i], edge_index=edge_index[i], edge_attr=edge_attr[i]))
    return data_list


def train(data, dataset, model, optimizer, criterion, scheduler, num_gpus, device):

    loss_list = []
    model.train()
    sg_nodes, sg_edges, sg_edges_index, _ = data

    train_y = dataset.y[dataset.train_idx]
    idx_clusters = list(np.arange(len(sg_nodes)))
    np.random.shuffle(idx_clusters)
#     idx_clusters = list(idx_clusters) * 2

    for i in range(int(math.ceil(len(idx_clusters) / float(num_gpus)))):
        clusters = idx_clusters[i*num_gpus : (i+1)*num_gpus]
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

        # step scheduler & zero out optimizer grads
        lr = scheduler.step()
        for g in optimizer.param_groups:
            g['lr'] = lr
        optimizer.zero_grad()

        pred = model(data_list)

        target = train_y[inter_idx].to(device)

        loss = criterion(pred[training_idx].to(torch.float32), target.to(torch.float32))
        loss.backward()
        nn.utils.clip_grad_value_(model.parameters(), 0.5)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        loss_list.append(loss.item())

    return statistics.mean(loss_list)


@torch.no_grad()
def multi_evaluate(valid_data_list, dataset, model, evaluator, num_gpus, device):
    model.eval()
    target = dataset.y.detach().numpy()

    train_pre_ordered_list = []
    valid_pre_ordered_list = []
    test_pre_ordered_list = []

    test_idx = dataset.test_idx.tolist()
    train_idx = dataset.train_idx.tolist()
    valid_idx = dataset.valid_idx.tolist()

    for valid_data_item in valid_data_list:
        sg_nodes, sg_edges, sg_edges_index, _ = valid_data_item
        idx_clusters = np.arange(len(sg_nodes))

        test_predict = []
        test_target_idx = []

        train_predict = []
        valid_predict = []

        train_target_idx = []
        valid_target_idx = []

        for i in range(int(math.ceil(len(idx_clusters) / float(num_gpus)))):
            clusters = idx_clusters[i*num_gpus : (i+1)*num_gpus]
            x = [dataset.x[sg_nodes[idx]].float() for idx in clusters]
            sg_nodes_idx = [torch.LongTensor(sg_nodes[idx]) for idx in clusters]
            sg_edges_ = [sg_edges[idx] for idx in clusters]
            sg_edges_attr = [dataset.edge_attr[sg_edges_index[idx]] for idx in clusters]

            data_list = create_datalist(x, sg_nodes_idx, sg_edges_, sg_edges_attr)

            # create training_idx, assuming cluster partitions are distinct
            all_nodes = np.concatenate([sg_nodes[idx] for idx in clusters], axis=0)
            mapper = {node: idx for idx, node in enumerate(all_nodes)}

            inter_tr_idx = intersection(all_nodes, train_idx)
            inter_v_idx = intersection(all_nodes, valid_idx)

            train_target_idx += inter_tr_idx
            valid_target_idx += inter_v_idx

            tr_idx = [mapper[tr_idx] for tr_idx in inter_tr_idx]
            v_idx = [mapper[v_idx] for v_idx in inter_v_idx]

            pred = model(data_list).cpu().detach()

            train_predict.append(pred[tr_idx])
            valid_predict.append(pred[v_idx])

            inter_te_idx = intersection(all_nodes, test_idx)
            test_target_idx += inter_te_idx

            te_idx = [mapper[te_idx] for te_idx in inter_te_idx]
            test_predict.append(pred[te_idx])

        train_pre = torch.cat(train_predict, 0).numpy()
        valid_pre = torch.cat(valid_predict, 0).numpy()
        test_pre = torch.cat(test_predict, 0).numpy()

        train_pre_ordered = train_pre[process_indexes(train_target_idx)]
        valid_pre_ordered = valid_pre[process_indexes(valid_target_idx)]
        test_pre_ordered = test_pre[process_indexes(test_target_idx)]

        train_pre_ordered_list.append(train_pre_ordered)
        valid_pre_ordered_list.append(valid_pre_ordered)
        test_pre_ordered_list.append(test_pre_ordered)

    train_pre_final = torch.mean(torch.Tensor(train_pre_ordered_list), dim=0)
    valid_pre_final = torch.mean(torch.Tensor(valid_pre_ordered_list), dim=0)
    test_pre_final = torch.mean(torch.Tensor(test_pre_ordered_list), dim=0)

    eval_result = {}

    input_dict = {"y_true": target[train_idx], "y_pred": train_pre_final}
    eval_result["train"] = evaluator.eval(input_dict)

    input_dict = {"y_true": target[valid_idx], "y_pred": valid_pre_final}
    eval_result["valid"] = evaluator.eval(input_dict)

    input_dict = {"y_true": target[test_idx], "y_pred": test_pre_final}
    eval_result["test"] = evaluator.eval(input_dict)

    return eval_result


def main():
    args = ArgsInit().save_exp()
    # wandb.init(project='deepgcn')

    if args.use_gpu:
        device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    else:
        device = torch.device("cpu")

    logging.info('%s' % device)

    dataset = OGBNDataset(dataset_name=args.dataset)
    # extract initial node features
    nf_path = dataset.extract_node_features(args.aggr)

    args.num_tasks = dataset.num_tasks
    args.nf_path = nf_path

    logging.info('%s' % args)

    evaluator = Evaluator(args.dataset)
    criterion = torch.nn.BCEWithLogitsLoss()

    valid_data_list = []

    for i in range(args.num_evals):
        parts = dataset.random_partition_graph(dataset.total_no_of_nodes,
                                               cluster_number=args.valid_cluster_number)
        valid_data = dataset.generate_sub_graphs(parts,
                                                 cluster_number=args.valid_cluster_number)
        valid_data_list.append(valid_data)

    sub_dir = 'random-train_{}-test_{}-num_evals_{}'.format(args.cluster_number,
                                                            args.valid_cluster_number,
                                                            args.num_evals)
    logging.info(sub_dir)

    model = DeeperGCN(args).to(device)
    model = DataParallel(model).cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = GradualScheduler(args.lr, int(args.warmup_ratio * args.cluster_number * args.epochs))

    results = {'highest_valid': 0,
               'final_train': 0,
               'final_test': 0,
               'highest_train': 0}

    start_time = time.time()

    loss_values, score_values = [], []
    for epoch in range(1, args.epochs + 1):
        # do random partition every epoch
        train_parts = dataset.random_partition_graph(dataset.total_no_of_nodes,
                                                     cluster_number=args.cluster_number)
        data = dataset.generate_sub_graphs(train_parts, cluster_number=args.cluster_number)

        epoch_loss = train(data, dataset, model, optimizer, criterion, scheduler, args.num_gpus, device)
        logging.info('Epoch {}, training loss {:.4f}'.format(epoch, epoch_loss))

        model.module.print_params(epoch=epoch)

        result = multi_evaluate(valid_data_list, dataset, model, evaluator, args.num_gpus, device)

        if epoch % 5 == 0:
            logging.info('%s' % result)

        train_result = result['train']['rocauc']
        valid_result = result['valid']['rocauc']
        test_result = result['test']['rocauc']

        # wandb.log({'loss': epoch_loss, 'train': train_result, 'valid': valid_result, 'test': test_result})

        if valid_result > results['highest_valid']:
            results['highest_valid'] = valid_result
            results['final_train'] = train_result
            results['final_test'] = test_result

            save_ckpt(model, optimizer, round(epoch_loss, 4),
                      epoch,
                      args.model_save_path, sub_dir,
                      name_post='valid_best')

        if train_result > results['highest_train']:
            results['highest_train'] = train_result

        loss_values.append(epoch_loss)
        score_values.append([result['train']['rocauc'], result['valid']['rocauc'], result['test']['rocauc']])

        if epoch % 10 == 0:
            np.save("loss_values.npy", loss_values)
            np.save("score_values.npy", score_values)

        if epoch % 500 == 0 and epoch > 0:
            torch.save("state_dict_epoch_{}.pth".format(epoch), {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch
            })

    logging.info("%s" % results)

    end_time = time.time()
    total_time = end_time - start_time
    logging.info('Total time: {}'.format(time.strftime('%H:%M:%S', time.gmtime(total_time))))


if __name__ == "__main__":
    main()
