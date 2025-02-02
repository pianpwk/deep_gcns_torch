import __init__
import torch
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
import wandb
import math


def pad_for_data_parallel(xs, sg_nodes_idxs, sg_edges_s, sg_edges_attrs):
    max_len_x = max([x.shape[0] for x in xs])
    max_len_edges = max([x.shape[0] for x in sg_edges_attrs])
    x, sg_nodes_idx, sg_edges_, sg_edges_attr = [], [], [], []
    for x_ in xs:
        if len(x_) < max_len_x:
            x_ = torch.cat([x_, torch.zeros(max_len_x - len(x_), 8, device=x_.device)], dim=0)
        x.append(x_)
    for val in sg_nodes_idxs:
        if len(val) < max_len_x:
            val = torch.cat([val, torch.zeros(max_len_x - len(val), device=val.device, dtype=val.dtype)], dim=0)
        sg_nodes_idx.append(val)
    for val in sg_edges_s:
        if val.size(1) < max_len_edges:
            val = torch.cat([val, torch.zeros(2, max_len_edges - val.size(1), device=val.device, dtype=val.dtype)], dim=1)
        sg_edges_s.append(val)
    for val in sg_edges_attrs:
        if len(val) < max_len_edges:
            val = torch.cat([val, torch.zeros(max_len_edges - len(val), 8, device=val.device)], dim=0)
        sg_edges_attr.append(val)
    return torch.stack(x), torch.stack(sg_nodes_idx), torch.stack(sg_edges_), torch.stack(sg_edges_attr)
    

def train_subclusters(num_iterations, train_parts, n_clusters, dataset, model, optimizer, criterion, device):

    loss_list = []
    model.train()

    train_y = dataset.y[dataset.train_idx]

    for i in range(num_iterations):

        nodes, edges, edges_index, edges_orig = dataset.sample_subclusters(train_parts, n_clusters, 5)
        nodes_cpu = nodes

        x = dataset.x[nodes].float().to(device)
        nodes = torch.LongTensor(nodes).to(device)

        edges = edges.to(device)
        edges_attr = dataset.edge_attr[edges_index].to(device)

        mapper = {node: idx for idx, node in enumerate(nodes_cpu)}

        inter_idx = intersection(nodes_cpu, dataset.train_idx.tolist())
        training_idx = [mapper[t_idx] for t_idx in inter_idx]

        optimizer.zero_grad()

        pred = model(x, nodes, edges, edges_attr)

        target = train_y[inter_idx].to(device)

        loss = criterion(pred[training_idx].to(torch.float32), target.to(torch.float32))
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())

        print(loss.item())

    return statistics.mean(loss_list)


def train(data, dataset, model, optimizer, criterion, device):

    loss_list = []
    model.train()
    sg_nodes, sg_edges, sg_edges_index, _ = data

    train_y = dataset.y[dataset.train_idx]
    idx_clusters = np.arange(len(sg_nodes))
    np.random.shuffle(idx_clusters)

    for i, idx in enumerate(idx_clusters):

        x = dataset.x[sg_nodes[idx]].float().to(device)
        sg_nodes_idx = torch.LongTensor(sg_nodes[idx]).to(device)

        sg_edges_ = sg_edges[idx].to(device)
        sg_edges_attr = dataset.edge_attr[sg_edges_index[idx]].to(device)

        mapper = {node: idx for idx, node in enumerate(sg_nodes[idx])}

        inter_idx = intersection(sg_nodes[idx], dataset.train_idx.tolist())
        training_idx = [mapper[t_idx] for t_idx in inter_idx]

        optimizer.zero_grad()

        st = time.time()
        pred = model(x, sg_nodes_idx, sg_edges_, sg_edges_attr)

        target = train_y[inter_idx].to(device)

        loss = criterion(pred[training_idx].to(torch.float32), target.to(torch.float32))
        st = time.time()
        loss.backward()

        optimizer.step()
        loss_list.append(loss.item())

        wandb.log({'loss': loss.item()})

    return statistics.mean(loss_list)


@torch.no_grad()
def multi_evaluate_subclusters(valid_parts, n_clusters, dataset, model, evaluator, device):
    model.eval()
    target = dataset.y.detach().numpy()

    train_predict = []
    valid_predict = []
    test_predict = []

    train_target_idx = []
    valid_target_idx = []
    test_target_idx = []

    test_idx = dataset.test_idx.tolist()
    train_idx = dataset.train_idx.tolist()
    valid_idx = dataset.valid_idx.tolist()

    n_subclusters = 5
    for i in range(int(math.ceil((np.max(valid_parts) + 1) / float(n_subclusters)))):

        print("eval", i, int(math.ceil(len(valid_parts) / float(n_subclusters))))

        nodes = np.arange(i * n_subclusters, (i + 1) * n_subclusters).astype(int)
        nodes, edges, edges_index, edges_orig = dataset.sample_subclusters(valid_parts, n_clusters, n_subclusters, idxs=nodes)
        nodes_cpu = nodes

        x = dataset.x[nodes].float().to(device)
        nodes = torch.LongTensor(nodes).to(device)

        mapper = {node: idx for idx, node in enumerate(nodes_cpu)}
        edges_attr = dataset.edge_attr[edges_index].to(device)

        inter_tr_idx = intersection(nodes_cpu, train_idx)
        inter_v_idx = intersection(nodes_cpu, valid_idx)

        tr_idx = [mapper[tr_idx] for tr_idx in inter_tr_idx]
        v_idx = [mapper[v_idx] for v_idx in inter_v_idx]

        pred = model(x, nodes, edges.to(device), edges_attr).cpu().detach()

        train_predict.append(pred[tr_idx])
        valid_predict.append(pred[v_idx])

        inter_te_idx = intersection(nodes_cpu, test_idx)

        te_idx = [mapper[te_idx] for te_idx in inter_te_idx]
        test_predict.append(pred[te_idx])

        train_target_idx += inter_tr_idx
        valid_target_idx += inter_v_idx
        test_target_idx += inter_te_idx

    train_pre = torch.cat(train_predict, 0).numpy()
    valid_pre = torch.cat(valid_predict, 0).numpy()
    test_pre = torch.cat(test_predict, 0).numpy()

    train_pre_final = train_pre[process_indexes(train_target_idx)]
    valid_pre_final = valid_pre[process_indexes(valid_target_idx)]
    test_pre_final = test_pre[process_indexes(test_target_idx)]

    eval_result = {}

    input_dict = {"y_true": target[train_idx], "y_pred": train_pre_final}
    eval_result["train"] = evaluator.eval(input_dict)

    input_dict = {"y_true": target[valid_idx], "y_pred": valid_pre_final}
    eval_result["valid"] = evaluator.eval(input_dict)

    input_dict = {"y_true": target[test_idx], "y_pred": test_pre_final}
    eval_result["test"] = evaluator.eval(input_dict)

    return eval_result


@torch.no_grad()
def multi_evaluate(valid_data_list, dataset, model, evaluator, device):
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

        for idx in idx_clusters:
            x = dataset.x[sg_nodes[idx]].float().to(device)
            sg_nodes_idx = torch.LongTensor(sg_nodes[idx]).to(device)

            mapper = {node: idx for idx, node in enumerate(sg_nodes[idx])}
            sg_edges_attr = dataset.edge_attr[sg_edges_index[idx]].to(device)

            inter_tr_idx = intersection(sg_nodes[idx], train_idx)
            inter_v_idx = intersection(sg_nodes[idx], valid_idx)

            train_target_idx += inter_tr_idx
            valid_target_idx += inter_v_idx

            tr_idx = [mapper[tr_idx] for tr_idx in inter_tr_idx]
            v_idx = [mapper[v_idx] for v_idx in inter_v_idx]

            pred = model(x, sg_nodes_idx, sg_edges[idx].to(device), sg_edges_attr).cpu().detach()

            train_predict.append(pred[tr_idx])
            valid_predict.append(pred[v_idx])

            inter_te_idx = intersection(sg_nodes[idx], test_idx)
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
    wandb.init(project='deepgcn')

    evaluator = Evaluator(args.dataset)
    criterion = torch.nn.BCEWithLogitsLoss()

    valid_data_list = []
    # if not args.metis:
    #     for i in range(args.num_evals):
    #         parts = dataset.random_partition_graph(dataset.total_no_of_nodes,
    #                                             cluster_number=args.valid_cluster_number)
    #         valid_data = dataset.generate_sub_graphs(parts,
    #                                                 cluster_number=args.valid_cluster_number)
    #         valid_data_list.append(valid_data)
    # else:
    #     valid_parts = dataset.random_partition_graph(dataset.total_no_of_nodes, cluster_number=args.valid_cluster_number, uniform=not args.metis, metis_subparts=args.metis_subparts)
    #     # valid_parts = dataset.random_partition_graph(dataset.total_no_of_nodes, cluster_number=10, uniform=True)
    for i in range(args.num_evals):
        parts = dataset.random_partition_graph(dataset.total_no_of_nodes, cluster_number=10, uniform=True)
        valid_data = dataset.generate_sub_graphs(parts, cluster_number=10)
        valid_data_list.append(valid_data)

    sub_dir = 'random-train_{}-test_{}-num_evals_{}'.format(args.cluster_number,
                                                            args.valid_cluster_number,
                                                            args.num_evals)
    logging.info(sub_dir)

    model = DeeperGCN(args).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    results = {'highest_valid': 0,
               'final_train': 0,
               'final_test': 0,
               'highest_train': 0}

    start_time = time.time()

    loss_values, score_values = [], []
 
    # do random partition every epoch
    # train_parts = dataset.random_partition_graph(dataset.total_no_of_nodes,
    #                                                 cluster_number=args.cluster_number, uniform=not args.metis, metis_subparts=args.metis_subparts)
    # if not args.metis:
    #     data = dataset.generate_sub_graphs(train_parts, cluster_number=args.cluster_number)
    # train_parts = dataset.random_partition_graph(dataset.total_no_of_nodes, cluster_number=10, uniform=True)
    # data = dataset.generate_sub_graphs(train_parts, cluster_number=10)

    for epoch in range(args.epochs):

        if args.metis and epoch % 10 == 0:
            if epoch < 100:
                train_parts = dataset.random_partition_graph(dataset.total_no_of_nodes,
                                                    cluster_number=args.cluster_number, uniform=not args.metis, metis_subparts=args.metis_subparts)
            else:
                train_parts = dataset.random_partition_graph(dataset.total_no_of_nodes, cluster_number=10, uniform=True)
                data = dataset.generate_sub_graphs(train_parts, cluster_number=10)

        if not args.metis or epoch > 100:
            epoch_loss = train(data, dataset, model, optimizer, criterion, device)
        elif args.metis and epoch < 100:
            epoch_loss = train_subclusters(
                int(args.cluster_number / 5) if args.metis_subparts is None else int(args.cluster_number * args.metis_subparts / 5),
                train_parts,
                args.cluster_number if args.metis_subparts is None else int(args.cluster_number * args.metis_subparts),
                dataset,
                model,
                optimizer,
                criterion,
                device
            )
        # epoch_loss = train(data, dataset, model, optimizer, criterion, device)
        logging.info('Epoch {}, training loss {:.4f}'.format(epoch, epoch_loss))

        model.print_params(epoch=epoch)

        if epoch % 5 == 0:
            # if not args.metis:
            #     result = multi_evaluate(valid_data_list, dataset, model, evaluator, device)
            # else:
            #     result = multi_evaluate_subclusters(
            #         valid_parts,
            #         args.valid_cluster_number if args.metis_subparts is None else int(args.valid_cluster_number * args.metis_subparts),
            #         dataset,
            #         model,
            #         evaluator,
            #         device
            #     )
            result = multi_evaluate(valid_data_list, dataset, model, evaluator, device)

            if epoch % 5 == 0:
                logging.info('%s' % result)

            train_result = result['train']['rocauc']
            valid_result = result['valid']['rocauc']
            test_result = result['test']['rocauc']

        loss_values.append(epoch_loss)
        if epoch % 5 == 0:
            score_values.append([train_result, valid_result, test_result])
            wandb.log({'loss': epoch_loss, 'train': train_result, 'valid': valid_result, 'test': test_result})
        else:
            wandb.log({'loss': epoch_loss})

        if epoch % 5 == 0:
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

        if epoch % 10 == 0 and epoch > 0:
            np.save("loss_values.npy", loss_values)
            np.save("score_values.npy", score_values)

    logging.info("%s" % results)

    end_time = time.time()
    total_time = end_time - start_time
    logging.info('Total time: {}'.format(time.strftime('%H:%M:%S', time.gmtime(total_time))))


if __name__ == "__main__":
    main()
