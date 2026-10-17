import argparse
from sklearn.metrics import roc_auc_score, r2_score, mean_squared_error
from utils_fp_downstream import *
from torch_geometric.loader import DataLoader
from model_gnn_fp_downstream import MolDEP
import torch.nn as nn
import os
import numpy as np

np.set_printoptions(threshold=np.inf)
import random


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # CPU
    torch.cuda.manual_seed(seed)  # GPU
    torch.cuda.manual_seed_all(seed)  # All GPU
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(model, device, data_loader, optimizer, epoch):
    print('Training on {} samples...'.format(len(data_loader.dataset)))
    model.train()
    train_pred = torch.Tensor()
    train_y = torch.Tensor()
    total_loss = 0.0
    loss_values = []
    for batch_idx, data in enumerate(data_loader):
        data = data.to(device)
        optimizer.zero_grad()
        out, y, w, out_loss, y_loss, w_loss = model(data)

        pred_loss = nn.Sigmoid()(out_loss)
        pred = nn.Sigmoid()(out)
        train_pred = torch.cat((train_pred, torch.Tensor(pred.cpu().data.numpy())), 0)
        train_y = torch.cat((train_y, torch.Tensor(y.cpu().data.numpy())), 0)
        loss = nn.BCELoss(weight=w_loss, reduction='mean')(pred_loss, y_loss)
        total_loss += loss.item()

        loss.backward()
        optimizer.step()
        # scheduler.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} \tLoss: {:.6f}'.format(epoch, total_loss / (batch_idx + 1)))

    avg_loss = total_loss / len(data_loader)
    return train_pred, train_y, avg_loss


def predicting(model, device, data_loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    val_loss = 0.0
    val_loss_values = []
    print('Make prediction for {} samples...'.format(len(data_loader.dataset)))
    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            out, y, w, out_loss, y_loss, w_loss = model(data)

            pred = nn.Sigmoid()(out)
            pred_loss = nn.Sigmoid()(out_loss)
            pred = pred.to('cpu')
            y_ = y.to('cpu')
            total_preds = torch.cat((total_preds, pred), 0)
            total_labels = torch.cat((total_labels, y_), 0)
            loss = nn.BCELoss(weight=w_loss, reduction='mean')(pred_loss, y_loss)
            val_loss += loss.item()
    avg_loss = val_loss / len(data_loader)

    return total_preds.numpy().flatten(), total_labels.numpy().flatten(), avg_loss


def caculate_auc(array1, array2, m):
    M = len(array1)
    auc_list = []

    for i in range(m):
        sub_array1 = array1[i::m]
        sub_array2 = array2[i::m]
        non_999_indices = sub_array1 != 999
        sub_array1 = sub_array1[non_999_indices]
        sub_array2 = sub_array2[non_999_indices]

        if np.unique(sub_array1).size > 1:
            auc = roc_auc_score(sub_array1, sub_array2)
            auc_list.append(auc)

    return np.mean(auc_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MolDEP')
    parser.add_argument('--path', default='down_task')
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--epochs', default=10000, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--task', default='bace', type=str,
                        choices=['bace', 'bbbp', 'clintox', 'sider', 'tox21', 'hiv', 'muv', 'toxcast'])
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument('--llm_model', nargs='?', default='llama',
                        help='Choose a LLM. [llama/bert/qwen/biobert/biogpt/pubmedbert]')
    parser.add_argument('--branch', default='three', type=str,
                        choices=['three', 'two', 'one']) # the number of branch

    args = parser.parse_args()
    print(args)

    batch_size, epochs = args.batch_size, args.epochs
    task, seed = args.task, args.seed
    branch = args.branch

    set_seed(seed)

    device = torch.device('cuda:%d' % args.gpu if torch.cuda.is_available() else 'cpu')

    LOG_INTERVAL = 20

    clr_tasks = {'bbbp': 1, 'hiv': 1, 'bace': 1, 'tox21': 12, 'clintox': 2, 'sider': 27, 'muv': 17, 'toxcast': 617}
    task_num = clr_tasks[task]

    train_data = TestbedDataset(root=args.path, task=task, dataset='train', llm_model=args.llm_model)
    valid_data = TestbedDataset(root=args.path, task=task, dataset='valid', llm_model=args.llm_model)
    test_data = TestbedDataset(root=args.path, task=task, dataset='test', llm_model=args.llm_model)

    train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_data_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)
    test_data_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    x_dim = train_data[0].x.shape[1]
    pubchem_dim = train_data[0].pubchem.shape[0]
    maccs_dim = train_data[0].maccs.shape[0]
    erg_dim = train_data[0].erg.shape[0]
    ecfp_dim = train_data[0].ecfp.shape[0]
    fp_dim = pubchem_dim + maccs_dim + erg_dim + ecfp_dim
    llm_dim = train_data[0].llm.shape[0]
    edge_dim = train_data[0].edge_attr.shape[1]
    model = MolDEP(x_input=x_dim, fp_input=fp_dim, llm_input=llm_dim, edge_dim=edge_dim, branch=branch, output=task_num).to(device)

    optimizer = torch.optim.Adam([{'params': model.parameters(), 'lr': args.lr, 'weight_decay': args.weight_decay}])

    if not os.path.exists('results/'):
        os.makedirs('results/')
    model_file_name = 'results/' + task + '_encoder.pkl'

    best_auc = 0
    stopping_monitor = 0
    train_losses = []
    valid_losses = []
    test_losses = []
    for epoch in range(epochs + 1):
        train_pred, train_y, test_loss = train(model, device, train_data_loader, optimizer, epoch + 1)
        train_losses.append(test_loss)
        valid_pred, valid_true, val_loss = predicting(model, device, valid_data_loader)
        valid_losses.append(val_loss)
        test_pred, test_true, test_loss = predicting(model, device, test_data_loader)
        test_losses.append(test_loss)

        if (epoch + 0) % 5 == 0:
            train_auc = caculate_auc(train_y, train_pred, task_num)
            AUCs = [epoch, train_auc]

            print('train_AUC:', train_auc)

            valid_auc = caculate_auc(valid_true, valid_pred, task_num)
            AUCs = [epoch, valid_auc]
            print('valid_AUC: ', AUCs)

            if best_auc < valid_auc:
                best_auc = valid_auc
                stopping_monitor = 0
                print('best_auc：', best_auc)
                print('save model weights')
                torch.save(model.state_dict(), model_file_name)
            else:
                stopping_monitor += 1
            if stopping_monitor > 0:
                print('stopping_monitor:', stopping_monitor)
            if stopping_monitor > 20:
                break

    model.load_state_dict(torch.load(model_file_name))
    test_pred, test_true, loss = predicting(model, device, test_data_loader)
    print('pred value：', test_pred)
    print('true value：', test_true)
    print('loss value：', loss)

    test_auc = caculate_auc(test_true, test_pred, task_num)
    AUCs = [0, test_auc]
    print(task, seed, 'test_AUC: ', AUCs)

