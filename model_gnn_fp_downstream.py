import torch.nn as nn
import torch.nn.functional as F
import torch
from torch_geometric.nn import GINEConv
from torch.nn import Linear
from torch_geometric.nn import global_add_pool as gmp


class GINNet(torch.nn.Module):
    def __init__(self, x_input, hidden_dim, edge_dim, eps=0., train_eps=True):
        super(GINNet, self).__init__()

        self.x_input = x_input
        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim

        self.conv1 = GINEConv(nn=Linear(x_input, hidden_dim), eps=eps, train_eps=train_eps)
        self.conv2 = GINEConv(nn=Linear(hidden_dim, hidden_dim * 2), eps=eps, train_eps=train_eps)
        self.conv3 = GINEConv(nn=Linear(hidden_dim * 2, hidden_dim), eps=eps, train_eps=train_eps)

        self.fc1 = Linear(edge_dim, x_input)
        self.fc2 = Linear(edge_dim, hidden_dim)
        self.fc3 = Linear(edge_dim, hidden_dim * 2)

    def forward(self, x, edge_index, edge_attr, batch):
        edge_attr1 = self.fc1(edge_attr)
        edge_attr2 = self.fc2(edge_attr)
        edge_attr3 = self.fc3(edge_attr)

        x = self.conv1(x, edge_index, edge_attr1)
        x = F.elu(x)

        x = self.conv2(x, edge_index, edge_attr2)
        x = F.elu(x)

        x = self.conv3(x, edge_index, edge_attr3)
        x = F.elu(x)

        x_mean = gmp(x, batch)

        return x_mean


class FullNet(nn.Module):
    def __init__(self, fp_input, hidden_dim=512, dropout=0.2):
        super(FullNet, self).__init__()
        self.fp_input = fp_input
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        self.fp_predict = nn.Sequential(
            nn.Linear(fp_input, hidden_dim * 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

    def forward(self, fp_x):
        fp_feature = self.fp_predict(fp_x)
        return fp_feature
    
    
class MolDEP(nn.Module):
    def __init__(self, x_input, fp_input, llm_input, edge_dim, branch='three', output=1, hidden_dim=512, dropout=0.2):
        super(MolDEP, self).__init__()
        self.x_input = x_input
        self.fp_input = fp_input
        self.llm_input = llm_input
        self.edge_dim = edge_dim
        self.output = output
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.branch = branch
        
        self.pathway1 = GINNet(x_input, hidden_dim, edge_dim)
        self.pathway2 = FullNet(fp_input, hidden_dim)
        self.pathway3 = FullNet(llm_input, hidden_dim)

        if self.branch == 'one':
            self.attention = None
        elif self.branch == 'two':
            self.attention = nn.Linear(hidden_dim * 2, 1)
        elif self.branch == 'three':
            self.attention = nn.Linear(hidden_dim * 3, 3)

        self.pre = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output)
        )


    def forward(self, data):
        x, edge_index, batch, y ,edge_attr, w = data.x, data.edge_index, data.batch, data.y, data.edge_attr, data.w

        out1 = self.pathway1(x, edge_index, edge_attr, batch)
        out1 = F.normalize(out1)

        pubchem = data.pubchem.reshape(data.num_graphs, -1)
        maccs = data.maccs.reshape(data.num_graphs, -1)
        erg = data.erg.reshape(data.num_graphs, -1)
        ecfp = data.ecfp.reshape(data.num_graphs, -1)
        fp_x = torch.cat((pubchem, maccs, erg, ecfp), dim=1)
        out2 = self.pathway2(fp_x)
        out2 = F.normalize(out2)

        llm_x = data.llm.reshape(data.num_graphs, -1)
        out3 = self.pathway3(llm_x)
        out3 = F.normalize(out3)

        if self.branch == 'one':
            out = out1
        elif self.branch == 'two':
            alpha = torch.sigmoid(self.attention(torch.cat([out1, out2], dim=-1)))
            out = alpha * out1 + (1 - alpha) * out2
        elif self.branch == 'three':
            alpha = torch.softmax(self.attention(torch.cat([out1, out2, out3], dim=-1)), dim=-1)
            out = alpha[:, 0:1] * out1 + alpha[:, 1:2] * out2 + alpha[:, 2:3] * out3

        out = self.pre(out)
        out = out.reshape(-1)

        non_999_indices = y != 999
        out_loss = out[non_999_indices]
        y_loss = y[non_999_indices]
        w_loss = w[non_999_indices]

        return out, y, w, out_loss, y_loss, w_loss
