import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class GCN(nn.Module):
    def __init__(self, configs):
        super(GCN, self).__init__()

        self.dropout_rate = 0.5  

        self.gcn1_t = GCNConv(116, 128)
        self.gcn2_t = GCNConv(128, 64)
        self.gcn3_t = GCNConv(64, 32)  


        self.gcn1_f = GCNConv(101, 128)
        self.gcn2_f = GCNConv(128, 64)
        self.gcn3_f = GCNConv(64, 32)  

        self.batch_norm1_t = nn.BatchNorm1d(128)
        self.batch_norm2_t = nn.BatchNorm1d(64)
        self.batch_norm3_t = nn.BatchNorm1d(32)
        self.batch_norm1_f = nn.BatchNorm1d(128)
        self.batch_norm2_f = nn.BatchNorm1d(64)
        self.batch_norm3_f = nn.BatchNorm1d(32)

        self.projector_t = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(64, 32)
        )

        self.projector_f = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(64, 32)
        )

    def forward(self, data_t, data_f):
        # === Time domain ===
        x_t = self.gcn1_t(data_t.x, data_t.edge_index, edge_weight=data_t.edge_attr)
        x_t = self.batch_norm1_t(x_t)
        x_t = F.relu(x_t)
        x_t = F.dropout(x_t, p=self.dropout_rate, training=self.training)

        x_t = self.gcn2_t(x_t, data_t.edge_index, edge_weight=data_t.edge_attr)
        x_t = self.batch_norm2_t(x_t)
        x_t = F.relu(x_t)
        x_t = F.dropout(x_t, p=self.dropout_rate, training=self.training)

        x_t = self.gcn3_t(x_t, data_t.edge_index, edge_weight=data_t.edge_attr)
        x_t = self.batch_norm3_t(x_t)
        x_t = F.relu(x_t)
        x_t = F.dropout(x_t, p=self.dropout_rate, training=self.training)

        x_t_pool = global_mean_pool(x_t, data_t.batch)
        h_time = x_t_pool
        z_time = F.normalize(self.projector_t(h_time), dim=1)  # Normalize

        # === Frequency domain ===
        x_f = self.gcn1_f(data_f.x, data_f.edge_index, edge_weight=data_f.edge_attr)
        x_f = self.batch_norm1_f(x_f)
        x_f = F.relu(x_f)
        x_f = F.dropout(x_f, p=self.dropout_rate, training=self.training)

        x_f = self.gcn2_f(x_f, data_f.edge_index, edge_weight=data_f.edge_attr)
        x_f = self.batch_norm2_f(x_f)
        x_f = F.relu(x_f)
        x_f = F.dropout(x_f, p=self.dropout_rate, training=self.training)

        x_f = self.gcn3_f(x_f, data_f.edge_index, edge_weight=data_f.edge_attr)
        x_f = self.batch_norm3_f(x_f)
        x_f = F.relu(x_f)
        x_f = F.dropout(x_f, p=self.dropout_rate, training=self.training)

        x_f_pool = global_mean_pool(x_f, data_f.batch)
        h_freq = x_f_pool
        z_freq = F.normalize(self.projector_f(h_freq), dim=1)  

        return h_time, z_time, h_freq, z_freq, x_t, x_f



class TargetClassifier(nn.Module):
    def __init__(self, configs):
        super(TargetClassifier, self).__init__()
        self.fc1 = nn.Linear(2*32, 16)
        self.logits = nn.Linear(16, configs.num_classes_target)
        self.dropout = nn.Dropout(0.7)

    def forward(self, emb):
        emb = F.relu(self.fc1(emb))
        emb = self.dropout(emb)
        return self.logits(emb)
      