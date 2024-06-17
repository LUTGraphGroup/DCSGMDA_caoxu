import torch
from torch import nn
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv
from load_data import low_feature
# from load_data import factorization_Network

torch.backends.cudnn.enabled = False

low_dim_miRNA, low_dim_disease = low_feature()
low_dim_miRNA = torch.from_numpy(low_dim_miRNA)
low_dim_disease = torch.from_numpy(low_dim_disease)
low_dim_miRNA = low_dim_miRNA.float()
low_dim_disease = low_dim_disease.float()


class CDSG(nn.Module):
    def __init__(self, args):
        """
        :param args: Arguments object.
        """
        super(CDSG, self).__init__()
        self.args = args
        self.gcn_mi1_f = GCNConv(self.args.fmi, self.args.fmi)
        self.gcn_mi2_f = GCNConv(self.args.fmi, self.args.fmi)
        self.gcn_mi3_f = GCNConv(self.args.fmi, self.args.fmi)

        self.gcn_dis1_f = GCNConv(self.args.fdis, self.args.fdis)
        self.gcn_dis2_f = GCNConv(self.args.fdis, self.args.fdis)
        self.gcn_dis3_f = GCNConv(self.args.fdis, self.args.fdis)
        
        

        self.cnn_mi = nn.Conv2d(in_channels=self.args.cnn,
                               out_channels=self.args.out_channels,
                               kernel_size=(5,1),
                               stride=1,
                               bias=True)
        self.cnn_dis = nn.Conv2d(in_channels=self.args.cnn,
                               out_channels=self.args.out_channels,
                               kernel_size=(5, 1),
                               stride=1,
                               bias=True)

        self.gat_mi1_f = GATConv(self.args.fmi, self.args.fmi,heads=4,concat=False,edge_dim=1)
        self.gat_mi2_f = GATConv(self.args.fmi, self.args.fmi, heads=4, concat=False, edge_dim=1)
        self.gat_mi3_f = GATConv(self.args.fmi, self.args.fmi, heads=4, concat=False, edge_dim=1)

        self.gat_dis1_f = GATConv(self.args.fdis, self.args.fdis,heads=4,concat=False,edge_dim=1)
        self.gat_dis2_f = GATConv(self.args.fdis, self.args.fdis, heads=4, concat=False, edge_dim=1)
        self.gat_dis3_f = GATConv(self.args.fdis, self.args.fdis, heads=4, concat=False, edge_dim=1)


    def forward(self, data):
        torch.manual_seed(1)
        x_mi = torch.randn(self.args.miRNA_number, self.args.fmi)  # （788，64）
        x_dis = torch.randn(self.args.disease_number, self.args.fdis)

        x_mi_f1 = torch.relu(self.gcn_mi1_f(x_mi, data['MM']['edges'], data['MM']['data_matrix'][
            data['MM']['edges'][0], data['MM']['edges'][1]]))
        x_mi_f2 = torch.relu(self.gcn_mi2_f(x_mi_f1, data['MM']['edges'], data['MM']['data_matrix'][
            data['MM']['edges'][0], data['MM']['edges'][1]]))
        x_mi_f3 = torch.relu(self.gcn_mi3_f(x_mi_f2, data['MM']['edges'], data['MM']['data_matrix'][
            data['MM']['edges'][0], data['MM']['edges'][1]]))

        x_mi_att1 = torch.relu(self.gat_mi1_f(x_mi_f3, data['MM']['edges'], data['MM']['data_matrix'][
            data['MM']['edges'][0], data['MM']['edges'][1]]))
        x_mi_att2 = torch.relu(self.gat_mi2_f(x_mi_att1, data['MM']['edges'], data['MM']['data_matrix'][
            data['MM']['edges'][0], data['MM']['edges'][1]]))
        x_mi_att3 = torch.relu(self.gat_mi3_f(x_mi_att2, data['MM']['edges'], data['MM']['data_matrix'][
            data['MM']['edges'][0], data['MM']['edges'][1]]))


        x_dis_f1 = torch.relu(self.gcn_dis1_f(x_dis, data['DD']['edges'], data['DD']['data_matrix'][
            data['DD']['edges'][0], data['DD']['edges'][1]]))
        x_dis_f2 = torch.relu(self.gcn_dis2_f(x_dis_f1, data['DD']['edges'], data['DD']['data_matrix'][
            data['DD']['edges'][0], data['DD']['edges'][1]]))
        x_dis_f3 = torch.relu(self.gcn_dis3_f(x_dis_f2, data['DD']['edges'], data['DD']['data_matrix'][
            data['DD']['edges'][0], data['DD']['edges'][1]]))

        x_dis_att1 = torch.relu(self.gat_dis1_f(x_dis_f3, data['DD']['edges'], data['DD']['data_matrix'][
            data['DD']['edges'][0], data['DD']['edges'][1]]))
        x_dis_att2 = torch.relu(self.gat_dis2_f(x_dis_att1, data['DD']['edges'], data['DD']['data_matrix'][
            data['DD']['edges'][0], data['DD']['edges'][1]]))
        x_dis_att3 = torch.relu(self.gat_dis3_f(x_dis_att2, data['DD']['edges'], data['DD']['data_matrix'][
            data['DD']['edges'][0], data['DD']['edges'][1]]))



        X_mi = torch.cat((x_mi_att3, low_dim_miRNA), 1).t()
        X_mi = X_mi.view(1, self.args.gcn_layers, self.args.fmi, -1)

        X_dis = torch.cat((x_dis_att3, low_dim_disease), 1).t()
        X_dis = X_dis.view(1, self.args.gcn_layers, self.args.fdis, -1)

        mi_fea = self.cnn_mi(X_mi)
        mi_fea = mi_fea.view(912, self.args.miRNA_number).t()#（912，788）
        dis_fea = self.cnn_dis(X_dis)
        dis_fea = dis_fea.view(912, self.args.disease_number).t()
        return mi_fea.mm(dis_fea.t()),mi_fea,dis_fea
