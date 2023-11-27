import os, sys
sys.path.append(os.path.abspath('../..'))
from cdbs.pkgs import *
from cdbs.general import *
from cdbs.custom import *



class EdgeConv(nn.Module):
    def __init__(self, in_channels, out_channels, num_neighbors, num_layers):
        super(EdgeConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_neighbors = num_neighbors
        self.num_layers = num_layers
        assert num_layers in [1, 2]
        if self.num_layers == 1:
            self.smlp = MLP(in_channels*2, out_channels, is_bn=True, nl='relu')
        if self.num_layers == 2:
            smlp_1 = MLP(in_channels*2, out_channels, is_bn=True, nl='relu')
            smlp_2 = MLP(out_channels, out_channels, is_bn=True, nl='relu')
            self.smlp = nn.Sequential(smlp_1, smlp_2)
    def forward(self, pc_ftr):
        num_neighbors = self.num_neighbors
        batch_size, num_points, in_channels = pc_ftr.size()
        knn_indices = knn_search(pc_ftr.detach(), pc_ftr.detach(), num_neighbors)
        nb_ftr = index_points(pc_ftr, knn_indices)
        pc_ftr_rep = pc_ftr.unsqueeze(2).repeat(1, 1, num_neighbors, 1)
        edge_ftr = torch.cat((pc_ftr_rep, nb_ftr-pc_ftr_rep), dim=-1)
        out_ftr = self.smlp(edge_ftr.view(batch_size, num_points*num_neighbors, -1)).view(batch_size, num_points, num_neighbors, -1)
        out_ftr_max_pooled = torch.max(out_ftr, dim=2)[0]
        return out_ftr_max_pooled


class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()
        self.econv_1 = EdgeConv(3, 64, 40, 2)
        self.econv_2 = EdgeConv(64, 64, 40, 2)
        self.econv_3 = EdgeConv(64, 64, 40, 1)
        self.pw_embedding = MLP(192, 320, True, 'relu')
        self.mlp = nn.Sequential(MLP(512, 128, True, 'relu'), MLP(128, 128, True, 'relu'))
    def forward(self, points):
        num_points = points.size(1)
        pw_ftr_1 = self.econv_1(points)
        pw_ftr_2 = self.econv_2(pw_ftr_1)
        pw_ftr_3 = self.econv_3(pw_ftr_2)
        pw_ftr_cat = torch.cat((pw_ftr_1, pw_ftr_2, pw_ftr_3), dim=-1)
        pw_ftr = self.pw_embedding(pw_ftr_cat)
        codeword = torch.max(pw_ftr, dim=1)[0]
        codeword_dup = codeword.unsqueeze(1).repeat(1, num_points, 1)
        pw_ftr_fused = self.mlp(torch.cat((codeword_dup, pw_ftr_1, pw_ftr_2, pw_ftr_3), dim=-1))
        return pw_ftr_fused


class GenNeuroGF(nn.Module):
    def __init__(self):
        super(GenNeuroGF, self).__init__()
        self.backbone = Backbone()
        self.fuse = MLP(128+3, 128, True, 'relu')
        self.head = nn.Sequential(MLP(128, 128, True, 'relu'), MLP(128, 64, True, 'relu'), MLP(64, 1, False, 'none'))
    def forward(self, P, S, T):
        # P: [B, N, 3]
        # S: [B, M, 3]
        # T: [B, M, 3]
        B, N, M = P.size(0), P.size(1), S.size(1)
        F = self.backbone(P) # [B, N, C]
        Fs = index_points(F, knn_search(P, S, 1).squeeze(-1)) # [B, M, C]
        Fs = self.fuse(torch.cat((Fs, S), dim=-1)) # [B, M, C]
        Ft = index_points(F, knn_search(P, T, 1).squeeze(-1)) # [B, M, C]
        Ft = self.fuse(torch.cat((Ft, T), dim=-1)) # [B, M, C]
        ratio_out = self.head((Fs - Ft).abs()).squeeze(-1) # [B, M]
        return ratio_out



