import os, sys
sys.path.append(os.path.abspath('../..'))
from cdbs.pkgs import *
from cdbs.general import *
from cdbs.custom import *



class SDist_Querier_Offline_Pretrainer(nn.Module):
    def __init__(self):
        super(SDist_Querier_Offline_Pretrainer, self).__init__()
        fc_1 = FC(3, 32, True, 'relu')
        fc_2 = FC(32, 64, True, 'relu')
        fc_3 = FC(64, 128, True, 'relu')
        fc_4 = FC(128, 256, True, 'relu')
        fc_5 = FC(256, 512, True, 'relu')
        fc_6 = FC(512, 128, True, 'relu')
        fc_7 = FC(128, 1, False, 'none')
        self.fc = nn.Sequential(fc_1, fc_2, fc_3, fc_4, fc_5, fc_6, fc_7)
    def forward(self, sdf_q):
        # sdf_q: [N, 3]
        sdist_out = self.fc(sdf_q).squeeze(-1) # [N]
        return sdist_out


class NeuroGF_Offline_Trainer(nn.Module):
    def __init__(self, D, num_pp_g, num_pp_l):
        super(NeuroGF_Offline_Trainer, self).__init__()
        self.num_pp_g = num_pp_g
        self.num_pp_l = num_pp_l
        self.lifting = nn.Sequential(FC(3, D//4, True, 'relu'), FC(D//4, D//2, True, 'relu'), FC(D//2, D, True, 'relu'))
        self.sdf_head = nn.Sequential(FC(D, D//4, True, 'relu'), FC(D//4, 64, True, 'relu'), FC(64, 1, False, 'none'))
        self.gdf_head = nn.Sequential(FC(D, D//4, True, 'relu'), FC(D//4, 64, True, 'relu'), FC(64, 1, False, 'none'))
        self.cdw_fc = FC(2*D, D, True, 'relu')
        mlp_1 = MLP(D+3, 128, True, 'relu')
        mlp_2 = MLP(128, 64, True, 'relu')
        mlp_3 = MLP(64, 32, True, 'relu')
        mlp_4 = MLP(32, 3, False, 'none')
        self.spf_head = nn.Sequential(mlp_1, mlp_2, mlp_3, mlp_4)
    def forward(self, sdf_q, gdf_q_g, gdf_q_l1, gdf_q_l2, spf_q_g, spf_q_l, lines_g, lines_l):
        # sdf_q -> [N, 3]
        # gdf_q_g, gdf_q_l1, gdf_q_l2 -> [Mg, 6], [Ml1, 6], [Ml2, 6]
        # spf_q_g, spf_q_l -> [Kg, 6], [Kl, 6]
        # lines_g, lines_l -> [Kg, num_pp_g, 3], [Kl, num_pp_l, 3]
        N, device = sdf_q.size(0), sdf_q.device
        Mg, Ml1, Ml2 = gdf_q_g.size(0), gdf_q_l1.size(0), gdf_q_l2.size(0)
        Kg, Kl = spf_q_g.size(0), spf_q_l.size(0)
        assert lines_g.size(0)==Kg and lines_l.size(0)==Kl
        gdf_qs_g = gdf_q_g[:, 0:3] # [Mg, 3]
        gdf_qe_g = gdf_q_g[:, 3:6] # [Mg, 3]
        gdf_g_cats = torch.cat((gdf_qs_g, gdf_qe_g), dim=0) # [2*Mg, 3]
        gdf_qs_l1 = gdf_q_l1[:, 0:3] # [Ml1, 3]
        gdf_qe_l1 = gdf_q_l1[:, 3:6] # [Ml1, 3]
        gdf_l1_cats = torch.cat((gdf_qs_l1, gdf_qe_l1), dim=0) # [2*Ml1, 3]
        gdf_qs_l2 = gdf_q_l2[:, 0:3] # [Ml2, 3]
        gdf_qe_l2 = gdf_q_l2[:, 3:6] # [Ml2, 3]
        gdf_l2_cats = torch.cat((gdf_qs_l2, gdf_qe_l2), dim=0) # [2*Ml2, 3]
        spf_qs_g = spf_q_g[:, 0:3] # [Kg, 3]
        spf_qe_g = spf_q_g[:, 3:6] # [Kg, 3]
        spf_g_cats = torch.cat((spf_qs_g, spf_qe_g), dim=0) # [2*Kg, 3]
        spf_qs_l = spf_q_l[:, 0:3] # [Kl, 3]
        spf_qe_l = spf_q_l[:, 3:6] # [Kl, 3]
        spf_l_cats = torch.cat((spf_qs_l, spf_qe_l), dim=0) # [2*Kl, 3]
        all_cats = torch.cat((sdf_q, gdf_g_cats, gdf_l1_cats, gdf_l2_cats, spf_g_cats, spf_l_cats), dim=0) # [total_batch_size, 3]
        # (1) lift all query points to high-dimensional features
        ftr__all = self.lifting(all_cats) # [total_batch_size, D]
        ftr__sdf_q = ftr__all[0:N] # [N, D]
        ftr__gdf_g_cats = ftr__all[(N):(N+2*Mg)] # [2*Mg, D]
        ftr__gdf_qs_g = ftr__gdf_g_cats[:Mg] # [Mg, D]
        ftr__gdf_qe_g = ftr__gdf_g_cats[Mg:] # [Mg, D]
        ftr__gdf_l1_cats = ftr__all[(N+2*Mg):(N+2*Mg+2*Ml1)] # [2*Ml1, D]
        ftr__gdf_qs_l1 = ftr__gdf_l1_cats[:Ml1] # [Ml1, D]
        ftr__gdf_qe_l1 = ftr__gdf_l1_cats[Ml1:] # [Ml1, D]
        ftr__gdf_l2_cats = ftr__all[(N+2*Mg+2*Ml1):(N+2*Mg+2*Ml1+2*Ml2)] # [2*Ml2, D]
        ftr__gdf_qs_l2 = ftr__gdf_l2_cats[:Ml2] # [Ml2, D]
        ftr__gdf_qe_l2 = ftr__gdf_l2_cats[Ml2:] # [Ml2, D]
        ftr__spf_g_cats = ftr__all[(N+2*Mg+2*Ml1+2*Ml2):(N+2*Mg+2*Ml1+2*Ml2+2*Kg)] # [2*Kg, D]
        ftr__spf_qs_g = ftr__spf_g_cats[:Kg] # [Kg, D]
        ftr__spf_qe_g = ftr__spf_g_cats[Kg:] # [Kg, D]
        ftr__spf_l_cats = ftr__all[(N+2*Mg+2*Ml1+2*Ml2+2*Kg):(N+2*Mg+2*Ml1+2*Ml2+2*Kg+2*Kl)] # [2*Kl, D]
        ftr__spf_qs_l = ftr__spf_l_cats[:Kl] # [Kl, D]
        ftr__spf_qe_l = ftr__spf_l_cats[Kl:] # [Kl, D]
        # (2) deduce signed distances
        sdist_out = self.sdf_head(ftr__sdf_q).squeeze(-1) # [N]
        # (3) deduce geodesic distances
        diff_g = (ftr__gdf_qs_g - ftr__gdf_qe_g).abs() # [Mg, D]
        diff_l1 = (ftr__gdf_qs_l1 - ftr__gdf_qe_l1).abs() # [Ml1, D]
        diff_l2 = (ftr__gdf_qs_l2 - ftr__gdf_qe_l2).abs() # [Ml2, D]
        diff_cats = torch.cat((diff_g, diff_l1, diff_l2), dim=0) # [Mg+Ml1+Ml2, D]
        gdist_out_cats = self.gdf_head(diff_cats).squeeze(-1) # [Mg+Ml1+Ml2]
        gdist_out_g = gdist_out_cats[0:Mg] # [Mg]
        gdist_out_l1 = gdist_out_cats[(Mg):(Mg+Ml1)] # [Ml1]
        gdist_out_l2 = gdist_out_cats[(Mg+Ml1):(Mg+Ml1+Ml2)] # [Ml2]
        # (4) deduce shortest paths
        a = torch.cat((ftr__spf_qs_g, ftr__spf_qe_g), dim=-1) # [Kg, 2*D]
        b = torch.cat((ftr__spf_qs_l, ftr__spf_qe_l), dim=-1) # [Kl, 2*D]
        ftr__spf_cats = torch.cat((a, b), dim=0) # [Kg+Kl, 2*D]
        cdw__spf_cats = self.cdw_fc(ftr__spf_cats) # [Kg+Kl, D]
        cdw_g = cdw__spf_cats[:Kg] # [Kg, D]
        cdw_l = cdw__spf_cats[Kg:] # [Kl, D]
        cdw_g_rep = cdw_g.unsqueeze(1).repeat(1, self.num_pp_g, 1) # [Kg, num_pp_g, D]
        cdw_l_rep = cdw_l.unsqueeze(1).repeat(1, self.num_pp_l, 1) # [Kl, num_pp_l, D]
        spath_out_g = self.spf_head(torch.cat((lines_g, cdw_g_rep), dim=-1)) # [Kg, num_pp_g, 3]
        spath_out_l = self.spf_head(torch.cat((lines_l, cdw_l_rep), dim=-1)) # [Kl, num_pp_lg, 3]
        return sdist_out, [gdist_out_g, gdist_out_l1, gdist_out_l2], [spath_out_g, spath_out_l]


class NeuroGF_Online_Querier_GDistOnly(nn.Module):
    def __init__(self, D):
        super(NeuroGF_Online_Querier_GDistOnly, self).__init__()
        self.lifting = nn.Sequential(FC(3, D//4, True, 'relu'), FC(D//4, D//2, True, 'relu'), FC(D//2, D, True, 'relu'))
        self.sdf_head = nn.Sequential(FC(D, D//4, True, 'relu'), FC(D//4, 64, True, 'relu'), FC(64, 1, False, 'none'))
        self.gdf_head = nn.Sequential(FC(D, D//4, True, 'relu'), FC(D//4, 64, True, 'relu'), FC(64, 1, False, 'none'))
        self.cdw_fc = FC(2*D, D, True, 'relu')
        mlp_1 = MLP(D+3, 128, True, 'relu')
        mlp_2 = MLP(128, 64, True, 'relu')
        mlp_3 = MLP(64, 32, True, 'relu')
        mlp_4 = MLP(32, 3, False, 'none')
        self.spf_head = nn.Sequential(mlp_1, mlp_2, mlp_3, mlp_4)
    def forward(self, gdf_q):
        # gdf_q: [N, 6]
        N = gdf_q.size(0)
        gdf_qs = gdf_q[:, 0:3] # [N, 3]
        gdf_qe = gdf_q[:, 3:6] # [N, 3]
        gdf_cats = torch.cat((gdf_qs, gdf_qe), dim=0) # [2*N, 3]
        with torch.no_grad():
            ftr__all = self.lifting(gdf_cats) # [2*N, D]
        ftr__gdf_qs = ftr__all[:N] # [N, D]
        ftr__gdf_qe = ftr__all[N:] # [N, D]
        diff = (ftr__gdf_qs - ftr__gdf_qe).abs() # [N, D]
        gdist_out = self.gdf_head(diff).squeeze(-1) # [N]
        return gdist_out


class NeuroGF_Online_Querier_SPathOnly(nn.Module):
    def __init__(self, D, num_pp):
        super(NeuroGF_Online_Querier_SPathOnly, self).__init__()
        self.num_pp = num_pp
        self.lifting = nn.Sequential(FC(3, D//4, True, 'relu'), FC(D//4, D//2, True, 'relu'), FC(D//2, D, True, 'relu'))
        self.sdf_head = nn.Sequential(FC(D, D//4, True, 'relu'), FC(D//4, 64, True, 'relu'), FC(64, 1, False, 'none'))
        self.gdf_head = nn.Sequential(FC(D, D//4, True, 'relu'), FC(D//4, 64, True, 'relu'), FC(64, 1, False, 'none'))
        self.cdw_fc = FC(2*D, D, True, 'relu')
        mlp_1 = MLP(D+3, 128, True, 'relu')
        mlp_2 = MLP(128, 64, True, 'relu')
        mlp_3 = MLP(64, 32, True, 'relu')
        mlp_4 = MLP(32, 3, False, 'none')
        self.spf_head = nn.Sequential(mlp_1, mlp_2, mlp_3, mlp_4)
    def forward(self, spf_q, lines):
        # spf_q: [N, 6]
        # lines: [N, num_pp, 3]
        N, device = spf_q.size(0), spf_q.device
        assert lines.size(0)==N and lines.size(1)==self.num_pp
        spf_qs = spf_q[:, 0:3] # [N, 3]
        spf_qe = spf_q[:, 3:6] # [N, 3]
        spf_cats = torch.cat((spf_qs, spf_qe), dim=0) # [2*N, 3]
        with torch.no_grad():
            ftr__all = self.lifting(spf_cats) # [2*N, D]
        ftr__spf_qs = ftr__all[:N] # [N, D]
        ftr__spf_qe = ftr__all[N:] # [N, D]        
        cdw = self.cdw_fc(torch.cat((ftr__spf_qs, ftr__spf_qe), dim=-1)) # [N, D]
        cdw_rep = cdw.unsqueeze(1).repeat(1, self.num_pp, 1) # [N, num_pp, D]
        spath_out = self.spf_head(torch.cat((lines, cdw_rep), dim=-1)) # [N, num_pp, 3]
        return spath_out


class NeuroGF_Online_Querier_SDistOnly(nn.Module):
    def __init__(self, D):
        super(NeuroGF_Online_Querier_SDistOnly, self).__init__()
        self.lifting = nn.Sequential(FC(3, D//4, True, 'relu'), FC(D//4, D//2, True, 'relu'), FC(D//2, D, True, 'relu'))
        self.sdf_head = nn.Sequential(FC(D, D//4, True, 'relu'), FC(D//4, 64, True, 'relu'), FC(64, 1, False, 'none'))
        self.gdf_head = nn.Sequential(FC(D, D//4, True, 'relu'), FC(D//4, 64, True, 'relu'), FC(64, 1, False, 'none'))
        self.cdw_fc = FC(2*D, D, True, 'relu')
        mlp_1 = MLP(D+3, 128, True, 'relu')
        mlp_2 = MLP(128, 64, True, 'relu')
        mlp_3 = MLP(64, 32, True, 'relu')
        mlp_4 = MLP(32, 3, False, 'none')
        self.spf_head = nn.Sequential(mlp_1, mlp_2, mlp_3, mlp_4)
    def forward(self, sdf_q):
        # sdf_q: [N, 3]
        N, device = sdf_q.size(0), sdf_q.device
        with torch.no_grad():
            ftr__sdf_q = self.lifting(sdf_q) # [N, D]
        sdist_out = self.sdf_head(ftr__sdf_q).squeeze(-1) # [N]
        return sdist_out


class QueryPointLifting(nn.Module):
    def __init__(self, D):
        super(QueryPointLifting, self).__init__()
        self.lifting = nn.Sequential(FC(3, D//4, True, 'relu'), FC(D//4, D//2, True, 'relu'), FC(D//2, D, True, 'relu'))
        self.sdf_head = nn.Sequential(FC(D, D//4, True, 'relu'), FC(D//4, 64, True, 'relu'), FC(64, 1, False, 'none'))
        self.gdf_head = nn.Sequential(FC(D, D//4, True, 'relu'), FC(D//4, 64, True, 'relu'), FC(64, 1, False, 'none'))
        self.cdw_fc = FC(2*D, D, True, 'relu')
        mlp_1 = MLP(D+3, 128, True, 'relu')
        mlp_2 = MLP(128, 64, True, 'relu')
        mlp_3 = MLP(64, 32, True, 'relu')
        mlp_4 = MLP(32, 3, False, 'none')
        self.spf_head = nn.Sequential(mlp_1, mlp_2, mlp_3, mlp_4)
    def forward(self, qry_pts):
        # qry_pts: [num_qry, 3]
        qry_ftr = self.lifting(qry_pts) # [num_qry, D]
        return qry_ftr


class NeuroGF_Offline_PostRefiner_SDistOnly(nn.Module):
    def __init__(self, D):
        super(NeuroGF_Offline_PostRefiner_SDistOnly, self).__init__()
        self.lifting = nn.Sequential(FC(3, D//4, True, 'relu'), FC(D//4, D//2, True, 'relu'), FC(D//2, D, True, 'relu'))
        self.sdf_head = nn.Sequential(FC(D, D//4, True, 'relu'), FC(D//4, 64, True, 'relu'), FC(64, 1, False, 'none'))
        self.gdf_head = nn.Sequential(FC(D, D//4, True, 'relu'), FC(D//4, 64, True, 'relu'), FC(64, 1, False, 'none'))
        self.cdw_fc = FC(2*D, D, True, 'relu')
        mlp_1 = MLP(D+3, 128, True, 'relu')
        mlp_2 = MLP(128, 64, True, 'relu')
        mlp_3 = MLP(64, 32, True, 'relu')
        mlp_4 = MLP(32, 3, False, 'none')
        self.spf_head = nn.Sequential(mlp_1, mlp_2, mlp_3, mlp_4)
    def forward(self, ftr__sdf_q):
        # ftr__sdf_q: [N, D]
        sdist_out = self.sdf_head(ftr__sdf_q).squeeze(-1) # [N]
        return sdist_out


class NeuroGF_Offline_PostRefiner_SPathOnly(nn.Module):
    def __init__(self, D, num_pp):
        super(NeuroGF_Offline_PostRefiner_SPathOnly, self).__init__()
        self.num_pp = num_pp
        self.lifting = nn.Sequential(FC(3, D//4, True, 'relu'), FC(D//4, D//2, True, 'relu'), FC(D//2, D, True, 'relu'))
        self.sdf_head = nn.Sequential(FC(D, D//4, True, 'relu'), FC(D//4, 64, True, 'relu'), FC(64, 1, False, 'none'))
        self.gdf_head = nn.Sequential(FC(D, D//4, True, 'relu'), FC(D//4, 64, True, 'relu'), FC(64, 1, False, 'none'))
        self.cdw_fc = FC(2*D, D, True, 'relu')
        mlp_1 = MLP(D+3, 128, True, 'relu')
        mlp_2 = MLP(128, 64, True, 'relu')
        mlp_3 = MLP(64, 32, True, 'relu')
        mlp_4 = MLP(32, 3, False, 'none')
        self.spf_head = nn.Sequential(mlp_1, mlp_2, mlp_3, mlp_4)
    def forward(self, ftr__all, lines):
        # ftr__all: [2*N, D]
        # lines: [N, num_pp, 3]
        N, device = ftr__all.size(0)//2, ftr__all.device
        assert lines.size(0)==N and lines.size(1)==self.num_pp
        ftr__spf_qs = ftr__all[:N] # [N, D]
        ftr__spf_qe = ftr__all[N:] # [N, D]
        cdw = self.cdw_fc(torch.cat((ftr__spf_qs, ftr__spf_qe), dim=-1)) # [N, D]
        cdw_rep = cdw.unsqueeze(1).repeat(1, self.num_pp, 1) # [N, num_pp, D]
        spath_out = self.spf_head(torch.cat((lines, cdw_rep), dim=-1)) # [N, num_pp, 3]
        return spath_out



