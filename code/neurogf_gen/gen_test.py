import os, sys
sys.path.append(os.path.abspath('../..'))
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from cdbs.pkgs import *
from cdbs.general import *
from cdbs.custom import *
sys.path.append('..')
from neurogf_gen_utils import *
from neurogf_gen_models import *
data_root = os.path.join('../..', 'data', 'ShapeNet13')
misc_root = os.path.join('../..', 'data', 'Misc')
ckpt_root = os.path.join('../..', 'ckpt', 'neurogf_gen')


ckpt_name = os.path.join(ckpt_root, 'GenNeuroGF')
net = GenNeuroGF().cuda()
net.load_state_dict(torch.load(ckpt_name + '.pth'))
net.eval()


all_test_data_container = load_all_test_data_gdist(data_root, unseen_category=True)
B_gdist_all = len(all_test_data_container['model_name'])
print('number of testing models: {}'.format(B_gdist_all))
P_test_all = torch.tensor(all_test_data_container['pc']).float().cuda()
S_test_all = torch.tensor(all_test_data_container['gdist_qp_s']).float().cuda()
T_test_all = torch.tensor(all_test_data_container['gdist_qp_t']).float().cuda()
gdist_gt_test_all = torch.tensor(all_test_data_container['gdist_gt']).float().cuda()
edist_test_all = ((S_test_all - T_test_all) ** 2).sum(dim=-1).sqrt()
ratio_gt_test_all = gdist_gt_test_all / (edist_test_all + 1e-8)
with torch.no_grad():
    ratio_out_test_all = []
    for test_idx in tqdm(range(B_gdist_all)):
        P_test_this = P_test_all[test_idx].unsqueeze(0)
        S_test_this = S_test_all[test_idx].unsqueeze(0)
        T_test_this = T_test_all[test_idx].unsqueeze(0)
        ratio_out_test_all.append(net(P_test_this, S_test_this, T_test_this))
    ratio_out_test_all = torch.cat(ratio_out_test_all, dim=0)
mre_test_all = ((ratio_out_test_all - ratio_gt_test_all).abs() / ratio_gt_test_all * 100).mean(dim=-1)
avg_test_mre = np.around(float(np.asarray(mre_test_all.mean().cpu())), 2)        
print('average mean-relative-error (unseen 5 categories): {}%'.format(avg_test_mre))


all_test_data_container = load_all_test_data_gdist(data_root, unseen_category=False)
B_gdist_all = len(all_test_data_container['model_name'])
print('number of testing models: {}'.format(B_gdist_all))
P_test_all = torch.tensor(all_test_data_container['pc']).float().cuda()
S_test_all = torch.tensor(all_test_data_container['gdist_qp_s']).float().cuda()
T_test_all = torch.tensor(all_test_data_container['gdist_qp_t']).float().cuda()
gdist_gt_test_all = torch.tensor(all_test_data_container['gdist_gt']).float().cuda()
edist_test_all = ((S_test_all - T_test_all) ** 2).sum(dim=-1).sqrt()
ratio_gt_test_all = gdist_gt_test_all / (edist_test_all + 1e-8)
with torch.no_grad():
    ratio_out_test_all = []
    for test_idx in tqdm(range(B_gdist_all)):
        P_test_this = P_test_all[test_idx].unsqueeze(0)
        S_test_this = S_test_all[test_idx].unsqueeze(0)
        T_test_this = T_test_all[test_idx].unsqueeze(0)
        ratio_out_test_all.append(net(P_test_this, S_test_this, T_test_this))
    ratio_out_test_all = torch.cat(ratio_out_test_all, dim=0)
mre_test_all = ((ratio_out_test_all - ratio_gt_test_all).abs() / ratio_gt_test_all * 100).mean(dim=-1)
avg_test_mre = np.around(float(np.asarray(mre_test_all.mean().cpu())), 2)        
print('average mean-relative-error (seen 8 categories): {}%'.format(avg_test_mre))



