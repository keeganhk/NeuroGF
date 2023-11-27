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
B_gdist = 64
Ns_gdist, Nt_gdist = 256, 128
N_gdist = Ns_gdist * Nt_gdist
net = GenNeuroGF().cuda()
max_lr = 1e-2
min_lr = 1e-4
num_itr = 20000
val_itv = 200
optimizer = optim.AdamW(net.parameters(), lr=max_lr, weight_decay=1e-8)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_itr, eta_min=min_lr)
criterion = nn.L1Loss()
all_test_data_container = load_all_test_data_gdist(data_root, unseen_category=True)
B_gdist_all = len(all_test_data_container['model_name'])
P_test_all = torch.tensor(all_test_data_container['pc']).float().cuda()
S_test_all = torch.tensor(all_test_data_container['gdist_qp_s']).float().cuda()
T_test_all = torch.tensor(all_test_data_container['gdist_qp_t']).float().cuda()
gdist_gt_test_all = torch.tensor(all_test_data_container['gdist_gt']).float().cuda()
edist_test_all = ((S_test_all - T_test_all) ** 2).sum(dim=-1).sqrt()
ratio_gt_test_all = gdist_gt_test_all / (edist_test_all + 1e-8)
train_stats = [0, 0]
for itr_index in tqdm(range(1, num_itr+1)):
    net.train()
    batch_container = load_batch_train_data_gdist(data_root, B_gdist, N_gdist, Ns_gdist, Nt_gdist)
    P = torch.tensor(batch_container['pc']).float().cuda()
    S = torch.tensor(batch_container['gdist_qp_s']).float().cuda()
    T = torch.tensor(batch_container['gdist_qp_t']).float().cuda()
    gdist_gt = torch.tensor(batch_container['gdist_gt']).float().cuda()
    edist = ((S - T) ** 2).sum(dim=-1).sqrt()
    ratio_gt = gdist_gt / (edist + 1e-8)
    optimizer.zero_grad()
    ratio_out = net(P, S, T)
    loss = criterion(ratio_out, ratio_gt)
    loss.backward()
    optimizer.step()
    scheduler.step()
    train_stats[0] += (loss.item() * B_gdist)
    train_stats[1] += B_gdist
    if np.mod(itr_index, val_itv) == 0:
        net.eval()
        with torch.no_grad():
            ratio_out_test_all = []
            for test_idx in range(B_gdist_all):
                P_test_this = P_test_all[test_idx].unsqueeze(0)
                S_test_this = S_test_all[test_idx].unsqueeze(0)
                T_test_this = T_test_all[test_idx].unsqueeze(0)
                ratio_out_test_all.append(net(P_test_this, S_test_this, T_test_this))
            ratio_out_test_all = torch.cat(ratio_out_test_all, dim=0)
        mre_test_all = ((ratio_out_test_all - ratio_gt_test_all).abs() / ratio_gt_test_all * 100).mean(dim=-1)
        avg_test_mre = np.around(float(np.asarray(mre_test_all.mean().cpu())), 2)        
        print('itr: {}, loss: {}, mre: {}%'.format(itr_index, np.around(train_stats[0]/train_stats[1], 6), avg_test_mre))
        torch.save(net.state_dict(), ckpt_name + '.pth')
        train_stats = [0, 0]



