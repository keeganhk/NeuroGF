import os, sys
sys.path.append(os.path.abspath('../../..'))
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from cdbs.pkgs import *
from cdbs.general import *
from cdbs.custom import *
sys.path.append('..')
from neurogf_ovft_utils import *
from neurogf_ovft_models import *
data_root = os.path.join('../../..', 'data', 'Models')
misc_root = os.path.join('../../..', 'data', 'Misc')
ckpt_root = os.path.join('../../..', 'ckpt', 'neurogf_ovft')


mesh_name = 'armadillo'


ckpt_name = os.path.join(ckpt_root, 'SDist_Querier_Offline_Pretrainer__' + mesh_name)
mm = vedo.load(os.path.join(data_root, mesh_name, mesh_name + '.obj'))
mv = mm.points().astype(np.float32)
mf = np.asarray(mm.faces()).astype(np.int64)
num_mv = mv.shape[0]
num_mf = mf.shape[0]
sp = np.loadtxt(os.path.join(data_root, mesh_name, mesh_name + '_surf_pts.xyz')).astype(np.float32)
num_sp = sp.shape[0]
net = SDist_Querier_Offline_Pretrainer()
max_lr = 1e-2
min_lr = 1e-4
num_epc = 100
num_itr = 200
optimizer = optim.AdamW(net.parameters(), lr=max_lr, weight_decay=1e-8)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epc, eta_min=min_lr)
criterion = nn.MSELoss()
N = 90000
best_loss = 1e8
for epc_index in range(1, num_epc+1):
    net.cuda().train()
    ts = time.time()
    np.random.seed()
    sdf_q, sdist_gt = fetch_train_data_sdf(N, mv, mf, sp)
    sdf_q = torch.tensor(sdf_q).float().cuda() # [N, 3]
    sdist_gt = torch.tensor(sdist_gt).float().cuda() # [N]
    for itr_index in range(1, num_itr+1):
        optimizer.zero_grad()
        sdist_out = net(sdf_q) # [N]
        loss = criterion(sdist_out, sdist_gt)
        loss.backward()
        optimizer.step()
    scheduler.step()
    te = time.time()
    dt = np.around(te - ts, 1)
    if loss.item() < best_loss:
        best_loss = loss.item()
        torch.save(net.cpu().state_dict(), ckpt_name + '.pth')
    print('epoch: {}, time: {} sec, loss: {}, best loss: {}\n'.format(epc_index, dt, np.around(loss.item(), 8), np.around(best_loss, 8)))



