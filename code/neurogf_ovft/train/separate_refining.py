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


D = 256
mesh_name = 'armadillo'


mm = vedo.load(os.path.join(data_root, mesh_name, mesh_name + '.obj'))
mv = mm.points().astype(np.float32)
mf = np.asarray(mm.faces()).astype(np.int64)
num_mv = mv.shape[0]
num_mf = mf.shape[0]
sp = np.loadtxt(os.path.join(data_root, mesh_name, mesh_name + '_surf_pts.xyz')).astype(np.float32)
num_sp = sp.shape[0]
qp_lift = QueryPointLifting(D)
qp_lift.load_state_dict(torch.load(os.path.join(ckpt_root, 'NeuroGF_Pretrainer__' + mesh_name + '_best.pth')))
qp_lift.cuda().eval()
load_root_gdf = os.path.join(data_root, mesh_name, 'gdf', 'train')
load_root_spf = os.path.join(data_root, mesh_name, 'spf', 'train')
num_pp_g = 128
num_pp_l = 32


sdist_refiner = NeuroGF_Offline_PostRefiner_SDistOnly(D)
sdist_refiner.load_state_dict(torch.load(os.path.join(ckpt_root, 'NeuroGF_Pretrainer__' + mesh_name + '_best.pth')))
sdist_refiner.cuda().train()
max_lr = 1e-3
min_lr = 1e-4
num_epc = 70
num_itr = 200
eval_itv = 10
optimizer = optim.AdamW(sdist_refiner.parameters(), lr=max_lr, weight_decay=1e-8)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epc, eta_min=min_lr)
criterion = nn.L1Loss()
for epc_index in tqdm(range(1, num_epc+1)):
    N_sdf = 90000
    sdf_q, sdist_gt = fetch_train_data_sdf(N_sdf, mv, mf, sp)
    sdf_q = torch.tensor(sdf_q).float().cuda()
    sdist_gt = torch.tensor(sdist_gt).float().cuda()
    with torch.no_grad():
        ftr__sdf_q = qp_lift(sdf_q)
    for itr_index in range(1, num_itr+1):
        optimizer.zero_grad()
        sdist_out = sdist_refiner(ftr__sdf_q)
        loss = criterion(sdist_out, sdist_gt)
        loss.backward()
        optimizer.step()
    scheduler.step()
    print('epoch: {}, loss: {}'.format(align_number(epc_index, 4), np.around(loss.item(), 8)))
refined_trainer = NeuroGF_Offline_Trainer(D, num_pp_g, num_pp_l)
refined_trainer.load_state_dict(torch.load(os.path.join(ckpt_root, 'NeuroGF_Pretrainer__' + mesh_name + '_best.pth')))
refined_trainer.sdf_head = sdist_refiner.cpu().sdf_head
torch.save(refined_trainer.cpu().state_dict(), os.path.join(ckpt_root, 'NeuroGF_Pretrainer__' + mesh_name + '_final.pth'))


num_pp = 128
spath_refiner = NeuroGF_Offline_PostRefiner_SPathOnly(D, num_pp)
spath_refiner.load_state_dict(torch.load(os.path.join(ckpt_root, 'NeuroGF_Pretrainer__' + mesh_name + '_final.pth')))
spath_refiner.cuda().train()
max_lr = 1e-3
min_lr = 1e-3
num_epc = 30
num_itr = 200
eval_itv = 10
optimizer = optim.AdamW(spath_refiner.parameters(), lr=max_lr, weight_decay=1e-8)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epc, eta_min=min_lr)
criterion = nn.L1Loss()
for epc_index in tqdm(range(1, num_epc+1)):
    N_spf_s = 100
    N_spf_t_g = 100
    spf_fetched = fetch_train_data_spf(N_spf_s, N_spf_t_g, 1, load_root_spf, num_pp_g, num_pp_l)
    spath_gt_g_uni = spf_fetched[0]
    spath_gt_g_flp = np.flip(spath_gt_g_uni, axis=1)
    spath_gt_g = np.concatenate((spath_gt_g_uni, spath_gt_g_flp), axis=0)
    spf_q_g = np.concatenate((spath_gt_g[:, 0, :], spath_gt_g[:, -1, :]), axis=-1)
    lines_g = generate_lines_from_end_points(spf_q_g[:, 0:3], spf_q_g[:, 3:6], num_pp_g)
    spf_q = torch.tensor(spf_q_g).float().cuda()
    spath_gt = torch.tensor(spath_gt_g).float().cuda()
    lines = torch.tensor(lines_g).float().cuda()
    with torch.no_grad():
        spf_qs = spf_q[:, 0:3]
        spf_qe = spf_q[:, 3:6]
        ftr__spf_qs = qp_lift(spf_qs)
        ftr__spf_qe = qp_lift(spf_qe)
        ftr__spf_q = torch.cat((ftr__spf_qs, ftr__spf_qe), dim=0)
    for itr_index in range(1, num_itr+1):
        optimizer.zero_grad()
        spath_out = spath_refiner(ftr__spf_q, lines)
        loss = criterion(spath_out, spath_gt)
        loss.backward()
        optimizer.step()
    scheduler.step()
    print('epoch: {}, loss: {}'.format(align_number(epc_index, 4), np.around(loss.item(), 8)))
refined_trainer = NeuroGF_Offline_Trainer(D, num_pp_g, num_pp_l)
refined_trainer.load_state_dict(torch.load(os.path.join(ckpt_root, 'NeuroGF_Pretrainer__' + mesh_name + '_final.pth')))
refined_trainer.cdw_fc = spath_refiner.cpu().cdw_fc
refined_trainer.spf_head = spath_refiner.cpu().spf_head
torch.save(refined_trainer.cpu().state_dict(), os.path.join(ckpt_root, 'NeuroGF_Pretrainer__' + mesh_name + '_final.pth'))



