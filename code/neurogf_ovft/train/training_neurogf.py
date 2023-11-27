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


ckpt_name = os.path.join(ckpt_root, 'NeuroGF_Pretrainer__' + mesh_name)
mm = vedo.load(os.path.join(data_root, mesh_name, mesh_name + '.obj'))
mv = mm.points().astype(np.float32)
mf = np.asarray(mm.faces()).astype(np.int64)
num_mv = mv.shape[0]
num_mf = mf.shape[0]
sp = np.loadtxt(os.path.join(data_root, mesh_name, mesh_name + '_surf_pts.xyz')).astype(np.float32)
num_sp = sp.shape[0]
load_root_gdf = os.path.join(data_root, mesh_name, 'gdf', 'train')
load_root_spf = os.path.join(data_root, mesh_name, 'spf', 'train')
file_path_list = glob.glob(os.path.join(data_root, mesh_name, 'gdf', 'test', '*.npy'))
num_files = len(file_path_list)
collection = np.zeros((num_files, num_mv-1, 7), dtype=np.float32)
for load_index, file_path in enumerate(file_path_list):
    file_data = np.load(file_path)
    assert file_data.shape[0]==(num_mv-1) and file_data.shape[1]==7
    collection[load_index, ...] = file_data
num_all_test = num_files * (num_mv-1)
all_test_data = collection.reshape(-1, 7)
num_splits = (num_all_test//100000)
split_all_test_data = np.array_split(all_test_data, num_splits)
num_pp_g = 128
num_pp_l = 32
net = NeuroGF_Offline_Trainer(D, num_pp_g, num_pp_l).cuda()
num_params = sum(p.numel() for p in net.parameters())
sdist_querier = SDist_Querier_Offline_Pretrainer()
sdist_querier.load_state_dict(torch.load(os.path.join(ckpt_root, 'SDist_Querier_Offline_Pretrainer__' + mesh_name + '.pth')))
sdist_querier.cuda().eval()
max_lr = 1e-2
min_lr = 1e-4
num_epc = 400
num_itr = 200
eval_itv = 10
optimizer = optim.AdamW(net.parameters(), lr=max_lr, weight_decay=1e-8)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epc, eta_min=min_lr)
criterion = nn.L1Loss()
best_mre = 1e8
for epc_index in tqdm(range(1, num_epc+1)):
    net.cuda().train()
    np.random.seed()
    ts = time.time()
    N_sdf = 30000
    sdf_q, sdist_gt = fetch_train_data_sdf(N_sdf, mv, mf, sp)
    sdf_q = torch.tensor(sdf_q).float().cuda()
    sdist_gt = torch.tensor(sdist_gt).float().cuda()
    N_gdf_s = 256
    N_gdf_t_g = 256
    N_gdf_t_l1 = 32
    N_gdf_t_l2 = 64
    gdf_fetched = fetch_train_data_gdf(N_gdf_s, N_gdf_t_g, N_gdf_t_l1, N_gdf_t_l2, load_root_gdf)
    gdf_q_g, gdf_q_l1, gdf_q_l2, gdist_gt_g, gdist_gt_l1, gdist_gt_l2 = gdf_fetched
    gdf_q_g = torch.tensor(gdf_q_g).float().cuda()
    gdf_q_l1 = torch.tensor(gdf_q_l1).float().cuda()
    gdf_q_l2 = torch.tensor(gdf_q_l2).float().cuda()
    gdist_gt_g = torch.tensor(gdist_gt_g).float().cuda()
    gdist_gt_l1 = torch.tensor(gdist_gt_l1).float().cuda()
    gdist_gt_l2 = torch.tensor(gdist_gt_l2).float().cuda()
    N_spf_s = 128
    N_spf_t_g = 32
    N_spf_t_l = 32
    spf_fetched = fetch_train_data_spf(N_spf_s, N_spf_t_g, N_spf_t_l, load_root_spf, num_pp_g, num_pp_l)
    spath_gt_g_uni = spf_fetched[0]
    spath_gt_g_flp = np.flip(spath_gt_g_uni, axis=1)
    spath_gt_l_uni = spf_fetched[1]
    spath_gt_l_flp = np.flip(spath_gt_l_uni, axis=1)
    spath_gt_g = np.concatenate((spath_gt_g_uni, spath_gt_g_flp), axis=0)
    spath_gt_l = np.concatenate((spath_gt_l_uni, spath_gt_l_flp), axis=0)
    spf_q_g = np.concatenate((spath_gt_g[:, 0, :], spath_gt_g[:, -1, :]), axis=-1)
    spf_q_l = np.concatenate((spath_gt_l[:, 0, :], spath_gt_l[:, -1, :]), axis=-1)
    lines_g = generate_lines_from_end_points(spf_q_g[:, 0:3], spf_q_g[:, 3:6], num_pp_g)
    lines_l = generate_lines_from_end_points(spf_q_l[:, 0:3], spf_q_l[:, 3:6], num_pp_l)
    spf_q_g = torch.tensor(spf_q_g).float().cuda()
    spf_q_l = torch.tensor(spf_q_l).float().cuda()
    spath_gt_g = torch.tensor(spath_gt_g).float().cuda()
    spath_gt_l = torch.tensor(spath_gt_l).float().cuda()
    lines_g = torch.tensor(lines_g).float().cuda()
    lines_l = torch.tensor(lines_l).float().cuda()
    for itr_index in range(1, num_itr+1):
        optimizer.zero_grad()
        sdist_out, gdist_out_list, spath_out_list = net(sdf_q, gdf_q_g, gdf_q_l1, gdf_q_l2, spf_q_g, spf_q_l, lines_g, lines_l)
        gdist_out_g, gdist_out_l1, gdist_out_l2 = gdist_out_list
        spath_out_g, spath_out_l = spath_out_list
        loss_sdist = criterion(sdist_out, sdist_gt)
        loss_gdist_g = criterion(gdist_out_g, gdist_gt_g)
        loss_gdist_l1 = criterion(gdist_out_l1, gdist_gt_l1)
        loss_gdist_l2 = criterion(gdist_out_l2, gdist_gt_l2)
        loss_gdist = loss_gdist_g + loss_gdist_l1 + loss_gdist_l2
        loss_spath_g = criterion(spath_out_g, spath_gt_g)
        loss_spath_l = criterion(spath_out_l, spath_gt_l)
        loss_spath = loss_spath_g + loss_spath_l 
        cstt_ccl = criterion(approx_path_len_batched(spath_out_g), approx_path_len_batched(spath_gt_g))
        sampled_idx_of_spath_out_g = np.random.choice(spath_out_g.shape[0], 1024, replace=False)
        cstt_dcp = sdist_querier(spath_out_g[sampled_idx_of_spath_out_g, :, :].view(-1, 3)).abs().mean()
        loss = 0.1*loss_sdist + 1.0*loss_gdist + 0.1*loss_spath + 0.1*cstt_ccl + 0.1*cstt_dcp
        loss.backward()
        optimizer.step()
    scheduler.step()
    torch.save(net.cpu().state_dict(), ckpt_name + '_current.pth')
    te = time.time()
    dt = int(te - ts)
    loss_sdist = np.around(loss_sdist.item(), 8)
    loss_gdist_g = np.around(loss_gdist_g.item(), 8)
    loss_gdist_l1 = np.around(loss_gdist_l1.item(), 8)
    loss_gdist_l2 = np.around(loss_gdist_l2.item(), 8)
    loss_spath_g = np.around(loss_spath_g.item(), 8)
    loss_spath_l = np.around(loss_spath_l.item(), 8)
    cstt_ccl = np.around(cstt_ccl.item(), 8)
    cstt_dcp = np.around(cstt_dcp.item(), 8)
    print('-----------------------------------------------------------')
    print('epoch: {}, time: {}s'.format(align_number(epc_index, 4), dt))
    print('loss_gdist_g: {}, loss_gdist_l1: {}, loss_gdist_l2: {}'.format(loss_gdist_g, loss_gdist_l1, loss_gdist_l2))
    print('loss_spath_g: {}, loss_spath_l: {}'.format(loss_spath_g, loss_spath_l))
    print('loss_sdist: {}'.format(loss_sdist))
    print('cstt_ccl: {}, cstt_dcp: {}'.format(cstt_ccl, cstt_dcp))
    if np.mod(epc_index, 50)==0 or (epc_index>=int(0.5*num_epc) and np.mod(epc_index, eval_itv)==0):
        eval_net = NeuroGF_Online_Querier_GDistOnly(D)
        eval_net.load_state_dict(torch.load(ckpt_name + '_current.pth'))
        eval_net.cuda().eval()
        gt_collection = []
        out_collection = []
        for split_index in range(num_splits):
            gdf_q = split_all_test_data[split_index][:, 0:6] # (B, 6)
            gdist_gt = split_all_test_data[split_index][:, -1] # (B,)
            gdf_q = torch.tensor(gdf_q).float().cuda() # [B, 6]
            gdist_gt = torch.tensor(gdist_gt).float().cuda() # [B]
            with torch.no_grad():
                gdist_out = eval_net(gdf_q)
            gt_collection.append(np.asarray(gdist_gt.cpu()))
            out_collection.append(np.asarray(gdist_out.cpu()))
        gt_collection = np.concatenate(gt_collection)
        out_collection = np.concatenate(out_collection)
        mre = (np.abs(out_collection - gt_collection) / (gt_collection)).mean()
        print('current mre: {}%'.format(np.around(mre*100, 4)))
        if mre < best_mre:
            best_mre = mre
            torch.save(net.cpu().state_dict(), ckpt_name + '_best.pth')
            print('updated the best checkpoint -> best mre: {}%'.format(np.around(best_mre*100, 4)))



