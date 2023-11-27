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
spf_test_data_file = os.path.join(data_root, mesh_name, 'spf', 'test', 'Ns_Nt_128_3.npy')
sources_to_targets_curves = np.load(spf_test_data_file).astype(np.float32)
all_test_data = sources_to_targets_curves.reshape(-1, 128, 3)
num_all_test = all_test_data.shape[0]
num_splits = (num_all_test//1000)
split_all_test_data = np.array_split(all_test_data, num_splits)
params = os.path.join(ckpt_root, 'NeuroGF_Pretrainer__' + mesh_name + '_final' + '.pth')
num_pp = 128
net = NeuroGF_Online_Querier_SPathOnly(D, num_pp)
net.load_state_dict(torch.load(params))
net.cuda().eval()
all_outputs = []
for split_index in range(num_splits):
    spath_gt = split_all_test_data[split_index].astype(np.float32)
    B = spath_gt.shape[0]
    spf_q = np.concatenate((spath_gt[:, 0, :], spath_gt[:, -1, :]), axis=-1)
    lines = generate_lines_from_end_points(spf_q[:, 0:3], spf_q[:, 3:6], num_pp)
    spath_gt = torch.tensor(spath_gt).float().cuda()
    spf_q = torch.tensor(spf_q).float().cuda()
    lines = torch.tensor(lines).float().cuda()
    with torch.no_grad():
        spath_out = net(spf_q, lines)
    all_outputs.append(np.asarray(spath_out.cpu()))
all_outputs = np.concatenate(all_outputs, axis=0)
cham = 0
counter = 0
for path_index in tqdm(range(num_all_test)):
    this_gt = all_test_data[path_index, ...]
    # this_gt = curve_interp(curve_interp(curve_interp(this_gt)))
    this_out = all_outputs[path_index, ...]
    cham += compute_chamfer_l1(this_out, this_gt)
    counter += 1
cham /= counter 
print('chamfer-l1: {}'.format(np.around(cham, 5)))



