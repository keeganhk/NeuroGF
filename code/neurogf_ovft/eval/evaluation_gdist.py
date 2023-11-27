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
params = os.path.join(ckpt_root, 'NeuroGF_Pretrainer__' + mesh_name + '_final' + '.pth')
net = NeuroGF_Online_Querier_GDistOnly(D)
net.load_state_dict(torch.load(params))
net.cuda().eval()
gt_collection = []
out_collection = []
for split_index in tqdm(range(num_splits)):
    gdf_q = split_all_test_data[split_index][:, 0:6]
    gdist_gt = split_all_test_data[split_index][:, -1]
    gdf_q = torch.tensor(gdf_q).float().cuda()
    gdist_gt = torch.tensor(gdist_gt).float().cuda()
    with torch.no_grad():
        gdist_out = net(gdf_q)
    gt_collection.append(np.asarray(gdist_gt.cpu()))
    out_collection.append(np.asarray(gdist_out.cpu()))
gt_collection = np.concatenate(gt_collection)
out_collection = np.concatenate(out_collection)
best_mre = (np.abs(out_collection-gt_collection) / (gt_collection)).mean()
print('mean relative error: {}%'.format(np.around(best_mre*100, 2)))



