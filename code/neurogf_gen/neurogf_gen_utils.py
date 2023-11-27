import os, sys
sys.path.append(os.path.abspath('../..'))
from cdbs.pkgs import *
from cdbs.general import *
from cdbs.custom import *



def load_batch_train_data_gdist(data_root, B_gdist, N_gdist, Ns_gdist, Nt_gdist):
    # N_gdist = Ns_gdist * Nt_gdist
    np.random.seed()
    data_root = os.path.abspath(data_root)
    class_list = ['airplane', 'car', 'chair', 'display', 'lamp', 'loudspeaker', 'table', 'watercraft']
    balanced_model_list_per_class = []
    for class_name in class_list:
        model_list_per_class = [it.split('/')[-1][:-4] for it in sorted(glob.glob(os.path.join(data_root, 'train', class_name, '*.obj')))]
        balanced_model_list_per_class += sorted(list(np.random.choice(model_list_per_class, 200, replace=False)))
    batch_model_list = sorted(list(np.random.choice(balanced_model_list_per_class, B_gdist, replace=False)))
    batch_container = {'model_name': [], 'mv': [], 'mf': [], 'pc': [], 'gdist_qp_s': [], 'gdist_qp_t': [], 'gdist_gt': []}
    for model_name in batch_model_list:
        class_name = model_name.split('_')[0]
        load_obj_path = os.path.join(data_root, 'train', class_name, model_name + '.obj')
        load_xyz_path = os.path.join(data_root, 'train', class_name, model_name + '_2048x3.xyz')
        load_npy_path = os.path.join(data_root, 'train', class_name, model_name + '_256x512x7.npy')
        # 1) load the mesh model
        mv, mf = pcu.load_mesh_vf(load_obj_path)
        mv, mf = mv.astype(np.float32), mf.astype(np.int32) # (num_mv, 3), (num_mf, 3)
        num_mv, num_mf = mv.shape[0], mf.shape[0]
        # 2) load the point cloud
        pc = np.loadtxt(load_xyz_path).astype(np.float32) # (num_pts, 3)
        # 3) load and sample the ground-truth geodesic distance data
        gdist_data = np.load(load_npy_path).astype(np.float32) # (256, 512, 3+3+1=7)
        assert Ns_gdist<=256 and Nt_gdist<=512
        sampled_gdist_data = gdist_data[np.random.choice(256, Ns_gdist, replace=False), :, :]
        sampled_gdist_data = sampled_gdist_data[:, np.random.choice(512, Nt_gdist, replace=False), :]
        sampled_gdist_data = sampled_gdist_data.reshape(-1, 7) # (N_gdist, 3+3+1=7)
        gdist_qp_s = sampled_gdist_data[:, 0:3] # (N_gdist, 3)
        gdist_qp_t = sampled_gdist_data[:, 3:6] # (N_gdist, 3)
        gdist_gt = sampled_gdist_data[:, -1] # (N_gdist,)
        batch_container['model_name'].append(model_name)
        batch_container['mv'].append(mv)
        batch_container['mf'].append(mf)
        batch_container['pc'].append(np.expand_dims(pc, axis=0))
        batch_container['gdist_qp_s'].append(np.expand_dims(gdist_qp_s, axis=0))
        batch_container['gdist_qp_t'].append(np.expand_dims(gdist_qp_t, axis=0))
        batch_container['gdist_gt'].append(np.expand_dims(gdist_gt, axis=0))
    batch_container['pc'] = np.concatenate(batch_container['pc'], axis=0) # (B_gdist, num_pts, 3)
    batch_container['gdist_qp_s'] = np.concatenate(batch_container['gdist_qp_s'], axis=0) # (B_gdist, N_gdist, 3)
    batch_container['gdist_qp_t'] = np.concatenate(batch_container['gdist_qp_t'], axis=0) # (B_gdist, N_gdist, 3)
    batch_container['gdist_gt'] = np.concatenate(batch_container['gdist_gt'], axis=0) # (B_gdist, N_gdist)
    return batch_container


def load_all_test_data_gdist(data_root, unseen_category=True):
    data_root = os.path.abspath(data_root)
    if unseen_category:
        class_list = ['bench', 'cabinet', 'rifle', 'sofa', 'telephone']
    else:
        class_list = ['airplane', 'car', 'chair', 'display', 'lamp', 'loudspeaker', 'table', 'watercraft']
    model_list_per_class = []
    for class_name in class_list:
        model_list_per_class += [it.split('/')[-1][:-4] for it in sorted(glob.glob(os.path.join(data_root, 'test', class_name, '*.obj')))] 
    B_gdist_all = len(model_list_per_class)
    all_test_data_container = {'model_name': [], 'mv': [], 'mf': [], 'pc': [], 'gdist_qp_s': [], 'gdist_qp_t': [], 'gdist_gt': []}
    for model_name in model_list_per_class:
        class_name = model_name.split('_')[0]
        load_obj_path = os.path.join(data_root, 'test', class_name, model_name + '.obj')
        load_xyz_path = os.path.join(data_root, 'test', class_name, model_name + '_2048x3.xyz')
        load_npy_path = os.path.join(data_root, 'test', class_name, model_name + '_256x512x7.npy')
        # 1) load the mesh model
        mv, mf = pcu.load_mesh_vf(load_obj_path)
        mv, mf = mv.astype(np.float32), mf.astype(np.int32) # (num_mv, 3), (num_mf, 3)
        num_mv, num_mf = mv.shape[0], mf.shape[0]
        # 2) load the point cloud
        pc = np.loadtxt(load_xyz_path).astype(np.float32) # (num_pts, 3)
        # 3) load the ground-truth geodesic distance data
        gdist_data = np.load(load_npy_path).astype(np.float32) # (256, 512, 3+3+1=7)
        N_gdist_all = gdist_data.shape[0] * gdist_data.shape[1]
        gdist_data = gdist_data.reshape(-1, 7) # (N_gdist_all, 3+3+1=7)
        gdist_qp_s = gdist_data[:, 0:3] # (N_gdist_all, 3)
        gdist_qp_t = gdist_data[:, 3:6] # (N_gdist_all, 3)
        gdist_gt = gdist_data[:, -1] # (N_gdist_all,)
        all_test_data_container['model_name'].append(model_name)
        all_test_data_container['mv'].append(mv)
        all_test_data_container['mf'].append(mf)
        all_test_data_container['pc'].append(np.expand_dims(pc, axis=0))
        all_test_data_container['gdist_qp_s'].append(np.expand_dims(gdist_qp_s, axis=0))
        all_test_data_container['gdist_qp_t'].append(np.expand_dims(gdist_qp_t, axis=0))
        all_test_data_container['gdist_gt'].append(np.expand_dims(gdist_gt, axis=0))
    all_test_data_container['pc'] = np.concatenate(all_test_data_container['pc'], axis=0) # (B_gdist_all, num_pts, 3)
    all_test_data_container['gdist_qp_s'] = np.concatenate(all_test_data_container['gdist_qp_s'], axis=0) # (B_gdist_all, N_gdist_all, 3)
    all_test_data_container['gdist_qp_t'] = np.concatenate(all_test_data_container['gdist_qp_t'], axis=0) # (B_gdist_all, N_gdist_all, 3)
    all_test_data_container['gdist_gt'] = np.concatenate(all_test_data_container['gdist_gt'], axis=0) # (B_gdist_all, N_gdist_all)
    return all_test_data_container



