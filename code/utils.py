import os, sys
prj_root = os.path.abspath(os.path.dirname(os.path.dirname(os.getcwd())))
sys.path.append(prj_root)
from cdbs.general_pkgs import *
from cdbs.general_funcs import *
from cdbs.general_modules import *
from cdbs.building_blocks import *



def fetch_train_data_sdf(num_fetch, mv, mf, sp):
    assert np.mod(num_fetch, 3) == 0
    num_fetch_each = (num_fetch//3)
    num_sp = sp.shape[0]
    np.random.seed()
    sdf_q_1 = ((np.random.rand(num_fetch_each, 3)-0.5)*2*1.01).astype(np.float32)
    np.random.seed()
    sdf_q_2 = sp[np.random.choice(num_sp, num_fetch_each, replace=False), :]
    np.random.seed()
    sdf_q_3 = random_jittering(sp[np.random.choice(num_sp, num_fetch_each, replace=False), :], 0.03, 0.03)
    sdf_q = np.concatenate((sdf_q_1, sdf_q_2, sdf_q_3), axis=0).astype(np.float32)
    sdist_gt = pcu.signed_distance_to_mesh(sdf_q, mv, mf)[0].astype(np.float32)
    return sdf_q, sdist_gt


def fetch_train_data_gdf(num_s, num_t_g, num_t_l1, num_t_l2, load_root_gdf):
    np.random.seed()
    folder_global = os.path.join(load_root_gdf, 'global')
    folder_local_1 = os.path.join(load_root_gdf, 'local_1')
    folder_local_2 = os.path.join(load_root_gdf, 'local_2')
    name_list = [item.split('/')[-1].split('.')[0] for item in glob.glob(os.path.join(folder_global, '*.npy'))]
    collection_g = np.zeros((num_s, num_t_g, 7), dtype=np.float32)
    collection_l1 = np.zeros((num_s, num_t_l1, 7), dtype=np.float32)
    collection_l2 = np.zeros((num_s, num_t_l2, 7), dtype=np.float32)
    for source_counter, selected_name in enumerate(np.random.choice(name_list, num_s)):
        si_str = selected_name.split('__')[-1]
        raw_data_g = np.load(os.path.join(folder_global, 'global__' + si_str + '.npy'))
        raw_data_l1 = np.load(os.path.join(folder_local_1, 'local_1__' + si_str + '.npy'))
        raw_data_l2 = np.load(os.path.join(folder_local_2, 'local_2__' + si_str + '.npy'))
        num_g = raw_data_g.shape[0]
        num_l1 = raw_data_l1.shape[0]
        num_l2 = raw_data_l2.shape[0]
        if num_t_g <= num_g:
            selected_idx_g = np.random.choice(np.arange(num_g), num_t_g, replace=False)
        else:
            selected_idx_g = np.random.choice(np.arange(num_g), num_t_g, replace=True)
        sampled_data_g = raw_data_g[selected_idx_g, :]
        collection_g[source_counter, ...] = sampled_data_g
        if num_t_l1 <= num_l1:
            selected_idx_l1 = np.random.choice(np.arange(num_l1), num_t_l1, replace=False)
        else:
            selected_idx_l1 = np.random.choice(np.arange(num_l1), num_t_l1, replace=True)
        sampled_data_l1 = raw_data_l1[selected_idx_l1, :]
        collection_l1[source_counter, ...] = sampled_data_l1
        if num_t_l2 <= num_l2:
            selected_idx_l2 = np.random.choice(np.arange(num_l2), num_t_l2, replace=False)
        else:
            selected_idx_l2 = np.random.choice(np.arange(num_l2), num_t_l2, replace=True)
        sampled_data_l2 = raw_data_l2[selected_idx_l2, :]
        collection_l2[source_counter, ...] = sampled_data_l2
    collection_g = collection_g.reshape(-1, 7).astype(np.float32)
    collection_l1 = collection_l1.reshape(-1, 7).astype(np.float32)
    collection_l2 = collection_l2.reshape(-1, 7).astype(np.float32)
    gdf_q_g = collection_g[:, 0:6]
    gdist_gt_g = collection_g[:, -1]
    gdf_q_l1 = collection_l1[:, 0:6]
    gdist_gt_l1 = collection_l1[:, -1]
    gdf_q_l2 = collection_l2[:, 0:6]
    gdist_gt_l2 = collection_l2[:, -1]
    return gdf_q_g, gdf_q_l1, gdf_q_l2, gdist_gt_g, gdist_gt_l1, gdist_gt_l2


def fetch_train_data_spf(num_s, num_t_g, num_t_l, load_root_spf, num_pp_g, num_pp_l):
    np.random.seed()
    folder_global = os.path.join(load_root_spf, 'global')
    folder_local = os.path.join(load_root_spf, 'local')
    name_list = [item.split('/')[-1][7:-4] for item in glob.glob(os.path.join(folder_global, '*.npy'))]
    collection_g = np.zeros((num_s, num_t_g, num_pp_g, 3), dtype=np.float32)
    collection_l = np.zeros((num_s, num_t_l, num_pp_l, 3), dtype=np.float32)
    for source_counter, selected_name in enumerate(np.random.choice(name_list, num_s)):
        raw_data_g = np.load(os.path.join(folder_global, 'global_' + selected_name + '.npy'))
        raw_data_l = np.load(os.path.join(folder_local, 'local_' + selected_name + '.npy'))
        num_g = raw_data_g.shape[0]
        num_l = raw_data_l.shape[0]
        assert raw_data_g.shape[1] == num_pp_g
        assert raw_data_l.shape[1] == num_pp_l
        if num_t_g <= num_g:
            selected_idx_g = np.random.choice(np.arange(num_g), num_t_g, replace=False)
        else:
            selected_idx_g = np.random.choice(np.arange(num_g), num_t_g, replace=True)
        sampled_data_g = raw_data_g[selected_idx_g, :, :]
        collection_g[source_counter, ...] = sampled_data_g
        if num_t_l <= num_l:
            selected_idx_l = np.random.choice(np.arange(num_l), num_t_l, replace=False)
        else:
            selected_idx_l = np.random.choice(np.arange(num_l), num_t_l, replace=True)
        sampled_data_l = raw_data_l[selected_idx_l, :, :]
        collection_l[source_counter, ...] = sampled_data_l   
    spath_gt_g = collection_g.reshape(-1, num_pp_g, 3).astype(np.float32)
    spath_gt_l = collection_l.reshape(-1, num_pp_l, 3).astype(np.float32)
    return spath_gt_g, spath_gt_l



