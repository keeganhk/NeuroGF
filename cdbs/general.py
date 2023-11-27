from .pkgs import *
from .KNN_CPU import nearest_neighbors as knn_cpu
from .EMD.emd import earth_mover_distance_unwrapped
from .CD.chamferdist.chamfer import knn_points as knn_gpu
from .GS_CPU.cpp_subsampling import grid_subsampling as cpp_grid_subsample



################################################################################
def knn_on_gpu(source_pts, query_pts, k):
    # source_pts: [B, N, C]
    # query_pts: [B, M, C]
    # knn_idx: [B, M, k] (sorted, from close to far)
    assert source_pts.device.type == 'cuda'
    assert query_pts.device.type == 'cuda'
    assert source_pts.size(0) == query_pts.size(0)
    assert source_pts.size(2) == query_pts.size(2)
    knn_idx = knn_gpu(p1=query_pts, p2=source_pts, K=k, return_nn=False, return_sorted=True)[1]
    return knn_idx


def knn_on_cpu(source_pts, query_pts, k):
    # source_pts: [B, N, C]
    # query_pts: [B, M, C]
    # knn_idx: [B, M, k] (sorted, from close to far)
    assert source_pts.device.type == 'cpu'
    assert query_pts.device.type == 'cpu'
    assert source_pts.size(0) == query_pts.size(0)
    assert source_pts.size(2) == query_pts.size(2)
    knn_idx = knn_cpu.knn_batch(source_pts, query_pts, k, omp=True)
    return knn_idx


def knn_search(source_pts, query_pts, k):
    # source_pts: [B, N, C]
    # query_pts: [B, M, C]
    # knn_idx: [B, M, k] (sorted, from close to far)
    assert source_pts.device.type == query_pts.device.type
    device_type = source_pts.device.type
    assert device_type in ['cpu', 'cuda']
    if device_type == 'cuda':
        knn_idx = knn_on_gpu(source_pts, query_pts, k)
    if device_type == 'cpu':
        knn_idx = knn_on_cpu(source_pts, query_pts, k)
    return knn_idx


def chamfer_distance_cuda(pts_s, pts_t, cpt_mode='max', return_detail=False):
    # pts_s: [B, Ns, C], source point cloud
    # pts_t: [B, Nt, C], target point cloud
    Bs, Ns, Cs, device_s = pts_s.size(0), pts_s.size(1), pts_s.size(2), pts_s.device
    Bt, Nt, Ct, device_t = pts_t.size(0), pts_t.size(1), pts_t.size(2), pts_t.device
    assert Bs == Bt
    assert Cs == Ct
    assert device_s == device_t
    assert device_s.type == 'cuda' and device_t.type == 'cuda'
    assert cpt_mode in ['max', 'avg']
    lengths_s = torch.ones(Bs, dtype=torch.long, device=device_s) * Ns
    lengths_t = torch.ones(Bt, dtype=torch.long, device=device_t) * Nt
    source_nn = knn_gpu(pts_s, pts_t, lengths_s, lengths_t, 1)
    target_nn = knn_gpu(pts_t, pts_s, lengths_t, lengths_s, 1)
    source_dist, source_idx = source_nn.dists.squeeze(-1), source_nn.idx.squeeze(-1) # [B, Ns]
    target_dist, target_idx = target_nn.dists.squeeze(-1), target_nn.idx.squeeze(-1) # [B, Nt]
    batch_dist = torch.cat((source_dist.mean(dim=-1, keepdim=True), target_dist.mean(dim=-1, keepdim=True)), dim=-1) # [B, 2]
    if cpt_mode == 'max':
        cd = batch_dist.max(dim=-1)[0].mean()
    if cpt_mode == 'avg':
        cd = batch_dist.mean(dim=-1).mean()
    if not return_detail:
        return cd
    else:
        return cd, source_dist, source_idx, target_dist, target_idx


def earth_mover_distance_cuda(pts_1, pts_2, return_detail=False):
    # pts_1: [B, N1, C=1,2,3]
    # pts_2: [B, N2, C=1,2,3]
    assert pts_1.size(0) == pts_2.size(0)
    assert pts_1.size(2) == pts_2.size(2)
    assert pts_1.device == pts_2.device
    B, N1, C, device = pts_1.size(0), pts_1.size(1), pts_1.size(2), pts_1.device
    B, N2, C, device = pts_2.size(0), pts_2.size(1), pts_2.size(2), pts_2.device
    assert device.type == 'cuda'
    assert C in [1, 2, 3]
    if C < 3:
        pts_1 = torch.cat((pts_1, torch.zeros(B, N1, 3-C).to(device)), dim=-1) # [B, N1, 3]
        pts_2 = torch.cat((pts_2, torch.zeros(B, N2, 3-C).to(device)), dim=-1) # [B, N2, 3]  
    dist_1 = earth_mover_distance_unwrapped(pts_1, pts_2, transpose=False) / N1 # [B]
    dist_2 = earth_mover_distance_unwrapped(pts_2, pts_1, transpose=False) / N2 # [B]
    emd = ((dist_1 + dist_2) / 2).mean()
    if not return_detail:
        return emd
    else:
        return emd, dist_1, dist_2


def grid_subsample_cpu(pts, grid_size):
    # pts: [num_input, 3]
    # the output "pts_gs" is not the subset of the input "pts"
    assert pts.ndim == 2
    assert pts.shape[1] == 3
    assert pts.device.type == 'cpu'
    pts_gs = cpp_grid_subsample.compute(pts, sampleDl=grid_size) # (num_gs, 3)
    return pts_gs


################################################################################
def seed_worker(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    np.random.seed(worker_info.seed % 2**32)


def align_number(raw_number, expected_num_digits):
    # align a number string
    string_number = str(raw_number)
    ori_num_digits = len(string_number)
    assert ori_num_digits <= expected_num_digits
    return (expected_num_digits - ori_num_digits) * '0' + string_number


def load_pc(load_path):
    pcd = o3d.io.read_point_cloud(load_path)
    assert pcd.has_points()
    points = np.asarray(pcd.points) # (num_points, 3)
    attributes = {'colors': None, 'normals': None}
    if pcd.has_colors():
        colors = np.asarray(pcd.colors) # (num_points, 3)
        attributes['colors'] = colors
    if pcd.has_normals():
        normals = np.asarray(pcd.normals) # (num_points, 3)
        attributes['normals'] = normals
    return points, attributes


def save_pc(save_path, points, colors=None, normals=None):
    assert save_path[-3:] == 'ply', 'not .ply file'
    if type(points) == torch.Tensor:
        points = np.asarray(points.detach().cpu())
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points) 
    if colors is not None:
        if type(colors) == torch.Tensor:
            colors = np.asarray(colors.detach().cpu())
        assert colors.min()>=0 and colors.max()<=1
        pcd.colors = o3d.utility.Vector3dVector(colors) # should be within the range of [0, 1]
    if normals is not None:
        if type(normals) == torch.Tensor:
            normals = np.asarray(normals.detach().cpu())
        pcd.normals = o3d.utility.Vector3dVector(normals)
    o3d.io.write_point_cloud(save_path, pcd, write_ascii=True) # should be saved as .ply file


################################################################################
def min_max_normalization(x):
    x_min = x.min()
    x_max = x.max()
    x_mmn = (x - x_min) / (x_max - x_min)
    return x_mmn


def random_sampling(pc, num_sample):
    # pc: (num_points, num_channels)
    # pc_sampled: # [num_sample, num_channels]
    num_points, num_channels = pc.shape
    assert num_sample < num_points
    selected_indices = np.random.choice(num_points, num_sample, replace=False) # (num_sample,)
    pc_sampled = pc[selected_indices, :]
    return pc_sampled


def farthest_point_sampling(pc, num_sample):
    # pc: (num_points, num_channels)
    # pc_sampled: [num_sample, num_channels]
    num_points, num_channels = pc.shape
    assert num_sample < num_points
    xyz = pc[:, 0:3] # sampling is based on spatial distance
    centroids = np.zeros((num_sample,))
    distance = np.ones((num_points,)) * 1e10
    farthest = np.random.randint(0, num_points)
    for i in range(num_sample):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    pc_sampled = pc[centroids.astype(np.int32)]
    return pc_sampled


def normalization_with_given_centroid(pc, ctr):
    # pc: (num_points, num_channels)
    # pc_normalized: (num_points, num_channels)
    num_points, num_channels = pc.shape
    xyz = pc[:, 0:3]
    attr = pc[:, 3:]
    xyz = xyz - ctr
    max_d = np.max(np.sqrt(np.abs(np.sum(xyz**2, axis=1)))) # a scalar
    xyz_normalized = xyz / max_d
    pc_normalized = np.concatenate((xyz_normalized, attr), axis=1)
    return pc_normalized


def centroid_normalization(pc):
    # pc: (num_points, num_channels)
    # pc_normalized: (num_points, num_channels)
    num_points, num_channels = pc.shape
    xyz = pc[:, 0:3]
    attr = pc[:, 3:]
    xyz = xyz - np.mean(xyz, axis=0)
    max_d = np.max(np.sqrt(np.abs(np.sum(xyz**2, axis=1)))) # a scalar
    xyz_normalized = xyz / max_d
    pc_normalized = np.concatenate((xyz_normalized, attr), axis=1)
    return pc_normalized


def bounding_box_normalization(pc):
    # pc: (num_points, num_channels)
    # pc_normalized: (num_points, num_channels)
    num_points, num_channels = pc.shape
    xyz = pc[:, 0:3]
    attr = pc[:, 3:]
    xyz = xyz - (np.min(xyz, axis=0) + np.max(xyz, axis=0))/2
    max_d = np.max(np.sqrt(np.abs(np.sum(xyz**2, axis=1)))) # a scalar
    xyz_normalized = xyz / max_d
    pc_normalized = np.concatenate((xyz_normalized, attr), axis=1)
    return pc_normalized


def random_shuffling(pc):
    # pc: (num_points, num_channels)
    # pc_shuffled: (num_points, num_channels)
    num_points, num_channels = pc.shape
    idx_shuffled = np.arange(num_points)
    np.random.shuffle(idx_shuffled)
    pc_shuffled = pc[idx_shuffled]
    return pc_shuffled


def random_jittering(pc, sigma, bound):
    # pc: (num_points, num_channels)
    # sigma: standard deviation of zero-mean Gaussian noise
    # bound: clip noise values
    # pc_jittered: [num_points, num_channels]
    num_points, num_channels = pc.shape
    assert sigma > 0
    assert bound > 0
    gaussian_noises = np.random.normal(0, sigma, size=(num_points, 3)).astype(np.float32) # (num_points, 3)
    bounded_gaussian_noises = np.clip(gaussian_noises, -bound, bound).astype(np.float32) # (num_points, 3)
    if num_channels == 3:
        pc_jittered = pc + bounded_gaussian_noises
    if num_channels > 3:
        xyz = pc[:, 0:3]
        attr = pc[:, 3:]
        pc_jittered = np.concatenate((xyz + bounded_gaussian_noises, attr), axis=1)
    return pc_jittered


def random_dropout(pc, min_dp_ratio, max_dp_ratio, return_num_dropped=False):
    # pc: (num_points, num_channels)
    # max_dp_ratio: (0, 1)
    # pc_dropped: [num_points, num_channels]
    num_points, num_channels = pc.shape
    assert min_dp_ratio>=0 and min_dp_ratio<=1
    assert max_dp_ratio>=0 and max_dp_ratio<=1
    assert min_dp_ratio <= max_dp_ratio
    dp_ratio = np.random.random() * (max_dp_ratio-min_dp_ratio) + min_dp_ratio
    num_dropped = int(num_points * dp_ratio)
    pc_dropped = pc.copy()
    if num_dropped > 0:
        dp_indices = np.random.choice(num_points, num_dropped, replace=False)
        pc_dropped[dp_indices, :] = pc_dropped[0, :] # all replaced by the first row of "pc"
    if not return_num_dropped:
        return pc_dropped
    else:
        return pc_dropped, num_dropped


def axis_rotation(pc, angle, axis):
    # pc: (num_points, num_channels=3/6)
    # angle: [0, 2*pi]
    # axis: 'x', 'y', 'z'
    # pc_rotated: (num_points, num_channels=3/6)
    num_points, num_channels = pc.shape
    assert num_channels in [3, 6]
    assert angle>=0 and angle <= (2*np.pi)
    assert axis in ['x', 'y', 'z']
    # generate the rotation matrix
    c = np.cos(angle).astype(np.float32)
    s = np.sin(angle).astype(np.float32)
    if axis == 'x':
        rot_mat = np.array([ [1, 0, 0], [0, c, -s], [0, s, c] ]).astype(np.float32)
    if axis == 'y':
        rot_mat = np.array([ [c, 0, s], [0, 1, 0], [-s, 0, c] ]).astype(np.float32)
    if axis == 'z':
        rot_mat = np.array([ [c, -s, 0], [s, c, 0], [0, 0, 1] ]).astype(np.float32)
    # apply the rotation matrix
    if num_channels == 3:
        pc_rotated = np.matmul(pc, rot_mat) # (num_points, 3)
    if num_channels == 6:
        pc_rotated = np.concatenate((np.matmul(pc[:, 0:3], rot_mat), np.matmul(pc[:, 3:6], rot_mat)), axis=1) # (num_points, 6)
    return pc_rotated


def random_axis_rotation(pc, axis, return_angle=False):
    # pc: (num_points, num_channels=3/6)
    # axis: 'x', 'y', 'z'
    # pc_rotated: (num_points, num_channels=3/6)
    num_points, num_channels = pc.shape
    assert num_channels in [3, 6]
    assert axis in ['x', 'y', 'z']
    # generate a random rotation matrix
    angle = np.random.uniform() * 2 * np.pi
    c = np.cos(angle).astype(np.float32)
    s = np.sin(angle).astype(np.float32)
    if axis == 'x':
        rot_mat = np.array([ [1, 0, 0], [0, c, -s], [0, s, c] ]).astype(np.float32)
    if axis == 'y':
        rot_mat = np.array([ [c, 0, s], [0, 1, 0], [-s, 0, c] ]).astype(np.float32)
    if axis == 'z':
        rot_mat = np.array([ [c, -s, 0], [s, c, 0], [0, 0, 1] ]).astype(np.float32)
    # apply the rotation matrix
    if num_channels == 3:
        pc_rotated = np.matmul(pc, rot_mat) # (num_points, 3)
    if num_channels == 6:
        pc_rotated = np.concatenate((np.matmul(pc[:, 0:3], rot_mat), np.matmul(pc[:, 3:6], rot_mat)), axis=1) # (num_points, 6)
    if not return_angle:
        return pc_rotated
    else:
        return pc_rotated, angle
    
    
def random_rotation(pc, return_angle=False):
    # pc: (num_points, num_channels=3/6)
    # pc_rotated: (num_points, num_channels=3/6)
    num_points, num_channels = pc.shape
    assert num_channels in [3, 6]
    rot_mat = scipy_R.random().as_matrix().astype(np.float32) # (3, 3)
    if num_channels == 3:
        pc_rotated = np.matmul(pc, rot_mat) # (num_points, 3)
    if num_channels == 6:
        pc_rotated = np.concatenate((np.matmul(pc[:, 0:3], rot_mat), np.matmul(pc[:, 3:6], rot_mat)), axis=1) # (num_points, 6)
    if not return_angle:
        return pc_rotated
    else:
        rot_ang = scipy_R.from_matrix(np.transpose(rot_mat)).as_euler('xyz', degrees=True).astype(np.float32) # (3,)
        for aid in range(3):
            if rot_ang[aid] < 0:
                rot_ang[aid] = 360.0 + rot_ang[aid]
        return pc_rotated, rot_ang


def random_isotropic_scaling(pc, min_s_ratio, max_s_ratio, return_iso_scaling_ratio=False):
    # pc: (num_points, num_channels)
    # pc_iso_scaled: [num_points, num_channels]
    num_points, num_channels = pc.shape
    assert min_s_ratio > 0 and min_s_ratio <= 1
    assert max_s_ratio >= 1
    iso_scaling_ratio = np.random.random() * (max_s_ratio - min_s_ratio) + min_s_ratio
    if num_channels == 3:
        pc_iso_scaled = pc * iso_scaling_ratio
    if num_channels > 3:
        xyz = pc[:, 0:3]
        attr = pc[:, 3:]
        pc_iso_scaled = np.concatenate((xyz * iso_scaling_ratio, attr), axis=1)
    if not return_iso_scaling_ratio:
        return pc_iso_scaled
    else:
        return pc_iso_scaled, iso_scaling_ratio
    
    
def random_anisotropic_scaling(pc, min_s_ratio, max_s_ratio, return_aniso_scaling_ratio=False):
    # pc: (num_points, num_channels)
    # pc_aniso_scaled: [num_points, num_channels]
    num_points, num_channels = pc.shape
    assert min_s_ratio > 0 and min_s_ratio <= 1
    assert max_s_ratio >= 1
    aniso_scaling_ratio = (np.random.random(3) * (max_s_ratio - min_s_ratio) + min_s_ratio).astype('float32')
    pc_aniso_scaled = pc.copy()
    pc_aniso_scaled[:, 0] *= aniso_scaling_ratio[0]
    pc_aniso_scaled[:, 1] *= aniso_scaling_ratio[1]
    pc_aniso_scaled[:, 2] *= aniso_scaling_ratio[2]
    if not return_aniso_scaling_ratio:
        return pc_aniso_scaled
    else:
        return pc_aniso_scaled, aniso_scaling_ratio


def random_translation(pc, max_offset, return_offset=False):
    # pc: (num_points, num_channels)
    # pc_translated: [num_points, num_channels]
    num_points, num_channels = pc.shape
    assert max_offset > 0
    offset = np.random.uniform(low=-max_offset, high=max_offset, size=[3]).astype('float32')
    pc_translated = pc.copy()
    pc_translated[:, 0] += offset[0]
    pc_translated[:, 1] += offset[1]
    pc_translated[:, 2] += offset[2]
    if not return_offset:
        return pc_translated
    else:
        return pc_translated, offset


def random_shuffling(pc):
    # pc: (num_points, num_channels)
    # pc_shuffled: (num_points, num_channels)
    num_points, num_channels = pc.shape
    idx_shuffled = np.arange(num_points)
    np.random.shuffle(idx_shuffled)
    pc_shuffled = pc[idx_shuffled]
    return pc_shuffled


################################################################################
def index_points(pc, idx):
    # pc: [B, N, C]
    # 1) idx: [B, S] -> pc_selected: [B, S, C]
    # 2) idx: [B, S, K] -> pc_selected: [B, S, K, C]
    device = pc.device
    B = pc.shape[0]
    view_shape = list(idx.shape) 
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B).to(device).view(view_shape).repeat(repeat_shape)
    pc_selected = pc[batch_indices, idx, :]
    return pc_selected


def get_fps_idx(xyz, num_sample):
    # xyz: torch.Tensor, [batch_size, num_input, 3]
    # fps_idx: [batch_size, num_sample]
    assert xyz.ndim==3 and xyz.size(2)==3
    batch_size, num_input, device = xyz.size(0), xyz.size(1), xyz.device
    batch_indices = torch.arange(batch_size, dtype=torch.long).to(device)
    fps_idx = torch.zeros(batch_size, num_sample, dtype=torch.long).to(device)
    distance = torch.ones(batch_size, num_input).to(device) * 1e10
    farthest = torch.randint(0, num_input, (batch_size,), dtype=torch.long).to(device)
    for i in range(num_sample):
        fps_idx[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(batch_size, 1, -1)
        dist = torch.sum((xyz-centroid)**2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return fps_idx


def get_fps_idx_zero_as_first(xyz, num_sample):
    # xyz: torch.Tensor, [batch_size, num_input, 3]
    # fps_idx: [batch_size, num_sample]
    assert xyz.ndim==3 and xyz.size(2)==3
    batch_size, num_input, device = xyz.size(0), xyz.size(1), xyz.device
    batch_indices = torch.arange(batch_size, dtype=torch.long).to(device)
    fps_idx = torch.zeros(batch_size, num_sample, dtype=torch.long).to(device)
    distance = torch.ones(batch_size, num_input).to(device) * 1e10
    farthest = torch.zeros(batch_size, dtype=torch.long).to(device) # torch.randint(0, num_input, (batch_size,), dtype=torch.long).to(device)
    for i in range(num_sample):
        fps_idx[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(batch_size, 1, -1)
        dist = torch.sum((xyz-centroid)**2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return fps_idx


def get_fps_idx_specified_first(xyz, num_sample, first):
    # xyz: torch.Tensor, [batch_size, num_input, 3]
    # fps_idx: [batch_size, num_sample]
    # first: [batch_size]
    assert xyz.ndim==3 and xyz.size(2)==3
    batch_size, num_input, device = xyz.size(0), xyz.size(1), xyz.device
    batch_indices = torch.arange(batch_size, dtype=torch.long).to(device)
    fps_idx = torch.zeros(batch_size, num_sample, dtype=torch.long).to(device)
    distance = torch.ones(batch_size, num_input).to(device) * 1e10
    farthest = first.long().to(device) # [batch_size]
    for i in range(num_sample):
        fps_idx[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(batch_size, 1, -1)
        dist = torch.sum((xyz-centroid)**2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return fps_idx


################################################################################
class MLP(nn.Module):
    def __init__(self, ic, oc, is_bn, nl):
        super(MLP, self).__init__()
        assert isinstance(is_bn, bool)
        assert nl in ['none', 'relu', 'tanh', 'sigmoid']
        self.is_bn = is_bn
        self.nl = nl
        self.conv = nn.Conv2d(in_channels=ic, out_channels=oc, kernel_size=1, bias=False)
        if self.is_bn:
            self.bn = nn.BatchNorm2d(oc)
        if nl == 'relu':
            self.activate = nn.ReLU(inplace=True)
        if nl == 'tanh':
            self.activate = nn.Tanh()
        if nl == 'sigmoid':
            self.activate = nn.Sigmoid()
    def forward(self, x):
        # x: [batch_size, num_points, ic]
        x = x.permute(0, 2, 1).contiguous().unsqueeze(-1) # [batch_size, ic, num_points, 1]
        y = self.conv(x) # [batch_size, oc, num_points, 1]
        if self.is_bn:
            y = self.bn(y)
        if self.nl != 'none':
            y = self.activate(y)   
        y = y.squeeze(-1).permute(0, 2, 1).contiguous() 
        return y # [batch_size, num_points, oc]


class FC(nn.Module):
    def __init__(self, ic, oc, is_bn, nl):
        super(FC, self).__init__()
        assert isinstance(is_bn, bool)
        assert nl in ['none', 'relu', 'tanh', 'sigmoid']
        self.is_bn = is_bn
        self.nl = nl
        self.linear = nn.Linear(ic, oc, bias=False)
        if self.is_bn:
            self.bn = nn.BatchNorm1d(oc)
        if nl == 'relu':
            self.activate = nn.ReLU(inplace=True)
        if nl == 'tanh':
            self.activate = nn.Tanh()
        if nl == 'sigmoid':
            self.activate = nn.Sigmoid()
    def forward(self, x):
        # x: [batch_size, ic]
        y = self.linear(x) # [batch_size, oc]
        if self.is_bn:
            y = self.bn(y)
        if self.nl != 'none':
            y = self.activate(y)
        return y # [batch_size, oc]


class ResMLP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResMLP, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mlp_1 = MLP(in_channels, in_channels, True, 'none')
        self.mlp_2 = MLP(in_channels, out_channels, True, 'none')
        if in_channels != out_channels:
            self.shortcut = MLP(in_channels, out_channels, True, 'none')
        self.nl = nn.ReLU(inplace=True)
    def forward(self, in_ftr):
        # in_ftr: [B, N, in_channels]
        out_ftr = self.mlp_2(self.nl(self.mlp_1(in_ftr)))
        if self.in_channels != self.out_channels:
            out_ftr = self.nl(self.shortcut(in_ftr) + out_ftr)
        else:
            out_ftr = self.nl(in_ftr + out_ftr)
        return out_ftr # [B, N, out_channels]



