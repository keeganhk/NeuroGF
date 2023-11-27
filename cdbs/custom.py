from .pkgs import *
from .general import *



def build_3d_grids(res_grids, margin):
    # res_grids: resolution of grids
    # the range of grid points is [-1.0-margin, +1.0+margin]
    num_grids = (res_grids ** 3)
    min_bound = -1 - margin
    max_bound = +1 + margin
    x = np.linspace(min_bound, max_bound, res_grids, dtype=np.float32)
    y = np.linspace(min_bound, max_bound, res_grids, dtype=np.float32)
    z = np.linspace(min_bound, max_bound, res_grids, dtype=np.float32)
    grids = np.array(list(itertools.product(x, y, z))) # (num_grids, 3)
    return grids


def curve_interp(curve_pts):
    # curve_pts: (N, 3)
    N = curve_pts.shape[0]
    assert N >= 2
    if N == 2:
        p1 = curve_pts[0, :].reshape(1, 3)
        p3 = curve_pts[1, :].reshape(1, 3)
        p2 = (p1 + p3) / 2.0
        curve_pts_interp = np.concatenate((p1, p2, p3), axis=0)
    else:
        t1 = curve_pts[0:N-1, :] # [N-1, 3]
        t2 = curve_pts[1:N, :] # [N-1, 3]
        mid_pts = (t1 + t2) / 2 # [N-1, 3]
        curve_pts_interp = np.zeros((2*N-1, 3))
        curve_pts_interp[np.arange(0, 2*N-1, 2), :] = curve_pts
        curve_pts_interp[np.arange(1, 2*N-1, 2), :] = mid_pts
    return curve_pts_interp.astype(np.float32)


def approx_path_len(path_pts):
    # path_pts: (N, 3)
    N = path_pts.shape[0]
    assert N > 1
    assert path_pts.shape[1] == 3
    if N == 2:
        d = ((path_pts[0]-path_pts[1])**2).sum() ** 0.5
    else:
        x = path_pts[0:N-1, :] # (N-1, 3)
        y = path_pts[1:N, :] # (N-1, 3)
        d = (((x-y)**2).sum(axis=-1) ** 0.5).sum()
    return d


def approx_path_len_batched(path_pts_batched):
    # path_pts_batched: [B, N, 3]
    B, N, _ = path_pts_batched.size()
    x = path_pts_batched[:, 0:N-1, :] # [B, N-1, 3]
    y = path_pts_batched[:, 1:N, :] # [B, N-1, 3]
    path_dists = ((((x-y)**2).sum(dim=-1))**0.5).sum(dim=-1) # [B]
    return path_dists


def colorcode_curve_points(seq_pts):
    # seq_pts: (N, 3), a discrete sequence of points
    N = seq_pts.shape[0]
    c_map = cm.jet
    cc = c_map(np.linspace(0, 1, N))[:, 0:3] # (N, 3)
    seq_pts_cc = np.concatenate((seq_pts, cc), axis=-1) # (N, 6)
    return seq_pts_cc


def generate_lines_from_end_points(ps, pe, num_points_per_line):
    # ps: (N, 3), starting points
    # pe: (N, 3), ending points
    ps = ps.astype(np.float32)
    pe = pe.astype(np.float32)
    assert ps.shape[0]==pe.shape[0] and ps.shape[1]==3 and pe.shape[1]==3
    N = ps.shape[0]
    lines = np.zeros((N, num_points_per_line, 3), dtype=np.float32)
    for index in range(N):
        ps_this = ps[index] # (3,)
        pe_this = pe[index] # (3,)
        x = np.linspace(ps_this[0], pe_this[0], num_points_per_line).astype(np.float32).reshape(-1, 1) # (num_points_per_line, 1)
        y = np.linspace(ps_this[1], pe_this[1], num_points_per_line).astype(np.float32).reshape(-1, 1) # (num_points_per_line, 1)
        z = np.linspace(ps_this[2], pe_this[2], num_points_per_line).astype(np.float32).reshape(-1, 1) # (num_points_per_line, 1)
        line_this = np.concatenate((x, y, z), axis=-1) # (num_points_per_line, 3)
        lines[index] = line_this
    return lines # (N, num_points_per_line, 3)


def compute_chamfer_l1(P1, P2):
    # P1: (N1, 3)
    # P2: (N2, 3)
    P1 = P1.astype(np.float32)
    P2 = P2.astype(np.float32)
    kd_tree_1 = cKDTree(P1)
    one_distances, one_vertex_ids = kd_tree_1.query(P2)
    chamfer_1_to_2 = np.mean(one_distances)
    kd_tree_2 = cKDTree(P2)
    two_distances, two_vertex_ids = kd_tree_2.query(P1)
    chamfer_2_to_1 = np.mean(two_distances)
    chamfer_l1 = (chamfer_1_to_2 + chamfer_2_to_1) / 2
    return chamfer_l1


def compute_chamfer_l2(P1, P2):
    # P1: (N1, 3)
    # P2: (N2, 3)
    P1 = P1.astype(np.float32)
    P2 = P2.astype(np.float32)
    kd_tree_1 = cKDTree(P1)
    one_distances, one_vertex_ids = kd_tree_1.query(P2)
    chamfer_1_to_2 = np.mean(np.square(one_distances))
    kd_tree_2 = cKDTree(P2)
    two_distances, two_vertex_ids = kd_tree_2.query(P1)
    chamfer_2_to_1 = np.mean(np.square(two_distances))
    chamfer_l2 = (chamfer_1_to_2 + chamfer_2_to_1) / 2
    return chamfer_l2


def select_salient_points(points, num_fps, num_salient):
    # points: (num_points, 3)
    # num_fps: use FPS to sample the initial candicate seeds
    num_points = points.shape[0]
    assert num_fps<=num_points and num_salient<=num_fps
    fps_seeds = farthest_point_sampling(points, num_fps) # (num_fps, 3)
    num_knn = int(num_points * 0.005)
    if num_knn < 8:
        num_knn = 8
    knn_idx = knn_search(torch.tensor(points).unsqueeze(0), torch.tensor(fps_seeds).unsqueeze(0), num_knn) # (1, num_fps, num_knn)
    pat_pts = np.asarray(index_points(torch.tensor(points).unsqueeze(0), knn_idx).squeeze(0)) # (num_fps, num_knn, 3)
    for k in range(num_fps):
        pat_pts[k] = bounding_box_normalization(pat_pts[k])
    _, Sigma, _ = np.linalg.svd(pat_pts, full_matrices=False, compute_uv=True)
    Sigma /= Sigma.sum(axis=1).reshape(-1, 1)
    saliency_scores = Sigma[:, -1] # (num_fps,)
    salient_seeds = fps_seeds[np.asarray(torch.tensor(saliency_scores).sort(descending=True)[1][0:num_salient])] # (num_salient, 3)
    return salient_seeds


def colorcode_geodesic_distance_field(points_with_gd):
    # points_with_gd: (N, 4)
    N = points_with_gd.shape[0]
    c_map = cm.jet
    cc = c_map(min_max_normalization(points_with_gd[:, -1]))[:, 0:3] # (N, 3)
    points_cc = np.concatenate((points_with_gd[:, 0:3], cc), axis=-1) # (N, 6)
    return points_cc


def encoding_function(p, L):
    # p: [num_points, 1]
    pe = []
    for l in range(L):
        w = (2**l) * np.pi
        pe.append(torch.sin(w * p))
        pe.append(torch.cos(w * p))
    pe = torch.cat(pe, dim=-1) # [num_points, 2L]
    return pe


def positional_encoding(xyz, L):
    # xyz: [num_points, 3]
    x = xyz[:, 0].view(-1, 1) # [num_points, 1]
    y = xyz[:, 1].view(-1, 1) # [num_points, 1]
    z = xyz[:, 2].view(-1, 1) # [num_points, 1]
    xe = encoding_function(x, L) # [num_points, 2L]
    ye = encoding_function(y, L) # [num_points, 2L]
    ze = encoding_function(z, L) # [num_points, 2L]
    xyz_e = torch.cat((xe, ye, ze), dim=-1) # [num_points, 6L]
    return xyz_e


