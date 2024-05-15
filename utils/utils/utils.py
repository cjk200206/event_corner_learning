"""
获取label标签
"""
import torch
from  .d2s import SpaceToDepth,DepthToSpace
import numpy as np
import torch.nn.functional as F

#superpoint的标签获取
def labels2Dto3D_flattened(labels, cell_size):
    '''
    Change the shape of labels into 3D. Batch of labels.

    :param labels:
        tensor [batch_size, 1, H, W]
        keypoint map.
    :param cell_size:
        8
    :return:
         labels: tensors[batch_size, 65, Hc, Wc]
    '''
    batch_size, channel, H, W = labels.shape
    Hc, Wc = H // cell_size, W // cell_size
    space2depth = SpaceToDepth(8)
    labels = space2depth(labels)

    dustbin = torch.ones((batch_size, 1, Hc, Wc)).cuda()
    labels = torch.cat((labels.cuda()*2, dustbin.view(batch_size, 1, Hc, Wc)), dim=1)
    # labels = torch.argmax(labels, dim=1)
    return labels

def labels2Dto3D(labels, cell_size, add_dustbin=True):
    '''
    Change the shape of labels into 3D. Batch of labels.

    :param labels:
        tensor [batch_size, 1, H, W]
        keypoint map.
    :param cell_size:
        8
    :return:
         labels: tensors[batch_size, 65, Hc, Wc]
    '''
    batch_size, channel, H, W = labels.shape
    Hc, Wc = H // cell_size, W // cell_size
    space2depth = SpaceToDepth(8)
    # labels = space2depth(labels).squeeze(0)
    labels = space2depth(labels)
    # labels = labels.view(batch_size, H, 1, W, 1)
    # labels = labels.view(batch_size, Hc, cell_size, Wc, cell_size)
    # labels = labels.transpose(1, 2).transpose(3, 4).transpose(2, 3)
    # labels = labels.reshape(batch_size, 1, cell_size ** 2, Hc, Wc)
    # labels = labels.view(batch_size, cell_size ** 2, Hc, Wc)
    if add_dustbin:
        dustbin = labels.sum(dim=1)
        dustbin = 1 - dustbin
        dustbin[dustbin < 1.] = 0
        # print('dust: ', dustbin.shape)
        # labels = torch.cat((labels, dustbin.view(batch_size, 1, Hc, Wc)), dim=1)
        labels = torch.cat((labels, dustbin.view(batch_size, 1, Hc, Wc)), dim=1)
        ## norm
        dn = labels.sum(dim=1)
        labels = labels.div(torch.unsqueeze(dn, 1))
    return labels

#superpoint的标签获取
def getLabels(labels_2D, cell_size, device="cpu"):
    """
    # transform 2D labels to 3D shape for training
    :param labels_2D:
    :param cell_size:
    :param device:
    :return:
    """
    labels3D_flattened = labels2Dto3D(
        labels_2D.to(device), cell_size=cell_size
    )
    labels3D_in_loss = labels3D_flattened
    return labels3D_in_loss

#做nms old
def getPtsFromHeatmap_old(heatmap, conf_thresh, nms_dist):
    '''
    :param self:
    :param heatmap:
        np (H, W)
    :return:
    '''

    border_remove = 4

    H, W = heatmap.shape[0], heatmap.shape[1]
    xs, ys = np.where(heatmap >= conf_thresh)  # Confidence threshold.
    sparsemap = (heatmap >= conf_thresh)
    if len(xs) == 0:
        return np.zeros((3, 0))
    pts = np.zeros((3, len(xs)))  # Populate point data sized 3xN.
    pts[0, :] = ys
    pts[1, :] = xs
    pts[2, :] = heatmap[xs, ys]
    pts, _ = nms_fast_old(pts, H, W, dist_thresh=nms_dist)  # Apply NMS.
    inds = np.argsort(pts[2, :])
    pts = pts[:, inds[::-1]]  # Sort by confidence.
    # Remove points along border.
    bord = border_remove
    toremoveW = np.logical_or(pts[0, :] < bord, pts[0, :] >= (W - bord))
    toremoveH = np.logical_or(pts[1, :] < bord, pts[1, :] >= (H - bord))
    toremove = np.logical_or(toremoveW, toremoveH)
    pts = pts[:, ~toremove]
    return pts

def nms_fast_old(in_corners, H, W, dist_thresh):
    """
    Run a faster approximate Non-Max-Suppression on numpy corners shaped:
      3xN [x_i,y_i,conf_i]^T
    Algo summary: Create a grid sized HxW. Assign each corner location a 1, rest
    are zeros. Iterate through all the 1's and convert them either to -1 or 0.
    Suppress points by setting nearby values to 0.
    Grid Value Legend:
    -1 : Kept.
     0 : Empty or suppressed.
     1 : To be processed (converted to either kept or supressed).
    NOTE: The NMS first rounds points to integers, so NMS distance might not
    be exactly dist_thresh. It also assumes points are within image boundaries.
    Inputs
      in_corners - 3xN numpy array with corners [x_i, y_i, confidence_i]^T.
      H - Image height.
      W - Image width.
      dist_thresh - Distance to suppress, measured as an infinty norm distance.
    Returns
      nmsed_corners - 3xN numpy matrix with surviving corners.
      nmsed_inds - N length numpy vector with surviving corner indices.
    """
    grid = np.zeros((H, W)).astype(int)  # Track NMS data.
    inds = np.zeros((H, W)).astype(int)  # Store indices of points.
    # Sort by confidence and round to nearest int.
    inds1 = np.argsort(-in_corners[2, :])
    corners = in_corners[:, inds1]
    rcorners = corners[:2, :].round().astype(int)  # Rounded corners.
    # Check for edge case of 0 or 1 corners.
    if rcorners.shape[1] == 0:
        return np.zeros((3, 0)).astype(int), np.zeros(0).astype(int)
    if rcorners.shape[1] == 1:
        out = np.vstack((rcorners, in_corners[2])).reshape(3, 1)
        return out, np.zeros((1)).astype(int)
    # Initialize the grid.
    for i, rc in enumerate(rcorners.T):
        grid[rcorners[1, i], rcorners[0, i]] = 1
        inds[rcorners[1, i], rcorners[0, i]] = i
    # Pad the border of the grid, so that we can NMS points near the border.
    pad = dist_thresh
    grid = np.pad(grid, ((pad, pad), (pad, pad)), mode='constant')
    # Iterate through points, highest to lowest conf, suppress neighborhood.
    count = 0
    for i, rc in enumerate(rcorners.T):
        # Account for top and left padding.
        pt = (rc[0] + pad, rc[1] + pad)
        if grid[pt[1], pt[0]] == 1:  # If not yet suppressed.
            grid[pt[1] - pad:pt[1] + pad + 1, pt[0] - pad:pt[0] + pad + 1] = 0
            grid[pt[1], pt[0]] = -1
            count += 1
    # Get all surviving -1's and return sorted array of remaining corners.
    keepy, keepx = np.where(grid == -1)
    keepy, keepx = keepy - pad, keepx - pad
    inds_keep = inds[keepy, keepx]
    out = corners[:, inds_keep]
    values = out[-1, :]
    inds2 = np.argsort(-values)
    out = out[:, inds2]
    out_inds = inds1[inds_keep[inds2]]
    return out, out_inds

def heatmap_nms_old(heatmap, nms_dist=8, conf_thresh=0.020):
    """
    input:
        heatmap: np [(1), H, W]
    """
    # nms_dist = self.config['model']['nms']
    # conf_thresh = self.config['model']['detection_threshold']
    heatmap = heatmap.squeeze()
    # print("heatmap: ", heatmap.shape)
    pts_nms = getPtsFromHeatmap_old(heatmap, conf_thresh, nms_dist)
    semi_thd_nms_sample = np.zeros_like(heatmap)
    semi_thd_nms_sample[
        pts_nms[1, :].astype(np.int), pts_nms[0, :].astype(np.int)
    ] = 1
    return semi_thd_nms_sample

def nms_fast(in_corners, H, W, dist_thresh):
    """
    Run a faster approximate Non-Max-Suppression on tensor corners shaped:
      3xN [x_i,y_i,conf_i]^T
    Algo summary: Create a grid sized HxW. Assign each corner location a 1, rest
    are zeros. Iterate through all the 1's and convert them either to -1 or 0.
    Suppress points by setting nearby values to 0.
    Grid Value Legend:
    -1 : Kept.
     0 : Empty or suppressed.
     1 : To be processed (converted to either kept or suppressed).
    NOTE: The NMS first rounds points to integers, so NMS distance might not
    be exactly dist_thresh. It also assumes points are within image boundaries.
    Inputs
      in_corners - 3xN tensor with corners [x_i, y_i, confidence_i]^T.
      H - Image height.
      W - Image width.
      dist_thresh - Distance to suppress, measured as an infinity norm distance.
    Returns
      nmsed_corners - 3xN tensor with surviving corners.
      nmsed_inds - N length tensor with surviving corner indices.
    """
    device = in_corners.device
    grid = torch.zeros((H, W), dtype=torch.int, device=device)  # Track NMS data.
    inds = torch.zeros((H, W), dtype=torch.int, device=device)  # Store indices of points.

    # Sort by confidence and round to nearest int.
    inds1 = torch.argsort(-in_corners[2, :])
    corners = in_corners[:, inds1]
    rcorners = corners[:2, :].round().to(torch.int)  # Rounded corners.

    # Check for edge case of 0 or 1 corners.
    if rcorners.shape[1] == 0:
        return torch.zeros((3, 0), dtype=torch.int, device=device), torch.zeros(0, dtype=torch.int, device=device)
    if rcorners.shape[1] == 1:
        out = torch.cat((rcorners, in_corners[2].view(1, -1)), dim=0).view(3, 1)
        return out, torch.zeros(1, dtype=torch.int, device=device)

    # Initialize the grid.
    for i, rc in enumerate(rcorners.T):
        grid[rc[1], rc[0]] = 1
        inds[rc[1], rc[0]] = i

    # Pad the border of the grid, so that we can NMS points near the border.
    pad = dist_thresh
    grid = torch.nn.functional.pad(grid, (pad, pad, pad, pad), mode='constant', value=0)

    # Iterate through points, highest to lowest conf, suppress neighborhood.
    count = 0
    for i, rc in enumerate(rcorners.T):
        # Account for top and left padding.
        pt = (rc[0] + pad, rc[1] + pad)
        if grid[pt[1], pt[0]] == 1:  # If not yet suppressed.
            grid[pt[1] - pad:pt[1] + pad + 1, pt[0] - pad:pt[0] + pad + 1] = 0
            grid[pt[1], pt[0]] = -1
            count += 1

    # Get all surviving -1's and return sorted array of remaining corners.
    keepy, keepx = torch.where(grid == -1)
    keepy, keepx = keepy - pad, keepx - pad
    inds_keep = inds[keepy, keepx].to(torch.long)
    out = corners[:, inds_keep]
    values = out[-1, :]
    inds2 = torch.argsort(-values)
    out = out[:, inds2]
    out_inds = inds1[inds_keep[inds2]]

    return out, out_inds


def getPtsFromHeatmap(heatmap, conf_thresh, nms_dist):
    """
    :param heatmap: torch tensor (H, W)
    :return:
    """

    device = heatmap.device
    border_remove = 4

    H, W = heatmap.shape[0], heatmap.shape[1]
    xs, ys = torch.where(heatmap >= conf_thresh)  # Confidence threshold.
    sparsemap = (heatmap >= conf_thresh)
    if xs.numel() == 0:
        return torch.zeros((3, 0))
    pts = torch.zeros((3, xs.numel()),device=device)  # Populate point data sized 3xN.
    pts[0, :] = ys
    pts[1, :] = xs
    pts[2, :] = heatmap[xs, ys]
    pts, _ = nms_fast(pts, H, W, dist_thresh=nms_dist)  # Apply NMS.
    inds = torch.argsort(pts[2, :])
    pts = pts[:, inds.flip(0)]  # Sort by confidence.
    # Remove points along border.
    bord = border_remove
    toremoveW = torch.logical_or(pts[0, :] < bord, pts[0, :] >= (W - bord))
    toremoveH = torch.logical_or(pts[1, :] < bord, pts[1, :] >= (H - bord))
    toremove = torch.logical_or(toremoveW, toremoveH)
    pts = pts[:, ~toremove]
    return pts


def heatmap_nms(heatmap, nms_dist=8, conf_thresh=0.020):
    """
    input:
        heatmap: torch tensor [(1), H, W]
    """
    # nms_dist = self.config['model']['nms']
    # conf_thresh = self.config['model']['detection_threshold']
    heatmap = heatmap.squeeze()
    # print("heatmap: ", heatmap.shape)
    pts_nms = getPtsFromHeatmap(heatmap, conf_thresh, nms_dist).to(torch.long)
    semi_thd_nms_sample = torch.zeros_like(heatmap)
    semi_thd_nms_sample[
        pts_nms[1, :], pts_nms[0, :]
    ] = 1
    return semi_thd_nms_sample



#数据增强,添加椒盐噪声
def add_salt_and_pepper_new(vox,type="default"):
    """ Add salt and pepper noise to an image """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vox = vox.to(device)
    noise = torch.randint(0, 256, size=vox.shape,device=device)
    if type == "default":
        # black = noise < 3
        white = noise > 254
        vox[white] = 255
        # vox[black] = 0
    elif type == "sae":
        # black = noise < 1
        white = noise > 254
        vox[white] = 2 * torch.rand(vox[white].shape,device=device) - 1
        # vox[black] = np.random.uniform(low=-1, high=0,size=vox[black].shape)

    return vox.cpu()

#HA变换的函数调用
def warp_points(points, homographies, device='cpu'):
    """
    Warp a list of points with the given homography.

    Arguments:
        points: list of N points, shape (N, 2(x, y))).
        homography: batched or not (shapes (B, 3, 3) and (...) respectively).

    Returns: a Tensor of shape (N, 2) or (B, N, 2(x, y)) (depending on whether the homography
            is batched) containing the new coordinates of the warped points.

    """
    # expand points len to (x, y, 1)
    no_batches = len(homographies.shape) == 2
    homographies = homographies.unsqueeze(0) if no_batches else homographies
    # homographies = homographies.unsqueeze(0) if len(homographies.shape) == 2 else homographies
    batch_size = homographies.shape[0]
    points = torch.cat((points.float(), torch.ones((points.shape[0], 1)).to(device)), dim=1)
    points = points.to(device)
    homographies = homographies.contiguous().view(batch_size*3,3)
    # warped_points = homographies*points
    # points = points.double()
    warped_points = homographies@points.transpose(0,1)
    # warped_points = np.tensordot(homographies, points.transpose(), axes=([2], [0]))
    # normalize the points
    warped_points = warped_points.view([batch_size, 3, -1])
    warped_points = warped_points.transpose(2, 1)
    warped_points = warped_points[:, :, :2] / warped_points[:, :, 2:]
    return warped_points[0,:,:] if no_batches else warped_points

#HA变换图像
def inv_warp_image_batch(img, mat_homo_inv, device='cpu', mode='bilinear'):
    '''
    Inverse warp images in batch

    :param img:
        batch of images
        tensor [batch_size, 1, H, W]
    :param mat_homo_inv:
        batch of homography matrices
        tensor [batch_size, 3, 3]
    :param device:
        GPU device or CPU
    :return:
        batch of warped images
        tensor [batch_size, 1, H, W]
    '''
    # compute inverse warped points
    if len(img.shape) == 2 or len(img.shape) == 3:
        img = img.view(1,1,img.shape[0], img.shape[1])
        # img = img.view(img.shape[0],1,img.shape[1], img.shape[2])
    if len(mat_homo_inv.shape) == 2:
        mat_homo_inv = mat_homo_inv.view(1,3,3)

    Batch, channel, H, W = img.shape
    coor_cells = torch.stack(torch.meshgrid(torch.linspace(-1, 1, W,device=device), torch.linspace(-1, 1, H,device=device)), dim=2)
    coor_cells = coor_cells.transpose(0, 1)
    coor_cells = coor_cells.to(device)
    coor_cells = coor_cells.contiguous()

    src_pixel_coords = warp_points(coor_cells.view([-1, 2]), mat_homo_inv, device)
    src_pixel_coords = src_pixel_coords.view([Batch, H, W, 2])
    src_pixel_coords = src_pixel_coords.float()

    warped_img = F.grid_sample(img, src_pixel_coords, mode=mode, align_corners=True)
    return warped_img

def inv_warp_image(img, mat_homo_inv, device='cpu', mode='bilinear'):
    '''
    Inverse warp images in batch

    :param img:
        batch of images
        tensor [H, W]
    :param mat_homo_inv:
        batch of homography matrices
        tensor [3, 3]
    :param device:
        GPU device or CPU
    :return:
        batch of warped images
        tensor [H, W]
    '''
    warped_img = inv_warp_image_batch(img, mat_homo_inv, device, mode)
    return warped_img.squeeze()

#SAE图
def get_timesurface(filename,img_size = (260,346),tau = 50e-3):
    infile = open(filename, 'r')
    ts, x, y, p = [], [], [], []
    for line in infile:
        words = line.split()
        x.append(int(words[0]))
        y.append(int(words[1]))
        ts.append(float(words[2])*10e-9)
        p.append(int(words[3]))
    infile.close()

    img_size = (260,346)

    # parameters for Time Surface
    t_ref = ts[-1]      # 'current' time
    tau = tau        # 50ms

    sae = np.zeros(img_size, np.float32)
    # calculate timesurface using expotential decay
    for i in range(len(ts)):
        if (p[i] > 0):
            sae[y[i], x[i]] = np.exp(-(t_ref-ts[i]) / tau)
        else:
            sae[y[i], x[i]] = -np.exp(-(t_ref-ts[i]) / tau)
        
        ## none-polarity Timesurface
        # sae[y[i], x[i]] = np.exp(-(t_ref-ts[i]) / tau)

    return sae

def get_timesurface_from_events(x:np.ndarray,y:np.ndarray,ts:np.ndarray,p:np.ndarray,img_size = (260,346),tau = 50e-3):
    
    x = x.tolist()
    y = y.tolist()
    ts = (ts*10e-6).tolist()
    p = p.tolist()
    
    img_size = img_size

    # parameters for Time Surface
    t_ref = ts[-1]      # 'current' time
    tau = 50e-3         # 50ms

    sae = np.zeros(img_size, np.float32)
    # calculate timesurface using expotential decay
    for i in range(len(ts)):
        if (p[i] > 0):
            sae[y[i], x[i]] = np.exp(-(t_ref-ts[i]) / tau)
        else:
            sae[y[i], x[i]] = -np.exp(-(t_ref-ts[i]) / tau)
        
        ## none-polarity Timesurface
        # sae[y[i], x[i]] = np.exp(-(t_ref-ts[i]) / tau)

    return sae

def sample_desc_from_points(coarse_desc, pts):
    """
        pts should be in format like [2,n]
    """
    # --- Process descriptor.
    H, W = coarse_desc.shape[2]*8, coarse_desc.shape[3]*8
    D = coarse_desc.shape[1]
    if pts.shape[1] == 0:
        desc = np.zeros((D, 0))
    else:
        # Interpolate into descriptor map using 2D point locations.
        samp_pts = torch.from_numpy(pts[:2, :].copy()).to(torch.float)
        samp_pts[0, :] = (samp_pts[0, :] / (float(W) / 2.)) - 1.
        samp_pts[1, :] = (samp_pts[1, :] / (float(H) / 2.)) - 1.
        samp_pts = samp_pts.transpose(0, 1).contiguous()
        samp_pts = samp_pts.view(1, 1, -1, 2)
        samp_pts = samp_pts.float()
        samp_pts = samp_pts.cuda()
        desc = torch.nn.functional.grid_sample(coarse_desc, samp_pts, align_corners=True)
        desc = desc.data.cpu().numpy().reshape(D, -1)
        desc /= np.linalg.norm(desc, axis=0)[np.newaxis, :]
    return desc

def warp_keypoints(keypoints, H):
    num_points = keypoints.shape[0]
    homogeneous_points = np.concatenate([keypoints, np.ones((num_points, 1))],
                                        axis=1)
    warped_points = np.dot(homogeneous_points, np.transpose(H))
    return warped_points[:, :2] / warped_points[:, 2:]