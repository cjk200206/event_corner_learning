from os.path import dirname
import argparse
import torch
import torchvision
import tqdm
import os
import numpy as np
import torch.nn.functional as F
import cv2

from utils.dataset import Syn_Superpoint
from torch.utils.data import DataLoader
from torch.utils.data import default_collate
from utils.models_superpoint import EventCornerSuperpoint
from utils.loss import compute_vox_loss,compute_superpoint_loss
from utils.utils.d2s import DepthToSpace,SpaceToDepth
from torch.utils.tensorboard import SummaryWriter

#做nms
def getPtsFromHeatmap(heatmap, conf_thresh, nms_dist):
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
    pts, _ = nms_fast(pts, H, W, dist_thresh=nms_dist)  # Apply NMS.
    inds = np.argsort(pts[2, :])
    pts = pts[:, inds[::-1]]  # Sort by confidence.
    # Remove points along border.
    bord = border_remove
    toremoveW = np.logical_or(pts[0, :] < bord, pts[0, :] >= (W - bord))
    toremoveH = np.logical_or(pts[1, :] < bord, pts[1, :] >= (H - bord))
    toremove = np.logical_or(toremoveW, toremoveH)
    pts = pts[:, ~toremove]
    return pts

def nms_fast(in_corners, H, W, dist_thresh):
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

def heatmap_nms(heatmap, nms_dist=4, conf_thresh=0.021):
    """
    input:
        heatmap: np [(1), H, W]
    """
    # nms_dist = self.config['model']['nms']
    # conf_thresh = self.config['model']['detection_threshold']
    heatmap = heatmap.squeeze()
    # print("heatmap: ", heatmap.shape)
    pts_nms = getPtsFromHeatmap(heatmap, conf_thresh, nms_dist)
    semi_thd_nms_sample = np.zeros_like(heatmap)
    semi_thd_nms_sample[
        pts_nms[1, :].astype(np.int), pts_nms[0, :].astype(np.int)
    ] = 255
    return semi_thd_nms_sample

def crop_and_resize_to_resolution(x, output_resolution=(224, 224)):
    B, C, H, W = x.shape
    if H > W:
        h = H // 2
        x = x[:, :, h - W // 2:h + W // 2, :]
    else:
        h = W // 2
        x = x[:, :, :, h - H // 2:h + H // 2]

    x = F.interpolate(x, size=output_resolution)

    return x

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
    labels = torch.argmax(labels, dim=1)
    return labels

#superpoint恢复到热力图
def flatten_64to1(semi, cell_size=8):
    """
    input: 
        semi: tensor[batch, cell_size*cell_size, Hc, Wc]
        (Hc = H/8)
    outpus:
        heatmap: tensor[batch, 1, H, W]
    """

    depth2space = DepthToSpace(cell_size)
    heatmap = depth2space(semi)
    return heatmap

#superpoint的标签获取
def getLabels(labels_2D, cell_size, device="cpu"):
    """
    # transform 2D labels to 3D shape for training
    :param labels_2D:
    :param cell_size:
    :param device:
    :return:
    """
    labels3D_flattened = labels2Dto3D_flattened(
        labels_2D.to(device), cell_size=cell_size
    )
    labels3D_in_loss = labels3D_flattened
    return labels3D_in_loss

def FLAGS():
    parser = argparse.ArgumentParser("""Train classifier using a learnt quantization layer.""")

    # test dataset
    parser.add_argument("--test_dataset", default="/home/cjk2002/code/syn2e/datasets/test")
    parser.add_argument("--checkpoint", default="log/superpoint_ckpt/model_best.pth")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", type=bool, default=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--output_path", default="/home/cjk2002/code/event_code/event_corner_learning/log/pics")


    flags = parser.parse_args()

    assert os.path.isdir(dirname(flags.checkpoint)), f"Checkpoint{flags.checkpoint} not found."
    assert os.path.isdir(flags.test_dataset), f"Test dataset directory {flags.test_dataset} not found."

    print(f"----------------------------\n"
          f"Starting testing with \n"
          f"checkpoint: {flags.checkpoint}\n"
          f"test_dataset: {flags.test_dataset}\n"
          f"batch_size: {flags.batch_size}\n"
          f"device: {flags.device}\n"
          f"----------------------------")

    return flags


if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = '1' #设置显卡可见

    flags = FLAGS()
    # datasets, add augmentation to training set
    test_dataset = Syn_Superpoint(flags.test_dataset,num_time_bins=3,grid_size=(260,346))
    # construct loader, handles data streaming to gpu
    test_loader = DataLoader(test_dataset,batch_size=flags.batch_size,
                               pin_memory=flags.pin_memory,collate_fn=default_collate)
  
    # model, and put to device
    model = EventCornerSuperpoint(voxel_dimension=(2,260,346))
    ckpt = torch.load(flags.checkpoint)
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(flags.device)

    model = model.eval()
    sum_accuracy = 0
    sum_loss = 0
    img_num = 0

    os.makedirs(flags.output_path,exist_ok=True)
        
    print("Test step")
        
    for event_vox,label_vox,heatmap in tqdm.tqdm(test_loader):
        
        heatmap = crop_and_resize_to_resolution(heatmap)
        label_vox = crop_and_resize_to_resolution(label_vox)
        # 把数据转到gpu
        event_vox = event_vox.to(flags.device)
        label_vox = label_vox.to(flags.device)
        heatmap = heatmap.to(flags.device)
        #转换标签
        label_3d = getLabels(label_vox[:,0,:,:].unsqueeze(1),8)
        with torch.no_grad():
            semi, _ = model(event_vox[:,0,:,:].unsqueeze(1))
            loss, accuracy = compute_superpoint_loss(semi, label_3d)

        semi = F.softmax(semi,dim=1)
        #输出热力图
        flatten_semi = flatten_64to1(semi[:,:-1,:,:])
    
        for nms_heatmap in flatten_semi:
            nms_semi = heatmap_nms(nms_heatmap.cpu())
            cv2.imwrite("{}/{:08d}.jpg".format(flags.output_path,img_num),nms_semi)
            img_num += 1

        sum_accuracy += accuracy
        sum_loss += loss

    test_loss = sum_loss.item() / len(test_loader)
    test_accuracy = sum_accuracy / len(test_loader)

    print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

        


