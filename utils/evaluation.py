import torch
import torch.nn.functional as F
import torchmetrics
import cv2
import numpy as np

from utils.utils.utils import warp_points,sample_desc_from_points,inv_warp_image,heatmap_nms,warp_keypoints

def compute_detection_correctness(flattened_semi, label_2d,threshold = 6):
    """
    计算准确度
    :param flatten_semi: superpoint网络输出，形状为 (height, width)
    :param label_2d: heatmap，与flatten_semi尺寸相同，形状为 (height, width),为计算匹配的数量
    :return: 正确的匹配
    """


    # 获取特征点的坐标
    coords1 = torch.nonzero(flattened_semi)
    coords2 = torch.nonzero(label_2d)

    # 计算每个特征点之间的距离
    distances = torch.cdist(coords2.float(), coords1.float())

    # 设置阈值
    threshold = threshold  # 设置为你认为合适的阈值

    # 对于每个特征点，在第一个张量中找到与之最近的点，并判断是否满足阈值
    matches = []
    min_dists = []
    if distances.shape[0] != 0 and distances.shape[1] != 0:
        for i in range(len(coords2)):
            min_dist, min_idx = torch.min(distances[i], dim=0)
            if min_dist <= threshold:
                matches.append((coords2[i], coords1[min_idx]))
                min_dists.append(min_dist)
    sum_dist = sum(min_dists)
    
    return matches,coords1,coords2,sum_dist

def compute_detection_repeatability(flattened_semi, inv_flattened_semi_transfromed,threshold = 6):
    """
    计算重复度
    :param flatten_semi: superpoint网络输出，形状为 (height, width)
    :param inv_flattened_semi_transfromed: 经过变换后superpoint检测的点再逆变换回来，形状为 (height, width)
    :return: 匹配数量和gt数量
    """
    matches_1,pred_1,gt_1,dist_1 = compute_detection_correctness(flattened_semi,inv_flattened_semi_transfromed,threshold)
    matches_2,pred_2,gt_2,dist_2 = compute_detection_correctness(inv_flattened_semi_transfromed,flattened_semi,threshold)

    return len(matches_1)+len(matches_2) , len(gt_1)+len(gt_2)

def nn_match_two_way(desc1, desc2, nn_thresh):
    """
    Performs two-way nearest neighbor matching of two sets of descriptors, such
    that the NN match from descriptor A->B must equal the NN match from B->A.

    Inputs:
        desc1 - NxM numpy matrix of N corresponding M-dimensional descriptors.
        desc2 - NxM numpy matrix of N corresponding M-dimensional descriptors.
        nn_thresh - Optional descriptor distance below which is a good match.

    Returns:
        matches - 3xL numpy array, of L matches, where L <= N and each column i is
                a match of two descriptors, d_i in image 1 and d_j' in image 2:
                [d_i index, d_j' index, match_score]^T
    """
    # assert desc1.shape[0] == desc2.shape[0]
    if desc1.shape[1] == 0 or desc2.shape[1] == 0:
        return np.zeros((3, 0))
    if nn_thresh < 0.0:
        raise ValueError('\'nn_thresh\' should be non-negative')
    # Compute L2 distance. Easy since vectors are unit normalized.
    dmat = np.dot(desc1.T, desc2)
    dmat = np.sqrt(2 - 2 * np.clip(dmat, -1, 1))
    # Get NN indices and scores.
    idx = np.argmin(dmat, axis=1)
    scores = dmat[np.arange(dmat.shape[0]), idx]
    # Threshold the NN matches.
    keep = scores < nn_thresh
    # Check if nearest neighbor goes both directions and keep those.
    idx2 = np.argmin(dmat, axis=0)
    keep_bi = np.arange(len(idx)) == idx2[idx]
    keep = np.logical_and(keep, keep_bi)
    idx = idx[keep]
    scores = scores[keep]
    # Get the surviving point indices.
    m_idx1 = np.arange(desc1.shape[1])[keep]
    m_idx2 = idx
    # Populate the final 3xN match data structure.
    matches = np.zeros((3, int(keep.sum())))
    matches[0, :] = m_idx1
    matches[1, :] = m_idx2
    matches[2, :] = scores
    return matches

def compute_descriptor_Nearest_neighbour_mAP(desc_raw,desc_transformed_raw,flattened_semi, inv_flattened_semi_transfromed, homography,inv_homography,threshold = 6):
    matches_1,pred_1,gt_1,dist_1 = compute_detection_correctness(flattened_semi,inv_flattened_semi_transfromed,threshold)
    # matches_2,pred_2,gt_2,dist_2 = compute_detection_correctness(inv_flattened_semi_transfromed,flattened_semi,threshold)
    H = 224
    W = 224

    #提取HA后的点的位置
    warped_semi = inv_warp_image(inv_flattened_semi_transfromed.cpu(),homography.cpu())
    warped_semi = heatmap_nms(warped_semi.cpu(),conf_thresh=0.020)
    gt_1_ha = torch.nonzero(warped_semi)
    
    
    desc = sample_desc_from_points(desc_raw.unsqueeze(0),pred_1.T.numpy())
    desc_ha = sample_desc_from_points(desc_transformed_raw.unsqueeze(0),gt_1_ha.T.numpy())

    norm1 = np.linalg.norm(desc,axis=0)
    norm2 = np.linalg.norm(desc_ha,axis=0)

    matches_desc = nn_match_two_way(desc/norm1,desc_ha/norm2,0.7)

    desc_match_number = matches_desc.shape[0]
    detector_match_number = len(matches_1)

    # # 获取点的百分比位置
    # sample_pred_1 = pred_1.clone().to(torch.float)
    # sample_pred_1[:,0] = (sample_pred_1[:,0]/(float(W)/2.)) - 1.
    # sample_pred_1[:,1] = (sample_pred_1[:,1]/(float(H)/2.)) - 1.

    # sample_gt_1 = gt_1.clone().to(torch.float)
    # sample_gt_1[:,0] = (sample_gt_1[:,0]/(float(W)/2.)) - 1.
    # sample_gt_1[:,1] = (sample_gt_1[:,1]/(float(H)/2.)) - 1.

    # # 将角点还原到HA图中
    # inv_sample_gt_1 = warp_keypoints(sample_gt_1.cpu(),homography.cpu())
    # inv_gt_1 = inv_sample_gt_1.copy()
    # inv_gt_1[:,0] = (inv_sample_gt_1[:,0]+1)*(float(W)/2.)
    # inv_gt_1[:,1] = (inv_sample_gt_1[:,1]+1)*(float(H)/2.)
    
    # desc_inv = sample_desc_from_points(desc_transformed_raw.unsqueeze(0),inv_gt_1.T)

    return desc_match_number,detector_match_number