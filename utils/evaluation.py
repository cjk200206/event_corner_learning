import torch
import torch.nn.functional as F
import torchmetrics

def compute_detection_correctness(flattened_semi, label_2d,threshold = 6):
    """
    计算损失函数
    :param flatten_semi: superpoint网络输出，形状为 (height, width)
    :param label_2d: heatmap，与flatten_semi尺寸相同，形状为 (height, width),为计算匹配的数量
    :return: 损失值
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

def compute_detection_repeatability(flattened_semi, inv_flattened_semi_transfromed,threshold = 9):
    """
    计算损失函数
    :param flatten_semi: superpoint网络输出，形状为 (height, width)
    :param inv_flattened_semi_transfromed: 经过变换后superpoint检测的点再逆变换回来，，形状为 (height, width)
    :return: 损失值
    """
    matches_1,pred_1,gt_1,dist_1 = compute_detection_correctness(flattened_semi,inv_flattened_semi_transfromed,threshold)
    matches_2,pred_2,gt_2,dist_2 = compute_detection_correctness(inv_flattened_semi_transfromed,flattened_semi,threshold)

    return len(matches_1)+len(matches_2) , len(gt_1)+len(gt_2)