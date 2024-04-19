import torch
import torch.nn.functional as F
import torchmetrics

def normPts(pts, shape):
    """
    normalize pts to [-1, 1]
    :param pts:
        tensor (y, x)
    :param shape:
        tensor shape (y, x)
    :return:
    """
    pts = pts/shape*2 - 1
    return pts

def denormPts(pts, shape):
    """
    denormalize pts back to H, W
    :param pts:
        tensor (y, x)
    :param shape:
        numpy (y, x)
    :return:
    """
    pts = (pts+1)*shape/2
    return pts


class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=2, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, input, target):
        ce_loss = F.cross_entropy(input, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            if self.alpha.type() != input.type():
                self.alpha = self.alpha.type_as(input)
            focal_loss = self.alpha[target] * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def cross_entropy_loss_and_accuracy(prediction, target):
    cross_entropy_loss = torch.nn.CrossEntropyLoss()
    loss = cross_entropy_loss(prediction, target)
    accuracy = (prediction.argmax(1) == target).float().mean()
    return loss, accuracy

def softmax_vox(vox):
    """
    对vox的每一层进行softmax操作
    :param vox: 输入的vox，形状为 (batch_size, num_time_bins, height, width)
    :return: 经过softmax处理后的vox (batch_size, num_time_bins, height*width)
    """
    softmax_vox = vox.view(vox.size(0), vox.size(1), -1)
    softmax_vox = F.softmax(softmax_vox, dim=2)  # 对每一层进行softmax操作
    return softmax_vox

def compute_vox_loss(vox, heatmap):
    """
    计算损失函数
    :param softmax_vox: 经过softmax处理后的vox，形状为 (batch_size, num_time_bins, height, width)
    :param heatmap: heatmap，与vox尺寸相同，形状为 (batch_size, num_time_bins, height, width)
    :return: 损失值
    """
    # 将softmax_vox和heatmap展平为二维张量，以便与交叉熵损失函数进行计算
    batch_size = vox.size(0)
    num_time_bins = vox.size(1)

    softmaxed_vox = softmax_vox(vox)
    heatmap_flat = heatmap.view(batch_size, num_time_bins, -1)
    
    # 计算交叉熵损失
    loss = 0
    acc = 0
    accuracy = torchmetrics.Accuracy()
    for i in range(batch_size):
        loss += F.cross_entropy(softmaxed_vox[i], heatmap_flat[i].argmax(dim=1))
    loss /= batch_size

    # 计算预测的类别
    predicted_classes = torch.argmax(softmaxed_vox, dim=2)
    # 计算真实的类别
    true_classes = torch.argmax(heatmap_flat, dim=2)
    
    acc = accuracy(predicted_classes.cpu().view(-1), true_classes.cpu().view(-1))
    
    return loss,acc.item()

def compute_superpoint_argmax_loss(vox, label_3d):
    """
    计算损失函数，标签是argmax过后的idx形状
    :param vox: superpoint网络输出，形状为 (batch_size, 64+1, height/8, width/8)
    :param heatmap: heatmap，与vox尺寸相同，形状为 (batch_size, 64+1, height/8, width/8)
    :return: 损失值
    """

    # 将softmax_vox和heatmap展平为二维张量，以便与交叉熵损失函数进行计算
    batch_size = vox.size(0)


    softmaxed_vox = F.softmax(vox,dim=1)
    label_3d = torch.argmax(label_3d,dim=1)

    # 计算交叉熵损失
    loss = 0
    acc = 0
    accuracy = torchmetrics.Accuracy()
    focal_loss = FocalLoss()
    cse_loss = torch.nn.CrossEntropyLoss()
    for i in range(batch_size):
        # loss += cse_loss(softmaxed_vox[i].cpu().view(65,-1).permute(1,0), label_3d[i].cpu().view(-1))
        loss += focal_loss(softmaxed_vox[i].view(65,-1).permute(1,0).cpu(), label_3d[i].view(-1).cpu())
    loss /= batch_size

    # 计算预测的类别
    predicted_classes = torch.argmax(softmaxed_vox, dim=1)
    # 计算真实的类别
    true_classes = label_3d
    
    acc = accuracy(predicted_classes.cpu().view(-1), true_classes.cpu().view(-1))
    
    return loss,acc.item()

def compute_superpoint_loss(vox, label_3d):
    """
    计算损失函数
    :param vox: superpoint网络输出，形状为 (batch_size, 64+1, height/8, width/8)
    :param heatmap: heatmap，与vox尺寸相同，形状为 (batch_size, 64+1, height/8, width/8)
    :return: 损失值
    """

    # 将softmax_vox和heatmap展平为二维张量，以便与交叉熵损失函数进行计算
    batch_size = vox.size(0)
    softmaxed_vox = F.softmax(vox,dim=1)

    # 计算交叉熵损失
    loss = 0
    acc = 0
    accuracy = torchmetrics.Accuracy()
    bce_loss = torch.nn.BCELoss(reduction="none")
    shape = vox.shape[2]*vox.shape[3]
    focal_loss = FocalLoss()
    for i in range(batch_size):
        loss += bce_loss(softmaxed_vox[i].cpu(), label_3d[i].cpu()).sum()/shape ###????????相当于是所有3d格子求和，再对特征图的面积求平均
        # loss += focal_loss(softmaxed_vox[i].cpu().view(65,-1).permute(1,0), label_3d[i].cpu().view(65,-1).permute(1,0))
    loss /= batch_size

    # 计算预测的类别
    predicted_classes = torch.argmax(softmaxed_vox, dim=1)
    # 计算真实的类别
    true_classes = torch.argmax(label_3d,dim=1)
    acc = accuracy(predicted_classes.cpu().view(-1), true_classes.cpu().view(-1))
    
    return loss,acc.item()

def descriptor_loss(descriptors, descriptors_warped, homographies, mask_valid=None, 
                    cell_size=8, lamda_d=250, device='cpu', descriptor_dist=4, **config):

    '''
    Compute descriptor loss from descriptors_warped and given homographies

    :param descriptors:
        Output from descriptor head
        tensor [batch_size, descriptors, Hc, Wc]
    :param descriptors_warped:
        Output from descriptor head of warped image
        tensor [batch_size, descriptors, Hc, Wc]
    :param homographies:
        known homographies
    :param cell_size:
        8
    :param device:
        gpu or cpu
    :param config:
    :return:
        loss, and other tensors for visualization
    '''

    # put to gpu
    homographies = homographies.to(device)
    # config
    from utils.utils import warp_points
    lamda_d = lamda_d # 250
    margin_pos = 1
    margin_neg = 0.2
    batch_size, Hc, Wc = descriptors.shape[0], descriptors.shape[2], descriptors.shape[3]
    #####
    # H, W = Hc.numpy().astype(int) * cell_size, Wc.numpy().astype(int) * cell_size
    H, W = Hc * cell_size, Wc * cell_size
    #####
    with torch.no_grad():
        # shape = torch.tensor(list(descriptors.shape[2:]))*torch.tensor([cell_size, cell_size]).type(torch.FloatTensor).to(device)
        shape = torch.tensor([H, W]).type(torch.FloatTensor).to(device)
        # compute the center pixel of every cell in the image

        coor_cells = torch.stack(torch.meshgrid(torch.arange(Hc), torch.arange(Wc)), dim=2)
        coor_cells = coor_cells.type(torch.FloatTensor).to(device)
        coor_cells = coor_cells * cell_size + cell_size // 2
        ## coord_cells is now a grid containing the coordinates of the Hc x Wc
        ## center pixels of the 8x8 cells of the image

        # coor_cells = coor_cells.view([-1, Hc, Wc, 1, 1, 2])
        coor_cells = coor_cells.view([-1, 1, 1, Hc, Wc, 2])  # be careful of the order
        # warped_coor_cells = warp_points(coor_cells.view([-1, 2]), homographies, device)
        warped_coor_cells = normPts(coor_cells.view([-1, 2]), shape)
        warped_coor_cells = torch.stack((warped_coor_cells[:,1], warped_coor_cells[:,0]), dim=1) # (y, x) to (x, y)
        warped_coor_cells = warp_points(warped_coor_cells, homographies, device)

        warped_coor_cells = torch.stack((warped_coor_cells[:, :, 1], warped_coor_cells[:, :, 0]), dim=2)  # (batch, x, y) to (batch, y, x)

        shape_cell = torch.tensor([H//cell_size, W//cell_size]).type(torch.FloatTensor).to(device)
        # warped_coor_mask = denormPts(warped_coor_cells, shape_cell)

        warped_coor_cells = denormPts(warped_coor_cells, shape)
        # warped_coor_cells = warped_coor_cells.view([-1, 1, 1, Hc, Wc, 2])
        warped_coor_cells = warped_coor_cells.view([-1, Hc, Wc, 1, 1, 2])
    #     print("warped_coor_cells: ", warped_coor_cells.shape)
        # compute the pairwise distance
        cell_distances = coor_cells - warped_coor_cells
        cell_distances = torch.norm(cell_distances, dim=-1)
        ##### check
    #     print("descriptor_dist: ", descriptor_dist)
        mask = cell_distances <= descriptor_dist # 0.5 # trick

        mask = mask.type(torch.FloatTensor).to(device)

    # compute the pairwise dot product between descriptors: d^t * d
    descriptors = descriptors.transpose(1, 2).transpose(2, 3)
    descriptors = descriptors.view((batch_size, Hc, Wc, 1, 1, -1))
    descriptors_warped = descriptors_warped.transpose(1, 2).transpose(2, 3)
    descriptors_warped = descriptors_warped.view((batch_size, 1, 1, Hc, Wc, -1))
    dot_product_desc = descriptors * descriptors_warped
    dot_product_desc = dot_product_desc.sum(dim=-1)
    ## dot_product_desc.shape = [batch_size, Hc, Wc, Hc, Wc, desc_len]

    # hinge loss
    positive_dist = torch.max(margin_pos - dot_product_desc, torch.tensor(0.).to(device))
    # positive_dist[positive_dist < 0] = 0
    negative_dist = torch.max(dot_product_desc - margin_neg, torch.tensor(0.).to(device))
    # negative_dist[neative_dist < 0] = 0
    # sum of the dimension

    if mask_valid is None:
        # mask_valid = torch.ones_like(mask)
        mask_valid = torch.ones(batch_size, 1, Hc*cell_size, Wc*cell_size)
    mask_valid = mask_valid.view(batch_size, 1, 1, mask_valid.shape[2], mask_valid.shape[3])

    loss_desc = lamda_d * mask * positive_dist + (1 - mask) * negative_dist
    loss_desc = loss_desc * mask_valid
        # mask_validg = torch.ones_like(mask)
    ##### bug in normalization
    normalization = (batch_size * (mask_valid.sum()+1) * Hc * Wc)
    pos_sum = (lamda_d * mask * positive_dist/normalization).sum()
    neg_sum = ((1 - mask) * negative_dist/normalization).sum()
    loss_desc = loss_desc.sum() / normalization
    # loss_desc = loss_desc.sum() / (batch_size * Hc * Wc)
    # return loss_desc, mask, mask_valid, positive_dist, negative_dist
    return loss_desc, mask, pos_sum, neg_sum


