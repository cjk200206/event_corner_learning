import torch
import torch.nn.functional as F
import torchmetrics


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
    focal_loss = FocalLoss()
    for i in range(batch_size):
        loss += focal_loss(softmaxed_vox[i].view(65,-1).permute(1,0), label_3d[i].view(-1))
    loss /= batch_size

    # 计算预测的类别
    predicted_classes = torch.argmax(softmaxed_vox, dim=1)
    # 计算真实的类别
    true_classes = label_3d
    
    acc = accuracy(predicted_classes.cpu().view(-1), true_classes.cpu().view(-1))
    
    return loss,acc.item()


