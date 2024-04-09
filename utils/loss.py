import torch
import torch.nn.functional as F
import torchmetrics


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

# # 示例数据
# batch_size = 3
# vox = torch.rand(batch_size, 5, 10, 10)  # 生成一个形状为 (batch_size, 5, 10, 10) 的示例vox
# heatmap = torch.randint(0, 10, (batch_size, 5, 10, 10))  # 生成一个形状为 (batch_size, 5, 10, 10) 的示例heatmap

# # 计算损失
# loss,acc = compute_vox_loss(vox, heatmap)

# print("损失值:", loss.item(),"\nacc:", acc)
