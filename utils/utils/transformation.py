import torch
import torch.nn.functional as F

def random_affine_transform(vox):
    # 随机生成旋转角度在(-45, 45)之间
    angle = torch.randint(-45, 45, (vox.size(0),))
    # 随机水平和垂直平移在(-30/260, 30/260)之间
    translate = torch.randint(-30, 30, (vox.size(0), 2))/vox.size(2)
    # 随机缩放在(0.8, 1.2)之间
    scale = torch.rand(vox.size(0)) * 0.4 + 0.8

    # 构建旋转矩阵
    theta = torch.zeros(vox.size(0),2, 3)
    theta[:, 0, 0] = torch.cos(angle * torch.pi / 180.0)
    theta[:, 0, 1] = -torch.sin(angle * torch.pi / 180.0)
    theta[:, 1, 0] = torch.sin(angle * torch.pi / 180.0)
    theta[:, 1, 1] = torch.cos(angle * torch.pi / 180.0)

    # 添加平移和缩放
    theta[:, :, 2] = translate.float()
    theta[:, 0, 0] *= scale
    theta[:, 1, 1] *= scale

    # 将变换矩阵记录下来
    # print("Affine transformation matrices:")
    # print(theta)

    # 创建仿射网格
    grid = F.affine_grid(theta, vox.size()).to(vox.device)  # 创建仿射网格

    # 对vox进行仿射变换
    transformed_vox = F.grid_sample(vox, grid)

    return transformed_vox, theta

# # 假设 vox 是一个形状为 (n, 260, 346) 的张量
# vox = torch.randn(2, 2, 260, 346)  # 生成一个示例张量

# # 执行仿射变换
# transformed_vox, theta = random_affine_transform(vox)

# # 输出变换后的张量形状
# print("Transformed vox shape:", transformed_vox.shape)


