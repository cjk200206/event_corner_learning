import torch
import torchvision.transforms.functional as TF

def random_transform(image_tensor):
    # 随机旋转角度在(-45, 45)之间
    angle = torch.randint(-45, 45, (1,))
    # 随机水平和垂直平移在(-30, 30)之间
    translate = (torch.randint(-30, 30, (1,)).item(), torch.randint(-30, 30, (1,)).item())
    # 随机缩放在(0.8, 1.2)之间
    scale = torch.rand(1).item() * 0.4 + 0.8
    # 随机水平和垂直形变在(-0.2, 0.2)之间
    shear = torch.rand(1).item() * 0.4 - 0.2
    # 随机水平翻转
    horizontal_flip = torch.rand(1).item() > 0.5
    # 随机垂直翻转
    vertical_flip = torch.rand(1).item() > 0.5

    # 对每个通道的图像进行随机变换
    transformed_images = []
    for i in range(image_tensor.size(0)):
        image = image_tensor[i]

        # 随机旋转
        image = TF.rotate(image, angle.item())

        # 随机平移
        image = TF.affine(image, angle=0, translate=translate, scale=scale, shear=shear)

        # 随机水平翻转
        if horizontal_flip:
            image = TF.hflip(image)

        # 随机垂直翻转
        if vertical_flip:
            image = TF.vflip(image)

        transformed_images.append(image)

    # 将变换后的图像堆叠成张量
    transformed_image_tensor = torch.stack(transformed_images, dim=0)

    return transformed_image_tensor
