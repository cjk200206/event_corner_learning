from os.path import dirname
import argparse
import torch
import torchvision
import tqdm
import os
import sys
import numpy as np
import torch.nn.functional as F
import cv2
sys.path.append("../")

from numpy.linalg import inv
from utils.dataset import Syn_Superpoint
from torch.utils.data import DataLoader
from torch.utils.data import default_collate
from utils.models_superpoint import EventCornerSuperpoint
from utils.loss import compute_vox_loss,compute_superpoint_loss
from utils.utils.utils import getLabels,heatmap_nms,inv_warp_image,inv_warp_image_batch
from utils.utils.transformation import random_affine_transform
from utils.utils.homographies import sample_homography_np
from utils.utils.d2s import flatten_64to1
from torch.utils.tensorboard import SummaryWriter

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

def FLAGS():
    parser = argparse.ArgumentParser("""Train classifier using a learnt quantization layer.""")

    # test dataset
    parser.add_argument("--test_dataset", default="/home/cjk2002/code/syn2e/datasets/test")
    parser.add_argument("--checkpoint", default="log/superpoint_ckpt/model_best.pth")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", type=bool, default=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--output_path", default="/home/cjk2002/code/event_code/event_corner_learning/log/event_superpoint")


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
    test_dataset = Syn_Superpoint(flags.test_dataset,num_time_bins=3,grid_size=(260,346),event_crop=False,test=True)
    # construct loader, handles data streaming to gpu
    test_loader = DataLoader(test_dataset,batch_size=flags.batch_size,
                               pin_memory=flags.pin_memory,collate_fn=default_collate)
  
    # model, and put to device
    model = EventCornerSuperpoint(voxel_dimension=(2,260,346))
    ckpt = torch.load(flags.checkpoint)
    model.load_state_dict(ckpt["state_dict"])
    # model.backbone.load_state_dict(ckpt,strict=False)
    model = model.to(flags.device)

    model = model.eval()
    sum_accuracy = 0
    sum_loss = 0
    img_num = 0

    #建文件夹
    os.makedirs(flags.output_path + "/heatmap",exist_ok=True)
    os.makedirs(flags.output_path + "/label",exist_ok=True)
    os.makedirs(flags.output_path + "/input_img",exist_ok=True) 
    os.makedirs(flags.output_path + "/input_img_cropped",exist_ok=True) 
    os.makedirs(flags.output_path + "/heatmap_transformed",exist_ok=True)
    os.makedirs(flags.output_path + "/label_transformed",exist_ok=True)
    os.makedirs(flags.output_path + "/input_img_transformed",exist_ok=True) 
    os.makedirs(flags.output_path + "/input_img_cropped_transformed",exist_ok=True) 
        
    print("Test step")
        
    for event_vox,label_vox,heatmap in tqdm.tqdm(test_loader):
        # 裁剪标签
        heatmap = crop_and_resize_to_resolution(heatmap)
        # 把数据转到gpu
        event_vox = event_vox.to(flags.device)
        label_vox = label_vox.to(flags.device)
        heatmap = heatmap.to(flags.device)
        #取出单通道的vox
        label_2d = label_vox[:,0,:,:]
        input_vox = event_vox[:,0,:,:]
        #做仿射变换
        vox_transform,_ = random_affine_transform(torch.cat((label_2d.unsqueeze(1),input_vox.unsqueeze(1)),dim=1))
        label_2d_transformed = vox_transform[:,0]
        input_vox_transformed =vox_transform[:,1]

        #HA变换测试
        homography = sample_homography_np(np.array([2, 2]))
        ##### use inverse from the sample homography
        homography = inv(homography)
        ######
        inv_homography = inv(homography)
        inv_homography = torch.tensor(inv_homography).to(torch.float32)
        homography = torch.tensor(homography).to(torch.float32).cuda()
        # images
        warped_img = inv_warp_image_batch(input_vox[0],homography,device=input_vox[0].device)
        

        # 裁剪图像
        event_vox_cropped = crop_and_resize_to_resolution(input_vox.unsqueeze(1))
        event_vox_cropped_transformed = crop_and_resize_to_resolution(input_vox_transformed.unsqueeze(1))
        label_2d = crop_and_resize_to_resolution(label_2d.unsqueeze(1))
        label_2d_transformed = crop_and_resize_to_resolution(label_2d_transformed.unsqueeze(1))
        #转换标签
        label_3d = getLabels(label_2d,8)
        label_3d_transform = getLabels(label_2d_transformed,8)


        with torch.no_grad():
            semi, _ = model(input_vox.unsqueeze(1))
            semi_transform, _ = model(input_vox_transformed.unsqueeze(1))
        
            loss_a, accuracy_a = compute_superpoint_loss(semi, label_3d)
            loss_b, accuracy_b = compute_superpoint_loss(semi_transform, label_3d_transform)
            loss = loss_a+loss_b
            accuracy = (accuracy_a+accuracy_b)/2

        semi = F.softmax(semi,dim=1)
        semi_transform = F.softmax(semi_transform, dim=1)
        #输出热力图
        flatten_semi = flatten_64to1(semi[:,:-1,:,:])
        flatten_semi_transformed = flatten_64to1(semi_transform[:,:-1,:,:])

        for input_img,nms_heatmap,cropped_img,label_heatmap,\
            input_img_transformed,nms_heatmap_transformed,cropped_img_transformed,label_heatmap_transformed \
                in zip(input_vox,flatten_semi,event_vox_cropped.squeeze(1),label_2d.squeeze(1),\
                       input_vox_transformed,flatten_semi_transformed,event_vox_cropped_transformed.squeeze(1),label_2d_transformed.squeeze(1)):
            nms_semi = heatmap_nms(nms_heatmap.cpu(),conf_thresh=0.020)
            nms_semi_transformed = heatmap_nms(nms_heatmap_transformed.cpu(),conf_thresh=0.020)
            
            cv2.imwrite("{}/heatmap/{:08d}.jpg".format(flags.output_path,img_num),nms_semi*255)
            cv2.imwrite("{}/label/{:08d}.jpg".format(flags.output_path,img_num),label_heatmap.cpu().numpy()*255)
            cv2.imwrite("{}/input_img/{:08d}.jpg".format(flags.output_path,img_num),input_img.cpu().numpy()*255)
            cv2.imwrite("{}/input_img_cropped/{:08d}.jpg".format(flags.output_path,img_num),cropped_img.cpu().numpy()*255)
            #变换后的
            cv2.imwrite("{}/heatmap_transformed/{:08d}.jpg".format(flags.output_path,img_num),nms_semi_transformed*255)
            cv2.imwrite("{}/label_transformed/{:08d}.jpg".format(flags.output_path,img_num),label_heatmap_transformed.cpu().numpy()*255)
            cv2.imwrite("{}/input_img_transformed/{:08d}.jpg".format(flags.output_path,img_num),input_img_transformed.cpu().numpy()*255)
            cv2.imwrite("{}/input_img_cropped_transformed/{:08d}.jpg".format(flags.output_path,img_num),cropped_img_transformed.cpu().numpy()*255)
            img_num += 1

        sum_accuracy += accuracy
        sum_loss += loss

    test_loss = sum_loss.item() / len(test_loader)
    test_accuracy = sum_accuracy / len(test_loader)

    print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

        


