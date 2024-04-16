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

from utils.dataset import Syn_Superpoint
from torch.utils.data import DataLoader
from torch.utils.data import default_collate
from utils.models_superpoint import EventCornerSuperpoint
from utils.loss import compute_vox_loss,compute_superpoint_loss
from utils.utils.utils import getLabels,heatmap_nms
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
    test_dataset = Syn_Superpoint(flags.test_dataset,num_time_bins=3,grid_size=(260,346),event_crop=False)
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
        
    print("Test step")
        
    for event_vox,label_vox,heatmap in tqdm.tqdm(test_loader):
        event_vox_cropped = crop_and_resize_to_resolution(event_vox)
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

        label_bin = torch.zeros_like(label_vox[0,0]).cpu()
        heatmap_bin = torch.zeros_like(flatten_semi[0,0]).cpu()
        img_bin = torch.zeros_like(event_vox[0,0,:,:]).cpu()
        cropped_img_bin = torch.zeros_like(event_vox_cropped[0,0,:,:]).cpu()
    
        for nms_heatmap,input_img,cropped_img,label_heatmap in zip(flatten_semi,event_vox[:,0,:,:],event_vox_cropped[:,0,:,:],label_vox[:,0,:,:]):
            nms_semi = heatmap_nms(nms_heatmap.cpu(),conf_thresh=0.020)

            # heatmap_bin += torch.from_numpy(nms_semi).cpu()
            # label_bin += label_heatmap.cpu()
            # img_bin += input_img.cpu()
            # cropped_img_bin += cropped_img.cpu()

            # #每5个可视化一次
            # if (img_num+1) % 5 ==0 : 
            #     heatmap_bin[heatmap_bin>1]=1
            #     label_bin[label_bin>1]=1
            #     img_bin[img_bin>1]=1
            #     cropped_img_bin[cropped_img_bin>1]=1

            #     cv2.imwrite("{}/heatmap/{:08d}.jpg".format(flags.output_path,img_num),heatmap_bin.cpu().numpy()*255)
            #     cv2.imwrite("{}/label/{:08d}.jpg".format(flags.output_path,img_num),label_bin.cpu().numpy()*255)
            #     cv2.imwrite("{}/input_img/{:08d}.jpg".format(flags.output_path,img_num),img_bin.cpu().numpy()*255)
            #     cv2.imwrite("{}/input_img_cropped/{:08d}.jpg".format(flags.output_path,img_num),cropped_img_bin.cpu().numpy()*255)

            #     label_bin = torch.zeros_like(label_vox[0,0]).cpu()
            #     heatmap_bin = torch.zeros_like(flatten_semi[0,0]).cpu()
            #     img_bin = torch.zeros_like(event_vox[0,0,:,:]).cpu()
            #     cropped_img_bin = torch.zeros_like(event_vox_cropped[0,0,:,:]).cpu()
            
            cv2.imwrite("{}/heatmap/{:08d}.jpg".format(flags.output_path,img_num),nms_semi*255)
            cv2.imwrite("{}/label/{:08d}.jpg".format(flags.output_path,img_num),label_heatmap.cpu().numpy()*255)
            cv2.imwrite("{}/input_img/{:08d}.jpg".format(flags.output_path,img_num),input_img.cpu().numpy()*255)
            cv2.imwrite("{}/input_img_cropped/{:08d}.jpg".format(flags.output_path,img_num),cropped_img.cpu().numpy()*255)
            img_num += 1

        sum_accuracy += accuracy
        sum_loss += loss

    test_loss = sum_loss.item() / len(test_loader)
    test_accuracy = sum_accuracy / len(test_loader)

    print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

        


