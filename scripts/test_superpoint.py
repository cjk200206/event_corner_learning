from os.path import dirname
import argparse
import torch
import torchvision
import tqdm
import os
import numpy as np
import torch.nn.functional as F
import cv2
import sys
sys.path.append("../")

from utils.dataset import Syn_Superpoint,Pic_Superpoint
from torch.utils.data import DataLoader
from torch.utils.data import default_collate
from utils.models_superpoint import EventCornerSuperpoint
from utils.loss import compute_vox_loss,compute_superpoint_loss,compute_superpoint_argmax_loss
from utils.utils.utils import getLabels,heatmap_nms
from utils.utils.d2s import flatten_64to1
from torch.utils.tensorboard import SummaryWriter

def crop_and_resize_to_resolution(x, output_resolution=(224, 224)):
    B, C, H, W = x.shape
    if H > W:
        h = H // 2
        x = x[:,:,h - W // 2:h + W // 2, :]
    else:
        h = W // 2
        x = x[:, : , :, h - H // 2:h + H // 2]

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
    test_dataset = Pic_Superpoint(flags.test_dataset,num_time_bins=10,grid_size=(260,346),event_crop=False)
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
        
    for img,label in tqdm.tqdm(test_loader):
        img_cropped = crop_and_resize_to_resolution(img.unsqueeze(1))/255
        label = crop_and_resize_to_resolution(label.unsqueeze(1))
        # 把数据转到gpu
        img = img.to(flags.device)/255
        label = label.to(flags.device)
        #转换标签
        label_3d = getLabels(label,8)
        with torch.no_grad():
            semi, _ = model(img.unsqueeze(1).to(torch.float))
            loss, accuracy = compute_superpoint_loss(semi, label_3d)
            # loss, accuracy = compute_superpoint_argmax_loss(semi, label_3d)

        semi = F.softmax(semi,dim=1)
        #输出热力图
        flatten_semi = flatten_64to1(semi[:,:-1,:,:])

    
        for nms_heatmap,input_img,cropped_img,label_heatmap in zip(flatten_semi,img,img_cropped,label):
            nms_semi = heatmap_nms(nms_heatmap.cpu(),conf_thresh=0.015)

            cv2.imwrite("{}/heatmap/{:08d}.jpg".format(flags.output_path,img_num),nms_semi*255)
            cv2.imwrite("{}/label/{:08d}.jpg".format(flags.output_path,img_num),label_heatmap[0].cpu().numpy()*255)
            cv2.imwrite("{}/input_img/{:08d}.jpg".format(flags.output_path,img_num),input_img.cpu().numpy()*255)
            cv2.imwrite("{}/input_img_cropped/{:08d}.jpg".format(flags.output_path,img_num),cropped_img[0].cpu().numpy()*255)
            img_num += 1

        sum_accuracy += accuracy
        sum_loss += loss

    test_loss = sum_loss.item() / len(test_loader)
    test_accuracy = sum_accuracy / len(test_loader)

    print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

        


