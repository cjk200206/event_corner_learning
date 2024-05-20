from os.path import dirname
import argparse
import torch
import torchvision
import tqdm
import os
import numpy as np
import torch.nn.functional as F
import sys
import cv2
sys.path.append("../")

from numpy.linalg import inv
from utils.dataset import DSEC
from torch.utils.data import DataLoader,default_collate
from utils.models_superpoint import SuperPointNet_RAW
from utils.utils.utils import getLabels,add_salt_and_pepper_new,inv_warp_image,inv_warp_image_batch,heatmap_nms_new
from utils.utils.homographies import sample_homography_np
from utils.utils.d2s import flatten_64to1
from torch.utils.tensorboard import SummaryWriter


def FLAGS():
    parser = argparse.ArgumentParser("""preprocess DSEC, generate sae and labels""")

    # test dataset
    parser.add_argument("--dataset", default="../DSEC")
    parser.add_argument("--checkpoint", default="../log/superpoint_ckpt/superpoint_v1.pth")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", type=bool, default=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--mode", default="raw_files")
    parser.add_argument("--output_path", default="../DSEC")


    flags = parser.parse_args()

    assert os.path.isdir(dirname(flags.checkpoint)), f"Checkpoint{flags.checkpoint} not found."
    assert os.path.isdir(flags.dataset), f"Test dataset directory {flags.dataset} not found."

    print(f"----------------------------\n"
          f"Starting testing with \n"
          f"checkpoint: {flags.checkpoint}\n"
          f"test_dataset: {flags.dataset}\n"
          f"batch_size: {flags.batch_size}\n"
          f"device: {flags.device}\n"
          f"----------------------------")

    return flags

if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = '1' #设置显卡可见
    flags = FLAGS()
    # datasets, add augmentation to training set
    test_dataset = DSEC(flags.dataset,mode=flags.mode)
    
    # construct loader, handles data streaming to gpu
    test_loader = DataLoader(test_dataset,batch_size=flags.batch_size,
                               pin_memory=flags.pin_memory,collate_fn=default_collate)

    # model, and put to device
    model = SuperPointNet_RAW()
    ckpt = torch.load(flags.checkpoint)
    model.load_state_dict(ckpt,strict=False)
    model = model.to(flags.device)
    model = model.eval()

    #建标签文件夹
    os.makedirs(os.path.join(flags.output_path,"labels"),exist_ok=True)

    iteration = 0
    img_num = 0

    for img,sae,sae_img,vox,vox_img,label,file_name in tqdm.tqdm(test_loader):
        
        # 把数据转到gpu
        input_vox = img.to(flags.device).to(torch.float)/255
        # input_vox = sae_img.to(flags.device).to(torch.float)/255

        with torch.no_grad():
            semi, _ = model(input_vox.unsqueeze(1).cuda())

            # 输出热力图
            flatten_semi = flatten_64to1(semi[:,:-1,:,:])
            error_flag = 0

            # 将标签输出
            for heatmap, file in zip(flatten_semi,file_name):
                nms_semi = heatmap_nms_new(heatmap,conf_thresh=0.025)

                name = file.split('/images')
                file_dir = "".join(name[1].split('/')[:-1])
                file_num = name[1].split('/')[-1].split('.')[0]
                if os.path.exists(os.path.join(flags.output_path,"labels",file_dir)) == False:
                    os.makedirs(os.path.join(flags.output_path,"labels",file_dir))
                
                cv2.imwrite("{}.jpg".format(os.path.join(flags.output_path,"labels",file_dir,file_num)),nms_semi.cpu().numpy()*255)

                img_num += 1

    print(f"Output number : {img_num} ")


        


