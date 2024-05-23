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
from utils.dataset import Syn_Superpoint
from torch.utils.data import DataLoader,default_collate
from utils.models_superpoint import EventCornerSuperpoint
from utils.utils.utils import getLabels,add_salt_and_pepper_new,inv_warp_image,inv_warp_image_batch,heatmap_nms
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
    parser.add_argument("--test_dataset", default="/remote-home/share/cjk/syn2e/datasets/val")
    parser.add_argument("--checkpoint", default="log/superpoint_ckpt/superpoint_05022138_best.pth")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", type=bool, default=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--mode", default="raw_files")
    parser.add_argument("--output_path", default="/home/cjk2002/code/event_code/event_corner_learning/pseudo_label")


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
    test_dataset = Syn_Superpoint(flags.test_dataset,num_time_bins=3,grid_size=(260,346),event_crop=False,mode=flags.mode)
    
    # construct loader, handles data streaming to gpu
    validation_loader = DataLoader(test_dataset,batch_size=flags.batch_size,
                               pin_memory=flags.pin_memory,collate_fn=default_collate)

    # model, and put to device
    model = EventCornerSuperpoint(crop_dimension=(224, 224),)
    ckpt = torch.load(flags.checkpoint)
    model.load_state_dict(ckpt["state_dict"],strict=False)
    model = model.to(flags.device)
    model = model.eval()

    #建文件夹
    os.makedirs(flags.output_path + "/label",exist_ok=True)
    os.makedirs(flags.output_path + "/input_img",exist_ok=True) 

    iteration = 0
    img_num = 0

    #HA变换的参数
    HA_params = {
        "translation": True,
        "rotation": True,
        "scaling": True,
        "perspective": True,
        "scaling_amplitude": 0.2,
        "perspective_amplitude_x": 0.2,
        "perspective_amplitude_y": 0.2,
        "allow_artifacts": True,
        "patch_ratio": 0.85,
    }

        
    for event_vox,label_vox,heatmap in tqdm.tqdm(validation_loader):
        event_vox = crop_and_resize_to_resolution(event_vox)

        # 把数据转到gpu
        event_vox = event_vox.to(flags.device)

        # #随机选择
        # rand_idx= np.random.randint(0,3)
        # input_vox = event_vox[:,rand_idx,:,:]
        input_vox = event_vox[:,0,:,:]

        #做HA变换
        homography = sample_homography_np(np.array([2, 2]),**HA_params)
        homography = inv(homography)
        inv_homography = inv(homography)
        inv_homography = torch.tensor(inv_homography).to(torch.float32)
        homography = torch.tensor(homography).to(torch.float32).cuda()
        #images
        warped_imgs = inv_warp_image_batch(input_vox.unsqueeze(1),\
                                        homography.unsqueeze(0).expand(input_vox.size(0),-1,-1),device=input_vox.device)
        input_vox_transformed = warped_imgs[:,0]
        #增加椒盐噪声
        input_vox = add_salt_and_pepper_new(input_vox)
        input_vox_transformed = add_salt_and_pepper_new(input_vox_transformed)

        input_vox_transformed = torch.where(input_vox_transformed.cuda() > 0, torch.tensor(1.0).cuda(), input_vox_transformed.cuda()) #大于0的地方全转到1
        input_vox_transformed = torch.where(input_vox_transformed.cuda() < 0, torch.tensor(0.0).cuda(), input_vox_transformed.cuda()) #小于0的地方全转到0


        with torch.no_grad():
            semi, _ = model(input_vox.unsqueeze(1).cuda())
            semi_transform, _ = model(input_vox_transformed.unsqueeze(1).cuda())

            #输出热力图
            flatten_semi = flatten_64to1(semi[:,:-1,:,:])
            error_flag = 0
            try:
                # Check for NaN semi_transform
                if torch.isnan(semi_transform[0,0,0,0]):
                    error_flag = 1
                    flatten_semi_transformed = flatten_semi
                    raise ValueError("NaN semi_transform detected")
                flatten_semi_transformed = flatten_64to1(semi_transform[:,:-1,:,:])



            except ValueError as e:
                print(f"Exception caught: {e}")
                print("Skipping current testing iteration...")

                continue  # Skip current iteration and move to next one

            # 将HA后的标签合并
            for heatmap,heatmap_transformed,input_img in zip(flatten_semi,flatten_semi_transformed,event_vox[:,0,:,:]):
                nms_semi = heatmap_nms(heatmap.cpu(),conf_thresh=0.03)
                nms_semi_transformed = heatmap_nms(heatmap_transformed.cpu(),conf_thresh=0.025)
                
                #将标签逆解
                warped_semi = inv_warp_image_batch(torch.from_numpy(nms_semi_transformed).unsqueeze(0).unsqueeze(1),\
                                        inv_homography.unsqueeze(0).expand(1,-1,-1),device=input_vox.device)
                #保证标签为1
                warped_semi = torch.where(warped_semi.cuda() >= 0.8, torch.tensor(1.0).cuda(), warped_semi.cuda()) #大于0.8的地方全转到1
                warped_semi = torch.where(warped_semi.cuda() < 0.8, torch.tensor(0.0).cuda(), warped_semi.cuda()) #小于0.8的地方全转到0
                #合并
                nms_semi = torch.from_numpy(nms_semi).unsqueeze(0).cuda()+warped_semi.squeeze(0).cuda()
                #再做一次nms
                nms_semi = heatmap_nms(heatmap.cpu(),conf_thresh=0.025)
                nms_semi = torch.from_numpy(nms_semi)
                nms_semi = torch.where(nms_semi.cuda() > 1, torch.tensor(1.0).cuda(), nms_semi.cuda()) #大于0的地方全转到1
                nms_semi = torch.where(nms_semi.cuda() < 1, torch.tensor(0.0).cuda(), nms_semi.cuda()) #小于0的地方全转到0
                nms_semi = nms_semi.cpu().numpy()

                cv2.imwrite("{}/label/{:08d}.jpg".format(flags.output_path,img_num),nms_semi*255)
                cv2.imwrite("{}/input_img/{:08d}.jpg".format(flags.output_path,img_num),input_img.cpu().numpy()*255)
                img_num += 1

    print(f"Output number : {img_num} ")


        


