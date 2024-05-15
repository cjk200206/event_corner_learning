from os.path import dirname
import argparse
import torch
import torchvision
import tqdm
import os
import numpy as np
import torch.nn.functional as F
import sys
from math import pi
sys.path.append("../")

from numpy.linalg import inv
from utils.dataset import Syn_Superpoint_SAE
from torch.utils.data import DataLoader
from torch.utils.data import default_collate
from utils.models_superpoint import EventCornerSuperpoint
from utils.loss import compute_vox_loss,compute_superpoint_loss,compute_superpoint_argmax_loss,descriptor_loss
from utils.utils.utils import getLabels,add_salt_and_pepper_new,inv_warp_image,inv_warp_image_batch,heatmap_nms
from utils.utils.transformation import random_affine_transform
from utils.utils.homographies import sample_homography_np
from torch.utils.tensorboard import SummaryWriter

import time 

import time




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

    # training / validation dataset
    parser.add_argument("--validation_dataset", default="/remote-home/share/cjk/syn2e/datasets/val")
    parser.add_argument("--training_dataset", default="/remote-home/share/cjk/syn2e/datasets/train")
    # parser.add_argument("--test_dataset", default="/remote-home/share/cjk/syn2e/datasets/test")
    parser.add_argument("--mode", default="raw_files")

    # logging options
    parser.add_argument("--log_dir", default="log/superpoint_sae")
    parser.add_argument("--pretrained",default=None)

    # loader and device options
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--pin_memory", type=bool, default=True)
    parser.add_argument("--batch_size", type=int, default=8)

    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--save_every_n_epochs", type=int, default=5)

    flags = parser.parse_args()

    assert os.path.isdir(dirname(flags.log_dir)), f"Log directory root {dirname(flags.log_dir)} not found."
    assert os.path.isdir(flags.validation_dataset), f"Validation dataset directory {flags.validation_dataset} not found."
    assert os.path.isdir(flags.training_dataset), f"Training dataset directory {flags.training_dataset} not found."
    # assert os.path.isdir(flags.test_dataset), f"Test dataset directory {flags.training_dataset} not found."

    print(f"----------------------------\n"
          f"Starting training with \n"
          f"num_epochs: {flags.num_epochs}\n"
          f"batch_size: {flags.batch_size}\n"
          f"device: {flags.device}\n"
          f"log_dir: {flags.log_dir}\n"
          f"training_dataset: {flags.training_dataset}\n"
          f"validation_dataset: {flags.validation_dataset}\n"
        #   f"test_dataset: {flags.test_dataset}\n"
          f"----------------------------")

    return flags

if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = '1' #设置显卡可见
    flags = FLAGS()
    # datasets, add augmentation to training set
    print("initialize training")
    training_dataset = Syn_Superpoint_SAE(flags.training_dataset,num_time_bins=1,grid_size=(260,346),mode=flags.mode)
    print("initialize validation")
    validation_dataset = Syn_Superpoint_SAE(flags.validation_dataset,num_time_bins=1,grid_size=(260,346),mode=flags.mode)

    # construct loader, handles data streaming to gpu
    training_loader = DataLoader(training_dataset,batch_size=flags.batch_size,
                               pin_memory=flags.pin_memory,collate_fn=default_collate)
    validation_loader = DataLoader(validation_dataset,batch_size=flags.batch_size,
                               pin_memory=flags.pin_memory,collate_fn=default_collate)

    # model, and put to device
    model = EventCornerSuperpoint(crop_dimension=(224, 224),)
    # resume from ckpt
    if flags.pretrained != None:
        ckpt = torch.load(flags.pretrained)
        model.load_state_dict(ckpt["state_dict"],strict=False)
    model = model.to(flags.device)

    # optimizer and lr scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.1)

    writer = SummaryWriter(flags.log_dir)

    iteration = 0
    min_validation_loss = 1000

    #HA变换的参数
    HA_params = {
        "translation": True,
        "rotation": True,
        "scaling": True,
        "perspective": True,
        "scaling_amplitude": 0.1,
        "perspective_amplitude_x": 0.1,
        "perspective_amplitude_y": 0.1,
        "allow_artifacts": False,
        "patch_ratio": 0.85,
        "max_angle": pi/12
    }

    for iter in range(flags.num_epochs):
        # val
        sum_accuracy = 0
        sum_loss = 0
        model = model.eval()
        print(f"Validation step [{iter:3d}/{flags.num_epochs:3d}]")
        
        if flags.mode != "preprocessed_files_server": 
            # for _,label_vox,_,sae,_,label_vox_first,_,sae_first in tqdm.tqdm(validation_loader):
            for event_vox, label_vox, heatmap, sae_50, sae_75, sae_100 in tqdm.tqdm(validation_loader): 
                
                ########## 以下是不用HA变化
                # #sae_50+sae_100            
                # label_vox = label_vox.to(flags.device)
                # sae_50 = sae_50.to(flags.device)
                # sae_100 = sae_100.to(flags.device)
                # label_vox = crop_and_resize_to_resolution(label_vox)
                # sae_50 = crop_and_resize_to_resolution(sae_50)
                # sae_100 = crop_and_resize_to_resolution(sae_100)

                # # 选择通道
                # label_2d = label_vox[:,0,:,:]
                # for i in range(label_2d.shape[0]):
                #     label_2d[i] = torch.from_numpy(heatmap_nms(label_2d[i].cpu())) #给标签使用nms，筛除噪点
                # input_vox = sae_50[:,0,:,:]
                # label_2d_transformed = label_2d
                # input_vox_transformed = sae_100[:,0,:,:]

                # #HA变换设为单位阵，即不变
                # homography = np.eye(3, 3)
                # homography = inv(homography)
                # inv_homography = inv(homography)
                # inv_homography = torch.tensor(inv_homography).to(torch.float32)
                # homography = torch.tensor(homography).to(torch.float32).cuda()

                ######### 以下是正常的HA变换
                # 把数据转到gpu
                label_vox = label_vox.to(flags.device)
                sae = sae_50.to(flags.device)
                label_vox = crop_and_resize_to_resolution(label_vox)
                sae = crop_and_resize_to_resolution(sae)
                # 选择通道
                label_2d = label_vox[:,0,:,:]
                for i in range(label_2d.shape[0]):
                    # label_2d[i] = torch.from_numpy(heatmap_nms(label_2d[i].cpu())) #给标签使用nms，筛除噪点
                    label_2d[i] = heatmap_nms(label_2d[i]) #给标签使用nms，筛除噪点
                input_vox = sae[:,0,:,:]
                
                #做HA变换
                homography = sample_homography_np(np.array([2, 2]),**HA_params)
                homography = inv(homography)
                inv_homography = inv(homography)
                inv_homography = torch.tensor(inv_homography).to(torch.float32)
                homography = torch.tensor(homography).to(torch.float32).cuda()
                #images
                
                warped_imgs = inv_warp_image_batch(torch.cat((label_2d,input_vox),dim=0).unsqueeze(1),\
                                                homography.unsqueeze(0).expand(label_2d.size(0)*2,-1,-1),device=input_vox.device)
                label_2d_transformed = warped_imgs[:label_2d.size(0),0]
                input_vox_transformed = warped_imgs[label_2d.size(0):,0]
                #增加椒盐噪声
                input_vox = add_salt_and_pepper_new(input_vox,type="sae")
                input_vox_transformed = add_salt_and_pepper_new(input_vox_transformed,type="sae")
                #保证标签为1
                for i in range(label_2d_transformed.shape[0]):
                    label_2d_transformed[i] = heatmap_nms(label_2d_transformed[i].cpu()) #给标签使用nms，筛除噪点
                # label_2d_transformed = torch.where(label_2d_transformed.cuda() >= 0.8, torch.tensor(1.0).cuda(), label_2d_transformed.cuda()) #大于0的地方全转到1
                # label_2d_transformed = torch.where(label_2d_transformed.cuda() < 0.8, torch.tensor(0.0).cuda(), label_2d_transformed.cuda()) #小于0的地方全转到0

                #转换标签
                label_3d = getLabels(label_2d.unsqueeze(1),8,device=flags.device)
                label_3d_transform = getLabels(label_2d_transformed.unsqueeze(1),8,device=flags.device)

                with torch.no_grad():
                    semi, desc = model(input_vox.unsqueeze(1).cuda())
                    semi_transform, desc_transform = model(input_vox_transformed.unsqueeze(1).cuda())

                    try:
                        # Check for NaN semi_transform
                        if torch.isnan(semi_transform[0,0,0,0]):
                            raise ValueError("NaN semi_transform detected")
                        
                        loss_a, accuracy_a = compute_superpoint_loss(semi, label_3d)
                        loss_b, accuracy_b = compute_superpoint_loss(semi_transform, label_3d_transform)
                        homographies = homography.unsqueeze(0).expand(desc.shape[0],-1,-1)
                        mask_valid = torch.ones_like(desc[:,0]).unsqueeze(1)
                        loss_d,_,_,_ = descriptor_loss(desc,desc_transform,homographies,mask_valid=mask_valid,device=desc.device)
                        
                        loss = loss_a+loss_b+ 0.001*loss_d
                        # loss = loss_a+loss_b
                        accuracy = (accuracy_a+accuracy_b)/2

                        sum_accuracy += accuracy
                        sum_loss += loss

                        iteration += 1
                    
                    except ValueError as e:
                        print(f"Exception caught: {e}")
                        print("Skipping current validation iteration...")
                        iteration += 1
                        continue  # Skip current iteration and move to next one
        else :
            # for input_vox, input_vox_transformed, label_3d, label_3d_transform,homography in tqdm.tqdm(validation_loader): 
                # input_vox = input_vox.to(flags.device).squeeze(1)
                # input_vox_transformed = input_vox_transformed.to(flags.device).squeeze(1)
                # label_3d = label_3d.to(flags.device).squeeze(1)
                # label_3d_transform = label_3d_transform.to(flags.device).squeeze(1)

            for label_vox,sae_50 in tqdm.tqdm(validation_loader): 
                ######### 以下是正常的HA变换
                # 把数据转到gpu
                start_time = time.time()
                start_time4 = time.time()

                label_vox = label_vox.to(flags.device)
                sae = sae_50.to(flags.device)
                label_vox = crop_and_resize_to_resolution(label_vox)
                sae = crop_and_resize_to_resolution(sae)

                # 要测量的代码块
                end_time4 = time.time()
                print(f"运行时间: {end_time4 - start_time4} 秒")
                # 选择通道
                label_2d = label_vox[:,0,:,:]
                for i in range(label_2d.shape[0]):
                    # label_2d[i] = torch.from_numpy(heatmap_nms(label_2d[i].cpu())) #给标签使用nms，筛除噪点
                    label_2d[i] = heatmap_nms(label_2d[i]) #给标签使用nms，筛除噪点
                input_vox = sae[:,0,:,:]

                #做HA变换
                homography = sample_homography_np(np.array([2, 2]),**HA_params)
                homography = inv(homography)
                inv_homography = inv(homography)
                inv_homography = torch.tensor(inv_homography).to(torch.float32)
                homography = torch.tensor(homography).to(torch.float32).cuda()
                #images
                warped_imgs = inv_warp_image_batch(torch.cat((label_2d,input_vox),dim=0).unsqueeze(1),\
                                                homography.unsqueeze(0).expand(label_2d.size(0)*2,-1,-1),device=input_vox.device)

                label_2d_transformed = warped_imgs[:label_2d.size(0),0]
                input_vox_transformed = warped_imgs[label_2d.size(0):,0]
                #增加椒盐噪声
                input_vox = add_salt_and_pepper_new(input_vox,type="sae").unsqueeze(1).cuda()
                input_vox_transformed = add_salt_and_pepper_new(input_vox_transformed,type="sae").unsqueeze(1).cuda()
                #保证标签为1
                for i in range(label_2d_transformed.shape[0]):
                    # label_2d_transformed[i] = torch.from_numpy(heatmap_nms(label_2d_transformed[i].cpu())) #给标签使用nms，筛除噪点
                    label_2d_transformed[i] = heatmap_nms(label_2d_transformed[i]) #给标签使用nms，筛除噪点

                #转换标签
                label_3d = getLabels(label_2d.unsqueeze(1),8,device=flags.device)
                label_3d_transform = getLabels(label_2d_transformed.unsqueeze(1),8,device=flags.device)

                # 要测量的代码块
                end_time = time.time()
                print(f"运行时间: {end_time - start_time} 秒")

                with torch.no_grad():
                    semi, desc = model(input_vox)
                    semi_transform, desc_transform = model(input_vox_transformed)

                    try:
                        # Check for NaN semi_transform
                        if torch.isnan(semi_transform[0,0,0,0]):
                            raise ValueError("NaN semi_transform detected")
                        
                        loss_a, accuracy_a = compute_superpoint_loss(semi, label_3d)
                        loss_b, accuracy_b = compute_superpoint_loss(semi_transform, label_3d_transform)
                        homographies = homography.expand(desc.shape[0],-1,-1)
                        mask_valid = torch.ones_like(desc[:,0]).unsqueeze(1)
                        loss_d,_,_,_ = descriptor_loss(desc,desc_transform,homographies,mask_valid=mask_valid,device=desc.device)
                        
                        loss = loss_a+loss_b+ 0.001*loss_d
                        # loss = loss_a+loss_b
                        accuracy = (accuracy_a+accuracy_b)/2

                        sum_accuracy += accuracy
                        sum_loss += loss

                        iteration += 1
                    
                    except ValueError as e:
                        print(f"Exception caught: {e}")
                        print("Skipping current validation iteration...")
                        iteration += 1
                        continue  # Skip current iteration and move to next one

        if len(validation_loader) != 0:
            validation_loss = sum_loss.item() / len(validation_loader)
            validation_accuracy = sum_accuracy / len(validation_loader)
        else:
            validation_loss = min_validation_loss
            validation_accuracy = 0
        print(f"Validation Loss {validation_loss:.4f}  Accuracy {validation_accuracy:.4f}")

        writer.add_scalar("validation/accuracy", validation_accuracy, iteration)
        writer.add_scalar("validation/loss", validation_loss, iteration)


        if validation_loss < min_validation_loss:
            min_validation_loss = validation_loss
            state_dict = model.state_dict()

            torch.save({
                "state_dict": state_dict,
                "min_val_loss": min_validation_loss,
                "iteration": iteration
            }, "../log/model_best.pth")
            print("New best at ", validation_loss)

        if (iter+1) % flags.save_every_n_epochs == 0:
            state_dict = model.state_dict()
            torch.save({
                "state_dict": state_dict,
                "min_val_loss": min_validation_loss,
                "iteration": iteration
            }, "../log/checkpoint_%05d_%.4f.pth" % (iteration, min_validation_loss))

        # train
        sum_accuracy = 0
        sum_loss = 0
        model = model.train()
        print(f"Training step [{iter:3d}/{flags.num_epochs:3d}]")
        
        if flags.mode != "preprocessed_files_server":

            for event_vox, label_vox, heatmap, sae_50, sae_75, sae_100 in tqdm.tqdm(training_loader):

                ########## 以下是不用HA变化
                #sae_50+sae_100            
                # label_vox = label_vox.to(flags.device)
                # sae_50 = sae_50.to(flags.device)
                # sae_100 = sae_100.to(flags.device)
                # label_vox = crop_and_resize_to_resolution(label_vox)
                # sae_50 = crop_and_resize_to_resolution(sae_50)
                # sae_100 = crop_and_resize_to_resolution(sae_100)

                # # 选择通道
                # label_2d = label_vox[:,0,:,:]
                # for i in range(label_2d.shape[0]):
                #     label_2d[i] = torch.from_numpy(heatmap_nms(label_2d[i].cpu())) #给标签使用nms，筛除噪点
                # input_vox = sae_50[:,0,:,:]
                # label_2d_transformed = label_2d
                # input_vox_transformed = sae_100[:,0,:,:]

                # #HA变换设为单位阵，即不变
                # homography = np.eye(3, 3)
                # homography = inv(homography)
                # inv_homography = inv(homography)
                # inv_homography = torch.tensor(inv_homography).to(torch.float32)
                # homography = torch.tensor(homography).to(torch.float32).cuda()

                ########## 以下是正常的HA变换
                # 把数据转到gpu
                label_vox = label_vox.to(flags.device)
                sae = sae_50.to(flags.device)
                label_vox = crop_and_resize_to_resolution(label_vox)
                sae = crop_and_resize_to_resolution(sae)
                #随机选一个
                label_2d = label_vox[:,0,:,:]
                for i in range(label_2d.shape[0]):
                    label_2d[i] = heatmap_nms(label_2d[i]) #给标签使用nms，筛除噪点
                input_vox = sae[:,0,:,:]

                #做HA变换
                homography = sample_homography_np(np.array([2, 2]),**HA_params)
                homography = inv(homography)
                inv_homography = inv(homography)
                inv_homography = torch.tensor(inv_homography).to(torch.float32)
                homography = torch.tensor(homography).to(torch.float32).cuda()
                #images
                warped_imgs = inv_warp_image_batch(torch.cat((label_2d,input_vox),dim=0).unsqueeze(1),\
                                                homography.unsqueeze(0).expand(label_2d.size(0)*2,-1,-1),device=input_vox.device)
                label_2d_transformed = warped_imgs[:label_2d.size(0),0]
                input_vox_transformed = warped_imgs[label_2d.size(0):,0]
                #增加椒盐噪声
                input_vox = add_salt_and_pepper_new(input_vox,type="sae").unsqueeze(1).cuda()
                input_vox_transformed = add_salt_and_pepper_new(input_vox_transformed,type="sae").unsqueeze(1).cuda()
                #保证标签为1
                for i in range(label_2d_transformed.shape[0]):
                    label_2d_transformed[i] = heatmap_nms(label_2d_transformed[i]) #给标签使用nms，筛除噪点
                
                # label_2d_transformed = torch.where(label_2d_transformed.cuda() >= 0.8, torch.tensor(1.0).cuda(), label_2d_transformed.cuda()) #大于0的地方全转到1
                # label_2d_transformed = torch.where(label_2d_transformed.cuda() < 0.8, torch.tensor(0.0).cuda(), label_2d_transformed.cuda()) #小于0的地方全转到0

                #标签转换
                label_3d = getLabels(label_2d.unsqueeze(1),8,device=flags.device)
                label_3d_transform = getLabels(label_2d_transformed.unsqueeze(1),8,device=flags.device)


                optimizer.zero_grad()

                semi, desc = model(input_vox)
                semi_transform, desc_transform = model(input_vox_transformed)

                try:
                    # Check for NaN semi_transform
                    if torch.isnan(semi_transform[0,0,0,0]):
                        raise ValueError("NaN semi_transform detected")
                    
                    loss_a, accuracy_a = compute_superpoint_loss(semi, label_3d)
                    loss_b, accuracy_b = compute_superpoint_loss(semi_transform, label_3d_transform)
                    homographies = homography.unsqueeze(0).expand(desc.shape[0],-1,-1)
                    mask_valid = torch.ones_like(desc[:,0]).unsqueeze(1)
                    loss_d,_,_,_ = descriptor_loss(desc,desc_transform,homographies,mask_valid=mask_valid,device=desc.device)
                    
                    loss = loss_a+loss_b+0.001*loss_d
                    # loss = loss_a+loss_b
                    accuracy = (accuracy_a+accuracy_b)/2
                    
                    loss.backward()

                    optimizer.step()

                    sum_accuracy += accuracy
                    sum_loss += loss

                    iteration += 1
                
                except ValueError as e:
                    print(f"Exception caught: {e}")
                    print("Skipping current training iteration...")
                    iteration += 1
                    continue  # Skip current iteration and move to next one
        else :
            # for input_vox, input_vox_transformed, label_3d, label_3d_transform,homography in tqdm.tqdm(validation_loader):
            #     input_vox = input_vox.to(flags.device).squeeze(1)
            #     input_vox_transformed = input_vox_transformed.to(flags.device).squeeze(1)
            #     label_3d = label_3d.to(flags.device).squeeze(1)
            #     label_3d_transform = label_3d_transform.to(flags.device).squeeze(1)
            for label_vox,sae_50 in tqdm.tqdm(training_loader): 
                ######### 以下是正常的HA变换
                # 把数据转到gpu
                label_vox = label_vox.to(flags.device)
                sae = sae_50.to(flags.device)
                label_vox = crop_and_resize_to_resolution(label_vox)
                sae = crop_and_resize_to_resolution(sae)
                # 选择通道
                label_2d = label_vox[:,0,:,:]
                for i in range(label_2d.shape[0]):
                    # label_2d[i] = torch.from_numpy(heatmap_nms(label_2d[i].cpu())) #给标签使用nms，筛除噪点
                    label_2d[i] = heatmap_nms(label_2d[i]) #给标签使用nms，筛除噪点
                input_vox = sae[:,0,:,:]
                
                #做HA变换
                homography = sample_homography_np(np.array([2, 2]),**HA_params)
                homography = inv(homography)
                inv_homography = inv(homography)
                inv_homography = torch.tensor(inv_homography).to(torch.float32)
                homography = torch.tensor(homography).to(torch.float32).cuda()
                #images
                warped_imgs = inv_warp_image_batch(torch.cat((label_2d,input_vox),dim=0).unsqueeze(1),\
                                                homography.unsqueeze(0).expand(label_2d.size(0)*2,-1,-1),device=input_vox.device)
                label_2d_transformed = warped_imgs[:label_2d.size(0),0]
                input_vox_transformed = warped_imgs[label_2d.size(0):,0]
                #增加椒盐噪声
                input_vox = add_salt_and_pepper_new(input_vox,type="sae")
                input_vox_transformed = add_salt_and_pepper_new(input_vox_transformed,type="sae")
                #保证标签为1
                for i in range(label_2d_transformed.shape[0]):
                    # label_2d_transformed[i] = torch.from_numpy(heatmap_nms(label_2d_transformed[i].cpu())) #给标签使用nms，筛除噪点
                    label_2d[i] = heatmap_nms(label_2d[i]) #给标签使用nms，筛除噪点

                #转换标签
                label_3d = getLabels(label_2d.unsqueeze(1),8,device=flags.device)
                label_3d_transform = getLabels(label_2d_transformed.unsqueeze(1),8,device=flags.device)


                optimizer.zero_grad()

                semi, desc = model(input_vox.unsqueeze(1).cuda())
                semi_transform, desc_transform = model(input_vox_transformed.unsqueeze(1).cuda())

                try:
                    # Check for NaN semi_transform
                    if torch.isnan(semi_transform[0,0,0,0]):
                        raise ValueError("NaN semi_transform detected")
                    
                    loss_a, accuracy_a = compute_superpoint_loss(semi, label_3d)
                    loss_b, accuracy_b = compute_superpoint_loss(semi_transform, label_3d_transform)
                    homographies = homography.expand(desc.shape[0],-1,-1)
                    mask_valid = torch.ones_like(desc[:,0]).unsqueeze(1)
                    loss_d,_,_,_ = descriptor_loss(desc,desc_transform,homographies,mask_valid=mask_valid,device=desc.device)
                    
                    loss = loss_a+loss_b+0.001*loss_d
                    # loss = loss_a+loss_b
                    accuracy = (accuracy_a+accuracy_b)/2
                    
                    loss.backward()

                    optimizer.step()

                    sum_accuracy += accuracy
                    sum_loss += loss

                    iteration += 1
                
                except ValueError as e:
                    print(f"Exception caught: {e}")
                    print("Skipping current training iteration...")
                    iteration += 1
                    continue  # Skip current iteration and move to next one    
        
        if iter % 10 == 9:
            lr_scheduler.step()
        if len(training_loader) != 0:
            training_loss = sum_loss.item() / len(training_loader)
            training_accuracy = sum_accuracy / len(training_loader)
        else:
            training_loss = min_validation_loss
            training_accuracy = 0
        print(f"Training Iteration {iteration:5d}  Loss {training_loss:.4f}  Accuracy {training_accuracy:.4f}")

        writer.add_scalar("training/accuracy", training_accuracy, iteration)
        writer.add_scalar("training/loss", training_loss, iteration)



