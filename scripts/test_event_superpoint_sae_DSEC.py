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
from math import pi
sys.path.append("../")

from numpy.linalg import inv
from utils.dataset import DSEC
from torch.utils.data import DataLoader
from torch.utils.data import default_collate
from utils.models_superpoint import EventCornerSuperpoint
from utils.loss import compute_vox_loss,compute_superpoint_loss,descriptor_loss
from utils.evaluation import compute_detection_correctness,compute_detection_repeatability,compute_descriptor_Nearest_neighbour_mAP
from utils.utils.utils import getLabels,heatmap_nms,heatmap_nms_new,inv_warp_image,inv_warp_image_batch,add_salt_and_pepper_new
from utils.utils.transformation import random_affine_transform
from utils.utils.homographies import sample_homography_np
from utils.utils.d2s import flatten_64to1
from utils.utils.match_descriptor import draw_matches
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
    parser.add_argument("--checkpoint", default="log/superpoint_ckpt/superpoint_05071311_sae.pth")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", type=bool, default=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--output_path", default="/home/cjk2002/code/event_code/event_corner_learning/log/event_superpoint_sae_DSEC")


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
    test_dataset = DSEC(flags.test_dataset,mode="preprocessed_files")
    # construct loader, handles data streaming to gpu
    test_loader = DataLoader(test_dataset,batch_size=flags.batch_size,
                               pin_memory=flags.pin_memory,collate_fn=default_collate)
  
    # model, and put to device
    model = EventCornerSuperpoint(crop_dimension=(224, 224))
    ckpt = torch.load(flags.checkpoint)
    model.load_state_dict(ckpt["state_dict"],strict=False)
    # model.backbone.load_state_dict(ckpt,strict=False)
    model = model.to(flags.device)

    model = model.eval()
    #评价指标初始化
    sum_accuracy = 0
    sum_loss = 0
    sum_loss_d = 0
    img_num = 0

    threshold_list = [3,6,9]
    match_numbers = [0]*len(threshold_list)
    pred_numbers = [0]*len(threshold_list)
    gt_numbers = [0]*len(threshold_list)
    sum_dist = [0]*len(threshold_list)
    rep_match_numbers = [0]*len(threshold_list)
    rep_gt_numbers = [0]*len(threshold_list)
    desc_num = [0]*len(threshold_list)
    total_match_num = [0]*len(threshold_list)

    #建文件夹
    os.makedirs(flags.output_path + "/heatmap",exist_ok=True)
    os.makedirs(flags.output_path + "/label",exist_ok=True)
    os.makedirs(flags.output_path + "/input_img",exist_ok=True) 

    os.makedirs(flags.output_path + "/heatmap_transformed",exist_ok=True)
    os.makedirs(flags.output_path + "/label_transformed",exist_ok=True)
    os.makedirs(flags.output_path + "/input_img_transformed",exist_ok=True)
    os.makedirs(flags.output_path + "/matched_result",exist_ok=True) 

        
    print("Test step")

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
        
    for img,sae,sae_img,label,img_file in tqdm.tqdm(test_loader):
        
        #把数据转到gpu
        label_vox = label.to(flags.device).unsqueeze(1).to(torch.float32)
        sae = sae.to(flags.device).unsqueeze(1).to(torch.float32)
        #取出单通道的vox
        label_vox = crop_and_resize_to_resolution(label_vox,(224,224))
        sae = crop_and_resize_to_resolution(sae,(224,224))
        label_2d = label_vox[:,0,:,:]
        input_vox = sae[:,0,:,:]
        for i in range(label_2d.shape[0]):
            # label_2d[i] = torch.from_numpy(heatmap_nms(label_2d[i].cpu())) #给标签使用nms，筛除噪点
            label_2d[i] = heatmap_nms_new(label_2d[i]) #给标签使用nms，筛除噪点

        # #取出比对组
        # label_vox_first = label_vox_first.to(flags.device)
        # sae_first = sae_first.to(flags.device)
        # label_vox_first = crop_and_resize_to_resolution(label_vox_first,(224,224))
        # sae_first = crop_and_resize_to_resolution(sae_first.unsqueeze(1),(224,224))
        # label_2d_transformed = label_vox_first[:,0,:,:]
        # input_vox_transformed = sae_first[:,0,:,:]

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
        # #增加椒盐噪声
        # input_vox = add_salt_and_pepper_new(input_vox,type="sae")
        # input_vox_transformed = add_salt_and_pepper_new(input_vox_transformed,type="sae")
        #保证标签为1
        for i in range(label_2d_transformed.shape[0]):
            # label_2d[i] = torch.from_numpy(heatmap_nms(label_2d[i].cpu())) #给标签使用nms，筛除噪点
            label_2d_transformed[i] = heatmap_nms_new(label_2d_transformed[i]) #给标签使用nms，筛除噪点
        # label_2d_transformed = torch.where(label_2d_transformed.cuda() >= 0.8, torch.tensor(1.0).cuda(), label_2d_transformed.cuda()) #大于0的地方全转到1
        # label_2d_transformed = torch.where(label_2d_transformed.cuda() < 0.8, torch.tensor(0.0).cuda(), label_2d_transformed.cuda()) #小于0的地方全转到0

        #转换标签
        label_3d = getLabels(label_2d.unsqueeze(1),8).cuda()
        label_3d_transform = getLabels(label_2d_transformed.unsqueeze(1),8).cuda()

        with torch.no_grad():
            semi, desc = model(input_vox.unsqueeze(1).cuda())
            semi_transform, desc_transformed = model(input_vox_transformed.unsqueeze(1).cuda())
        
            loss_a, accuracy_a = compute_superpoint_loss(semi, label_3d)
            loss_b, accuracy_b = compute_superpoint_loss(semi_transform, label_3d_transform)

            homographies = homography.unsqueeze(0).expand(desc.shape[0],-1,-1)
            inv_homographies = inv_homography.unsqueeze(0).expand(desc.shape[0],-1,-1)
            mask_valid = torch.ones_like(desc[:,0]).unsqueeze(1)
            loss_d,_,_,_ = descriptor_loss(desc,desc_transformed,inv_homographies,mask_valid=mask_valid,device=desc.device)
            
            loss = loss_a+loss_b+0.001*loss_d
            sum_loss_d  += loss_d
            # loss = loss_a+loss_b
            accuracy = (accuracy_a+accuracy_b)/2

        semi = F.softmax(semi,dim=1)
        semi_transform = F.softmax(semi_transform, dim=1)
        #输出热力图
        flatten_semi = flatten_64to1(semi[:,:-1,:,:])
        flatten_semi_transformed = flatten_64to1(semi_transform[:,:-1,:,:])

        for input_img,nms_heatmap,label_heatmap,desc_1,\
            input_img_transformed,nms_heatmap_transformed,label_heatmap_transformed,desc_2 \
                in zip(input_vox,flatten_semi,label_2d.squeeze(1),desc,\
                       input_vox_transformed,flatten_semi_transformed,label_2d_transformed.squeeze(1),desc_transformed):
            nms_semi = heatmap_nms_new(nms_heatmap.cpu(),conf_thresh=0.020)
            nms_semi_transformed = heatmap_nms_new(nms_heatmap_transformed.cpu(),conf_thresh=0.020)
            label_heatmap = heatmap_nms_new(label_heatmap.cpu(),conf_thresh=0.020)
            label_heatmap_transformed = heatmap_nms_new(label_heatmap_transformed.cpu(),conf_thresh=0.020)
            
            #将预测结果变回来
            inv_warped_semi = inv_warp_image(nms_semi_transformed,inv_homography)
            inv_warped_semi = heatmap_nms_new(inv_warped_semi.cpu(),conf_thresh=0.020)
            
            # #将预测结果变回来
            # #获取点
            # transformed_pnts = torch.nonzero(torch.from_numpy(nms_semi_transformed))
            # W=224
            # H=224
            # #把点o变到和原图一样
            # transformed_pnts_sample = transformed_pnts.clone().to(torch.float)
            # transformed_pnts_sample[:,0] = (transformed_pnts_sample[:,0]/(float(W)/2.)) - 1.
            # transformed_pnts_sample[:,1] = (transformed_pnts_sample[:,1]/(float(H)/2.)) - 1.
            # inv_transformed_pnts_sample = warp_points(transformed_pnts_sample,inv_homography)
            # inv_transformed_pnts_sample[:,0] = ((inv_transformed_pnts_sample[:,0]+1)*(float(W)/2.)).to(torch.int)
            # inv_transformed_pnts_sample[:,1] = ((inv_transformed_pnts_sample[:,1]+1)*(float(H)/2.)).to(torch.int)
            # #填充成图
            # inv_warped_semi = torch.zeros_like(torch.from_numpy(nms_semi_transformed))
            # #筛除越界点
            # valid_indices = (inv_transformed_pnts_sample[:, 0] >= 0) & (inv_transformed_pnts_sample[:, 0] < W) & \
            #         (inv_transformed_pnts_sample[:, 1] >= 0) & (inv_transformed_pnts_sample[:, 1] < H)
            # inv_transformed_pnts_sample = inv_transformed_pnts_sample[valid_indices]
            # inv_warped_semi[inv_transformed_pnts_sample.to(torch.long)[:,0],inv_transformed_pnts_sample.to(torch.long)[:,1]] = 1
            # inv_warped_semi = inv_warped_semi.numpy()



            ## 评价指标计算
            for index in range(len(threshold_list)):
                threshold = threshold_list[index]
                matches,pred,gt,dist = compute_detection_correctness(nms_semi,label_heatmap,threshold)
                rep_matches,rep_gt = compute_detection_repeatability(nms_semi,inv_warped_semi,threshold)
                num_1,num_2,matched_points_1,matched_points_2,points_1,points_2,desc_matches = compute_descriptor_Nearest_neighbour_mAP(desc_1,desc_2,nms_semi,inv_warped_semi,nms_semi_transformed,homography,inv_homography,threshold)
                matched_img = draw_matches((input_img.cpu().numpy()+1)*127.5,matched_points_1,(input_img_transformed.cpu().numpy()+1)*127.5,matched_points_2,points_1,points_2)

                rep_match_numbers[index] += rep_matches
                rep_gt_numbers[index] += rep_gt
                match_numbers[index] += len(matches)
                pred_numbers[index] += len(pred)
                gt_numbers[index] += len(gt)
                sum_dist[index] += dist
                total_match_num[index] += num_2
                desc_num[index] += num_1

            cv2.imwrite("{}/heatmap/{:08d}.jpg".format(flags.output_path,img_num),nms_semi.numpy()*255)
            # cv2.imwrite("{}/label/{:08d}.jpg".format(flags.output_path,img_num),label_heatmap.cpu().numpy()*255)
            cv2.imwrite("{}/label/{:08d}.jpg".format(flags.output_path,img_num),label_heatmap.numpy()*255)
            cv2.imwrite("{}/input_img/{:08d}.jpg".format(flags.output_path,img_num),(input_img.cpu().numpy()+1)*127.5)
            #变换后的
            cv2.imwrite("{}/heatmap_transformed/{:08d}.jpg".format(flags.output_path,img_num),nms_semi_transformed.numpy()*255)
            # cv2.imwrite("{}/label_transformed/{:08d}.jpg".format(flags.output_path,img_num),label_heatmap_transformed.cpu().numpy()*255)
            cv2.imwrite("{}/label_transformed/{:08d}.jpg".format(flags.output_path,img_num),label_heatmap_transformed.numpy()*255)
            cv2.imwrite("{}/input_img_transformed/{:08d}.jpg".format(flags.output_path,img_num),(input_img_transformed.cpu().numpy()+1)*127.5)
            #保存匹配结果图像
            cv2.imwrite("{}/matched_result/{:08d}.jpg".format(flags.output_path,img_num), matched_img)

            img_num += 1

        sum_accuracy += accuracy
        sum_loss += loss

    test_loss = sum_loss.item() / len(test_loader)
    test_accuracy = sum_accuracy / len(test_loader)
    test_loss_d = sum_loss_d.item()/len(test_loader)

    
    print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}, Test Loss_d: {test_loss_d}")
          
    for index in range(len(threshold_list)):
        threshold = threshold_list[index]
        test_recall = match_numbers[index] / gt_numbers[index]
        test_precision = match_numbers[index] / pred_numbers[index]
        localization_error = sum_dist[index]/match_numbers[index]
        repeatability = rep_match_numbers[index]/rep_gt_numbers[index]
        NN_mAP = desc_num[index]/total_match_num[index]
        print(f"\nThreshold: {threshold} \nTest Recall: {test_recall}, Test Precision: {test_precision}, Localization Error: {localization_error}, Repeatability: {repeatability}, NN_mAP: {NN_mAP}")

        


