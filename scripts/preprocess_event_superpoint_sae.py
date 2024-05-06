from os.path import dirname
import argparse
import torch
import torchvision
import tqdm
import os
import numpy as np
import torch.nn.functional as F
import sys
sys.path.append("../")

from utils.dataset import Syn_Superpoint_SAE
from torch.utils.data import DataLoader
from torch.utils.data import default_collate
from utils.models_superpoint import EventCornerSuperpoint
from utils.loss import compute_vox_loss,compute_superpoint_loss,compute_superpoint_argmax_loss
from utils.utils.utils import getLabels
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
    parser = argparse.ArgumentParser("""preprocess_data.""")

    # training / validation dataset
    parser.add_argument("--validation_dataset","-val", default="/remote-home/share/cjk/syn2e/datasets/val")
    parser.add_argument("--training_dataset","-train", default="/remote-home/share/cjk/syn2e/datasets/train")
    # parser.add_argument("--test_dataset", default="/remote-home/share/cjk/syn2e/datasets/test")
    parser.add_argument("--mode", default="raw_files")
    
    # logging options
    # loader and device options
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--pin_memory", type=bool, default=True)
    flags = parser.parse_args()

    assert os.path.isdir(flags.validation_dataset), f"Validation dataset directory {flags.validation_dataset} not found."
    assert os.path.isdir(flags.training_dataset), f"Training dataset directory {flags.training_dataset} not found."
    # assert os.path.isdir(flags.test_dataset), f"Test dataset directory {flags.training_dataset} not found."

    return flags

if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = '1' #设置显卡可见
    flags = FLAGS()
    # datasets, add augmentation to training set
    training_dataset = Syn_Superpoint_SAE(flags.training_dataset,num_time_bins=1,grid_size=(260,346),mode=flags.mode)
    validation_dataset = Syn_Superpoint_SAE(flags.validation_dataset,num_time_bins=1,grid_size=(260,346),mode=flags.mode)

    # construct loader, handles data streaming to gpu
    training_loader = DataLoader(training_dataset,batch_size=1,
                               pin_memory=flags.pin_memory,collate_fn=default_collate)
    validation_loader = DataLoader(validation_dataset,batch_size=1,
                               pin_memory=flags.pin_memory,collate_fn=default_collate)

    #建立预处理集
    preprocessed_train_path=os.path.join(flags.training_dataset,"preprocessed_sae")
    preprocessed_val_path=os.path.join(flags.validation_dataset,"preprocessed_sae")
    os.makedirs(preprocessed_train_path,exist_ok=True)
    os.makedirs(preprocessed_val_path,exist_ok=True)

    val_counter = 0
    for event_vox,label_vox,heatmap,sae in tqdm.tqdm(validation_loader):
        event_vox = event_vox.numpy()
        label_vox = label_vox.numpy()
        heatmap = heatmap.numpy()
        sae = sae.numpy()
        np.savez_compressed("{}/{:08d}.npz".format(preprocessed_val_path,val_counter),event_vox=event_vox,label_vox=label_vox,heatmap=heatmap,sae = sae)
        val_counter+=1

    train_counter = 0
    for event_vox,label_vox,heatmap,sae in tqdm.tqdm(training_loader):
        event_vox = event_vox.numpy()
        label_vox = label_vox.numpy()
        heatmap = heatmap.numpy()
        sae = sae.numpy()
        np.savez_compressed("{}/{:08d}.npz".format(preprocessed_train_path,train_counter),event_vox=event_vox,label_vox=label_vox,heatmap=heatmap,sae = sae)
        train_counter+=1


