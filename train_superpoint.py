from os.path import dirname
import argparse
import torch
import torchvision
import tqdm
import os
import numpy as np
import torch.nn.functional as F

from utils.dataset import Syn_Superpoint,Pic_Superpoint
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
    parser = argparse.ArgumentParser("""Train classifier using a learnt quantization layer.""")

    # training / validation dataset
    parser.add_argument("--validation_dataset", default="/remote-home/share/cjk/syn2e/datasets/val")
    parser.add_argument("--training_dataset", default="/remote-home/share/cjk/syn2e/datasets/train")
    # parser.add_argument("--test_dataset", default="/remote-home/share/cjk/syn2e/datasets/test")
    # logging options
    parser.add_argument("--log_dir", default="log/superpoint")

    # loader and device options
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", type=bool, default=True)
    parser.add_argument("--batch_size", type=int, default=4)

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
    training_dataset = Pic_Superpoint(flags.training_dataset,num_time_bins=10,grid_size=(260,346))
    validation_dataset = Pic_Superpoint(flags.validation_dataset,num_time_bins=10,grid_size=(260,346))

    # construct loader, handles data streaming to gpu
    training_loader = DataLoader(training_dataset,batch_size=flags.batch_size,
                               pin_memory=flags.pin_memory,collate_fn=default_collate)
    validation_loader = DataLoader(validation_dataset,batch_size=flags.batch_size,
                               pin_memory=flags.pin_memory,collate_fn=default_collate)

    # model, and put to device
    model = EventCornerSuperpoint(voxel_dimension=(2,260,346))
    model = model.to(flags.device)

    # optimizer and lr scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3,weight_decay=0.0001)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.5)

    writer = SummaryWriter(flags.log_dir)

    iteration = 0
    min_validation_loss = 1000

    for i in range(flags.num_epochs):
        # val
        sum_accuracy = 0
        sum_loss = 0
        model = model.eval()
        print(f"Validation step [{i:3d}/{flags.num_epochs:3d}]")
        
        for img,label in tqdm.tqdm(validation_loader):
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

            sum_accuracy += accuracy
            sum_loss += loss
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
            }, "log/model_best.pth")
            print("New best at ", validation_loss)

        if (i+1) % flags.save_every_n_epochs == 0:
            state_dict = model.state_dict()
            torch.save({
                "state_dict": state_dict,
                "min_val_loss": min_validation_loss,
                "iteration": iteration
            }, "log/checkpoint_%05d_%.4f.pth" % (iteration, min_validation_loss))

        # train
        sum_accuracy = 0
        sum_loss = 0
        model = model.train()
        print(f"Training step [{i:3d}/{flags.num_epochs:3d}]")

        for img,label in tqdm.tqdm(training_loader):
            label = crop_and_resize_to_resolution(label.unsqueeze(1))
            # 把数据转到gpu
            img = img.to(flags.device)/255
            label = label.to(flags.device)

            optimizer.zero_grad()

            #转换标签,随机挑一个切片
            label_3d = getLabels(label,8)
            semi, _ = model(img.unsqueeze(1).to(torch.float))
            # with torch.autograd.detect_anomaly():
            loss, accuracy = compute_superpoint_loss(semi, label_3d)
            # loss, accuracy = compute_superpoint_argmax_loss(semi, label_3d)
            loss.backward()

            optimizer.step()

            sum_accuracy += accuracy
            sum_loss += loss

            iteration += 1

        if i % 10 == 9:
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



