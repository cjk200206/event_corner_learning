from os.path import dirname
import argparse
import torch
import torchvision
import tqdm
import os
import numpy as np
import torch.nn.functional as F

from utils.dataset import Syn_Events
from utils.dataset import Syn_Heatmaps
from torch.utils.data import DataLoader
from torch.utils.data import default_collate
from utils.models_heatmap import EventCornerHeatmap
from utils.loss import cross_entropy_loss_and_accuracy,compute_vox_loss
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
    parser.add_argument("--log_dir", default="log/events_corner")

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
    training_dataset = Syn_Heatmaps(flags.training_dataset,num_time_bins=10,grid_size=(260,346))
    validation_dataset = Syn_Heatmaps(flags.validation_dataset,num_time_bins=10,grid_size=(260,346))

    # construct loader, handles data streaming to gpu
    training_loader = DataLoader(training_dataset,batch_size=flags.batch_size,
                               pin_memory=flags.pin_memory,collate_fn=default_collate)
    validation_loader = DataLoader(validation_dataset,batch_size=flags.batch_size,
                               pin_memory=flags.pin_memory,collate_fn=default_collate)

    # model, and put to device
    model = EventCornerHeatmap(voxel_dimension=(10,260,346))
    model = model.to(flags.device)

    # optimizer and lr scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
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
        for event_vox,label_vox,heatmap in tqdm.tqdm(validation_loader):
            heatmap = crop_and_resize_to_resolution(heatmap)
            
            # 把数据转到gpu
            event_vox = event_vox.to(flags.device)
            label_vox = label_vox.to(flags.device)
            heatmap = heatmap.to(flags.device)
            
            with torch.no_grad():
                pred, vox = model(event_vox)
                loss, accuracy = compute_vox_loss(pred, heatmap)

            sum_accuracy += accuracy
            sum_loss += loss

        validation_loss = sum_loss.item() / len(validation_loader)
        validation_accuracy = sum_accuracy / len(validation_loader)
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

        if i % flags.save_every_n_epochs == 0:
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
        for event_vox,label_vox,heatmap in tqdm.tqdm(training_loader):
            # heatmap = crop_and_resize_to_resolution(heatmap)
            # 把数据转到gpu
            event_vox = event_vox.to(flags.device)
            label_vox = label_vox.to(flags.device)
            heatmap = heatmap.to(flags.device)
            optimizer.zero_grad()

            pred, vox = model(event_vox)
            with torch.autograd.detect_anomaly():
                loss, accuracy = compute_vox_loss(pred, heatmap)
                loss.backward()

            optimizer.step()

            sum_accuracy += accuracy
            sum_loss += loss

            iteration += 1

        if i % 10 == 9:
            lr_scheduler.step()

        training_loss = sum_loss.item() / len(training_loader)
        training_accuracy = sum_accuracy / len(training_loader)
        print(f"Training Iteration {iteration:5d}  Loss {training_loss:.4f}  Accuracy {training_accuracy:.4f}")

        writer.add_scalar("training/accuracy", training_accuracy, iteration)
        writer.add_scalar("training/loss", training_loss, iteration)



