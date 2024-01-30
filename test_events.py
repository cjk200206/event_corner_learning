from os.path import dirname
import argparse
import torch
import tqdm
import os

from utils.dataset import Syn_Events
from torch.utils.data import DataLoader

def FLAGS():
    parser = argparse.ArgumentParser("""Train classifier using a learnt quantization layer.""")

    # training / validation dataset
    # parser.add_argument("--validation_dataset", default="", required=True)
    # parser.add_argument("--training_dataset", default="", required=True)
    parser.add_argument("--test_dataset", default="/remote-home/share/cjk/syn2e/datasets/test")
    # logging options
    # parser.add_argument("--log_dir", default="", required=True)

    # loader and device options
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", type=bool, default=True)
    parser.add_argument("--batch_size", type=int, default=4)

    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--save_every_n_epochs", type=int, default=5)

    flags = parser.parse_args()

    # assert os.path.isdir(dirname(flags.log_dir)), f"Log directory root {dirname(flags.log_dir)} not found."
    # assert os.path.isdir(flags.validation_dataset), f"Validation dataset directory {flags.validation_dataset} not found."
    # assert os.path.isdir(flags.training_dataset), f"Training dataset directory {flags.training_dataset} not found."
    assert os.path.isdir(flags.test_dataset), f"Training dataset directory {flags.training_dataset} not found."

    print(f"----------------------------\n"
          f"Starting training with \n"
          f"num_epochs: {flags.num_epochs}\n"
          f"batch_size: {flags.batch_size}\n"
          f"device: {flags.device}\n"
        #   f"log_dir: {flags.log_dir}\n"
        #   f"training_dataset: {flags.training_dataset}\n"
        #   f"validation_dataset: {flags.validation_dataset}\n"
          f"test_dataset: {flags.test_dataset}\n"
          f"----------------------------")

    return flags

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '1' #设置显卡可见
    flags = FLAGS()

    # dataset_root = "/remote-home/share/cjk/syn2e/datasets"
    # train_root = os.join(dataset_root,"train")
    
    test_dataset = Syn_Events(flags.test_dataset)

    print("test dataset validation")

    # construct loader, responsible for streaming data to gpu
    test_loader = DataLoader(test_dataset,batch_size=flags.batch_size,
                               pin_memory=flags.pin_memory)

    # model, load and put to device


    print("Test step")
    for events,labels in tqdm.tqdm(iter(test_loader)):
        the_last_pairs = {"events":events[:,-1,:],"labels":labels[:,-1]}
        # print("events:{},event_corners:{}".format(events,labels))
        print("the last pair:{}".format(the_last_pairs))

