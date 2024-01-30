from os.path import dirname
import argparse
import torch
import tqdm
import os
import numpy as np

from utils.dataset import Syn_Events
from torch.utils.data import DataLoader
from torch.utils.data import default_collate
from utils.models_new import EventCornerClassifier
from utils.loss import cross_entropy_loss_and_accuracy

"""
    用于测试dataset和dataloader的读写状态，不是测试集
"""

#收集事件并处理，主要是处理事件长度不同的问题
def collate_events(data): 
    last_events = []
    last_labels = []
    events = []
    for i, d in enumerate(data):
        #把最后一个事件和标签记下来即可
        last_events.append(d[0][-1])
        last_labels.append(d[1][-1])
        
        ev = np.concatenate([d[0], i*np.ones((len(d[0]),1), dtype=np.float32)],1)
        events.append(ev)
    events = torch.from_numpy(np.concatenate(events,0))
    last_events = default_collate(last_events)
    last_labels = default_collate(last_labels)
    last_pairs = (last_events,last_labels)
    return events, last_pairs

def FLAGS():
    parser = argparse.ArgumentParser("""Train classifier using a learnt quantization layer.""")

    # training / validation dataset
    parser.add_argument("--test_dataset", default="/remote-home/share/cjk/syn2e/datasets/test")
    # logging options
    parser.add_argument("--log_dir", default="log/events_corner")

    # loader and device options
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", type=bool, default=True)
    parser.add_argument("--batch_size", type=int, default=4)

    flags = parser.parse_args()
    assert os.path.isdir(flags.test_dataset), f"Test dataset directory {flags.training_dataset} not found."

    print(f"----------------------------\n"
          f"Starting training with \n"
          f"batch_size: {flags.batch_size}\n"
          f"device: {flags.device}\n"
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
                               pin_memory=flags.pin_memory,collate_fn=collate_events)

    # model, load and put to device

    print("Test step")
    for events,last_pairs in tqdm.tqdm(iter(test_loader)):
        print("the last pair:{}".format(last_pairs))

