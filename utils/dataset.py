import numpy as np
from os import listdir
from os.path import join
import os
from torch.utils.data import Dataset

def random_shift_events(events, max_shift=20, resolution=(180, 240)):
    H, W = resolution
    x_shift, y_shift = np.random.randint(-max_shift, max_shift+1, size=(2,))
    events[:,0] += x_shift
    events[:,1] += y_shift

    valid_events = (events[:,0] >= 0) & (events[:,0] < W) & (events[:,1] >= 0) & (events[:,1] < H)
    events = events[valid_events]

    return events

def random_flip_events_along_x(events, resolution=(180, 240), p=0.5):
    H, W = resolution
    if np.random.random() < p:
        events[:,0] = W - 1 - events[:,0]
    return events


class NCaltech101:
    def __init__(self, root, augmentation=False):
        self.classes = listdir(root)

        self.files = []
        self.labels = []

        self.augmentation = augmentation

        for i, c in enumerate(self.classes):
            new_files = [join(root, c, f) for f in listdir(join(root, c))]
            self.files += new_files
            self.labels += [i] * len(new_files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        """
        returns events and label, loading events from aedat
        :param idx:
        :return: x,y,t,p,  label
        """
        label = self.labels[idx]
        f = self.files[idx]
        events = np.load(f).astype(np.float32)

        if self.augmentation:
            events = random_shift_events(events)
            events = random_flip_events_along_x(events)

        return events, label
    
class Syn_Events(Dataset):
    """
        syn_corner数据集通过syn2e建立,具体格式如下：
        /datasets
            /train
                /syn_polygon
                    /events
                        /0
                            /0000000000.txt
                            /0000000001.txt
                            /others
                        /1
                        /2
                        /others
                    /event_corners
                    /others
                /syn_mutiple_polygons
                /others
            /val
    """
    def __init__(self,root): #这里的root从/train或者/val开始
        self.events_paths = [] # e.g. /datasets/train/syn_polygon/events/0
        self.event_corners_paths = [] # e.g. /datasets/train/syn_polygon/event_corners/0

        self.events_files = [] # e.g. /datasets/train/syn_polygon/events/0/0000000000.txt
        self.event_corners_files = []

        for path, dirs, files in os.walk(root):
            if path.split('/')[-1] == 'events':
                for dir in sorted(dirs): # 加入文件夹/0 -> /xxx
                    self.events_paths.append(join(path,dir))
                    for file in sorted(listdir(join(path,dir))): #加入文件/0/00000000.txt -> /0/xxxxxxxx.txt
                        self.events_files.append(join(path,dir,file))

            elif path.split('/')[-1] =='event_corners':
                for dir in sorted(dirs):
                    self.event_corners_paths.append(join(path,dir))
                    for file in sorted(listdir(join(path,dir))):
                        self.event_corners_files.append(join(path,dir,file))
            else:
                continue
    
    def __len__(self):
        return len(self.events_files)
    
    def __getitem__(self, idx):
        """
        returns events and event_corners, load from txts
        :param idx:
        :return: x,y,t,p x,y,t,p
        """
        e_f = self.events_files[idx]
        e_c_f = self.event_corners_files[idx]
        events = np.loadtxt(e_f).astype(np.float32)
        event_corners = np.loadtxt(e_c_f).astype(np.float32)

        return events, event_corners

if __name__ =='__main__':
    dataset_root = "/remote-home/share/cjk/syn2e/datasets"
    train_root = join(dataset_root,"train")
    
    train_dataset = Syn_Events(train_root)

    print("test dataset validation")


        


        