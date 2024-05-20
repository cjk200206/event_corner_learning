import numpy as np
from os import listdir
from os.path import join
import os
import torch 
from pathlib import Path
import tqdm
from torch.utils.data import Dataset
from utils.utils.utils import add_salt_and_pepper_new,get_timesurface,get_timesurface_from_events,getLabels,inv_warp_image,inv_warp_image_batch,heatmap_nms
from utils.utils.homographies import sample_homography_np
from utils.scripts.visualization.eventreader import EventReader
from scipy.ndimage import zoom
from numpy.linalg import inv
from math import pi
import torch.nn.functional as F
import cv2

import tqdm

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

def event_cropping(events,length,percent = 0.1): #随机裁剪一个事件的片段，作为输入,最后一位单独作为需检验的。
    start_idx = np.random.randint(0,length-percent*length-1)
    end_idx = int(start_idx+percent*length)

    cropped_events = events[start_idx:end_idx]

    return cropped_events,start_idx,end_idx

#构建Vox和heatmap_label
def events_to_vox_and_heatmap(events, num_time_bins=10, grid_size=(260, 346)):
    vox = torch.zeros((num_time_bins, *grid_size))
    label_vox = torch.zeros((num_time_bins, *grid_size))
    heatmap = torch.zeros((num_time_bins, *grid_size))
    time_bins = np.linspace(events[:, 2].min(), events[:, 2].max(), num=num_time_bins+1)

    for i in range(num_time_bins):
        mask = (events[:, 2] >= time_bins[i]) & (events[:, 2] < time_bins[i + 1])
        event_subset = events[mask]
        x_indices = event_subset[:, 0].astype(int)
        y_indices = event_subset[:, 1].astype(int)
        vox[i, y_indices, x_indices] = 1

        #这是全的label_vox
        label_mask = (event_subset[:,4] == 1)
        event_corner_subset = event_subset[label_mask]
        x_indices = event_corner_subset[:, 0].astype(int)
        y_indices = event_corner_subset[:, 1].astype(int)
        label_vox[i, y_indices, x_indices] = 1

        #选择一个random点标为1，得到heatmap
        if event_corner_subset.shape[0] != 0:
            random_idx = torch.randint(0,event_corner_subset.shape[0],(1,))
            random_corner = event_corner_subset[random_idx]
            x_idx = random_corner[0].astype(int)
            y_idx = random_corner[1].astype(int)
            heatmap[i, y_idx, x_idx] = 1

    return vox,label_vox,heatmap

#无增强角点标记的事件数据，转化成vox
def events_to_vox(events, num_time_bins=10, grid_size=(260, 346)):
    vox = torch.zeros((num_time_bins, *grid_size))
    time_bins = np.linspace(events[:, 2].min(), events[:, 2].max(), num=num_time_bins+1)

    for i in range(num_time_bins):
        mask = (events[:, 2] >= time_bins[i]) & (events[:, 2] < time_bins[i + 1])
        event_subset = events[mask]
        x_indices = event_subset[:, 0].astype(int)
        y_indices = event_subset[:, 1].astype(int)
        vox[i, y_indices, x_indices] = 1

    return vox

#构建img的label
def corner_to_heatmap(label, grid_size=(260, 346)):
    heatmap = torch.zeros(grid_size)

    mask = (label[:,0]<346) & (label[:,0]>=0) & (label[:,1]<260) & (label[:,1]>=0 )

    x_indices = label[mask,0].astype(int)
    y_indices = label[mask,1].astype(int)
    heatmap[y_indices, x_indices] = 1

    return heatmap


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
    
class NCaltech101_Superpoint:
    def __init__(self, root, event_crop=False,num_time_bins = 10,grid_size=(260,346)):
        self.classes = listdir(root)

        self.files = []
        self.labels = []

        self.event_crop = event_crop # 决定是否需要裁剪事件片段
        self.num_time_bins = num_time_bins
        self.grid_size = grid_size

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

        #决定要不要裁剪
        if self.event_crop:
            events,_,_ = event_cropping(events,len(events),percent=0.5)

        #将事件转换到vox
        event_vox = events_to_vox(events, num_time_bins=self.num_time_bins, grid_size=self.grid_size)

        return event_vox, label


class Syn_Events(Dataset):
    """
        syn_corner数据集通过syn2e建立,具体格式如下：
        /datasets
            /train
                /syn_polygon
                    /augmented_events
                        /0
                            /0000000000.txt
                            /0000000001.txt
                            /others
                        /1
                        /2
                        /others
                    /event_corners
                    /events
                    /others
                /syn_mutiple_polygons
                /others
            /val
    """
    def __init__(self,root,event_crop = True): #这里的root从/train或者/val开始
        self.events_paths = [] # e.g. /datasets/train/syn_polygon/events/0
        self.event_corners_paths = [] # e.g. /datasets/train/syn_polygon/event_corners/0

        self.events_files = [] # e.g. /datasets/train/syn_polygon/events/0/0000000000.txt
        self.event_crop = event_crop # 决定是否需要裁剪事件片段

        for path, dirs, files in os.walk(root,followlinks=True):
            if path.split('/')[-1] == 'augmented_events':
                for dir in sorted(dirs): # 加入文件夹/0 -> /100
                    self.events_paths.append(join(path,dir))
                    for file in sorted(listdir(join(path,dir))): #加入文件/0/00000000.txt -> /0/xxxxxxxx.txt
                        self.events_files.append(join(path,dir,file))
            else:
                continue
    
    def __len__(self):
        return len(self.events_files)
    
    def __getitem__(self, idx):
        """
        returns events and event_corners, load from txts
        :param idx:
        :return: x,y,t,p  label
        """
        e_f = self.events_files[idx]
        augmented_events = np.loadtxt(e_f).astype(np.float32)


        if self.event_crop:
            # augmented_events,_,_ = event_cropping(augmented_events,len(augmented_events),piece=1001)
            augmented_events,_,_ = event_cropping(augmented_events,len(augmented_events),percent=0.1)
        
        events = augmented_events[:,0:4]
        labels = augmented_events[:,-1].astype(int)

        return events,labels

class Syn_Heatmaps(Dataset):
    """
        syn_corner数据集通过syn2e建立,具体格式如下：
        /datasets
            /train
                /syn_polygon
                    /augmented_events
                        /0
                            /0000000000.txt
                            /0000000001.txt
                            /others
                        /1
                        /2
                        /others
                    /event_corners
                    /events
                    /others
                /syn_mutiple_polygons
                /others
            /val
    """
    def __init__(self,root,event_crop = True,num_time_bins = 10,grid_size=(260, 346)): #这里的root从/train或者/val开始
        self.events_paths = [] # e.g. /datasets/train/syn_polygon/events/0
        self.event_corners_paths = [] # e.g. /datasets/train/syn_polygon/event_corners/0
        self.events_files = [] # e.g. /datasets/train/syn_polygon/events/0/0000000000.txt
        self.event_crop = event_crop # 决定是否需要裁剪事件片段
        self.num_time_bins = num_time_bins
        self.grid_size = grid_size

        for path, dirs, files in os.walk(root,followlinks=True):
            if path.split('/')[-1] == 'augmented_events':
                for dir in sorted(dirs): # 加入文件夹/0 -> /100
                    self.events_paths.append(join(path,dir))
                    for file in sorted(listdir(join(path,dir))): #加入文件/0/00000000.txt -> /0/xxxxxxxx.txt
                        self.events_files.append(join(path,dir,file))
            else:
                continue
    
    def __len__(self):
        return len(self.events_files)
    
    def __getitem__(self, idx):
        """
        returns events and event_corners, load from txts
        :param idx:
        :return: x,y,t,p  label
        """
        e_f = self.events_files[idx]
        augmented_events = np.loadtxt(e_f).astype(np.float32)

        #决定要不要裁剪
        if self.event_crop:
            augmented_events,_,_ = event_cropping(augmented_events,len(augmented_events),percent=0.5)
        
        #将事件转换到vox和heatmap
        event_vox, label_vox, heatmap = events_to_vox_and_heatmap(augmented_events, num_time_bins=self.num_time_bins, grid_size=self.grid_size)
        # #原始的事件和label
        # events = augmented_events[:,0:4]
        # labels = augmented_events[:,-1].astype(int)

        return event_vox, label_vox, heatmap

class Syn_Superpoint(Dataset):
    """
        syn_corner数据集通过syn2e建立,具体格式如下：
        /datasets
            /train
                /syn_polygon
                    /augmented_events
                        /0
                            /0000000000.txt
                            /0000000001.txt
                            /others
                        /1
                        /2
                        /others
                    /event_corners
                    /events
                    /others
                /syn_mutiple_polygons
                /others
            /val
    """
    def __init__(self,root,event_crop = True,num_time_bins = 10,grid_size=(260, 346),mode = "raw_files",test = False): #这里的root从/train或者/val开始
        self.events_paths = [] # e.g. /datasets/train/syn_polygon/events/0
        self.event_corners_paths = [] # e.g. /datasets/train/syn_polygon/event_corners/0
        self.events_files = [] # e.g. /datasets/train/syn_polygon/events/0/0000000000.txt
        self.preprocessed_files = []
        self.event_crop = event_crop # 决定是否需要裁剪事件片段
        self.num_time_bins = num_time_bins
        self.grid_size = grid_size
        self.mode = mode
        self.test = test

        for path, dirs, files in os.walk(root,followlinks=True):
            if self.mode == "raw_files":
                if path.split('/')[-1] == 'augmented_events':
                    for dir in sorted(dirs): # 加入文件夹/0 -> /100
                        self.events_paths.append(join(path,dir))
                        for file in sorted(listdir(join(path,dir))): #加入文件/0/00000000.txt -> /0/xxxxxxxx.txt
                            self.events_files.append(join(path,dir,file))
                else:
                    continue
            elif self.mode == "preprocessed_files":
                if path.split('/')[-1] == 'preprocessed':
                    for file in files: #加入文件/preprocessed/00000000.npz
                        self.preprocessed_files.append(join(path,file))
                else:
                    continue
    
    def __len__(self):
        if self.mode == "raw_files":
            return len(self.events_files)
        elif self.mode == "preprocessed_files":
            return len(self.preprocessed_files)
    
    def __getitem__(self, idx):
        """
        returns events and event_corners, load from txts
        :param idx:
        :return: x,y,t,p  label
        """
        if self.mode == "raw_files":
            e_f = self.events_files[idx]
            augmented_events = np.loadtxt(e_f).astype(np.float32)

            #决定要不要裁剪
            if self.event_crop:
                augmented_events,_,_ = event_cropping(augmented_events,len(augmented_events),percent=0.5)
            
            #将事件转换到vox和heatmap
            event_vox, label_vox, heatmap = events_to_vox_and_heatmap(augmented_events, num_time_bins=self.num_time_bins, grid_size=self.grid_size)

            if self.test == True:
                #数据增强加噪声
                # event_vox = add_salt_and_pepper_new(event_vox)
                pass

            # #原始的事件和label
            # events = augmented_events[:,0:4]
            # labels = augmented_events[:,-1].astype(int)

        elif self.mode == "preprocessed_files":
            loaded_data = np.load(self.preprocessed_files[idx])
            event_vox = loaded_data["event_vox"]
            label_vox = loaded_data["label_vox"]
            heatmap = loaded_data["heatmap"]

            event_vox = torch.from_numpy(event_vox).squeeze(0)
            label_vox = torch.from_numpy(label_vox).squeeze(0)
            heatmap = torch.from_numpy(heatmap).squeeze(0)

            #数据增强加噪声
            # event_vox = add_salt_and_pepper_new(event_vox)

        return event_vox, label_vox, heatmap

class VECtor(Dataset):
    """
        syn_corner数据集通过syn2e建立,具体格式如下：
        /datasets
            /train
                /0
                    /imgs
                    /labels
            /val
    """
    def __init__(self,root,mode = "predict"): #这里的root从/train或者/val开始,predict状态直接从两者上级目录开始
        self.img_paths = [] # e.g. /datasets/train/0/imgs
        self.sae_50_paths = [] # e.g. /datasets/train/0/sae_50
        self.sae_75_paths = [] # e.g. /datasets/train/0/sae_75
        self.sae_100_paths = [] # e.g. /datasets/train/0/sae_100
        self.label_paths = [] # e.g. /datasets/train/0/labels

        self.mode = mode
        

        for path, dirs, files in os.walk(root,followlinks=True):
            if self.mode == "predict":
                if path.split('/')[-1] == 'imgs':
                    for file in sorted(listdir(path)): #加入文件/0/imgs/0.jpg -> /0/imgs/xxx.jpg 
                        self.img_paths.append(join(path,file))
                elif path.split('/')[-1] == 'sae_50':
                    for file in sorted(listdir(path)): #加入文件/0/labels/0.jpg -> /0/labels/xxx.jpg 
                        self.sae_50_paths.append(join(path,file))
                elif path.split('/')[-1] == 'sae_75':
                    for file in sorted(listdir(path)): #加入文件/0/labels/0.jpg -> /0/labels/xxx.jpg 
                        self.sae_75_paths.append(join(path,file))
                elif path.split('/')[-1] == 'sae_100':
                    for file in sorted(listdir(path)): #加入文件/0/labels/0.jpg -> /0/labels/xxx.jpg 
                        self.sae_100_paths.append(join(path,file))
        
            elif self.mode == "train":
                if path.split('/')[-1] == 'imgs':
                    for file in sorted(listdir(path)): #加入文件/0/imgs/0.jpg -> /0/imgs/xxx.jpg 
                        self.img_paths.append(join(path,file))
                elif path.split('/')[-1] == 'sae_50':
                    for file in sorted(listdir(path)): #加入文件/0/labels/0.jpg -> /0/labels/xxx.jpg 
                        self.sae_50_paths.append(join(path,file))
                elif path.split('/')[-1] == 'sae_75':
                    for file in sorted(listdir(path)): #加入文件/0/labels/0.jpg -> /0/labels/xxx.jpg 
                        self.sae_75_paths.append(join(path,file))
                elif path.split('/')[-1] == 'sae_100':
                    for file in sorted(listdir(path)): #加入文件/0/labels/0.jpg -> /0/labels/xxx.jpg 
                        self.sae_100_paths.append(join(path,file))
                elif path.split('/')[-1] == 'labels':
                    for file in sorted(listdir(path)): #加入文件/0/labels/0.jpg -> /0/labels/xxx.jpg 
                        self.label_paths.append(join(path,file))
                    
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        """
        returns events and event_corners, load from txts
        :param idx:
        :return: x,y,t,p  label
        """

        # 只读入图像
        if self.mode == "predict":
            img_path = self.img_paths[idx]
            img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
            img = np.array(img)

            sae_50_path = self.sae_50_paths[idx]
            sae_50 = cv2.imread(sae_50_path,cv2.IMREAD_GRAYSCALE)
            sae_50 = np.array(sae_50)

            sae_75_path = self.sae_75_paths[idx]
            sae_75 = cv2.imread(sae_75_path,cv2.IMREAD_GRAYSCALE)
            sae_75 = np.array(sae_75)

            sae_100_path = self.sae_100_paths[idx]
            sae_100 = cv2.imread(sae_100_path,cv2.IMREAD_GRAYSCALE)
            sae_100 = np.array(sae_100)


            label = img

        # 读入事件图和标签
        elif self.mode == "train":
            img_path = self.img_paths[idx]
            img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
            img = np.array(img)

            sae_50_path = self.sae_50_paths[idx]
            sae_50 = cv2.imread(sae_50_path,cv2.IMREAD_GRAYSCALE)
            sae_50 = np.array(sae_50)

            sae_75_path = self.sae_75_paths[idx]
            sae_75 = cv2.imread(sae_75_path,cv2.IMREAD_GRAYSCALE)
            sae_75 = np.array(sae_75)

            sae_100_path = self.sae_100_paths[idx]
            sae_100 = cv2.imread(sae_100_path,cv2.IMREAD_GRAYSCALE)
            sae_100 = np.array(sae_100)

            label_path = self.label_paths[idx]
            label = cv2.imread(label_path,cv2.IMREAD_GRAYSCALE)
            label = np.array(label)
            
        img = torch.from_numpy(img).to(torch.float)
        sae_50 = torch.from_numpy(sae_50).to(torch.float)
        sae_75 = torch.from_numpy(sae_75).to(torch.float)
        sae_100 = torch.from_numpy(sae_100).to(torch.float)
        label = torch.from_numpy(label).to(torch.float)

        img = torch.where(img.cuda() > 0, torch.tensor(1.0).cuda(), img.cuda()) #大于0的地方全转到1
        img = torch.where(img.cuda() < 0, torch.tensor(0.0).cuda(), img.cuda()) #小于0的地方全转到0
        # 转回[-1,1]
        sae_50 = sae_50/127.5 - 1
        sae_75 = sae_75/127.5 - 1
        sae_100 = sae_100/127.5 - 1

        # # 转回[-1,1]
        # sae_50 = sae_50/255
        # sae_75 = sae_75/255
        # sae_100 = sae_100/255

        label = torch.where(label.cuda() > 0, torch.tensor(1.0).cuda(), label.cuda()) #大于0的地方全转到1
        label = torch.where(label.cuda() < 0, torch.tensor(0.0).cuda(), label.cuda()) #小于0的地方全转到0

        return img.cpu(),label.cpu(),sae_50,sae_75,sae_100,img_path

class Syn_Superpoint_SAE(Dataset):
    """
        syn_corner数据集通过syn2e建立,具体格式如下：
        /datasets
            /train
                /syn_polygon
                    /augmented_events
                        /0
                            /0000000000.txt
                            /0000000001.txt
                            /others
                        /1
                        /2
                        /others
                    /event_corners
                    /events
                    /others
                /syn_mutiple_polygons
                /others
            /val
    """
    def __init__(self,root,num_time_bins = 1,grid_size=(260, 346),mode = "raw_files",test = False): #这里的root从/train或者/val开始
        self.events_files = [] # e.g. /datasets/train/syn_polygon/events/0/0000000000.txt
        self.events_files_first = [] # The first file of the list, used for comparison.
        self.preprocessed_files = [] # e.g. /datasets/train/syn_polygon/preprocessed/0/0000000000.npz
        self.preprocessed_files_first = [] 
        self.num_time_bins = num_time_bins
        self.grid_size = grid_size
        self.mode = mode
        self.test = test


        self.label_vox = []
        self.sae_50 = []
        self.preprocessed_label_3d = []
        self.preprocessed_label_3d_transformed = []
        self.preprocessed_input_vox = []
        self.preprocessed_input_vox_transformed = []
        self.homography = []

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

        for path, dirs, files in os.walk(root,followlinks=True):
            if self.mode == "raw_files":
                if path.split('/')[-1] == 'augmented_events':
                    for dir in sorted(dirs): # 加入文件夹/0 -> /100
                        first_file = sorted(listdir(join(path,dir)))[0]
                        for file in sorted(listdir(join(path,dir))): #加入文件/0/00000000.txt -> /0/xxxxxxxx.txt
                            self.events_files_first.append(join(path,dir,first_file))
                            self.events_files.append(join(path,dir,file))
                else:
                    continue
            elif self.mode == "preprocessed_files":
                if path.split('/')[-1] == 'preprocessed_sae':
                    for dir in sorted(dirs): # 加入文件夹/0 -> /100
                        first_file = sorted(listdir(join(path,dir)))[0]
                        for file in sorted(listdir(join(path,dir))): #加入文件/0/00000000.npz -> /0/xxxxxxxx.npz
                            self.preprocessed_files_first.append(join(path,dir,first_file))
                            self.preprocessed_files.append(join(path,dir,file))
                else:
                    continue
            elif self.mode == "preprocessed_files_server":
                if path.split('/')[-1] == 'preprocessed_sae':
                    for dir in tqdm.tqdm(sorted(dirs)): # 加入文件夹/0 -> /100
                        first_file = sorted(listdir(join(path,dir)))[0]
                        for file in sorted(listdir(join(path,dir))): #加入文件/0/00000000.npz -> /0/xxxxxxxx.npz
                            self.preprocessed_files_first.append(join(path,dir,first_file))
                            self.preprocessed_files.append(join(path,dir,file))
                            data = np.load(join(path,dir,file))
                            label_vox = torch.from_numpy(data["label_vox"]).cpu()
                            sae_50 = torch.from_numpy(data["sae_50"]).cpu().unsqueeze(0)

                            self.label_vox.append(label_vox)
                            self.sae_50.append(sae_50)

                            # ######### 以下是正常的HA变换
                            # # 把数据转到gpu
                            # label_vox = label_vox.cpu()
                            # sae = sae_50.cpu()
                            # label_vox = crop_and_resize_to_resolution(label_vox)
                            # sae = crop_and_resize_to_resolution(sae)
                            # # 选择通道
                            # label_2d = label_vox[:,0,:,:]
                            # for i in range(label_2d.shape[0]):
                            #     label_2d[i] = torch.from_numpy(heatmap_nms(label_2d[i].cpu())) #给标签使用nms，筛除噪点
                            # input_vox = sae[:,0,:,:]
                            
                            # #做HA变换
                            # homography = sample_homography_np(np.array([2, 2]),**HA_params)
                            # homography = inv(homography)
                            # inv_homography = inv(homography)
                            # inv_homography = torch.tensor(inv_homography).to(torch.float32)
                            # homography = torch.tensor(homography).to(torch.float32).cpu()
                            # #images
                            # warped_imgs = inv_warp_image_batch(torch.cat((label_2d,input_vox),dim=0).unsqueeze(1),\
                            #                                 homography.unsqueeze(0).expand(label_2d.size(0)*2,-1,-1),device=input_vox.device)
                            # label_2d_transformed = warped_imgs[:label_2d.size(0),0]
                            # for i in range(label_2d_transformed.shape[0]):
                            #     label_2d_transformed[i] = torch.from_numpy(heatmap_nms(label_2d_transformed[i].cpu())) #给标签使用nms，筛除噪点
                            # input_vox_transformed = warped_imgs[label_2d.size(0):,0]
                            # #增加椒盐噪声
                            # input_vox = add_salt_and_pepper_new(input_vox,type="sae").unsqueeze(1).cpu()
                            # input_vox_transformed = add_salt_and_pepper_new(input_vox_transformed,type="sae").unsqueeze(1).cpu()
                            # #保证标签为1
                            # for i in range(label_2d_transformed.shape[0]):
                            #     label_2d_transformed[i] = torch.from_numpy(heatmap_nms(label_2d_transformed[i].cpu())) #给标签使用nms，筛除噪点

                            # #转换标签
                            # label_3d = getLabels(label_2d.unsqueeze(1),8).cpu()
                            # label_3d_transform = getLabels(label_2d_transformed.unsqueeze(1),8).cpu()

                            # self.preprocessed_input_vox.append(input_vox)
                            # self.preprocessed_input_vox_transformed.append(input_vox_transformed)
                            # self.preprocessed_label_3d.append(label_3d)
                            # self.preprocessed_label_3d_transformed.append(label_3d_transform)
                            # self.homography.append(homography)


                else:
                    continue
    
    def __len__(self):
        if self.mode == "raw_files":
            return len(self.events_files)
        elif self.mode == "preprocessed_files":
            return len(self.preprocessed_files)
        elif self.mode == "preprocessed_files_server":
            return len(self.label_vox)
            # return len(self.preprocessed_input_vox)
    
    def __getitem__(self, idx):
        """
        returns events and event_corners, load from txts
        :param idx:
        :return: x,y,t,p  label
        """
        if self.mode == "raw_files":
            e_f = self.events_files[idx]
            augmented_events = np.loadtxt(e_f).astype(np.float32)
            #转换到sae
            sae_50 = get_timesurface(e_f,img_size=self.grid_size,tau = 50e-3)
            sae_75 = get_timesurface(e_f,img_size=self.grid_size,tau = 75e-3)
            sae_100 = get_timesurface(e_f,img_size=self.grid_size,tau = 100e-3)
            #将事件转换到vox和heatmap
            event_vox, label_vox, heatmap = events_to_vox_and_heatmap(augmented_events, num_time_bins=self.num_time_bins, grid_size=self.grid_size)

            # if self.test == True:
            #     # 返回参照事件
            #     first_e_f = self.events_files_first[idx]
            #     first_augmented_events = np.loadtxt(first_e_f).astype(np.float32)
            #     #转换到sae
            #     sae_first = get_timesurface(first_e_f,img_size=self.grid_size)
            #     #将事件转换到vox和heatmap
            #     event_vox_first, label_vox_first, heatmap_first = events_to_vox_and_heatmap(first_augmented_events, num_time_bins=self.num_time_bins, grid_size=self.grid_size)


            #     return event_vox, label_vox, heatmap, sae, event_vox_first, label_vox_first, heatmap_first, sae_first
                
            return event_vox, label_vox, heatmap, sae_50, sae_75, sae_100, e_f


        elif self.mode == "preprocessed_files":
            loaded_data = np.load(self.preprocessed_files[idx])
            event_vox = loaded_data["event_vox"]
            label_vox = loaded_data["label_vox"]
            heatmap = loaded_data["heatmap"]
            sae_50 = loaded_data["sae_50"]
            sae_75 = loaded_data["sae_75"]
            sae_100 = loaded_data["sae_100"]

            event_vox = torch.from_numpy(event_vox).squeeze(0)
            label_vox = torch.from_numpy(label_vox).squeeze(0)
            heatmap = torch.from_numpy(heatmap).squeeze(0)
            sae_50 = torch.from_numpy(sae_50)

            # loaded_data = np.load(self.preprocessed_files_first[idx])
            # event_vox_first = loaded_data["event_vox"]
            # label_vox_first = loaded_data["label_vox"]
            # heatmap_first = loaded_data["heatmap"]
            # sae_first = loaded_data["sae"]

            # event_vox_first = torch.from_numpy(event_vox_first).squeeze(0)
            # label_vox_first = torch.from_numpy(label_vox_first).squeeze(0)
            # heatmap_first = torch.from_numpy(heatmap_first).squeeze(0)
            # sae_first = torch.from_numpy(sae_first)

            return event_vox, label_vox, heatmap, sae_50, sae_75, sae_100\
                #   event_vox_first, label_vox_first, heatmap_first, sae_first
        
        elif self.mode == "preprocessed_files_server":

            # input_vox = self.preprocessed_input_vox[idx]
            # input_vox_transformed = self.preprocessed_input_vox_transformed[idx]
            # label_3d = self.preprocessed_label_3d[idx]
            # label_3d_transform = self.preprocessed_label_3d_transformed[idx]
            # homography = self.homography[idx]
            
            # return input_vox,input_vox_transformed,label_3d,label_3d_transform,homography

            label_vox = self.label_vox[idx].squeeze(1)
            sae_50 = self.sae_50[idx].squeeze(1)

            return label_vox,sae_50

        
        else:
            raise ValueError("use preprocessed_files or raw_files!")
        
#临时修改成图片
class Pic_Superpoint(Dataset):
    """
        syn_corner数据集通过syn2e建立,具体格式如下：
        /datasets
            /train
                /syn_polygon
                    /augmented_events
                        /0
                            /0000000000.txt
                            /0000000001.txt
                            /others
                        /1
                        /2
                        /others
                    /event_corners
                    /events
                    /others
                /syn_mutiple_polygons
                /others
            /val
    """
    def __init__(self,root,event_crop = True,num_time_bins = 10,grid_size=(260, 346)): #这里的root从/train或者/val开始
        self.img_paths = [] # e.g. /datasets/train/syn_polygon/events/0
        self.corners_paths = [] # e.g. /datasets/train/syn_polygon/event_corners/0
        self.img_files = [] # e.g. /datasets/train/syn_polygon/events/0/0000000000.txt
        self.corners_files = [] # e.g. /datasets/train/syn_polygon/event_corners/0
        self.event_crop = event_crop # 决定是否需要裁剪事件片段
        self.num_time_bins = num_time_bins
        self.grid_size = grid_size

        for path, dirs, files in os.walk(root,followlinks=True):
            if path.split('/')[-1] == 'img':
                for dir in sorted(dirs): # 加入文件夹/0 -> /100
                    self.img_paths.append(join(path,dir))
                    for file in sorted(listdir(join(path,dir))): #加入文件/0/00000000.txt -> /0/xxxxxxxx.txt
                        self.img_files.append(join(path,dir,file))
            elif path.split('/')[-1] == 'points':
                for dir in sorted(dirs): # 加入文件夹/0 -> /100
                    self.corners_paths.append(join(path,dir))
                    for file in sorted(listdir(join(path,dir))): #加入文件/0/00000000.txt -> /0/xxxxxxxx.txt
                        self.corners_files.append(join(path,dir,file))
            else:
                continue
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        """
        returns events and event_corners, load from txts
        :param idx:
        :return: x,y,t,p  label
        """
        img_file = self.img_files[idx]
        img = cv2.imread(img_file,cv2.IMREAD_GRAYSCALE)
        img = np.array(img)
        corner_file = self.corners_files[idx]
        corner = np.loadtxt(corner_file).astype(np.float32)

        #决定要不要裁剪
        # if self.event_crop:
            # augmented_events,_,_ = event_cropping(augmented_events,len(augmented_events),percent=0.5)
        
        #将事件转换到vox和heatmap
        # event_vox, label_vox, heatmap = events_to_vox_and_heatmap(augmented_events, num_time_bins=self.num_time_bins, grid_size=self.grid_size)
        # #原始的事件和label
        # events = augmented_events[:,0:4]
        # labels = augmented_events[:,-1].astype(int)

        heatmap = corner_to_heatmap(corner)

        return img, heatmap

class DSEC(Dataset):
    def __init__(self, root, mode="raw_files"):
        """
            /root
                /events
                    /zurich_city...
                        /events.h5
                        /rectify_map.h5
                    /interlake_...
                /images
                    /zurich_city...
                        /000000.png
                        ...
                    /interlake_...
                /timestamps
                    /interlake_xxx.txt  # timestamps for images
                    ...
        """
        self.dirs = listdir(join(root,"events")) # different sequence names
        # 原始的数据
        self.img_files = []
        self.event_files = []
        self.timestamps_files = []
        # 经过预处理的数据
        self.labels = []
        self.sae_files = []

        self.mode = mode # raw_files or preprocessed_files
        self.img_size = (1080,1440)
        self.event_size = (480,640)

        new_time_stamp_files = [join(root,"timestamps",f) for f in listdir(join(root,"timestamps"))]
        self.timestamps_files += new_time_stamp_files

        for iter, c in enumerate(self.dirs):
            name = c.split('events')
            name.insert(1,'images_rectified')
            img_name = "".join(name)

            new_event_files = [join(root,"events",c,"events.h5")]
            new_img_files = [join(root,"images",img_name, f) for f in listdir(join(root,"images",img_name))]

            self.event_files += new_event_files
            self.img_files += new_img_files

            if self.mode == "preprocessed_files" or "preprocessed_files_server":
                new_labels = [join(root,"labels",img_name, f) for f in listdir(join(root,"labels",img_name))]
                self.labels += new_labels
                
            
            # 预处理原始事件到分割事件
            if self.mode == "raw_files":
                if os.path.exists(join(root,"saes")) == False:
                    os.makedirs(join(root,"saes"))
                if os.path.exists(join(root,"saes",c)) == False:
                    os.makedirs(join(root,"saes",c))
                    print("preprocess raw events to saes ...")
                    for i,events in tqdm.tqdm(enumerate(EventReader(Path(new_event_files[0]), 50))):
                        x,y,t,p = events['x'],events['y'],events['t'],events['p']
                        sae = get_timesurface_from_events(x,y,t,p,self.event_size,scaling_factor=10e-3)
                        vox = events_to_vox(np.array([x,y,t,p]).transpose(),1,self.event_size)
                        np.savez_compressed(join(root,"saes",c,"{:06d}".format(i+1)),sae=sae,vox=vox) # except first and last img
            
            new_sae = [join(root,"saes",c, f) for f in listdir(join(root,"saes",c))]
            self.sae_files += new_sae
            # assert len(new_img_files)-2 == len(new_sae), "saes and imgs must be the same length" # except first and last img
            # assert len(new_labels) == len(new_sae), "saes and labels must be the same length"

        # mode for server, load data to memory
        if self.mode == "preprocessed_files_server":
            self.img = []
            self.label = []
            self.sae = []
            self.vox = []
            for idx in tqdm.tqdm(range(self.__len__())):
                img_file = self.img_files[idx+1]
                img = cv2.imread(img_file)
                img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

                sae_file = self.sae_files[idx]
                data = np.load(sae_file)
                sae = data['sae']
                vox = data['vox']

                label = 0
                
                label_file = self.labels[idx]
                label = cv2.imread(label_file) 
                label = cv2.cvtColor(label,cv2.COLOR_RGB2GRAY)
                
                # 计算缩放因子
                zoom_factors = (480 / 1080, 640 / 1440)

                # 缩放图像
                label = zoom(label, zoom_factors)
                label = label/255

                self.img.append(img)
                self.label.append(label)
                self.sae.append(sae)
                self.vox.append(vox)


    def __len__(self):
        return len(self.sae_files) # except first and last img

    def __getitem__(self, idx):
        """
        returns events and label, loading events from aedat
        :param idx:
        :return: x,y,t,p,  label
        """
        if self.mode != "preprocessed_files_server":
            img_file = self.img_files[idx+1]
            img = cv2.imread(img_file)
            img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

            sae_file = self.sae_files[idx]
            data = np.load(sae_file)
            sae = data['sae']
            vox = data['vox']
            sae_img = ((sae+1)*127.5).astype(np.int8)
            vox_img = vox*255

            label = 0
            if self.mode == "preprocessed_files":
                label_file = self.labels[idx]
                label = cv2.imread(label_file) 
                label = cv2.cvtColor(label,cv2.COLOR_RGB2GRAY)
                
                # 计算缩放因子
                zoom_factors = (480 / 1080, 640 / 1440)

                # 缩放图像
                label = zoom(label, zoom_factors)
                label = label/255
        else:
            img = self.img[idx]
            sae = self.sae[idx]
            vox = self.vox[idx]
            label = self.label[idx]
            img_file = self.img_files[idx+1]
            sae_img = ((sae+1)*127.5).astype(np.int8)
            vox_img = vox*255

        return img,sae,sae_img,vox,vox_img,label,img_file


if __name__ =='__main__':
    dataset_root = "/remote-home/share/cjk/syn2e/datasets"
    train_root = join(dataset_root,"train")
    
    train_dataset = Syn_Events(train_root)

    print("test dataset validation")


        


        