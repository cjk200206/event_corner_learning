import argparse
from pathlib import Path
import cv2
import numpy as np

from tqdm import tqdm
import sys
import os
sys.path.append('..')

from utils.scripts.visualization.eventreader import EventReader
from utils.utils.utils import get_timesurface_from_events


def render_for_model(x: np.ndarray, y: np.ndarray, pol: np.ndarray, H: int, W: int) -> np.ndarray:
    assert x.size == y.size == pol.size
    assert H > 0
    assert W > 0
    img = np.full((H,W), fill_value=0,dtype='uint8')
    mask = np.zeros((H,W),dtype='int32')
    mask1 = (x>=0)&(y>=0)&(W>x)&(H>y)
    mask[y[mask1],x[mask1]]= 1
    img[mask==1]= 255
    return img


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Visualize Events')
    parser.add_argument('--event_file_dir', type=str, default="/home/cjk2002/datasets/VECTOR/events",help='Path to events.h5 file')
    parser.add_argument('--output_path', default="/home/cjk2002/code/event_code/event_corner_learning/output/VECtor",help='Path to write file')
    parser.add_argument('--delta_time_ms', '-dt_ms', type=float, default=100.0, help='Time window (in milliseconds) to summarize events for visualization')
    args = parser.parse_args()

    event_filepath_dir = args.event_file_dir
    event_files = os.listdir(event_filepath_dir)
    event_filepaths = []
    for event_file in event_files:
        event_file = os.path.join(event_filepath_dir,event_file)
        event_filepaths.append(Path(event_file))
    output_path = Path(args.output_path)
    dt = args.delta_time_ms
    
    height = 480
    width = 640
    
    dir_counter = 0
    for event_filepath in event_filepaths:
        counter = 0
        os.makedirs(os.path.join(output_path,str(dir_counter),"imgs"),exist_ok=True)
        os.makedirs(os.path.join(output_path,str(dir_counter),"sae_50"),exist_ok=True)
        os.makedirs(os.path.join(output_path,str(dir_counter),"sae_75"),exist_ok=True)
        os.makedirs(os.path.join(output_path,str(dir_counter),"sae_100"),exist_ok=True)
        os.makedirs(os.path.join(output_path,str(dir_counter),"labels"),exist_ok=True)
        for events in tqdm(EventReader(event_filepath, dt)):
            p = events['p']
            x = events['x']
            y = events['y']
            t = events['t']
            img = render_for_model(x, y, p, height, width)
            sae_50 = get_timesurface_from_events(x,y,t,p,(height,width),50e-3)
            sae_75 = get_timesurface_from_events(x,y,t,p,(height,width),75e-3)
            sae_100 = get_timesurface_from_events(x,y,t,p,(height,width),100e-3)
            cv2.imwrite("{}/{}/imgs/{}.jpg".format(output_path,dir_counter,counter),img)
            test = (sae_50+1)*127.5
            cv2.imwrite("{}/{}/sae_50/{}.jpg".format(output_path,dir_counter,counter),(sae_50+1)*127.5)
            cv2.imwrite("{}/{}/sae_75/{}.jpg".format(output_path,dir_counter,counter),(sae_75+1)*127.5)
            cv2.imwrite("{}/{}/sae_100/{}.jpg".format(output_path,dir_counter,counter),(sae_100+1)*127.5)
            counter += 1
        dir_counter += 1 
    
    print("finished")