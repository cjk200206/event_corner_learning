import h5py
import hdf5plugin
import sys

from utils.eventslicer import EventSlicer

if __name__ == "__main__":

    path = "/home/cjk2002/datasets/VECTOR/corridors_dolly1.synced.left_event.hdf5"

    with h5py.File(path,"r") as f:
        # print("Keys: %s" % f.keys())
        # event_group = f['events']
        # ms_to_idx_dataset = f['ms_to_idx']
        # t_offset_dataset = f['t_offset']
        # print("Keys: %s" % event_group.keys())
        # x_dataset = event_group['x']
        # y_dataset = event_group['y']
        # t_dataset = event_group['t']
        # p_dataset = event_group['p']

        # x = x_dataset[:]
        # y = y_dataset[:]
        # t = t_dataset[:]
        # p = p_dataset[:]
        # ms_to_idx = ms_to_idx_dataset[:]
        # t_offset = t_offset_dataset[:]

        # print('finished')

        eventslicer = EventSlicer(f)
        
        start_time_us = eventslicer.get_start_time_us()
        final_time_us = eventslicer.get_final_time_us()
        interval = final_time_us - start_time_us

        events = eventslicer.get_events(start_time_us,start_time_us+interval/10)
        idx_start, idx_end = eventslicer.get_time_indices_offsets(events['t'],start_time_us,final_time_us)

        print('finished')

        