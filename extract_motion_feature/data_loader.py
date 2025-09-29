import os

import pandas as pd
from PIL import Image

import torch
from torch.utils import data
import numpy as np
import scipy.io as scio
import cv2


class VideoDataset_NR_SlowFast_feature(data.Dataset):
    """Read data from the original dataset for feature extraction"""
    def __init__(self, data_dir, transform, resize):
        super(VideoDataset_NR_SlowFast_feature, self).__init__()



        self.video_names = [
            os.path.join(root, file)
            for root, dirs, files in os.walk(data_dir)
            for file in files
            if file.endswith(".mp4")
        ]
        print(f"total data: {len(self.video_names)}")

        

        self.transform = transform           
        self.videos_dir = data_dir
        self.resize = resize
        self.length = len(self.video_names)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        video_name = self.video_names[idx]
        video_name_str = video_name[:-4]
        filename=os.path.join(self.videos_dir, video_name)

        # if os.path.exists(os.path.join('/data2/xxx-1/video_database/SlowFast_feature/jitter2/', video_name_str)):
        #     print(video_name, "has been processed!")
        #     return torch.ones((3, 224, 224)), video_name_str

        try:

            video_capture = cv2.VideoCapture()
            video_capture.open(filename)
            cap=cv2.VideoCapture(filename)

            video_channel = 3
            
            video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_frame_rate = int(round(cap.get(cv2.CAP_PROP_FPS)))

            if video_frame_rate == 0:
                video_clip = 10
            else:
                video_clip = int(video_length/video_frame_rate)

            video_clip_min = 8

            video_length_clip = 32             

            transformed_frame_all = torch.zeros([video_length, video_channel, self.resize, self.resize])

            transformed_video_all = []
            
            video_read_index = 0
            for i in range(video_length):
                has_frames, frame = video_capture.read()
                if has_frames:
                    read_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    read_frame = self.transform(read_frame)
                    transformed_frame_all[video_read_index] = read_frame
                    video_read_index += 1


            if video_read_index < video_length:
                for i in range(video_read_index, video_length):
                    transformed_frame_all[i] = transformed_frame_all[video_read_index - 1]
    
            video_capture.release()

            for i in range(video_clip):
                transformed_video = torch.zeros([video_length_clip, video_channel, self.resize, self.resize])
                if (i*video_frame_rate + video_length_clip) <= video_length:
                    transformed_video = transformed_frame_all[i*video_frame_rate : (i*video_frame_rate + video_length_clip)]
                else:
                    transformed_video[:(video_length - i*video_frame_rate)] = transformed_frame_all[i*video_frame_rate :]
                    for j in range((video_length - i*video_frame_rate), video_length_clip):
                        transformed_video[j] = transformed_video[video_length - i*video_frame_rate - 1]
                transformed_video_all.append(transformed_video)

            if video_clip < video_clip_min:
                for i in range(video_clip, video_clip_min):
                    transformed_video_all.append(transformed_video_all[video_clip - 1])
        
            return transformed_video_all, video_name_str

        except:
            print(video_name, "error!")
            return torch.ones((3, 224, 224)), video_name_str