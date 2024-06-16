import numpy as np
import torch
import pandas as pd


train_caption_embed_path = '/home/mona/SSVLI/dataset/epic_kitchens/EPIC_100_train_mixed_captions.pt'
epic_captions_train_path = '../../home/mona/SSVLI/dataset/epic_kitchens/epic_captions_train.csv'


train_caption_embed = torch.load(train_caption_embed_path)
epic_captions_train = pd.read_csv(epic_captions_train_path)

video_text_dict = {}
epic_captions_train = pd.read_csv(epic_captions_train_path)
for i in range(len(epic_captions_train)):
    video = epic_captions_train['video'][i].split("/")[-1].split(".")[-2]
    video_text_dict[video] = train_caption_embed[i].cpu()

torch.save(video_text_dict, '/home/mona/FRIL/avion/datasets/EK100/epic_embedded_mix_captions_train_dict.pt')


