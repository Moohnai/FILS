import os
import numpy as np
import decord
import glob
import os.path as osp
import csv
import pandas as pd

from pytorchvideo.transforms import RandAugment, Normalize
import torch
from torch.utils.data._utils.collate import default_collate
import torchvision

from FILS.data.random_erasing import RandomErasing
from FILS.data.transforms import Permute, AdaptiveTemporalCrop, SpatialCrop
from FILS.data.clip_dataset import get_frame_ids, get_video_reader

def datetime2sec(str):
    hh, mm, ss = str.split(':')
    return int(hh) * 3600 + int(mm) * 60 + float(ss)

def read_metadata(metadata_fname, root=None, args=None, mode='train'):
    samples = []
    if args.dataset == 'ek100_cls':
        video_list = glob.glob(osp.join(root, '*/*.MP4'))
        fps_dict = {video: decord.VideoReader(video + '/0.MP4').get_avg_fps() for video in video_list}
        with open(metadata_fname) as f:
            csv_reader = csv.reader(f)
            _ = next(csv_reader)  # skip the header
            for idx, row in enumerate(csv_reader):
                pid, vid = row[1:3]
                start_timestamp, end_timestamp = datetime2sec(row[4]), datetime2sec(row[5])
                narration = row[8]
                verb, noun = int(row[10]), int(row[12])
                vid_path = '{}/{}'.format(pid, vid)
                fps = fps_dict[osp.join(root, vid_path + '.MP4')]
                # start_frame = int(np.round(fps * start_timestamp))
                # end_frame = int(np.ceil(fps * end_timestamp))
                samples.append((vid_path, start_timestamp, end_timestamp, fps, narration, verb, noun, idx))

        
        # ###########################################################################################################
        # if mode == 'train':
        #     a=[args.label_mapping['{}:{}'.format(x[-3], x[-2])] for x in samples]
        #     a_unique = list(set(a))
        #     a_unique.sort()
        #     counter = {x:0 for x in a_unique}
        #     for x in a:
        #         counter[x] += 1
        #     # sort the dictionary based on the values
        #     counter = {k: v for k, v in sorted(counter.items(), key=lambda item: item[1])}
        #     #save in a text file
        #     with open(os.path.join(os.path.dirname(root), 'whole.txt'), 'w') as f:
        #         for key, value in counter.items():
        #             f.write('%s:%s\n' % (key, value))

        # ############### classes that have more than 90 and less than 110 videos

        #     b_unique = [x for x in a_unique if counter[x] > 90 and counter[x] < 110] #3510data_35classes
        #     selected_samples = [x for x in samples if args.label_mapping['{}:{}'.format(x[-3], x[-2])] in b_unique]
        

        #     b=[args.label_mapping['{}:{}'.format(x[-3], x[-2])] for x in selected_samples]
        #     b_unique = list(set(b)) 
        #     b_unique.sort()
        #     counter = {args.label_mapping['{}:{}'.format(x[-3], x[-2])]:0 for x in selected_samples}
        #     for x in b:
        #         counter[x] += 1
        #     # sort the dictionary based on the values
        #     counter = {k: v for k, v in sorted(counter.items(), key=lambda item: item[1])}
        #     #save in a text file
        #     with open(os.path.join(os.path.dirname(root), 'sub_epic_middle_train.txt'), 'w') as f:
        #         for key, value in counter.items():
        #             f.write('%s:%s\n' % (key, value))

        #     samples = selected_samples#[:80]
            
        #     # store the unique classes args
        #     args.sub_unique_classes = b_unique
        # ###########################################################################################################
        # else:
        #     selected_samples = [x for x in samples if args.label_mapping['{}:{}'.format(x[-3], x[-2])] in args.sub_unique_classes]
            
        #     b=[args.label_mapping['{}:{}'.format(x[-3], x[-2])] for x in selected_samples]
        #     b_unique = list(set(b)) 
        #     b_unique.sort()
        #     counter = {args.label_mapping['{}:{}'.format(x[-3], x[-2])]:0 for x in selected_samples}
        #     for x in b:
        #         counter[x] += 1
        #     # sort the dictionary based on the values
        #     counter = {k: v for k, v in sorted(counter.items(), key=lambda item: item[1])}
        #     #save in a text file
        #     with open(os.path.join(os.path.dirname(root), 'sub_epic_middle_val.txt'), 'w') as f:
        #         for key, value in counter.items():
        #             f.write('%s:%s\n' % (key, value))
            
        #     samples = selected_samples#[:80]

        # if mode == 'test':
        #     # update label_mapping based on the selected classes
        #     args.label_mapping = {k: v for k, v in args.label_mapping.items() if v in args.sub_unique_classes}
        #     # reset values of the label_mapping to be in the range of 0 to len(args.label_mapping)
        #     args.label_mapping = {k: i for i, (k, v) in enumerate(args.label_mapping.items())}
        #     # update the mapping_act2v, mapping_act2n, and actions
        #     args.mapping_act2v = {i: int(vn.split(':')[0]) for (vn, i) in args.label_mapping.items()}
        #     args.mapping_act2n = {i: int(vn.split(':')[1]) for (vn, i) in args.label_mapping.items()}
        #     args.actions = pd.DataFrame.from_dict({'verb': args.mapping_act2v.values(), 'noun': args.mapping_act2n.values()})
        # ###########################################################################################################

    elif args.dataset == 'ssv2':
        with open(metadata_fname) as split_f:
            data = split_f.readlines()
            for id, line in enumerate(data):
                line_info = line.split(' ')
                assert len(line_info) == 3
                samples.append((line_info[0], line_info[1], int(line_info[2]), id))

    else: 
        with open(metadata_fname) as split_f:
            data = split_f.readlines()
            for line in data:
                line_info = line.split(' ')
                assert len(line_info) == 3
                samples.append((line_info[0], int(line_info[1]), int(line_info[2])))
    return samples

class VideoClsDataset(torch.utils.data.Dataset):
    def __init__ (self, root, metadata, mode='train',
                  clip_length=16, clip_stride=4,
                  threads=1,
                  crop_size=224, shorter_side_size=224,
                  new_height=256, new_width=340,
                  keep_aspect_ratio=True,
                  # fused augmentations need to be specified here
                  fast_rrc=False,
                  rrc_params=(224, (0.5, 1.0)),
                  fast_cc=False,
                  cc_params=(224,),
                  hflip_prob=0.5,
                  vflip_prob=0.,
                  num_segment=1, num_crop=1,
                  test_num_segment=5, test_num_crop=3,
                  args=None):
        self.root = root
        self.samples = read_metadata(metadata)
        assert mode in ['train', 'validation', 'test']
        self.mode = mode
        self.clip_length = clip_length
        self.clip_stride = clip_stride
        self.threads = threads
        self.crop_size = crop_size
        self.shorter_side_size = shorter_side_size
        self.new_height = new_height
        self.new_width = new_width
        self.keep_aspect_ratio = keep_aspect_ratio
        self.fast_rrc = fast_rrc
        self.rrc_params = rrc_params
        self.fast_cc = fast_cc
        self.cc_params = cc_params
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob
        self.num_segment = num_segment
        self.num_crop = num_crop
        self.test_num_segment = test_num_segment
        self.test_num_crop = test_num_crop
        self.args = args
        self.aug = False
        self.rand_erase = False
        if mode == 'train':
            self.aug = True
            if self.args.reprob > 0:
                self.rand_erase = True

        if mode == 'train' and not fast_rrc:
            transforms_list = [
                Permute([3, 0, 1, 2]),    # T H W C -> C T H W
                torchvision.transforms.RandomResizedCrop(self.crop_size, scale=(0.08, 1.0), ratio=(0.75, 1.3333)),
                torchvision.transforms.RandomHorizontalFlip(p=0.5),
            ]
            transforms_list += [
                Permute([1, 0, 2, 3]),
                RandAugment(magnitude=7, num_layers=4),
                Permute([1, 0, 2, 3]),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
            if self.rand_erase:
                transforms_list += [
                    Permute([1, 0, 2, 3]),
                    RandomErasing(probability=self.args.reprob, mode='pixel', max_count=1, num_splits=1, cube=True, device='cpu'),
                    Permute([1, 0, 2, 3]),
                ]
            self.data_transform = torchvision.transforms.Compose(transforms_list)
        elif mode == 'validation' and not fast_cc:
            self.data_transform = torchvision.transforms.Compose([
                Permute([3, 0, 1, 2]),
                torchvision.transforms.CenterCrop(self.crop_size),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        elif mode == 'test':
            self.data_transform = torchvision.transforms.Compose([
                Permute([3, 0, 1, 2]),
                torchvision.transforms.Resize(self.shorter_side_size),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                AdaptiveTemporalCrop(self.clip_length, self.test_num_segment, self.clip_stride),
                SpatialCrop(crop_size=self.shorter_side_size, num_crops=self.test_num_crop),
            ])
        else:
            assert (mode == 'train' and fast_rrc) or (mode == 'validation' and fast_cc)
            self.data_transform = None

    def __getitem__(self, index):
        if self.mode == 'train' and not self.fast_rrc:
            args = self.args
            buffer = self._load_frames(self.root, self.samples[index][0], norm=True)
            if len(buffer) == 0:
                while len(buffer) == 0:
                    index = np.random.randint(self.__len__())
                    buffer = self._load_frames(self.root, self.samples[index][0], norm=True)
            
            if args.repeated_aug > 1:
                frame_list = []
                label_list = []
                index_list = []
                for _ in range(args.repeated_aug):
                    buffer = torch.from_numpy(buffer.astype(np.float32))
                    new_frames = self.data_transform(buffer)
                    frame_list.append(new_frames)
                    label_list.append(self.samples[index][2])
                    index_list.append(index)
                return frame_list, label_list, index_list, {}
            else:
                buffer = torch.from_numpy(buffer.astype(np.float32))
                new_frames = self.data_transform(buffer)
                return new_frames, self.samples[index][2], index, {}
        elif self.mode == 'train' and self.fast_rrc:
            args = self.args
            buffer = self._load_frames(
                self.root, self.samples[index][0], norm=True,
                fast_rrc=self.fast_rrc, rrc_params=self.rrc_params,
                hflip_prob=self.hflip_prob, vflip_prob=self.vflip_prob,
            )
            if len(buffer) == 0:
                while len(buffer) == 0:
                    index = np.random.randint(self.__len__())
                    buffer = self._load_frames(
                        self.root, self.samples[index][0], norm=True,
                        fast_rrc=self.fast_rrc, rrc_params=self.rrc_params,
                        hflip_prob=self.hflip_prob, vflip_prob=self.vflip_prob,
                    )
                    buffer = torch.from_numpy(buffer.astype(np.float32))

            if args.repeated_aug > 1:
                frame_list = [buffer, ]
                label_list = [self.samples[index][2], ]
                index_list = [index, ]
                for _ in range(args.repeated_aug - 1):
                    buffer = self._load_frames(
                        self.root, self.samples[index][0], norm=True,
                        fast_rrc=self.fast_rrc, rrc_params=self.rrc_params,
                        hflip_prob=self.hflip_prob, vflip_prob=self.vflip_prob,
                    )
                    buffer = torch.from_numpy(buffer.astype(np.float32))
                    frame_list.append(buffer)
                    label_list.append(self.samples[index][2])
                    index_list.append(index)
                return frame_list, label_list, index_list, {}
            else:
                return buffer, self.samples[index][2], index, {}
        elif self.mode == 'validation' and not self.fast_cc:
            buffer = self._load_frames(self.root, self.samples[index][0], norm=True)
            if len(buffer) == 0:
                while len(buffer) == 0:
                    index = np.random.randint(self.__len__())
                    buffer = self._load_frames(self.root, self.samples[index][0], norm=True)
            buffer = torch.from_numpy(buffer.astype(np.float32))
            buffer = self.data_transform(buffer)
            return buffer, self.samples[index][2], index, self.samples[index][0]
        elif self.mode == 'validation' and self.fast_cc:
            buffer = self._load_frames(
                self.root, self.samples[index][0], norm=True,
                fast_cc=True, cc_params=self.cc_params,
            )
            if len(buffer) == 0:
                while len(buffer) == 0:
                    index = np.random.randint(self.__len__())
                    buffer = self._load_frames(
                        self.root, self.samples[index][0], norm=True,
                        fast_cc=True, cc_params=self.cc_params,
                    )
            buffer = torch.from_numpy(buffer.astype(np.float32))
            return buffer, self.samples[index][2], index, self.samples[index][0]
        elif self.mode == 'test':
            buffer = self._load_frames(self.root, self.samples[index][0], norm=True)
            buffer = torch.from_numpy(buffer.astype(np.float32))
            buffer = self.data_transform(buffer)
            if isinstance(buffer, list):
                buffer = torch.stack(buffer)
            return buffer, self.samples[index][2], index, self.samples[index][0]


    def _load_frames(self, root, vid, norm=False,
                     fast_rrc=False, rrc_params=(224, (0.5, 1.0)),
                     fast_cc=False, cc_params=(224,),
                     hflip_prob=0., vflip_prob=0.):
        fname = os.path.join(root, vid + '.mp4')

        if not os.path.exists(fname):
            print('No such video: ', fname)
            return []

        if os.path.getsize(fname) < 1 * 1024:
            print('SKIP: ', fname, " - ", os.path.getsize(fname))
            return []

        try:
            if self.keep_aspect_ratio:
                if fast_rrc:
                    width, height = rrc_params[0], rrc_params[0]
                elif fast_cc:
                    width, height = cc_params[0], cc_params[0]
                else:
                    width, height = -1, -1
                vr = decord.VideoReader(
                    fname, num_threads=self.threads,
                    width=width,
                    height=height,
                    use_rrc=fast_rrc, scale_min=rrc_params[1][0], scale_max=rrc_params[1][1],
                    use_centercrop=fast_cc,
                    hflip_prob=hflip_prob, vflip_prob=vflip_prob,
                )
            else:
                vr = decord.VideoReader(
                    fname, num_treads=self.threads,
                    width=self.new_width, height=self.new_height)
        except (IndexError, decord.DECORDError) as error:
            print(error)
            print("Fail to load video: ", fname)
            return []

        if self.mode == 'test':
            all_index = [x for x in range(0, len(vr))]
            while len(all_index) < self.clip_length * self.clip_stride:
                all_index.append(all_index[-1])
            vr.seek(0)
            buffer = vr.get_batch(all_index).asnumpy()
            if norm:
                buffer = buffer.astype(np.float32)
                buffer /= 255.
            return buffer

        # handle temporal segments
        total_length = int(self.clip_length * self.clip_stride)
        seg_len = len(vr) // self.num_segment

        all_index = []
        for i in range(self.num_segment):
            if seg_len <= total_length:
                index = np.arange(0, seg_len, step=self.clip_stride)
                index = np.concatenate((index, np.ones(self.clip_length - len(index)) * seg_len))
                index = np.clip(index, 0, seg_len - 1).astype(np.int64)
            else:
                end_id = np.random.randint(total_length, seg_len)
                start_id = end_id - total_length
                index = np.linspace(start_id, end_id, num=self.clip_length)
                index = np.clip(index, start_id, end_id - 1).astype(np.int64)
            index = index + i * seg_len
            all_index.extend(list((index)))

        vr.seek(0)
        buffer = vr.get_batch(all_index).asnumpy()
        if norm:
            buffer = buffer.astype(np.float32)
            buffer /= 255.
        return buffer

    def __len__(self):
        return len(self.samples)
    
class VideoClsDataset_FRIL(torch.utils.data.Dataset):
    def __init__ (self, root, metadata, mode='train',
                  clip_length=16, clip_stride=4,
                  threads=1,
                  crop_size=224, shorter_side_size=224,
                  new_height=256, new_width=340,
                  keep_aspect_ratio=True,
                  # fused augmentations need to be specified here
                  fast_rrc=False,
                  rrc_params=(224, (0.5, 1.0)),
                  fast_cc=False,
                  cc_params=(224,),
                  hflip_prob=0.5,
                  vflip_prob=0.,
                  num_segment=1, num_crop=1,
                  test_num_segment=5, test_num_crop=3,
                  args=None):
        self.root = root
        self.samples = read_metadata(metadata, root, args, mode)
        assert mode in ['train', 'validation', 'test']
        self.mode = mode
        self.clip_length = clip_length
        self.clip_stride = clip_stride
        self.threads = threads
        self.crop_size = crop_size
        self.shorter_side_size = shorter_side_size
        self.new_height = new_height
        self.new_width = new_width
        self.keep_aspect_ratio = keep_aspect_ratio
        self.fast_rrc = fast_rrc
        self.rrc_params = rrc_params
        self.fast_cc = fast_cc
        self.cc_params = cc_params
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob
        self.num_segment = num_segment
        self.num_crop = num_crop
        self.test_num_segment = test_num_segment
        self.test_num_crop = test_num_crop
        self.args = args
        self.aug = False
        self.rand_erase = False
        if mode == 'train':
            self.aug = True
            if self.args.reprob > 0:
                self.rand_erase = True

        if mode == 'train' and not fast_rrc:
            transforms_list = [
                Permute([3, 0, 1, 2]),    # T H W C -> C T H W
                torchvision.transforms.RandomResizedCrop(self.crop_size, scale=(0.08, 1.0), ratio=(0.75, 1.3333), antialias=True),
                torchvision.transforms.RandomHorizontalFlip(p=0.5),
            ]
            transforms_list += [
                Permute([1, 0, 2, 3]),
                RandAugment(magnitude=7, num_layers=4),
                Permute([1, 0, 2, 3]),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
            if self.rand_erase:
                transforms_list += [
                    Permute([1, 0, 2, 3]),
                    RandomErasing(probability=self.args.reprob, mode='pixel', max_count=1, num_splits=1, cube=True, device='cpu'),
                    Permute([1, 0, 2, 3]),
                ]
            self.data_transform = torchvision.transforms.Compose(transforms_list)
        elif mode == 'validation' and not fast_cc:
            self.data_transform = torchvision.transforms.Compose([
                Permute([3, 0, 1, 2]),
                torchvision.transforms.CenterCrop(self.crop_size),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        elif mode == 'test':
            self.data_transform = torchvision.transforms.Compose([
                Permute([3, 0, 1, 2]),
                torchvision.transforms.Resize(self.shorter_side_size, antialias=True),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                AdaptiveTemporalCrop(self.clip_length, self.test_num_segment, self.clip_stride),
                SpatialCrop(crop_size=self.shorter_side_size, num_crops=self.test_num_crop),
            ])
        else:
            assert (mode == 'train' and fast_rrc) or (mode == 'validation' and fast_cc)
            self.data_transform = None
            
    def __getitem__(self, index):
        if self.mode == 'train' and not self.fast_rrc:
            args = self.args
            if args.dataset == 'ek100_cls':
                buffer = self._load_frames_Epic(
                    self.root, self.samples[index][0], 
                    'MP4', self.samples[index][1], self.samples[index][2], norm=True,
                    chunk_len=self.args.video_chunk_length, 
                    clip_length=self.clip_length,
                    fast_rrc=self.fast_rrc, rrc_params=self.rrc_params,
                    fast_cc=self.fast_cc, cc_params=self.cc_params,
                    jitter=True, threads=self.threads,
                    hflip_prob=self.hflip_prob, vflip_prob=self.vflip_prob,
                )
            else:
                buffer = self._load_frames(self.root, self.samples[index][0], norm=True)
            if len(buffer) == 0:
                while len(buffer) == 0:
                    index = np.random.randint(self.__len__())
                    if args.dataset == 'ek100_cls':
                        buffer = self._load_frames_Epic(
                            self.root, self.samples[index][0],
                            'MP4', self.samples[index][1], self.samples[index][2], norm=True,
                            chunk_len=self.args.video_chunk_length,
                            clip_length=self.clip_length,
                            fast_rrc=self.fast_rrc, rrc_params=self.rrc_params,
                            fast_cc=self.fast_cc, cc_params=self.cc_params,
                            jitter=True, threads=self.threads,
                            hflip_prob=self.hflip_prob, vflip_prob=self.vflip_prob,
                        )
                    else:
                        buffer = self._load_frames(self.root, self.samples[index][0], norm=True)
            
            if args.repeated_aug > 1:
                frame_list = []
                label_list = []
                index_list = []
                for _ in range(args.repeated_aug):
                    buffer = torch.from_numpy(buffer.astype(np.float32))
                    new_frames = self.data_transform(buffer)
                    frame_list.append(new_frames)
                    if self.args.dataset == 'ek100_cls':
                        label_list.append(self.args.label_mapping['{}:{}'.format(self.samples[index][-3], self.samples[index][-2])])
                    else:
                        label_list.append(self.samples[index][2])
                    index_list.append(index)
                return frame_list, label_list, index_list, {}
            else:
                buffer = torch.from_numpy(buffer.astype(np.float32))
                new_frames = self.data_transform(buffer)
                if self.args.dataset == 'ek100_cls':
                    return new_frames, self.args.label_mapping['{}:{}'.format(self.samples[index][-3], self.samples[index][-2])], index, {}
                else:
                    return new_frames, self.samples[index][2], index, {}
        elif self.mode == 'train' and self.fast_rrc:
            args = self.args
            if args.dataset == 'ek100_cls':
                buffer = self._load_frames_Epic(
                    self.root, self.samples[index][0],
                    'MP4', self.samples[index][1], self.samples[index][2], norm=True,
                    chunk_len=self.args.video_chunk_length,
                    clip_length=self.clip_length,
                    fast_rrc=self.fast_rrc, rrc_params=self.rrc_params,
                    fast_cc=self.fast_cc, cc_params=self.cc_params,
                    jitter=True, threads=self.threads,
                    hflip_prob=self.hflip_prob, vflip_prob=self.vflip_prob,
                )
            else:
                buffer = self._load_frames(
                    self.root, self.samples[index][0], norm=True,
                    fast_rrc=self.fast_rrc, rrc_params=self.rrc_params,
                    hflip_prob=self.hflip_prob, vflip_prob=self.vflip_prob,
                )
            if len(buffer) == 0:
                while len(buffer) == 0:
                    index = np.random.randint(self.__len__())
                    if args.dataset == 'ek100_cls':
                        buffer = self._load_frames_Epic(
                            self.root, self.samples[index][0],
                            'MP4', self.samples[index][1], self.samples[index][2], norm=True,
                            chunk_len=self.args.video_chunk_length,
                            clip_length=self.clip_length,
                            fast_rrc=self.fast_rrc, rrc_params=self.rrc_params,
                            fast_cc=self.fast_cc, cc_params=self.cc_params,
                            jitter=True, threads=self.threads,
                            hflip_prob=self.hflip_prob, vflip_prob=self.vflip_prob,
                        )
                    else:
                        buffer = self._load_frames(
                            self.root, self.samples[index][0], norm=True,
                            fast_rrc=self.fast_rrc, rrc_params=self.rrc_params,
                            hflip_prob=self.hflip_prob, vflip_prob=self.vflip_prob,
                        )
                    buffer = torch.from_numpy(buffer.astype(np.float32))

            if args.repeated_aug > 1:
                frame_list = [buffer, ]
                if self.args.dataset == 'ek100_cls':
                    label_list = [self.args.label_mapping['{}:{}'.format(self.samples[index][-3], self.samples[index][-2])], ]
                else:
                    label_list = [self.samples[index][2], ]
                index_list = [index, ]
                for _ in range(args.repeated_aug - 1):
                    if args.dataset == 'ek100_cls':
                        buffer = self._load_frames_Epic(
                            self.root, self.samples[index][0],
                            'MP4', self.samples[index][1], self.samples[index][2], norm=True,
                            chunk_len=self.args.video_chunk_length,
                            clip_length=self.clip_length,
                            fast_rrc=self.fast_rrc, rrc_params=self.rrc_params,
                            fast_cc=self.fast_cc, cc_params=self.cc_params,
                            jitter=True, threads=self.threads,
                            hflip_prob=self.hflip_prob, vflip_prob=self.vflip_prob,
                        )
                    else:
                        buffer = self._load_frames(
                            self.root, self.samples[index][0], norm=True,
                            fast_rrc=self.fast_rrc, rrc_params=self.rrc_params,
                            hflip_prob=self.hflip_prob, vflip_prob=self.vflip_prob,
                        )
                    buffer = torch.from_numpy(buffer.astype(np.float32))
                    frame_list.append(buffer)
                    if self.args.dataset == 'ek100_cls':
                        label_list.append(self.args.label_mapping['{}:{}'.format(self.samples[index][-3], self.samples[index][-2])])
                    else:
                        label_list.append(self.samples[index][2])
                    index_list.append(index)
                return frame_list, label_list, index_list, {}
            else:
                if self.args.dataset == 'ek100_cls':
                    return buffer, self.args.label_mapping['{}:{}'.format(self.samples[index][-3], self.samples[index][-2])], index, {}
                else:
                    return buffer, self.samples[index][2], index, {}
        elif self.mode == 'validation' and not self.fast_cc:
            if self.args.dataset == 'ek100_cls':
                buffer = self._load_frames_Epic(
                    self.root, self.samples[index][0],
                    'MP4', self.samples[index][1], self.samples[index][2], norm=True,
                    chunk_len=self.args.video_chunk_length,
                    clip_length=self.clip_length,
                    fast_rrc=self.fast_rrc, rrc_params=self.rrc_params,
                    fast_cc=self.fast_cc, cc_params=self.cc_params,
                    jitter=False, threads=self.threads,
                    hflip_prob=self.hflip_prob, vflip_prob=self.vflip_prob,
                )
            else:
                buffer = self._load_frames(self.root, self.samples[index][0], norm=True)
            if len(buffer) == 0:
                while len(buffer) == 0:
                    index = np.random.randint(self.__len__())
                    if self.args.dataset == 'ek100_cls':
                        buffer = self._load_frames_Epic(
                            self.root, self.samples[index][0],
                            'MP4', self.samples[index][1], self.samples[index][2], norm=True,
                            chunk_len=self.args.video_chunk_length,
                            clip_length=self.clip_length,
                            fast_rrc=self.fast_rrc, rrc_params=self.rrc_params,
                            fast_cc=self.fast_cc, cc_params=self.cc_params,
                            jitter=False, threads=self.threads,
                            hflip_prob=self.hflip_prob, vflip_prob=self.vflip_prob,
                        )
                    else:
                        buffer = self._load_frames(self.root, self.samples[index][0], norm=True)
            buffer = torch.from_numpy(buffer.astype(np.float32))
            buffer = self.data_transform(buffer)
            if self.args.dataset == 'ek100_cls':
                return buffer, self.args.label_mapping['{}:{}'.format(self.samples[index][-3], self.samples[index][-2])], index, self.samples[index][0]
            else:
                return buffer, self.samples[index][2], index, self.samples[index][0]
        elif self.mode == 'validation' and self.fast_cc:
            if self.args.dataset == 'ek100_cls':
                buffer = self._load_frames_Epic(
                    self.root, self.samples[index][0],
                    'MP4', self.samples[index][1], self.samples[index][2], norm=True,
                    chunk_len=self.args.video_chunk_length,
                    clip_length=self.clip_length,
                    fast_rrc=self.fast_rrc, rrc_params=self.rrc_params,
                    fast_cc=True, cc_params=self.cc_params,
                    jitter=False, threads=self.threads,
                    hflip_prob=self.hflip_prob, vflip_prob=self.vflip_prob,
                )
            else:
                buffer = self._load_frames(
                    self.root, self.samples[index][0], norm=True,
                    fast_cc=True, cc_params=self.cc_params,
                )
            if len(buffer) == 0:
                while len(buffer) == 0:
                    index = np.random.randint(self.__len__())
                    if self.args.dataset == 'ek100_cls':
                        buffer = self._load_frames_Epic(
                            self.root, self.samples[index][0],
                            'MP4', self.samples[index][1], self.samples[index][2], norm=True,
                            chunk_len=self.args.video_chunk_length,
                            clip_length=self.clip_length,
                            fast_rrc=self.fast_rrc, rrc_params=self.rrc_params,
                            fast_cc=True, cc_params=self.cc_params,
                            jitter=False, threads=self.threads,
                            hflip_prob=self.hflip_prob, vflip_prob=self.vflip_prob,
                        )
                    else:
                        buffer = self._load_frames(
                            self.root, self.samples[index][0], norm=True,
                            fast_cc=True, cc_params=self.cc_params,
                        )
            buffer = torch.from_numpy(buffer.astype(np.float32))
            if self.args.dataset == 'ek100_cls':
                return buffer, self.args.label_mapping['{}:{}'.format(self.samples[index][-3], self.samples[index][-2])], index, self.samples[index][0]
            else:
                return buffer, self.samples[index][2], index, self.samples[index][0]
        elif self.mode == 'test':
            if self.args.dataset == 'ek100_cls':
                buffer = self._load_frames_Epic(
                    self.root, self.samples[index][0],
                    'MP4', self.samples[index][1], self.samples[index][2], norm=True,
                    chunk_len=self.args.video_chunk_length,
                    clip_length=self.clip_length,
                    fast_rrc=self.fast_rrc, rrc_params=self.rrc_params,
                    fast_cc=self.fast_cc, cc_params=self.cc_params,
                    jitter=False, threads=self.threads,
                    hflip_prob=self.hflip_prob, vflip_prob=self.vflip_prob,
                )
            else:
                buffer = self._load_frames(self.root, self.samples[index][0], norm=True)
            buffer = torch.from_numpy(buffer.astype(np.float32))
            buffer = self.data_transform(buffer)
            if isinstance(buffer, list):
                buffer = torch.stack(buffer)
            if self.args.dataset == 'ek100_cls':
                return buffer, self.args.label_mapping['{}:{}'.format(self.samples[index][-3], self.samples[index][-2])], index, self.samples[index][0]
            else:
                return buffer, self.samples[index][2], index, self.samples[index][0]


    def _load_frames(self, root, vid, norm=False,
                     fast_rrc=False, rrc_params=(224, (0.5, 1.0)),
                     fast_cc=False, cc_params=(224,),
                     hflip_prob=0., vflip_prob=0.):
        fname = os.path.join(root, vid + '.mp4')

        if not os.path.exists(fname):
            print('No such video: ', fname)
            return []

        if os.path.getsize(fname) < 1 * 1024:
            print('SKIP: ', fname, " - ", os.path.getsize(fname))
            return []

        try:
            if self.keep_aspect_ratio:
                if fast_rrc:
                    width, height = rrc_params[0], rrc_params[0]
                elif fast_cc:
                    width, height = cc_params[0], cc_params[0]
                else:
                    width, height = -1, -1
                vr = decord.VideoReader(
                    fname, num_threads=self.threads,
                    width=width,
                    height=height,
                    use_rrc=fast_rrc, scale_min=rrc_params[1][0], scale_max=rrc_params[1][1],
                    use_centercrop=fast_cc,
                    hflip_prob=hflip_prob, vflip_prob=vflip_prob,
                )
            else:
                vr = decord.VideoReader(
                    fname, num_treads=self.threads,
                    width=self.new_width, height=self.new_height)
        except (IndexError, decord.DECORDError) as error:
            print(error)
            print("Fail to load video: ", fname)
            return []

        if self.mode == 'test':
            all_index = [x for x in range(0, len(vr))]
            while len(all_index) < self.clip_length * self.clip_stride:
                all_index.append(all_index[-1])
            vr.seek(0)
            buffer = vr.get_batch(all_index).asnumpy()
            if norm:
                buffer = buffer.astype(np.float32)
                buffer /= 255.
            return buffer

        # handle temporal segments
        total_length = int(self.clip_length * self.clip_stride)
        seg_len = len(vr) // self.num_segment

        all_index = []
        for i in range(self.num_segment):
            if seg_len <= total_length:
                index = np.arange(0, seg_len, step=self.clip_stride)
                index = np.concatenate((index, np.ones(self.clip_length - len(index)) * seg_len))
                index = np.clip(index, 0, seg_len - 1).astype(np.int64)
            else:
                end_id = np.random.randint(total_length, seg_len)
                start_id = end_id - total_length
                index = np.linspace(start_id, end_id, num=self.clip_length)
                index = np.clip(index, start_id, end_id - 1).astype(np.int64)
            index = index + i * seg_len
            all_index.extend(list((index)))

        vr.seek(0)
        buffer = vr.get_batch(all_index).asnumpy()
        if norm:
            buffer = buffer.astype(np.float32)
            buffer /= 255.
        return buffer

    def _load_frames_Epic(self, root, vid, 
                     ext, second, end_second, norm=False,
                     chunk_len=300, fps=30, clip_length=32,
                     fast_rrc=False, rrc_params=(224, (0.5, 1.0)),
                     fast_cc=False, cc_params=(224,),
                     jitter=False, threads=1,
                     hflip_prob=0., vflip_prob=0.):
        
        assert fps > 0, 'fps should be greater than 0'
        fast_rcc = fast_cc
        rcc_params = cc_params
        
        fname = os.path.join(root, vid + '.' + ext)

        if not os.path.exists(fname):
            print('No such video: ', fname)
            return []

        if os.path.getsize(fname) < 1 * 1024:
            print('SKIP: ', fname, " - ", os.path.getsize(fname))
            return []

        try:
            # if self.keep_aspect_ratio:
            #     if fast_rrc:
            #         width, height = rrc_params[0], rrc_params[0]
            #     elif fast_cc:
            #         width, height = cc_params[0], cc_params[0]
            #     else:
            #         width, height = -1, -1
            #     vr = decord.VideoReader(
            #         fname, num_threads=self.threads,
            #         width=width,
            #         height=height,
            #         use_rrc=fast_rrc, scale_min=rrc_params[1][0], scale_max=rrc_params[1][1],
            #         use_centercrop=fast_cc,
            #         hflip_prob=hflip_prob, vflip_prob=vflip_prob,
            #     )
            # else:
            #     vr = decord.VideoReader(
            #         fname, num_treads=self.threads,
            #         width=self.new_width, height=self.new_height)
            
            ########################
            chunk_start = int(second) // chunk_len * chunk_len
            chunk_end = int(end_second) // chunk_len * chunk_len
            while True:
                video_filename = osp.join(root, '{}.{}'.format(vid, ext), '{}.{}'.format(chunk_end, ext))
                if not osp.exists(video_filename):
                    # print("{} does not exists!".format(video_filename))
                    chunk_end -= chunk_len
                else:
                    vr = decord.VideoReader(video_filename)
                    end_second = min(end_second, (len(vr) - 1) / fps + chunk_end)
                    assert chunk_start <= chunk_end
                    break
                
            # calculate frame_ids
            frame_ids = get_frame_ids(
                int(np.round(second * fps)),
                int(np.round(end_second * fps)),
                num_segments=clip_length, jitter=jitter
            )
            all_frame_ids = np.arange(int(np.round(second * fps)), int(np.round(end_second * fps)), dtype=np.int64)
            
            all_frames = []
            all_selected_frames = []
            # allocate absolute frame-ids into the relative ones
            for chunk in range(chunk_start, chunk_end + chunk_len, chunk_len):
                rel_frame_ids = list(filter(lambda x: int(chunk * fps) <= x < int((chunk + chunk_len) * fps), frame_ids))
                rel_frame_ids = [int(frame_id - chunk * fps) for frame_id in rel_frame_ids]
                rel_all_frame_ids = list(filter(lambda x: int(chunk * fps) <= x < int((chunk + chunk_len) * fps), all_frame_ids))
                rel_all_frame_ids = [int(frame_id - chunk * fps) for frame_id in rel_all_frame_ids]
                vr = get_video_reader(
                    osp.join(root, '{}.{}'.format(vid, ext), '{}.{}'.format(chunk, ext)),
                    num_threads=threads,
                    fast_rrc=fast_rrc, rrc_params=rrc_params,
                    fast_rcc=fast_rcc, rcc_params=rcc_params,
                )
                try:
                    # frames = vr.get_batch(rel_frame_ids).asnumpy()
                    frames = vr.get_batch(rel_all_frame_ids).asnumpy()
                except decord.DECORDError as error:
                    print(error)
                    # frames = vr.get_batch([0] * len(rel_frame_ids)).asnumpy()
                    frames = vr.get_batch([0] * len(rel_all_frame_ids)).asnumpy()
                except IndexError:
                    print(root, vid, ext, second, end_second)
                all_frames.append(frames)
                rel_frame_ids = [x - rel_all_frame_ids[0] for x in rel_frame_ids]
                all_selected_frames.append(frames[rel_frame_ids])
                if sum(map(lambda x: x.shape[0], all_selected_frames)) == clip_length:
                    break
            raw_buffer = np.concatenate(all_frames, axis=0).astype(np.float32)
            raw_selected_buffer = np.concatenate(all_selected_frames, axis=0).astype(np.float32)
            ########################
            
        except (IndexError, decord.DECORDError) as error:
            print(error)
            print("Fail to load video: ", fname)
            return []

        if self.mode == 'test':
            all_index = [x for x in range(0, len(vr))]
            while len(all_index) < self.clip_length * self.clip_stride:
                all_index.append(all_index[-1])
            # vr.seek(0)
            # buffer = vr.get_batch(all_index).asnumpy()
            # buffer = raw_buffer[all_index]
            buffer = raw_buffer
            if norm:
                buffer = buffer.astype(np.float32)
                buffer /= 255.
            return buffer

        # sample frames
        self.random_clip_sampling = True
        self.allow_clip_overlap = False
        fpc = clip_length
        fstp = self.clip_stride
        clip_len = int(fpc * fstp)
        partition_len = len(raw_buffer) // self.num_segment
        all_indices, clip_indices = [], []
        for i in range(self.num_segment):

            if partition_len > clip_len:
                # If partition_len > clip len, then sample a random window of
                # clip_len frames within the segment
                end_indx = clip_len
                if self.random_clip_sampling:
                    end_indx = np.random.randint(clip_len, partition_len)
                start_indx = end_indx - clip_len
                indices = np.linspace(start_indx, end_indx, num=fpc)
                indices = np.clip(indices, start_indx, end_indx-1).astype(np.int64)
                # --
                indices = indices + i * partition_len
            else:
                # If partition overlap not allowed and partition_len < clip_len
                # then repeatedly append the last frame in the segment until
                # we reach the desired clip length
                if not self.allow_clip_overlap:
                    indices = np.linspace(0, partition_len, num=partition_len // fstp)
                    indices = np.concatenate((indices, np.ones(fpc - partition_len // fstp) * partition_len,))
                    indices = np.clip(indices, 0, partition_len-1).astype(np.int64)
                    # --
                    indices = indices + i * partition_len

                # If partition overlap is allowed and partition_len < clip_len
                # then start_indx of segment i+1 will lie within segment i
                else:
                    sample_len = min(clip_len, len(vr)) - 1
                    indices = np.linspace(0, sample_len, num=sample_len // fstp)
                    indices = np.concatenate((indices, np.ones(fpc - sample_len // fstp) * sample_len,))
                    indices = np.clip(indices, 0, sample_len-1).astype(np.int64)
                    # --
                    clip_step = 0
                    if len(vr) > clip_len:
                        clip_step = (len(vr) - clip_len) // (self.num_segment - 1)
                    indices = indices + i * clip_step

            clip_indices.append(indices)
            all_indices.extend(list(indices))
        

        # vr.seek(0)
        # buffer = raw_selected_buffer
        buffer = raw_buffer[clip_indices[0]]
        if norm:
            buffer = buffer.astype(np.float32)
            buffer /= 255.
        return buffer

    def __len__(self):
        return len(self.samples)


def multiple_samples_collate(batch, fold=False):
    """
    Collate function for repeated augmentation. Each instance in the batch has
    more than one sample.
    Args:
        batch (tuple or list): data batch to collate.
    Returns:
        (tuple): collated data batch.
    """
    inputs, labels, video_idx, extra_data = zip(*batch)
    inputs = [item for sublist in inputs for item in sublist]
    labels = [item for sublist in labels for item in sublist]
    video_idx = [item for sublist in video_idx for item in sublist]
    inputs, labels, video_idx, extra_data = (
        default_collate(inputs),
        default_collate(labels),
        default_collate(video_idx),
        default_collate(extra_data),
    )
    if fold:
        return [inputs], labels, video_idx, extra_data
    else:
        return inputs, labels, video_idx, extra_data
