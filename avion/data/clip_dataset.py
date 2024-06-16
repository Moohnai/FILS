import csv
import glob
import math
import os.path as osp
import pickle
import random
import numpy as np
import pandas as pd
import torch
import orjson
from torchvision import tv_tensors
from avion.data.kinetics_dataset import KineticsDataset_FRIL

import decord


def datetime2sec(str):
    hh, mm, ss = str.split(':')
    return int(hh) * 3600 + int(mm) * 60 + float(ss)


def get_frame_ids(start_frame, end_frame, num_segments=32, jitter=True):
    frame_ids = np.convolve(np.linspace(start_frame, end_frame, num_segments + 1), [0.5, 0.5], mode='valid')
    if jitter:
        seg_size = float(end_frame - start_frame - 1) / num_segments
        shift = (np.random.rand(num_segments) - 0.5) * seg_size
        frame_ids += shift
    return frame_ids.astype(int).tolist()

def video_loader_by_frames(root, vid, frame_ids, start_frame=None, end_frame=None, num_clips=1, clip_length=16, jitter=False):
    vr = decord.VideoReader(osp.join(root, vid))
    # make sure the frame_ids are within the range
    vid_len = len(vr)
    if frame_ids[-1] >= vid_len:
        frame_ids = get_frame_ids(start_frame, vid_len, num_segments=num_clips * clip_length, jitter=jitter)
    try:
        frames = vr.get_batch(frame_ids).asnumpy()
        frames = [torch.tensor(frame, dtype=torch.float32) for frame in frames]
    except (IndexError, decord.DECORDError) as error:
        print(error)
        print("Erroneous video: ", vid)
        frames = [torch.zeros((240, 320, 3)) for _ in range(len(frame_ids))]
    return torch.stack(frames, dim=0)


def get_video_reader(videoname, num_threads, fast_rrc, rrc_params, fast_rcc, rcc_params):
    video_reader = None
    if fast_rrc:
        video_reader = decord.VideoReader(
            videoname,
            num_threads=num_threads,
            width=rrc_params[0], height=rrc_params[0],
            use_rrc=True, scale_min=rrc_params[1][0], scale_max=rrc_params[1][1],
        )
    elif fast_rcc:
        video_reader = decord.VideoReader(
            videoname,
            num_threads=num_threads,
            width=rcc_params[0], height=rcc_params[0],
            use_rcc=True,
        )
    else:
        video_reader = decord.VideoReader(videoname, num_threads=num_threads)
    return video_reader


def video_loader(root, vid, ext, second, end_second,
                 chunk_len=300, fps=30, clip_length=32,
                 threads=1,
                 fast_rrc=False, rrc_params=(224, (0.5, 1.0)),
                 fast_rcc=False, rcc_params=(224, ),
                 jitter=False):
    # assert fps > 0, 'fps should be greater than 0'

    if chunk_len == -1:
        vr = get_video_reader(
            osp.join(root, '{}.{}'.format(vid, ext)),
            num_threads=threads,
            fast_rrc=fast_rrc, rrc_params=rrc_params,
            fast_rcc=fast_rcc, rcc_params=rcc_params,
        )
        if fps == -1:
            fps = vr.get_avg_fps()
        end_second = min(end_second, len(vr) / fps)

        # calculate frame_ids
        frame_offset = int(np.round(second * fps))
        total_duration = max(int((end_second - second) * fps), clip_length)
        frame_ids = get_frame_ids(frame_offset, min(frame_offset + total_duration, len(vr)), num_segments=clip_length, jitter=jitter)

        # load frames
        assert max(frame_ids) < len(vr)
        try:
            frames = vr.get_batch(frame_ids).asnumpy()
        except decord.DECORDError as error:
            print(error)
            frames = vr.get_batch([0] * len(frame_ids)).asnumpy()
    
        return torch.from_numpy(frames.astype(np.float32)), frame_ids

    else:
        chunk_start = int(second) // chunk_len * chunk_len
        chunk_end = int(end_second) // chunk_len * chunk_len
        while True:
            video_filename = osp.join(root, '{}.{}'.format(vid, ext), '{}.{}'.format(chunk_end, ext))
            if not osp.exists(video_filename):
                # print("{} does not exists!".format(video_filename))
                chunk_end -= chunk_len
            else:
                vr = decord.VideoReader(video_filename)
                if fps == -1:
                    fps = vr.get_avg_fps()
                end_second = min(end_second, (len(vr) - 1) / fps + chunk_end)
                assert chunk_start <= chunk_end
                break
        # calculate frame_ids
        frame_ids = get_frame_ids(
            int(np.round(second * fps)),
            int(np.round(end_second * fps)),
            num_segments=clip_length, jitter=jitter
        )
        all_frames = []
        # allocate absolute frame-ids into the relative ones
        for chunk in range(chunk_start, chunk_end + chunk_len, chunk_len):
            rel_frame_ids = list(filter(lambda x: int(chunk * fps) <= x < int((chunk + chunk_len) * fps), frame_ids))
            rel_frame_ids = [int(frame_id - chunk * fps) for frame_id in rel_frame_ids]
            vr = get_video_reader(
                osp.join(root, '{}.{}'.format(vid, ext), '{}.{}'.format(chunk, ext)),
                num_threads=threads,
                fast_rrc=fast_rrc, rrc_params=rrc_params,
                fast_rcc=fast_rcc, rcc_params=rcc_params,
            )
            try:
                frames = vr.get_batch(rel_frame_ids).asnumpy()
            except decord.DECORDError as error:
                print(error)
                frames = vr.get_batch([0] * len(rel_frame_ids)).asnumpy()
            except IndexError:
                print(root, vid, ext, second, end_second)
            all_frames.append(frames)
            if sum(map(lambda x: x.shape[0], all_frames)) == clip_length:
                break
        res = torch.from_numpy(np.concatenate(all_frames, axis=0).astype(np.float32))
        assert res.shape[0] == clip_length, "{}, {}, {}, {}, {}, {}, {}".format(root, vid, second, end_second, res.shape[0], rel_frame_ids, frame_ids)
        return res, rel_frame_ids


class VideoCaptionDatasetBase(torch.utils.data.Dataset):
    def __init__(self, dataset, root, metadata, is_trimmed=True, label_mapping=None):
        self.dataset = dataset
        self.root = root
        self.metadata = metadata
        self.is_trimmed = is_trimmed

        if self.dataset == 'ego4d':
            with open(metadata, 'rb') as f:
                self.samples = pickle.load(f)
        elif self.dataset in ['ek100_cls', 'ek100_mir']:
            video_list = glob.glob(osp.join(self.root, '*/*.MP4'))
            fps_dict = {video: decord.VideoReader(video + '/0.MP4').get_avg_fps() for video in video_list}
            self.samples = []
            with open(metadata) as f:
                csv_reader = csv.reader(f)
                _ = next(csv_reader)  # skip the header
                for idx, row in enumerate(csv_reader):
                    pid, vid = row[1:3]
                    start_timestamp, end_timestamp = datetime2sec(row[4]), datetime2sec(row[5])
                    narration = row[8]
                    verb, noun = int(row[10]), int(row[12])
                    vid_path = '{}/{}'.format(pid, vid)
                    fps = fps_dict[osp.join(self.root, vid_path + '.MP4')]
                    # start_frame = int(np.round(fps * start_timestamp))
                    # end_frame = int(np.ceil(fps * end_timestamp))
                    self.samples.append((vid_path, start_timestamp, end_timestamp, fps, narration, verb, noun, idx))

            
            # ###########################################################################################################
            # a=[label_mapping['{}:{}'.format(x[-3], x[-2])] for x in self.samples]
            # a_unique = list(set(a))
            # a_unique.sort()
            # counter = {x:0 for x in a_unique}
            # for x in a:
            #     counter[x] += 1
            # # sort the dictionary based on the values
            # counter = {k: v for k, v in sorted(counter.items(), key=lambda item: item[1])}
            # #save in a text file
            # with open('/home/mona/whole.txt', 'w') as f:
            #     for key, value in counter.items():
            #         f.write('%s:%s\n' % (key, value))

            # ############### classes that have more than 90 and less than 110 videos

            # b_unique = [x for x in a_unique if counter[x] > 90 and counter[x] < 110] #3510data_35classes
            # selected_samples = [x for x in self.samples if label_mapping['{}:{}'.format(x[-3], x[-2])] in b_unique]
        

            # b=[label_mapping['{}:{}'.format(x[-3], x[-2])] for x in selected_samples]
            # b_unique = list(set(b)) 
            # b_unique.sort()
            # counter = {label_mapping['{}:{}'.format(x[-3], x[-2])]:0 for x in selected_samples}
            # for x in b:
            #     counter[x] += 1
            # # sort the dictionary based on the values
            # counter = {k: v for k, v in sorted(counter.items(), key=lambda item: item[1])}
            # #save in a text file
            # with open('/home/mona/sub_epic_middle.txt', 'w') as f:
            #     for key, value in counter.items():
            #         f.write('%s:%s\n' % (key, value))

            # self.samples = selected_samples
            # ###########################################################################################################


            if self.dataset == 'ek100_mir':
                self.metadata_sentence = pd.read_csv(metadata[:metadata.index('.csv')] + '_sentence.csv')
                if 'train' in metadata:
                    self.relevancy_mat = pickle.load(open(osp.join(osp.dirname(metadata), 'relevancy', 'caption_relevancy_EPIC_100_retrieval_train.pkl'), 'rb'))
                elif 'test' in metadata:
                    self.relevancy_mat = pickle.load(open(osp.join(osp.dirname(metadata), 'relevancy', 'caption_relevancy_EPIC_100_retrieval_test.pkl'), 'rb'))
                else:
                    raise ValueError('{} should contain either "train" or "test"!'.format(metadata))
                self.relevancy = .1

        elif self.dataset == 'EGTEA':
            video_list = glob.glob(osp.join(self.root, '*/*'))
            len_dict = {video: len(decord.VideoReader(video)) for video in video_list}

            vn_list, labels = [], []
            for row in open(osp.join(osp.dirname(metadata), 'action_idx.txt')):
                row = row.strip()
                vn = int(row.split(' ')[-1])
                vn_list.append(vn)
                narration = ' '.join(row.split(' ')[:-1])
                labels.append(narration.replace('_', ' ').lower())
                # labels.append(narration)
            mapping_act2narration = {vn: narration for vn, narration in zip(vn_list, labels)}

            self.samples = []
            with open(metadata) as f:
                for row in f:
                    clip_id, action_idx = row.strip().split(' ')[:2]
                    video_id = '-'.join(clip_id.split('-')[:3])
                    vid_relpath = osp.join(video_id, '{}.mp4'.format(clip_id))
                    vid_fullpath = osp.join(self.root, video_id, '{}.mp4'.format(clip_id))
                    self.samples.append((vid_relpath, 0, len_dict[vid_fullpath], mapping_act2narration[int(action_idx)]))
        elif self.dataset == 'charades_ego':
            video_list = glob.glob(osp.join(self.root, '*.mp4'))
            fps_dict = {video: decord.VideoReader(video).get_avg_fps() for video in video_list}
            self.samples = []
            if ".csv" in metadata:
                with open(metadata) as f:
                    csv_reader = csv.reader(f)
                    _ = next(csv_reader)  # skip the header
                    for row in csv_reader:
                        video_id = row[0]
                        if self.is_trimmed:
                            for action_tuple in row[9].split(';'):
                                if not action_tuple:
                                    continue
                                action, start_timestamp, end_timestamp = action_tuple.split(' ')
                                script = row[6]
                                start_timestamp, end_timestamp = float(start_timestamp), float(end_timestamp)
                                vid_path = '{}.mp4'.format(video_id)
                                fps = fps_dict[osp.join(self.root, vid_path)]
                                start_frame = int(np.round(fps * start_timestamp))
                                end_frame = int(np.ceil(fps * end_timestamp))
                                self.samples.append((vid_path, start_frame, end_frame, action))
                        else:
                            if not row[9]:
                                action_list = []
                            else:
                                action_list = [action_tuple.split(' ')[0] for action_tuple in row[9].split(';')]
                            vid_path = '{}.mp4'.format(video_id)
                            fps = fps_dict[osp.join(self.root, vid_path)]
                            duration = fps * float(row[10])
                            self.samples.append((vid_path, 0, duration, action_list))
            elif ".pkl" in metadata:
                with open(metadata, 'rb') as f:
                    csv_reader = pickle.load(f)
                    for row in csv_reader:
                        action, start_timestamp, end_timestamp = row[-1], row[1], row[2]
                        start_timestamp, end_timestamp = float(start_timestamp), float(end_timestamp)
                        vid_path = '{}.mp4'.format(row[0])
                        fps = fps_dict[osp.join(self.root, vid_path)]
                        start_frame = int(np.round(fps * start_timestamp))
                        end_frame = int(np.ceil(fps * end_timestamp))
                        self.samples.append((vid_path, start_frame, end_frame, action))
        elif self.dataset == 'charades_ego_trimmed':
            with open(metadata, 'rb') as f:
                self.samples = pickle.load(f)
        else:
            raise NotImplementedError

    def get_raw_item(
        self, i, is_training=True, num_clips=1,
        chunk_len=300, clip_length=32, clip_stride=2,
        sparse_sample=False,
        narration_selection='random',
        threads=1,
        fast_rrc=False, rrc_params=(224, (0.5, 1.0)),
        fast_rcc=False, rcc_params=(224,),
    ):
        if self.dataset == 'ego4d':
            vid, start_second, end_second, narration = self.samples[i][:4]
            frames, frame_ids = video_loader(self.root, vid, 'mp4',
                                  start_second, end_second,
                                  chunk_len=chunk_len,
                                  clip_length=clip_length,
                                  threads=threads,
                                  fast_rrc=fast_rrc,
                                  rrc_params=rrc_params,
                                  fast_rcc=fast_rcc,
                                  rcc_params=rcc_params,
                                  jitter=is_training)
            if isinstance(narration, list):
                if narration_selection == 'random':
                    narration = random.choice(narration)
                elif narration_selection == 'concat':
                    narration = '. '.join(narration)
                elif narration_selection == 'list':
                    pass
                else:
                    raise ValueError
            return frames, narration, frame_ids
        elif self.dataset == 'ek100_mir':
            vid_path, start_second, end_second, fps, narration, verb, noun = self.samples[i]
            frames, frame_ids = video_loader(self.root, vid_path, 'MP4',
                                  start_second, end_second,
                                  chunk_len=chunk_len, fps=fps,
                                  clip_length=clip_length,
                                  threads=threads,
                                  fast_rrc=fast_rrc,
                                  rrc_params=rrc_params,
                                  fast_rcc=fast_rcc,
                                  rcc_params=rcc_params,
                                  jitter=is_training)
            if is_training:
                positive_list = np.where(self.relevancy_mat[i] > self.relevancy)[0].tolist()
                if positive_list != []:
                    pos = random.sample(positive_list, min(len(positive_list), 1))[0]
                    if pos < len(self.metadata_sentence) and pos < self.relevancy_mat.shape[1]:
                        return frames, (self.metadata_sentence.iloc[pos][1], self.relevancy_mat[i][pos]), frame_ids
            else:
                return frames, (narration, 1), frame_ids
        elif self.dataset == 'ek100_cls':
            vid_path, start_second, end_second, fps, narration, verb, noun, idx = self.samples[i]
            frames, frame_ids = video_loader(self.root, vid_path, 'MP4',
                                  start_second, end_second,
                                  chunk_len=chunk_len, fps=fps,
                                  clip_length=clip_length,
                                  threads=threads,
                                  fast_rrc=fast_rrc,
                                  rrc_params=rrc_params,
                                  fast_rcc=fast_rcc,
                                  rcc_params=rcc_params,
                                  jitter=is_training)
            return frames, '{}:{}'.format(verb, noun), frame_ids, idx
        
        elif self.dataset == 'EGTEA':
            vid_path, start_frame, end_frame, sentence = self.samples[i]
            if is_training:
                assert num_clips == 1
                if end_frame < clip_length * clip_stride:
                    frames = video_loader_by_frames(self.root, vid_path, list(np.arange(0, end_frame)))
                    zeros = torch.zeros((clip_length * clip_stride - end_frame, *frames.shape[1:]))
                    frames = torch.cat((frames, zeros), dim=0)
                    frames = frames[::clip_stride]
                else:
                    start_id = np.random.randint(0, end_frame - clip_length * clip_stride + 1)
                    frame_ids = np.arange(start_id, start_id + clip_length * clip_stride, clip_stride)
                    frames = video_loader_by_frames(self.root, vid_path, frame_ids)
            else:
                if end_frame < clip_length * clip_stride:
                    frames = video_loader_by_frames(self.root, vid_path, list(np.arange(0, end_frame)))
                    zeros = torch.zeros((clip_length * clip_stride - end_frame, *frames.shape[1:]))
                    frames = torch.cat((frames, zeros), dim=0)
                    frames = frames[::clip_stride]
                    frames = frames.repeat(num_clips, 1, 1, 1)
                else:
                    frame_ids = []
                    for start_id in np.linspace(0, end_frame - clip_length * clip_stride, num_clips, dtype=int):
                        frame_ids.extend(np.arange(start_id, start_id + clip_length * clip_stride, clip_stride))
                    frames = video_loader_by_frames(self.root, vid_path, frame_ids)
            return frames, sentence
        elif self.dataset == 'charades_ego':
            if len(self.samples[i]) == 4:
                vid_path, start_frame, end_frame, action_list = self.samples[i]
            else:
                vid_path, start_frame, end_frame, _, action_list = self.samples[i]
                action_list = action_list[:77]
            if sparse_sample:
                frame_ids = get_frame_ids(start_frame, end_frame, num_segments=num_clips * clip_length, jitter=is_training)
                frames = video_loader_by_frames(self.root, vid_path, frame_ids, start_frame, end_frame, num_clips, clip_length, jitter=is_training)
            else:
                if end_frame < clip_length * clip_stride:
                    frames = video_loader_by_frames(self.root, vid_path, list(np.arange(0, end_frame)))
                    zeros = torch.zeros((clip_length * clip_stride - end_frame, *frames.shape[1:]))
                    frames = torch.cat((frames, zeros), dim=0)
                    frames = frames[::clip_stride]
                    frames = frames.repeat(num_clips, 1, 1, 1)
                else:
                    frame_ids = []
                    for start_id in np.linspace(0, end_frame - clip_length * clip_stride, num_clips, dtype=int):
                        frame_ids.extend(np.arange(start_id, start_id + clip_length * clip_stride, clip_stride))
                    # print('frame_ids:', frame_ids)
                    frames = video_loader_by_frames(self.root, vid_path, frame_ids)
            return frames, action_list
        elif self.dataset == 'charades_ego_trimmed':
            vid, start_second, end_second, narration = self.samples[i]
            frames, frame_ids = video_loader(root=self.root, vid=vid, ext='mp4', second=start_second,
                                  end_second=end_second,
                                  chunk_len=-1,  # no chunk for CharadesEgo
                                  fps=-1,  # could be variable fps
                                  clip_length=clip_length,
                                  jitter=is_training)
            return frames, narration
        else:
            raise NotImplementedError

    def __len__(self):
        return len(self.samples)


class VideoCaptionDatasetCLIP(VideoCaptionDatasetBase):
    def __init__(self, dataset, root, metadata, transform=None,
                 is_training=True, tokenizer=None,
                 chunk_len=300,
                 clip_length=32, clip_stride=2,
                 threads=1,
                 fast_rrc=False,
                 rrc_params=(224, (0.5, 1.0)),
                 fast_rcc=False,
                 rcc_params=(224,),
                 subsample_stride=None,
                 **kwargs,
                 ):
        super().__init__(dataset, root, metadata, **kwargs)

        self.full_samples = self.samples.copy()
        if isinstance(subsample_stride, int):
            self.samples = self.samples[::subsample_stride]
        self.transform = transform
        self.is_training = is_training
        self.tokenizer = tokenizer
        self.chunk_len = chunk_len
        self.clip_length = clip_length
        self.clip_stride = clip_stride
        self.threads = threads
        self.fast_rrc = fast_rrc
        self.rrc_params = rrc_params
        self.fast_rcc = fast_rcc
        self.rcc_params = rcc_params

    def __getitem__(self, i):
        output = self.get_raw_item(
            i, is_training=self.is_training,
            chunk_len=self.chunk_len,
            clip_length=self.clip_length,
            clip_stride=self.clip_stride,
            threads=self.threads,
            fast_rrc=self.fast_rrc,
            rrc_params=self.rrc_params,
            fast_rcc=self.fast_rcc,
            rcc_params=self.rcc_params,
        )

        if self.dataset == 'ek100_cls':
            if len(output) == 4:
                frames, label, frame_ids, idx = output
                caption = self.samples[i][-4]
            else:
                frames, label, frame_ids, idx, caption = output
        else:
            frames, caption = output

        # ek100_mir will also output relevancy value
        if isinstance(caption, tuple):
            caption, relevancy = caption
        else:
            relevancy = 0.

        # apply transformation
        if self.transform is not None:
            frames = self.transform(frames)
            if isinstance(frames, tuple):
                frames, _ = frames

        # tokenize caption
        if self.tokenizer is not None:
            caption = self.tokenizer(caption).squeeze(0)

        if isinstance(caption, tuple):
            caption, mask = caption
            return frames, caption, mask, relevancy
        else:
            return frames, caption, relevancy


class VideoClassyDataset(VideoCaptionDatasetBase):
    def __init__(
        self, dataset, root, metadata, transform=None,
        is_training=True, label_mapping=None,
        num_clips=1,
        chunk_len=300,
        clip_length=32, clip_stride=2,
        threads=1,
        fast_rrc=False,
        rrc_params=(224, (0.5, 1.0)),
        fast_rcc=False,
        rcc_params=(224,),
        sparse_sample=False,
        is_trimmed=True):
        super().__init__(dataset, root, metadata, is_trimmed=is_trimmed)

        self.transform = transform
        self.is_training = is_training
        self.label_mapping = label_mapping
        self.num_clips = num_clips
        self.chunk_len = chunk_len
        self.clip_length = clip_length
        self.clip_stride = clip_stride
        self.threads = threads
        self.fast_rrc = fast_rrc
        self.rrc_params = rrc_params
        self.fast_rcc = fast_rcc
        self.rcc_params = rcc_params
        self.sparse_sample = sparse_sample

    def __getitem__(self, i):
        frames, label = self.get_raw_item(
            i, is_training=self.is_training,
            chunk_len=self.chunk_len,
            num_clips=self.num_clips,
            clip_length=self.clip_length,
            clip_stride=self.clip_stride,
            threads=self.threads,
            fast_rrc=self.fast_rrc,
            rrc_params=self.rrc_params,
            fast_rcc=self.fast_rcc,
            rcc_params=self.rcc_params,
            sparse_sample=self.sparse_sample,
        )

        # apply transformation
        if self.transform is not None:
            frames = self.transform(frames)
            if isinstance(frames, tuple):
                frames, _ = frames

        if self.label_mapping is not None:
            if isinstance(label, list):
                # multi-label case
                res_array = np.zeros(len(self.label_mapping))
                for lbl in label:
                    res_array[self.label_mapping[lbl]] = 1.
                label = res_array
            else:
                label = self.label_mapping[label]

        return frames, label


class VideoClassyDataset_FRIL(VideoCaptionDatasetBase):
    def __init__(
        self, dataset, root, metadata, transform=None,
        is_training=True, label_mapping=None,
        num_clips=1,
        chunk_len=300,
        clip_length=32, clip_stride=2,
        threads=1,
        fast_rrc=False,
        rrc_params=(224, (0.5, 1.0)),
        fast_rcc=False,
        rcc_params=(224,),
        sparse_sample=False,
        is_trimmed=True,
        patch_yab_strategy = 'fully_included', # 'fully_included' or 'partially_included'
        motion_boxes = None,
        text_embeddings = None,
        patch_size = (16, 16),
        ):
        super().__init__(dataset, root, metadata, is_trimmed=is_trimmed, label_mapping=label_mapping)

        self.transform = transform
        self.is_training = is_training
        self.label_mapping = label_mapping
        self.num_clips = num_clips
        self.chunk_len = chunk_len
        self.clip_length = clip_length
        self.clip_stride = clip_stride
        self.threads = threads
        self.fast_rrc = fast_rrc
        self.rrc_params = rrc_params
        self.fast_rcc = fast_rcc
        self.rcc_params = rcc_params
        self.sparse_sample = sparse_sample
        self.patch_yab_strategy = patch_yab_strategy
        self.motion_boxes = motion_boxes
        self.text_embeddings = text_embeddings
        self.patch_size = patch_size


    def __getitem__(self, i):
        frames, label, frame_ids, vid_index= self.get_raw_item(
            i, is_training=self.is_training,
            chunk_len=self.chunk_len,
            num_clips=self.num_clips,
            clip_length=self.clip_length,
            clip_stride=self.clip_stride,
            threads=self.threads,
            fast_rrc=self.fast_rrc,
            rrc_params=self.rrc_params,
            fast_rcc=self.fast_rcc,
            rcc_params=self.rcc_params,
            sparse_sample=self.sparse_sample,
        )

        # filter out the motion box based on frame ids
        frames_motion_bbs = []
        frame_ids = np.arange(len(self.motion_boxes[f'video_{vid_index}'])) ## check it
        for idx, c in enumerate(frame_ids):
            union_frame_bboxs = np.array([[x['box2d']["x1"], x['box2d']["y1"], x['box2d']["x2"], x['box2d']["y2"]] for x in self.motion_boxes[f'video_{vid_index}'][c]['labels']]).reshape(-1) # x1, y1, x2, y2
            frames_motion_bbs.append(union_frame_bboxs)

        frames_motion_bbs = np.array(frames_motion_bbs)  # x1, y1, x2, y2

        # create a union bbox of all the frames
        union_bbx = np.array([np.min(frames_motion_bbs[:, 0]), np.min(frames_motion_bbs[:, 1]), np.max(frames_motion_bbs[:, 2]), np.max(frames_motion_bbs[:, 3])])
        union_frame_bb = tv_tensors.BoundingBoxes(union_bbx, format="XYXY", canvas_size=(frames.shape[1], frames.shape[2]))
        # frames_motion_bbs = [union_bbx]*len(frames_motion_bbs)



        # apply transformation
        if self.transform is not None:
            frames, cropped_union_frame_bb = self.transform(frames, union_frame_bb)

        if self.label_mapping is not None:
            if isinstance(label, list):
                # multi-label case
                res_array = np.zeros(len(self.label_mapping))
                for lbl in label:
                    res_array[self.label_mapping[lbl]] = 1.
                label = res_array
            else:
                label = self.label_mapping[label]

        #create a matrix with the size of the image and fill it with 1 in the bbox area
        motion_patch_yab_size = [ frames.size()[-2]//self.patch_size[0], frames.size()[-1]//self.patch_size[1]]
        motion_patch_yab = torch.zeros(motion_patch_yab_size[-2], motion_patch_yab_size[-1])
        union_bbx = cropped_union_frame_bb[0]
        if self.patch_yab_strategy == 'partially_included':
            x_start = math.ceil(union_bbx[0]/self.patch_size[0])
            x_end = math.floor(union_bbx[2]/self.patch_size[0])
            y_start = math.ceil(union_bbx[1]/self.patch_size[1])
            y_end = math.floor(union_bbx[3]/self.patch_size[1])
            motion_patch_yab[x_start:x_end-1, y_start:y_end-1] = 1
            
        if self.patch_yab_strategy == 'fully_included':
            x_start = math.floor(union_bbx[0]/self.patch_size[0])
            x_end = math.ceil(union_bbx[2]/self.patch_size[0])
            y_start = math.floor(union_bbx[1]/self.patch_size[1])
            y_end = math.ceil(union_bbx[3]/self.patch_size[1])
            motion_patch_yab[x_start:x_end-1, y_start:y_end-1] = 1



        return frames, label, motion_patch_yab.transpose(1, 0).flatten(), self.text_embeddings[f'{vid_index}'] #[0]
    


def get_downstream_dataset(transform, crop_size, args, subset='train', label_mapping=None):
    if subset == 'train':
        return VideoClassyDataset(
            args.dataset, args.root, args.train_metadata, transform,
            is_training=True, label_mapping=label_mapping,
            num_clips=args.num_clips,
            chunk_len=args.video_chunk_length,
            clip_length=args.clip_length, clip_stride=args.clip_stride,
            threads=args.decode_threads,
            fast_rrc=args.fused_decode_crop, rrc_params=(crop_size, (0.5, 1.0)),
        )
    elif subset == 'val':
        return VideoClassyDataset(
            args.dataset, args.root, args.val_metadata, transform,
            is_training=False, label_mapping=label_mapping,
            num_clips=args.num_clips,
            chunk_len=args.video_chunk_length,
            clip_length=args.clip_length, clip_stride=args.clip_stride,
            threads=args.decode_threads,
            fast_rcc=args.fused_decode_crop, rcc_params=(crop_size, ),
            is_trimmed=not args.dataset == 'charades_ego',
        )
    else:
        assert ValueError("subset should be either 'train' or 'val'")


def get_pretrain_dataset_FRIL(transform, crop_size, args, subset='train', label_mapping=None):
    if subset == 'train':
        # load motion box
        with open(args.motion_box_path, "r", encoding="utf-8") as f:
            motion_boxes = orjson.loads(f.read())

        # load text embeddings
        text_embeddings = torch.load(args.embedded_text_path)
        for k,v in text_embeddings.items():
            text_embeddings[k] = v.cpu().numpy() #v[0].cpu().numpy()

        if args.dataset.lower() == "ssv2":
            return KineticsDataset_FRIL(
            args.root, args.train_metadata, transform=transform, is_training=True, 
            clip_length=args.clip_length, clip_stride=args.clip_stride,
            threads=args.decode_threads,
            fast_rrc=False, rrc_params=(224, (0.5, 1.0)),
            fast_msc=args.fused_decode_crop, msc_params=(224, ),
            fast_cc=False, cc_params=(224, ),
            hflip_prob=0.5, vflip_prob=0.,
            mask_type='later',  # do masking in batches
            window_size=args.window_size, mask_ratio=args.mask_ratio,
            verbose=args.verbose,
            motion_boxes=motion_boxes,
            text_embeddings=text_embeddings,
            args=args,
        )
        else:
            return VideoClassyDataset_FRIL(
                args.dataset, args.root, args.train_metadata, transform,
                is_training=True, label_mapping=label_mapping,
                num_clips=args.num_clips,
                chunk_len=args.video_chunk_length,
                clip_length=args.clip_length, clip_stride=args.clip_stride,
                threads=args.decode_threads,
                fast_rrc=args.fused_decode_crop, rrc_params=(crop_size, (0.5, 1.0)),
                motion_boxes=motion_boxes,
                text_embeddings=text_embeddings,
                patch_size=args.patch_size,
            )
    # elif subset == 'val':
    #     return VideoClassyDataset_FRIL(
    #         args.dataset, args.root, args.val_metadata, transform,
    #         is_training=False, label_mapping=label_mapping,
    #         num_clips=args.num_clips,
    #         chunk_len=args.video_chunk_length,
    #         clip_length=args.clip_length, clip_stride=args.clip_stride,
    #         threads=args.decode_threads,
    #         fast_rcc=args.fused_decode_crop, rcc_params=(crop_size, ),
    #         is_trimmed=not args.dataset == 'charades_ego',
    #     )
    else:
        assert ValueError("subset should be either 'train' or 'val'")

