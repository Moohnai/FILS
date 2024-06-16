import datetime
import math
import numpy as np
import os

import decord
import torch
from torchvision import tv_tensors

from avion.data.transforms import TubeMaskingGenerator


def read_metadata(metadata_fname, root=None, args=None, mode='train'):
    samples = []
    if args.dataset == 'ssv2':
        with open(metadata_fname) as split_f:
            data = split_f.readlines()
            for id, line in enumerate(data):
                line_info = line.split(' ')
                # assert len(line_info) == 3
                samples.append((line_info[0], eval(" ".join(line_info[1:-1])), int(line_info[-1]), id))

    else:
        with open(metadata_fname) as split_f:
            data = split_f.readlines()
            for line in data:
                line_info = line.split(' ')
                assert len(line_info) == 3
                samples.append((line_info[0], int(line_info[1]), int(line_info[2])))
    return samples

# def video_loader_by_frames(
#     root, vid, frame_ids, threads=1,
#     fast_rrc=False, rrc_params=(224, (0.5, 1.0)),
#     fast_msc=False, msc_params=(224,),
#     fast_cc=False, cc_params=(224,),
#     hflip_prob=0., vflip_prob=0.,
# ):
#     if fast_rrc:
#         width, height = rrc_params[0], rrc_params[0]
#     elif fast_msc:
#         width, height = msc_params[0], msc_params[0]
#     elif fast_cc:
#         width, height = cc_params[0], cc_params[0]
#     else:
#         width, height = -1, -1
#     vr = decord.VideoReader(
#         os.path.join(root, vid), num_threads=threads,
#         width=width, height=height,
#         use_rrc=fast_rrc, scale_min=rrc_params[1][0], scale_max=rrc_params[1][1],
#         use_msc=fast_msc,
#         use_centercrop=fast_cc,
#         hflip_prob=hflip_prob, vflip_prob=vflip_prob,
#     )
#     try:
#         frames = vr.get_batch(frame_ids).asnumpy()
#     except (IndexError, decord.DECORDError) as error:
#         print(error)
#         print("Erroneous video: ", vid)
#         frames = torch.zeros((len(frame_ids), 240, 320, 3)).numpy()
#     return torch.from_numpy(frames.astype(np.float32))



class KineticsDataset(torch.utils.data.Dataset):
    def __init__(self, root, metadata,
                 transform=None,
                 is_training=True,
                 clip_length=32,
                 clip_stride=2,
                 threads=1,
                 # fused augmentations need to be specified here
                 fast_rrc=False,
                 rrc_params=(224, (0.5, 1.0)),
                 fast_msc=False,
                 msc_params=(224, ),
                 fast_cc=False,
                 cc_params=(224,),
                 hflip_prob=0.5,
                 vflip_prob=0.,
                 verbose=False,
                 # for masking
                 mask_type='tube',
                 window_size=(8, 14, 14),
                 mask_ratio=0.9,
                 # for quick prototype
                 subsample_stride=None,
                 args = None,):
        super().__init__()

        self.root = root
        self.samples = read_metadata(metadata, args=args)
        self.full_samples = self.samples.copy()
        if isinstance(subsample_stride, int):
            self.samples = self.samples[::subsample_stride]
        self.transform = transform
        self.is_training = is_training
        self.clip_length = clip_length
        self.clip_stride = clip_stride
        self.threads = threads
        self.fast_rrc = fast_rrc
        self.rrc_params = rrc_params
        self.fast_msc = fast_msc
        self.msc_params = msc_params
        self.fast_cc = fast_cc
        self.cc_params = cc_params
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob
        self.verbose = verbose

        if mask_type == 'tube':
            self.masked_position_generator = TubeMaskingGenerator(
                window_size, mask_ratio,
            )
        elif mask_type == 'later':
            self.masked_position_generator = None
        else:
            raise NotImplementedError
        
    def video_loader_by_frames(self,
        root, vid, threads=1,
        fast_rrc=False, rrc_params=(224, (0.5, 1.0)),
        fast_msc=False, msc_params=(224,),
        fast_cc=False, cc_params=(224,),
        hflip_prob=0., vflip_prob=0.,
    ):
        if fast_rrc:
            width, height = rrc_params[0], rrc_params[0]
        elif fast_msc:
            width, height = msc_params[0], msc_params[0]
        elif fast_cc:
            width, height = cc_params[0], cc_params[0]
        else:
            width, height = -1, -1
        # vr = decord.VideoReader(
        #     os.path.join(root, vid), num_threads=threads,
        #     width=width, height=height,
        #     use_rrc=fast_rrc, scale_min=rrc_params[1][0], scale_max=rrc_params[1][1],
        #     use_msc=fast_msc,
        #     use_centercrop=fast_cc,
        #     hflip_prob=hflip_prob, vflip_prob=vflip_prob,
        # )
        vr = decord.VideoReader(
            os.path.join(root, vid), num_threads=threads,
        )

        num_frames = len(vr)
        if num_frames > (self.clip_length + 1) * self.clip_stride:
            start_id = np.random.randint(num_frames - (self.clip_length + 1) * self.clip_stride)
        else:
            start_id = 0
        frame_ids = np.arange(start_id, start_id + self.clip_length * self.clip_stride, step=self.clip_stride)
        if self.is_training:
            shift = np.random.randint(self.clip_stride, size=self.clip_length)
            frame_ids += shift
        frame_ids = frame_ids % num_frames


        try:
            frames = vr.get_batch(frame_ids).asnumpy()
        except (IndexError, decord.DECORDError) as error:
            print(error)
            print("Erroneous video: ", vid)
            frames = torch.zeros((len(frame_ids), 240, 320, 3)).numpy()
        return torch.from_numpy(frames.astype(np.float32))

    def __getitem__(self, i):
        if self.verbose:
            print("[{}] __getitem__() starts at {}".format(os.getpid(), datetime.datetime.now()))

        video_id, _, label, _ = self.samples[i]

        if self.is_training:
            frames = self.video_loader_by_frames(
                self.root, video_id + '.mp4' if '.' not in video_id else video_id, threads=self.threads,
                fast_rrc=self.fast_rrc, rrc_params=self.rrc_params,
                fast_msc=self.fast_msc, msc_params=self.msc_params,
                fast_cc=False, cc_params=self.cc_params,
                hflip_prob=self.hflip_prob, vflip_prob=self.vflip_prob,
            )
        else:
            frames = self.video_loader_by_frames(
                self.root, video_id + '.mp4' if '.' not in video_id else video_id, threads=self.threads,
                fast_rrc=False, rrc_params=self.rrc_params,
                fast_cc=True, cc_params=self.cc_params,
                hflip_prob=0., vflip_prob=0.,
            )

        # apply transformation
        if self.transform is not None:
            frames = self.transform(frames)

        if self.verbose:
            print("[{}] __getitem__() end at {}".format(os.getpid(), datetime.datetime.now()))

        if self.masked_position_generator is None:
            return frames, label
        else:
            return frames, self.masked_position_generator()

    def __len__(self):
        return len(self.samples)


class KineticsDataset_FRIL(torch.utils.data.Dataset):
    def __init__(self, root, metadata,
                 transform=None,
                 is_training=True,
                 clip_length=32,
                 clip_stride=2,
                 threads=1,
                 # fused augmentations need to be specified here
                 fast_rrc=False,
                 rrc_params=(224, (0.5, 1.0)),
                 fast_msc=False,
                 msc_params=(224, ),
                 fast_cc=False,
                 cc_params=(224,),
                 hflip_prob=0.5,
                 vflip_prob=0.,
                 verbose=False,
                 # for masking
                 mask_type='later',
                 window_size=(8, 14, 14),
                 mask_ratio=0.9,
                 # for quick prototype
                 subsample_stride=None,
                 motion_boxes = None,
                 text_embeddings = None,
                 patch_yab_strategy = 'fully_included', # 'fully_included' or 'partially_included'
                 args = None,
                 ):
        super().__init__()

        self.root = root
        self.samples = read_metadata(metadata, args.root, args, mode='train' if is_training else 'val')
        self.full_samples = self.samples.copy()
        if isinstance(subsample_stride, int):
            self.samples = self.samples[::subsample_stride]
        self.transform = transform
        self.is_training = is_training
        self.clip_length = clip_length
        self.clip_stride = clip_stride
        self.threads = threads
        self.fast_rrc = fast_rrc
        self.rrc_params = rrc_params
        self.fast_msc = fast_msc
        self.msc_params = msc_params
        self.fast_cc = fast_cc
        self.cc_params = cc_params
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob
        self.verbose = verbose
        self.motion_boxes = motion_boxes
        self.text_embeddings = text_embeddings
        self.patch_yab_strategy = patch_yab_strategy
        self.patch_size = args.patch_size

        if mask_type == 'tube':
            self.masked_position_generator = TubeMaskingGenerator(
                window_size, mask_ratio,
            )
        elif mask_type == 'later':
            self.masked_position_generator = None
        else:
            raise NotImplementedError

    
    def video_loader_by_frames(self,
        root, vid, threads=1,
        fast_rrc=False, rrc_params=(224, (0.5, 1.0)),
        fast_msc=False, msc_params=(224,),
        fast_cc=False, cc_params=(224,),
        hflip_prob=0., vflip_prob=0.,
    ):
        if fast_rrc:
            width, height = rrc_params[0], rrc_params[0]
        elif fast_msc:
            width, height = msc_params[0], msc_params[0]
        elif fast_cc:
            width, height = cc_params[0], cc_params[0]
        else:
            width, height = -1, -1
        # vr = decord.VideoReader(
        #     os.path.join(root, vid), num_threads=threads,
        #     width=width, height=height,
        #     use_rrc=fast_rrc, scale_min=rrc_params[1][0], scale_max=rrc_params[1][1],
        #     use_msc=fast_msc,
        #     use_centercrop=fast_cc,
        #     hflip_prob=hflip_prob, vflip_prob=vflip_prob,
        # )
        vr = decord.VideoReader(
            os.path.join(root, vid), num_threads=threads,
        )

        num_frames = len(vr)
        if num_frames > (self.clip_length + 1) * self.clip_stride:
            start_id = np.random.randint(num_frames - (self.clip_length + 1) * self.clip_stride)
        else:
            start_id = 0
        frame_ids = np.arange(start_id, start_id + self.clip_length * self.clip_stride, step=self.clip_stride)
        if self.is_training:
            shift = np.random.randint(self.clip_stride, size=self.clip_length)
            frame_ids += shift
        frame_ids = frame_ids % num_frames


        try:
            frames = vr.get_batch(frame_ids).asnumpy()
        except (IndexError, decord.DECORDError) as error:
            print(error)
            print("Erroneous video: ", vid)
            frames = torch.zeros((len(frame_ids), 240, 320, 3))
        return torch.from_numpy(frames.astype(np.float32))

    def __getitem__(self, i):
        if self.verbose:
            print("[{}] __getitem__() starts at {}".format(os.getpid(), datetime.datetime.now()))

        video_id, data_caption, label, vid_index = self.samples[i]
        video_id = video_id.split('.')[0]

        
        if self.is_training:
            frames = self.video_loader_by_frames(
                self.root, video_id + '.mp4' if '.' not in video_id else video_id, threads=self.threads,
                fast_rrc=self.fast_rrc, rrc_params=self.rrc_params,
                fast_msc=self.fast_msc, msc_params=self.msc_params,
                fast_cc=False, cc_params=self.cc_params,
                hflip_prob=self.hflip_prob, vflip_prob=self.vflip_prob,
            )
        else:
            frames = self.video_loader_by_frames(
                self.root, video_id + '.mp4' if '.' not in video_id else video_id, threads=self.threads,
                fast_rrc=False, rrc_params=self.rrc_params,
                fast_cc=True, cc_params=self.cc_params,
                hflip_prob=0., vflip_prob=0.,
            )

        # filter out the motion box based on frame ids
        try:
            frames_motion_bbs = []
            frame_ids = np.arange(len(self.motion_boxes[f'{video_id}'])) ## check it
            for idx, c in enumerate(frame_ids):
                union_frame_bboxs = np.array([[x['box2d']["x1"], x['box2d']["y1"], x['box2d']["x2"], x['box2d']["y2"]] for x in self.motion_boxes[f'{video_id}'][c]['labels']]).reshape(-1) # x1, y1, x2, y2
                frames_motion_bbs.append(union_frame_bboxs)

            frames_motion_bbs = np.array(frames_motion_bbs)  # x1, y1, x2, y2
        except:
            # if there is no motion box, then create a center bbox with 50% of the frame size
            frames_motion_bbs = np.array([[frames.size()[-3]//4, frames.size()[-2]//4, frames.size()[-3]//4*3, frames.size()[-2]//4*3]]*len(frames))

        # create a union bbox of all the frames
        union_bbx = np.array([np.min(frames_motion_bbs[:, 0]), np.min(frames_motion_bbs[:, 1]), np.max(frames_motion_bbs[:, 2]), np.max(frames_motion_bbs[:, 3])])
        union_frame_bb = tv_tensors.BoundingBoxes(union_bbx, format="XYXY", canvas_size=(frames.shape[1], frames.shape[2]))
        # frames_motion_bbs = [union_bbx]*len(frames_motion_bbs)



        # apply transformation
        if self.transform is not None:
            frames, cropped_union_frame_bb = self.transform(frames, union_frame_bb)


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

        if self.verbose:
            print("[{}] __getitem__() end at {}".format(os.getpid(), datetime.datetime.now()))

        if self.masked_position_generator is None:
            return frames, label, motion_patch_yab.transpose(1, 0).flatten(), self.text_embeddings[f'{video_id}'] #[0]
        else:
            return frames, self.masked_position_generator()

    def __len__(self):
        return len(self.samples)
