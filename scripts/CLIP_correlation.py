import argparse
from collections import OrderedDict
import cv2, os
import numpy as np
import torch
import torch.cuda.amp as amp
import pandas as pd

import clip
from pytorch_grad_cam.utils.image import show_cam_on_image

import torchvision
from pytorchvideo.transforms import Normalize

from decord import VideoReader, cpu
from torchvision.transforms import v2
import kornia as K

# find the path to the current file
current_path = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.dirname(current_path)
# add parent path to the system path
import sys
sys.path.append(parent_path)

from FILS.data.classification_dataset import VideoClsDataset_FRIL
from FILS.data.clip_dataset import VideoCaptionDatasetCLIP
import FILS.models.model_FRIL as model_FRIL
from FILS.utils.misc import generate_label_map
from FILS.data.clip_dataset import get_pretrain_dataset_FRIL
from FILS.data.transforms import  Permute_BB

def spatial_mask_creator(mask_height=4, mask_width=4, height=224, width=224, num_frames=16, start_x=0, start_y=0):
    mask = torch.zeros(3, num_frames, height, width)
    mask[:, :, start_y:start_y + mask_height, start_x:start_x + mask_width] = 1
    return mask

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ek100_cls', help='Dataset name')
    parser.add_argument('--model', default='FRILSCLIP_VITB16', type=str)
    parser.add_argument('--nb-classes', default=3806, type=int, help='number of classes, EK100: 3806, SSV2: 174, EGTEA: 106')
    parser.add_argument('--use-registers', default=False, type=bool, help='Use registers for the model')
    parser.add_argument('--clip_length', type=int, default=16, help='Number of frames in a clip')
    parser.add_argument('--clip_stride', type=int, default=4, help='Stride for sampling frames')
    parser.add_argument('--video-chunk-length', default=15, type=int)
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                            help='Random erase prob (default: 0.25)')
    parser.add_argument('--use-cuda', action='store_true', default=True,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument(
        '--image-path',
        type=str,
        default='./1.jpg',
        help='Input image path')
    parser.add_argument('--aug_smooth', action='store_true', default=False,
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument(
        '--eigen_smooth',
        action='store_true',
        default=True,
        help='Reduce noise by taking the first principle componenet'
        'of cam_weights*activations')

    parser.add_argument(
        '--method',
        type=str,
        default='gradcam',
        help='Can be gradcam/gradcam++/scorecam/xgradcam/ablationcam')

    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')

    return args


def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0),
                                      height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result

def load_model(model, pret_path):
    if os.path.isfile(pret_path):
        print("=> loading resume checkpoint '{}'".format(pret_path))
        checkpoint = torch.load(pret_path, map_location='cpu')
        # remove prefix 'module.' for the keys of the state_dict
        new_state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            if k.startswith('module.'):
                k = k.replace('module.', '')
            new_state_dict[k] = v
        result = model.load_state_dict(new_state_dict, strict=True)
        print(f"Missing keys: {result.missing_keys}")
        print(f"Unexpected keys: {result.unexpected_keys}")

def loadvideo_decord(sample, sample_rate_scale=1):
        """Load video content using Decord"""
        fname = sample

        if not (os.path.exists(fname)):
            return []

        # avoid hanging issue
        if os.path.getsize(fname) < 1 * 1024:
            print('SKIP: ', fname, " - ", os.path.getsize(fname))
            return []
        try:
            # if self.keep_aspect_ratio:
            if True:
                vr = VideoReader(fname, num_threads=1, ctx=cpu(0))
            else:
                vr = VideoReader(fname, width=320, height=256,
                                 num_threads=1, ctx=cpu(0))
        except:
            print("video cannot be loaded by decord: ", fname)
            return []

        # if self.mode == 'test':
        all_index = []#///////////////////////////////////////////
        tick = len(vr) / float(16)#//////////////////////////////////
        all_index = list(np.array([int(tick / 2.0 + tick * x) for x in range(16)] +
                            [int(tick * x) for x in range(16)]))#//////////////////////////////////
        while len(all_index) < (16 * 2):#//////////////////////////////////
            all_index.append(all_index[-1])
        all_index = list(np.sort(np.array(all_index))) 
        vr.seek(0)
        buffer = vr.get_batch(all_index).asnumpy()
        return buffer


if __name__ == '__main__':
    """ python vit_gradcam.py --image-path <path_to_image>
    Example usage of using cam-methods on a VIT network.

    """

    # create args parser
    args = get_args()

    device = torch.device("cuda" if args.use_cuda else "cpu")

    if args.dataset == 'ek100_cls':
        root = os.path.join(parent_path, 'datasets/EK100/EK100_320p_15sec_30fps_libx264')
        train_metadata = os.path.join(parent_path, 'datasets/EK100/epic-kitchens-100-annotations/EPIC_100_train.csv')
        val_metadata = os.path.join(parent_path, 'datasets/EK100/epic-kitchens-100-annotations/EPIC_100_validation.csv')

        mean, std = [0.485 * 255, 0.456 * 255, 0.406 * 255], [0.229 * 255, 0.224 * 255, 0.225 * 255]
        normalize = K.enhance.Normalize(mean=mean, std=std)

        ####################
        crop_size = 336 if args.model.endswith("_336PX") else 224

        base_train_transform_ls = [
            Permute_BB([0, 3, 1, 2]),
            v2.RandomResizedCrop(crop_size, scale=(0.5, 1.0), antialias=True),
            v2.RandomHorizontalFlip(0.5),
            Permute_BB([1, 0, 2, 3]),
        ]

        base_val_transform_ls = [
            Permute_BB([0, 3, 1, 2]),
            v2.Resize(crop_size, antialias=True),
            v2.CenterCrop(crop_size),
            Permute_BB([1, 0, 2, 3]),
        ]

        train_transform = v2.Compose(base_train_transform_ls)
        val_transform = v2.Compose(base_val_transform_ls)

        total_labels, mapping_vn2act = generate_label_map(args.dataset, root=parent_path)
        # add mamapping_vn2act to args
        args.label_mapping = mapping_vn2act
        args.mapping_act2v = {i: int(vn.split(':')[0]) for (vn, i) in mapping_vn2act.items()}
        args.mapping_act2n = {i: int(vn.split(':')[1]) for (vn, i) in mapping_vn2act.items()}
        args.actions = pd.DataFrame.from_dict({'verb': args.mapping_act2v.values(), 'noun': args.mapping_act2n.values()})

        args.num_clips = 1
        tokenizer = clip.tokenize
        train_dataset = VideoCaptionDatasetCLIP(
            args.dataset, root, train_metadata,
            transform=train_transform, is_training=True, tokenizer=tokenizer,
            clip_length=args.clip_length, clip_stride=args.clip_stride,
            chunk_len=args.video_chunk_length,
            threads=1,
            fast_rrc=False, rrc_params=(crop_size, (0.5, 1.0)),
            label_mapping=args.label_mapping,
        )
        args.num_clips = 1
        val_dataset = VideoCaptionDatasetCLIP(
            args.dataset, root, val_metadata,
            transform=val_transform, is_training=False, tokenizer=tokenizer,
            clip_length=args.clip_length, clip_stride=args.clip_stride,
            chunk_len=args.video_chunk_length,
            threads=1,
            fast_rrc=False, rrc_params=(crop_size, (0.5, 1.0)),
            label_mapping=args.label_mapping,
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=1, shuffle=False,
            num_workers=8, pin_memory=True,
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=1, shuffle=False,
            num_workers=8, pin_memory=True,
        )     

    #####
    # create a folder to save results
    save_dir = os.path.join('CLIP_corr_results', args.dataset)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    model_CLIP = model = getattr(model_FRIL, args.model)(
        freeze_temperature=True,
        use_grad_checkpointing=True,
        patch_dropout=0,
        num_frames=args.clip_length,
        drop_path_rate=0,
        use_fast_conv1=True,
        use_flash_attn=True,
        pretrain_zoo="frils",
        pretrain_path='/home/mona/FRIL/FILS/results/pretrain_FRILS/pretrain_CLIP_vidcaption_vifi_all_EK100_decoder_head=6__MSE_scale=0__CLIP_scale=1__FR_scale=0__ssvli_iter=1_800_epochs_totalbatch=200_lr=0.00015_CLIP_strategy=patch-average/checkpoint_00280.pt',
        text_pretrain_path='/home/mona/FRIL/FILS/results/vifi_clip_10_epochs_k400_full_finetuned.pth',
    ).to(device)
    
    model_FR_CLIP = model = getattr(model_FRIL, args.model)(
        freeze_temperature=True,
        use_grad_checkpointing=True,
        patch_dropout=0,
        num_frames=args.clip_length,
        drop_path_rate=0,
        use_fast_conv1=True,
        use_flash_attn=True,
        pretrain_zoo="frils",
        pretrain_path='/home/mona/FRIL/FILS/results/pretrain_FRILS/pretrain_FR_CLIP_vidcaption_vifi_full_all_EK_decoder_head=6__MSE_scale=0__CLIP_scale=1__FR_scale=1__ssvli_iter=1_800_epochs_totalbatch=240_lr=0.00015_CLIP_strategy=patch-average/checkpoint_00800.pt',
        text_pretrain_path='/home/mona/FRIL/FILS/results/vifi_clip_10_epochs_k400_full_finetuned.pth',
    ).to(device)
    
    # set the models to evaluation mode
    model_FR_CLIP.eval()
    model_CLIP.eval()

    # load caption csv file
    caption_csv = pd.read_csv('/home/mona/FRIL/FILS/datasets/EK100/epic_captions_train.csv')
    caption_csv['video'] = caption_csv['video'].apply(lambda x: x.split('/')[-1].split('.')[0].split('_')[-1])

    cutoff_threshold = 0.4 # 0.3

    # read a video from the dataset
    loader = train_loader
    for i, inputs in enumerate(loader):
        vid_name = f"val_{i}_{loader.dataset.samples[i][0]}".replace('/', '_')
        vid_id = loader.dataset.samples[i][-1]

        videos = inputs[0].clone()
        dataset_caption = loader.dataset.samples[i][-4]
        captions = {
            'verb':' '.join(dataset_caption.split(' ')[:-1]),
            'noun': dataset_caption.split(' ')[-1],
            'verb_noun': dataset_caption,
            'generated_caption': caption_csv.loc[caption_csv['video'] == str(vid_id)]['video_caption'].values[0],
        }
        for caption_name, caption in captions.items():
            
            # take only the first 77 words in caption
            caption = ' '.join(caption.split(' ')[:77])
            text = tokenizer(caption).to(device)

            inputs = [tensor.to(device) for tensor in inputs]

            # normalize videos
            norm_videos = normalize(inputs[0])
            # relevancies = inputs.pop()

            with amp.autocast():
                with torch.no_grad():
                    image_features_CLIP, text_features_CLIP, _ = model_CLIP(norm_videos, text)
                    image_features_FR_CLIP, text_features_FR_CLIP, _ = model_FR_CLIP(norm_videos, text)

                    # cosine similarity as logits
                    logits_per_image_CLIP = image_features_CLIP @ text_features_CLIP.t()
                    # logits_per_image_CLIP = torch.softmax(logits_per_image_CLIP, dim=1)
                    logits_per_image_FR_CLIP = image_features_FR_CLIP @ text_features_FR_CLIP.t()
                    # logits_per_image_FR_CLIP = torch.softmax(logits_per_image_FR_CLIP, dim=1)

            # now we have attentions for 1568 patches, extend it to the original video frames --> 224x224
            # each of the 14x14 patches corresponds to 16x16 pixels
            CLIP_attn_maps = []
            FR_CLIP_attn_maps = []
            if caption_name == 'verb' or caption_name == 'verb_noun':
                for m in range(8):
                    attn_map_CLIP = logits_per_image_CLIP.reshape(-1)[m*14*14:(m+1)*14*14]
                    attn_map_FR_CLIP = logits_per_image_FR_CLIP.reshape(-1)[m*14*14:(m+1)*14*14]
                    # each element in attn_map must be repeated by 16x16 times
                    attn_map_CLIP = attn_map_CLIP.reshape(14,14).repeat_interleave(16, dim=0).repeat_interleave(16, dim=1).cpu().numpy().astype(np.float32)
                    attn_map_FR_CLIP = attn_map_FR_CLIP.reshape(14,14).repeat_interleave(16, dim=0).repeat_interleave(16, dim=1).cpu().numpy().astype(np.float32)

                    # first remove negative values
                    attn_map_CLIP[attn_map_CLIP < 0] = 0
                    attn_map_FR_CLIP[attn_map_FR_CLIP < 0] = 0

                    CLIP_attn_maps.append(attn_map_CLIP)
                    FR_CLIP_attn_maps.append(attn_map_FR_CLIP)

                # take average of the attentions
                attn_map_CLIP = np.mean(CLIP_attn_maps, axis=0)
                attn_map_FR_CLIP = np.mean(FR_CLIP_attn_maps, axis=0)
                # smooth the attention map
                attn_map_CLIP = cv2.GaussianBlur(attn_map_CLIP, (0, 0), 7)
                attn_map_FR_CLIP = cv2.GaussianBlur(attn_map_FR_CLIP, (0, 0), 7)
                # normalize the attention map between 0 and 1
                attn_map_CLIP = (attn_map_CLIP - attn_map_CLIP.min()) / (attn_map_CLIP.max() - attn_map_CLIP.min())
                attn_map_FR_CLIP = (attn_map_FR_CLIP - attn_map_FR_CLIP.min()) / (attn_map_FR_CLIP.max() - attn_map_FR_CLIP.min())
                #remove values less than 0.3
                attn_map_CLIP[attn_map_CLIP < cutoff_threshold] = 0
                attn_map_FR_CLIP[attn_map_FR_CLIP < cutoff_threshold] = 0
                # keep values between 0 and 1
                attn_map_CLIP = np.clip(attn_map_CLIP, 0, 1)
                attn_map_FR_CLIP = np.clip(attn_map_FR_CLIP, 0, 1)

                CLIP_attn_maps = [attn_map_CLIP] * 8
                FR_CLIP_attn_maps = [attn_map_FR_CLIP] * 8

            else:

                for m in range(8):
                    attn_map_CLIP = logits_per_image_CLIP.reshape(-1)[m*14*14:(m+1)*14*14]
                    attn_map_FR_CLIP = logits_per_image_FR_CLIP.reshape(-1)[m*14*14:(m+1)*14*14]
                    # each element in attn_map must be repeated by 16x16 times
                    attn_map_CLIP = attn_map_CLIP.reshape(14,14).repeat_interleave(16, dim=0).repeat_interleave(16, dim=1).cpu().numpy().astype(np.float32)
                    attn_map_FR_CLIP = attn_map_FR_CLIP.reshape(14,14).repeat_interleave(16, dim=0).repeat_interleave(16, dim=1).cpu().numpy().astype(np.float32)

                    # first remove negative values
                    attn_map_CLIP[attn_map_CLIP < 0] = 0
                    attn_map_FR_CLIP[attn_map_FR_CLIP < 0] = 0

                    # smooth the attention map
                    attn_map_CLIP = cv2.GaussianBlur(attn_map_CLIP, (0, 0), 7)
                    attn_map_FR_CLIP = cv2.GaussianBlur(attn_map_FR_CLIP, (0, 0), 7)

                    # normalize the attention map between 0 and 1
                    attn_map_CLIP = (attn_map_CLIP - attn_map_CLIP.min()) / (attn_map_CLIP.max() - attn_map_CLIP.min())
                    attn_map_FR_CLIP = (attn_map_FR_CLIP - attn_map_FR_CLIP.min()) / (attn_map_FR_CLIP.max() - attn_map_FR_CLIP.min())

                    # # pick the top 5% of the attention map
                    # attn_map_CLIP[attn_map_CLIP < np.percentile(attn_map_CLIP, 95)] = 0
                    # attn_map_FR_CLIP[attn_map_FR_CLIP < np.percentile(attn_map_FR_CLIP, 95)] = 0

                    # remove values less than 0.3
                    attn_map_CLIP[attn_map_CLIP < cutoff_threshold] = 0
                    attn_map_FR_CLIP[attn_map_FR_CLIP < cutoff_threshold] = 0

                    # keep values between 0 and 1
                    attn_map_CLIP = np.clip(attn_map_CLIP, 0, 1)
                    attn_map_FR_CLIP = np.clip(attn_map_FR_CLIP, 0, 1)

                    CLIP_attn_maps.append(attn_map_CLIP)
                    FR_CLIP_attn_maps.append(attn_map_FR_CLIP)

            input_tensor_org = videos.clone()
            frame_number_list = [0, 7, 15]

            for frame_number in frame_number_list:
                    rgb_img = videos.squeeze(0).permute(1, 2, 3, 0).cpu().numpy()[frame_number] / 255
                    rgb_img = np.clip(rgb_img, 0, 1)

                    # save the original image
                    rgb_img_org = (rgb_img * 255).astype(np.uint8)
            
                    cam_image_FR_CLIP = show_cam_on_image(rgb_img, FR_CLIP_attn_maps[frame_number//2])
                    cam_image_CLIP = show_cam_on_image(rgb_img, CLIP_attn_maps[frame_number//2])

                    # save the images
                    if not os.path.exists(os.path.join(save_dir, vid_name, caption_name)):
                        os.makedirs(os.path.join(save_dir, vid_name, caption_name), exist_ok=True)
                    dataset_caption = '_'.join(dataset_caption.split(' '))
                    cv2.imwrite(os.path.join(save_dir, vid_name, caption_name, f"{vid_name}_frame_{frame_number}_org_{dataset_caption}.jpg"), rgb_img_org[:,:,::-1])
                    cv2.imwrite(os.path.join(save_dir, vid_name, caption_name, f"{vid_name}_frame_{frame_number}_FR_CLIP.jpg"), cam_image_FR_CLIP)
                    cv2.imwrite(os.path.join(save_dir, vid_name, caption_name, f"{vid_name}_frame_{frame_number}_CLIP.jpg"), cam_image_CLIP)
