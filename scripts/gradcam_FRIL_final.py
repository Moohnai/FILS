import argparse
from collections import OrderedDict
import cv2, os
import numpy as np
import torch
import torch.cuda.amp as amp
import pandas as pd

import torchvision
from pytorchvideo.transforms import Normalize
from torchvision.transforms import v2

import clip
import kornia as K

from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad

from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    preprocess_image
from pytorch_grad_cam.ablation_layer import AblationLayerVit
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from decord import VideoReader, cpu

# find the path to the current file
current_path = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.dirname(current_path)
# add parent path to the system path
import sys
sys.path.append(parent_path)

from FILS.data.classification_dataset import VideoClsDataset_FRIL
from FILS.data.transforms import Permute, TemporalCrop, SpatialCrop, AdaptiveTemporalCrop, Permute_BB
from FILS.data.kinetics_dataset import KineticsDataset
from FILS.data.clip_dataset import VideoClassyDataset, VideoCaptionDatasetCLIP
import FILS.models.model_FRIL as model_FRIL
from FILS.utils.misc import generate_label_map


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ek100_cls', help='Dataset name', choices=['ek100_cls', 'SSV2', 'EGTEA', 'charades_ego'])
    parser.add_argument('--model', default='vit_base_patch16_224', type=str)
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

        args.repeated_aug = 1

        train_dataset = VideoClsDataset_FRIL(
            root, train_metadata, mode='validation', 
            clip_length=args.clip_length, clip_stride=args.clip_stride,
            args=args,
        )

        val_dataset = VideoClsDataset_FRIL(
            root, val_metadata, mode='validation',
            clip_length=args.clip_length, clip_stride=args.clip_stride,
            args=args,
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=1, shuffle=False,
            num_workers=8, pin_memory=True,
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=1, shuffle=False,
            num_workers=8, pin_memory=True,
        )

        inv_normalize = torchvision.transforms.Compose([
            Normalize(
                mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                std=[1/0.229, 1/0.224, 1/0.255]
            )
        ])

        total_labels, mapping_vn2act = generate_label_map(args.dataset, root=parent_path)
        # add mamapping_vn2act to args
        args.label_mapping = mapping_vn2act
        args.mapping_act2v = {i: int(vn.split(':')[0]) for (vn, i) in mapping_vn2act.items()}
        args.mapping_act2n = {i: int(vn.split(':')[1]) for (vn, i) in mapping_vn2act.items()}
        args.actions = pd.DataFrame.from_dict({'verb': args.mapping_act2v.values(), 'noun': args.mapping_act2n.values()})

    elif args.dataset == "EGTEA":
        crop_size = 224 if '336PX' not in args.model else 336
        root = os.path.join(parent_path, 'datasets/EGTEA/cropped_clips')
        train_metadata = os.path.join(parent_path, 'datasets/EGTEA/train_split1.txt')
        val_metadata = os.path.join(parent_path, 'datasets/EGTEA/test_split1.txt')

        transforms_list = [
            Permute([3, 0, 1, 2]),    # T H W C -> C T H W
            torchvision.transforms.RandomResizedCrop(crop_size, scale=(0.5, 1.0), antialias=True),
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
        ]
        transforms_list.append(Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]))
        train_transform = torchvision.transforms.Compose(transforms_list)

        val_transform = torchvision.transforms.Compose([
                Permute([3, 0, 1, 2]),    # T H W C -> C T H W
                torchvision.transforms.Resize(crop_size, antialias=True),
                torchvision.transforms.CenterCrop(crop_size),
                Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
                TemporalCrop(frames_per_clip=args.clip_length, stride=args.clip_length),
                SpatialCrop(crop_size=crop_size, num_crops=1),
            ])
        
        test_transform = torchvision.transforms.Compose([
                Permute([3, 0, 1, 2]),    # T H W C -> C T H W
                torchvision.transforms.Resize(crop_size, antialias=True),
                Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
                TemporalCrop(frames_per_clip=args.clip_length, stride=args.clip_stride),
                SpatialCrop(crop_size=crop_size, num_crops=1),
            ])
        
        total_labels, mapping_vn2act = generate_label_map(args.dataset, root=parent_path)
    
        train_dataset = VideoClassyDataset(
                args.dataset, root, train_metadata, train_transform,
                is_training=True, label_mapping=mapping_vn2act,
                num_clips=1,
                chunk_len=args.video_chunk_length,
                clip_length=args.clip_length, clip_stride=args.clip_stride,
                threads=False,
                rrc_params=(crop_size, (0.5, 1.0)),
            )
        
        val_dataset = VideoClassyDataset(
                args.dataset, root, val_metadata, val_transform,
                is_training=False, label_mapping=mapping_vn2act,
                num_clips=1,
                chunk_len=args.video_chunk_length,
                clip_length=args.clip_length, clip_stride=args.clip_stride,
                threads=False,
                rrc_params=(crop_size, (0.5, 1.0)),
            )
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=1, shuffle=False,
            num_workers=8, pin_memory=True,
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=1, shuffle=False,
            num_workers=8, pin_memory=True,
        )

        inv_normalize = torchvision.transforms.Compose([
            Normalize(
                mean=[-123.675/58.395, -116.28/57.12, -103.53/57.375],
                std=[1/58.395, 1/57.12, 1/57.375],
            )
        ])

    elif args.dataset == "ssv2":
        crop_size = 224 if '336PX' not in args.model else 336
        root = '/mnt/welles/scratch/datasets/SSV2/mp4_videos'
        train_metadata = os.path.join(parent_path, 'datasets/SSV2/annotation/train.csv')
        val_metadata = os.path.join(parent_path, 'datasets/SSV2/annotation/val.csv')

        transforms_list = [
            Permute([3, 0, 1, 2]),    # T H W C -> C T H W
            torchvision.transforms.RandomResizedCrop(crop_size, scale=(0.5, 1.0), antialias=True),
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
        ]
        transforms_list.append(Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]))
        train_transform = torchvision.transforms.Compose(transforms_list)

        val_transform = torchvision.transforms.Compose([
                Permute([3, 0, 1, 2]),    # T H W C -> C T H W
                torchvision.transforms.Resize(crop_size, antialias=True),
                torchvision.transforms.CenterCrop(crop_size),
                Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
            ])
        
        train_dataset = KineticsDataset(
            root, train_metadata, transform=train_transform, is_training=True, 
            clip_length=args.clip_length, clip_stride=args.clip_stride,
            threads=False,
            fast_rrc=False, rrc_params=(224, (0.5, 1.0)),
            msc_params=(224, ),
            fast_cc=False, cc_params=(224, ),
            hflip_prob=0.5, vflip_prob=0.,
            mask_type='later',  # do masking in batches
            args=args,
        )

        val_dataset = KineticsDataset(
            root, val_metadata, transform=val_transform, is_training=False, 
            clip_length=args.clip_length, clip_stride=args.clip_stride,
            threads=False,
            fast_rrc=False, rrc_params=(224, (0.5, 1.0)),
            msc_params=(224, ),
            fast_cc=False, cc_params=(224, ),
            hflip_prob=0.5, vflip_prob=0.,
            mask_type='later',  # do masking in batches
            args=args,
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=1, shuffle=False,
            num_workers=8, pin_memory=True,
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=1, shuffle=False,
            num_workers=8, pin_memory=True,
        )

        inv_normalize = torchvision.transforms.Compose([
            Normalize(
                mean=[-123.675/58.395, -116.28/57.12, -103.53/57.375],
                std=[1/58.395, 1/57.12, 1/57.375],
            )
        ])

    elif args.dataset == 'charades_ego':
        crop_size = 224 if '336PX' not in args.model else 336
        root = os.path.join(parent_path, 'datasets/CharadesEgo/CharadesEgo_v1_480')
        train_metadata = os.path.join(parent_path, 'datasets/CharadesEgo/CharadesEgo/metadata_filtered_train.pkl')
        val_metadata = os.path.join(parent_path, 'datasets/CharadesEgo/CharadesEgo/CharadesEgo_v1_test_only1st.csv')

        tokenizer = clip.tokenize

        mean, std = [0.485 * 255, 0.456 * 255, 0.406 * 255], [0.229 * 255, 0.224 * 255, 0.225 * 255]
        normalize = K.enhance.Normalize(mean=mean, std=std)


        base_train_transform_ls = [
            Permute_BB([0, 3, 1, 2]),
            v2.RandomResizedCrop(crop_size, scale=(0.5, 1.0), antialias=True),
            # v2.RandomHorizontalFlip(0.5),
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

        train_dataset = VideoCaptionDatasetCLIP(
            'charades_ego_trimmed', root, train_metadata,
            transform=train_transform, is_training=True, tokenizer=tokenizer,
            clip_length=args.clip_length, clip_stride=args.clip_stride
        )
        labels, mapping_vn2act = generate_label_map(args.dataset, root=parent_path)
        val_dataset = VideoClassyDataset(
            args.dataset, root, val_metadata,
            transform=val_transform, is_training=False,
            label_mapping=mapping_vn2act, 
            is_trimmed=False,
            # is_trimmed=True,
            num_clips=1, 
            clip_length=args.clip_length, clip_stride=args.clip_stride,
            # sparse_sample=args.sparse_sample,
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=1, shuffle=False,
            num_workers=8, pin_memory=True,
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=1, shuffle=False,
            num_workers=8, pin_memory=True,
        )

        inv_normalize = torchvision.transforms.Compose([
            Normalize(
                mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                std=[1/0.229, 1/0.224, 1/0.255]
            )
        ])
        

    methods = \
        {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM,
         "layercam": LayerCAM,
         "fullgrad": FullGrad}

    if args.method not in list(methods.keys()):
        raise Exception(f"method should be one of {list(methods.keys())}")

    #####
    # create a folder to save results
    save_dir = os.path.join('CAM_results', args.dataset, args.method)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    if args.dataset == 'charades_ego':
        args.model = 'FRILSCLIP_VITB16'
        model_MSE = getattr(model_FRIL, args.model)(
            freeze_temperature=True,
            use_grad_checkpointing=True,
            patch_dropout=0,
            num_frames=args.clip_length,
            drop_path_rate=0,
            use_fast_conv1=True,
            use_flash_attn=True,
            pretrain_zoo="frils",#args.pretrain_zoo,
            pretrain_path='/home/mona/FRIL/FILS/results/pretrain_FRILS/pretrain_FR_CLIP_vidcaption_vifi_full_all_EK_decoder_head=6__MSE_scale=0__CLIP_scale=1__FR_scale=1__ssvli_iter=1_800_epochs_totalbatch=240_lr=0.00015_CLIP_strategy=patch-average/checkpoint_00800.pt',
            text_pretrain_path=os.path.join(parent_path, 'results/vifi_clip_10_epochs_k400_full_finetuned.pth'),
        ).to(device)

        model_FR = getattr(model_FRIL, args.model)(
            freeze_temperature=True,
            use_grad_checkpointing=True,
            patch_dropout=0,
            num_frames=args.clip_length,
            drop_path_rate=0,
            use_fast_conv1=True,
            use_flash_attn=True,
            pretrain_zoo="frils",#args.pretrain_zoo,
            pretrain_path='/home/mona/FRIL/FILS/results/pretrain_FRILS/pretrain_FR_CLIP_vidcaption_vifi_full_all_EK_decoder_head=6__MSE_scale=0__CLIP_scale=1__FR_scale=1__ssvli_iter=1_800_epochs_totalbatch=240_lr=0.00015_CLIP_strategy=patch-average/checkpoint_00800.pt',
            text_pretrain_path=os.path.join(parent_path, 'results/vifi_clip_10_epochs_k400_full_finetuned.pth'),
        ).to(device)

        model_FR_CLIP = getattr(model_FRIL, args.model)(
            freeze_temperature=True,
            use_grad_checkpointing=True,
            patch_dropout=0,
            num_frames=args.clip_length,
            drop_path_rate=0,
            use_fast_conv1=True,
            use_flash_attn=True,
            pretrain_zoo="frils",#args.pretrain_zoo,
            pretrain_path='/home/mona/FRIL/FILS/results/pretrain_FRILS/pretrain_FR_CLIP_vidcaption_vifi_full_all_EK_decoder_head=6__MSE_scale=0__CLIP_scale=1__FR_scale=1__ssvli_iter=1_800_epochs_totalbatch=240_lr=0.00015_CLIP_strategy=patch-average/checkpoint_00800.pt',
            text_pretrain_path=os.path.join(parent_path, 'results/vifi_clip_10_epochs_k400_full_finetuned.pth'),
        ).to(device)

        # load weights
        load_model(model_FR_CLIP, '/home/mona/FRIL/FILS/results/charades/Finetuning_FR_CLIP_vidcaption_vifi_all_Charades/checkpoint_best.pt')
        # load_model(model_FR, '/home/mona/FRIL/FILS/results/finetune_FRILS/Finetune_FR_FRILS_800__decoder_head=6_all_EK_100_epochs_totalbatch=256_lr=0.0015/checkpoint_best.pt')
        # load_model(model_MSE, '/home/mona/FRIL/FILS/results/finetune_FRILS/Finetune_MSE_FRILS_800__decoder_head=6_all_EK_100_epochs_totalbatch=256_lr=0.0015/checkpoint_best.pt')

    else:

        model_MSE = getattr(model_FRIL, args.model)(
            pretrained=False,
            num_classes=args.nb_classes,
            fc_drop_rate = 0.0,
            drop_rate = 0.0,
            drop_path_rate=0.1,
            attn_drop_rate=0.0,
            use_flash_attn=True,
            use_checkpoint=True,
            channel_last=False,
            args=args,
        ).to(device)

        model_FR = getattr(model_FRIL, args.model)(
            pretrained=False,
            num_classes=args.nb_classes,
            fc_drop_rate = 0.0,
            drop_rate = 0.0,
            drop_path_rate=0.1,
            attn_drop_rate=0.0,
            use_flash_attn=True,
            use_checkpoint=True,
            channel_last=False,
            args=args,
        ).to(device)
        
        model_FR_CLIP = getattr(model_FRIL, args.model)(
            pretrained=False,
            num_classes=args.nb_classes,
            fc_drop_rate = 0.0,
            drop_rate = 0.0,
            drop_path_rate=0.1,
            attn_drop_rate=0.0,
            use_flash_attn=True,
            use_checkpoint=True,
            channel_last=False,
            args=args,
        ).to(device)

        # load weights
        load_model(model_FR_CLIP, '/home/mona/FRIL/FILS/results/finetune_FRILS/Finetune_FR_CLIP_FRILS_800__decoder_head=6_all_EK_100_epochs_totalbatch=256_lr=0.0015/checkpoint_best.pt')
        load_model(model_FR, '/home/mona/FRIL/FILS/results/finetune_FRILS/Finetune_FR_FRILS_800__decoder_head=6_all_EK_100_epochs_totalbatch=256_lr=0.0015/checkpoint_best.pt')
        load_model(model_MSE, '/home/mona/FRIL/FILS/results/finetune_FRILS/Finetune_MSE_FRILS_800__decoder_head=6_all_EK_100_epochs_totalbatch=256_lr=0.0015/checkpoint_best.pt')
    
    
    # set the models to evaluation mode
    model_FR_CLIP.eval()
    model_FR.eval()
    model_MSE.eval()


    video_prf_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 140, 160, 200, 230, 250, 280]

    # read a video from the dataset
    loader = train_loader
    for i, data in enumerate(loader):
        # data = next(iter(loader))
        vid_name = f"val_{i}_{loader.dataset.samples[i][0]}".replace('/', '_')

        if isinstance(data[0], list):
            videos = data[0][0]
        else:
            videos = data[0]
        videos = videos
        label = data[1]

        if args.dataset == 'charades_ego':
            videos = normalize(videos)
            label = 0


        input_tensor_org = videos.clone()
        frame_number_list = [0, 7, 15]
        

        # if the model has the correct prediction, then plot its cam
        with torch.no_grad():
            with amp.autocast():
                if args.use_cuda:
                    input_tensor = input_tensor_org.clone().cuda()

                if args.dataset == 'charades_ego':
                    # MSE model
                    output_MSE = model_MSE(input_tensor, torch.zeros([1, 77]).to(torch.long).to(device))
                    # FR model
                    output_FR = model_FR(input_tensor, torch.zeros([1, 77]).to(torch.long).to(device))
                    # FR-CLIP model
                    output_FR_CLIP = model_FR_CLIP(input_tensor, torch.zeros([1, 77]).to(torch.long).to(device))
                    prediction_MSE, prediction_FR, prediction_FR_CLIP = 1, 1, 0
                else:
                    # MSE model
                    output_MSE = model_MSE(input_tensor)
                    prediction_MSE = torch.argmax(output_MSE, dim=1).item()
                    # FR model
                    output_FR = model_FR(input_tensor)
                    prediction_FR = torch.argmax(output_FR, dim=1).item()
                    # FR-CLIP model
                    output_FR_CLIP = model_FR_CLIP(input_tensor)
                    prediction_FR_CLIP = torch.argmax(output_FR_CLIP, dim=1).item()

        if prediction_FR_CLIP == label and prediction_MSE != label and prediction_FR != label:

            # if not i in video_prf_list:
            #     continue

            for frame_number in frame_number_list:
                rgb_img = inv_normalize(videos.squeeze(0)).permute(1, 2, 3, 0).cpu().numpy()[frame_number]
                if args.dataset == 'ek100_cls' or args.dataset == 'charades_ego':
                    rgb_img = np.clip(rgb_img, 0, 1)

                    # save the original image
                    rgb_img_org = (rgb_img * 255).astype(np.uint8)
                elif args.dataset == 'EGTEA' or args.dataset == 'ssv2':
                    rgb_img /= 255
                    rgb_img = np.clip(rgb_img, 0, 1)

                    # save the original image
                    rgb_img_org = (rgb_img * 255).astype(np.uint8)

                if args.dataset == 'ek100_cls':
                    label_name = loader.dataset.samples[i][-4]
                elif args.dataset == 'EGTEA':
                    label_name = loader.dataset.samples[i][-1].replace('/', '_')
                elif args.dataset == 'ssv2':
                    label_name = loader.dataset.samples[i][1]

                # If None, returns the map for the highest scoring category.
                # Otherwise, targets the requested category.
                if args.dataset == 'charades_ego':
                    targets = None
                else:
                    targets = [ClassifierOutputTarget(label)]

                MSE_cam_dict = {'aug': [], 'eigen': [], 'eigen_aug': [], 'cam': []}
                FR_cam_dict = {'aug': [], 'eigen': [], 'eigen_aug': [], 'cam': []}
                FR_CLIP_cam_dict = {'aug': [], 'eigen': [], 'eigen_aug': [], 'cam': []}

                for layer in range(0, 12):
                    if args.dataset == 'charades_ego':
                        # layer-wise
                        target_layers_MSE = [
                            model_MSE.visual.encoder.blocks[layer].norm1,
                        ]
                        target_layers_FR = [
                            model_FR.visual.encoder.blocks[layer].norm1,
                        ]
                        target_layers_FR_CLIP = [
                            model_FR_CLIP.visual.encoder.blocks[layer].norm1,
                        ]
                    else:
                        # layer-wise
                        target_layers_MSE = [
                            model_MSE.blocks[layer].norm1,
                        ]
                        target_layers_FR = [
                            model_FR.blocks[layer].norm1,
                        ]
                        target_layers_FR_CLIP = [
                            model_FR_CLIP.blocks[layer].norm1,
                        ]
                # for layer in range(0, 1):
                #     # all layers
                #     target_layers_MSE = [model_MSE.blocks[x].norm1 for x in range(12)]
                #     target_layers_FR_CLIP = [model_FR_CLIP.blocks[x].norm1 for x in range(12)]
                #     # specific layers
                #     target_layers_MSE = [
                #         model_MSE.blocks[0].norm1,
                #         model_MSE.blocks[6].norm1,
                #         model_MSE.blocks[11].norm1,
                #     ]
                #     target_layers_FR_CLIP = [
                #         model_FR_CLIP.blocks[0].norm1,
                #         model_FR_CLIP.blocks[6].norm1,
                #         model_FR_CLIP.blocks[11].norm1,
                #     ]
                    
                    start, end = (frame_number//2)*14*14, ( (frame_number//2) + 1 )*14*14
                    def reshape_transform_2(tensor, height=14, width=14):
                        
                        result = tensor[:, start:end, :].reshape(tensor.size(0),
                                                        height, width, tensor.size(2))

                        # Bring the channels to the first dimension,
                        # like in CNNs.
                        result = result.transpose(2, 3).transpose(1, 2)
                        return result

                    if args.method not in methods:
                        raise Exception(f"Method {args.method} not implemented")

                    if args.method == "ablationcam":
                        cam_FR_CLIP = methods[args.method](model=model_FR_CLIP,
                                                target_layers=target_layers_FR_CLIP,
                                                # use_cuda=args.use_cuda,
                                                reshape_transform=reshape_transform_2,
                                                ablation_layer=AblationLayerVit())
                        
                        cam_FR = methods[args.method](model=model_FR,
                                                target_layers=target_layers_FR,
                                                # use_cuda=args.use_cuda,
                                                reshape_transform=reshape_transform_2,
                                                ablation_layer=AblationLayerVit())
                        
                        cam_MSE = methods[args.method](model=model_MSE,
                                                target_layers=target_layers_MSE,
                                                # use_cuda=args.use_cuda,
                                                reshape_transform=reshape_transform_2,
                                                ablation_layer=AblationLayerVit())
                    else:
                        cam_FR_CLIP = methods[args.method](model=model_FR_CLIP,
                                                target_layers=target_layers_FR_CLIP,
                                                # use_cuda=args.use_cuda,
                                                reshape_transform=reshape_transform_2)
                        
                        cam_FR = methods[args.method](model=model_FR,
                                                target_layers=target_layers_FR,
                                                # use_cuda=args.use_cuda,
                                                reshape_transform=reshape_transform_2)
                        
                        cam_MSE = methods[args.method](model=model_MSE,
                                                target_layers=target_layers_MSE,
                                                # use_cuda=args.use_cuda,
                                                reshape_transform=reshape_transform_2)

                    # AblationCAM and ScoreCAM have batched implementations.
                    # You can override the internal batch size for faster computation.
                    cam_FR_CLIP.batch_size = 32
                    cam_FR.batch_size = 32
                    cam_MSE.batch_size = 32

                    ## Added
                    for eigen in [True, False]:
                        for aug in [True, False]:

                            with amp.autocast():

                                grayscale_cam_FR_CLIP = cam_FR_CLIP(input_tensor=input_tensor,
                                                    targets=targets,
                                                    eigen_smooth=eigen,
                                                    aug_smooth=aug)
                                grayscale_cam_FR = cam_FR(input_tensor=input_tensor,
                                                    targets=targets,
                                                    eigen_smooth=eigen,
                                                    aug_smooth=aug)
                                grayscale_cam_MSE = cam_MSE(input_tensor=input_tensor,
                                                    targets=targets,
                                                    eigen_smooth=eigen,
                                                    aug_smooth=aug)

                            # Here grayscale_cam has only one image in the batch
                            grayscale_cam_FR_CLIP = grayscale_cam_FR_CLIP[0, :]
                            grayscale_cam_FR = grayscale_cam_FR[0, :]
                            grayscale_cam_MSE = grayscale_cam_MSE[0, :]

                            cam_image_FR_CLIP = show_cam_on_image(rgb_img, grayscale_cam_FR_CLIP)
                            cam_image_FR = show_cam_on_image(rgb_img, grayscale_cam_FR)
                            cam_image_MSE = show_cam_on_image(rgb_img, grayscale_cam_MSE)
                            if aug and eigen:
                                img_name_MSE = f'{vid_name}_{args.method}_eigen_aug_cam_{layer}_MSE.jpg'
                                img_name_FR = f'{vid_name}_{args.method}_eigen_aug_cam_{layer}_FR.jpg'
                                img_name_FR_CLIP = f'{vid_name}_{args.method}_eigen_aug_cam_{layer}_FR_CLIP.jpg'
                                MSE_cam_dict['eigen_aug'].append(grayscale_cam_MSE)
                                FR_cam_dict['eigen_aug'].append(grayscale_cam_FR)
                                FR_CLIP_cam_dict['eigen_aug'].append(grayscale_cam_FR_CLIP)
                            elif aug:
                                img_name_MSE = f'{vid_name}_{args.method}_aug_cam_{layer}_MSE.jpg'
                                img_name_FR = f'{vid_name}_{args.method}_aug_cam_{layer}_FR.jpg'
                                img_name_FR_CLIP = f'{vid_name}_{args.method}_aug_cam_{layer}_FR_CLIP.jpg'
                                MSE_cam_dict['aug'].append(grayscale_cam_MSE)
                                FR_cam_dict['aug'].append(grayscale_cam_FR)
                                FR_CLIP_cam_dict['aug'].append(grayscale_cam_FR_CLIP)
                            elif eigen:
                                img_name_MSE = f'{vid_name}_{args.method}_eigen_cam_{layer}_MSE.jpg'
                                img_name_FR = f'{vid_name}_{args.method}_eigen_cam_{layer}_FR.jpg'
                                img_name_FR_CLIP = f'{vid_name}_{args.method}_eigen_cam_{layer}_FR_CLIP.jpg'
                                MSE_cam_dict['eigen'].append(grayscale_cam_MSE)
                                FR_cam_dict['eigen'].append(grayscale_cam_FR)
                                FR_CLIP_cam_dict['eigen'].append(grayscale_cam_FR_CLIP)
                            else:
                                img_name_MSE = f'{vid_name}_{args.method}_cam_{layer}_MSE.jpg'
                                img_name_FR = f'{vid_name}_{args.method}_cam_{layer}_FR.jpg'
                                img_name_FR_CLIP = f'{vid_name}_{args.method}_cam_{layer}_FR_CLIP.jpg'
                                MSE_cam_dict['cam'].append(grayscale_cam_MSE)
                                FR_cam_dict['cam'].append(grayscale_cam_FR)
                                FR_CLIP_cam_dict['cam'].append(grayscale_cam_FR_CLIP)

                            img_name_MSE = os.path.join(save_dir, vid_name, str(frame_number), img_name_MSE)
                            img_name_FR = os.path.join(save_dir, vid_name, str(frame_number), img_name_FR)
                            img_name_FR_CLIP = os.path.join(save_dir, vid_name, str(frame_number), img_name_FR_CLIP)
                            frame_name = os.path.join(save_dir, vid_name, str(frame_number), f'frame_{frame_number}_{label_name}.jpg')
                            # check whether the folder exists
                            if not os.path.exists(os.path.dirname(img_name_MSE)):
                                os.makedirs(os.path.dirname(img_name_MSE), exist_ok=True)
                            cv2.imwrite(img_name_MSE, cam_image_MSE)
                            cv2.imwrite(img_name_FR, cam_image_FR)
                            cv2.imwrite(img_name_FR_CLIP, cam_image_FR_CLIP)
                            cv2.imwrite(frame_name, rgb_img_org[:,:,::-1])
                            print(f"Generated {img_name_MSE} for the input video")
                    
                # save the average cam for all layers
                preferred_layer_list = [8, 9, 11]
                for k, v in MSE_cam_dict.items():
                    MSE_cam = np.stack(MSE_cam_dict[k])[preferred_layer_list].mean(0)
                    FR_cam = np.stack(FR_cam_dict[k])[preferred_layer_list].mean(0)
                    FR_CLIP_cam = np.stack(FR_CLIP_cam_dict[k])[preferred_layer_list].mean(0)
                    cam_image_FR_CLIP = show_cam_on_image(rgb_img, FR_CLIP_cam)
                    cam_image_FR = show_cam_on_image(rgb_img, FR_cam)
                    cam_image_MSE = show_cam_on_image(rgb_img, MSE_cam)
                    img_name_MSE = f'{vid_name}_{args.method}_cam_MSE_{k}_avg.jpg'
                    img_name_FR = f'{vid_name}_{args.method}_cam_FR_{k}_avg.jpg' 
                    img_name_FR_CLIP = f'{vid_name}_{args.method}_cam_FR_CLIP_{k}_avg.jpg'
                    img_name_MSE = os.path.join(save_dir, vid_name, str(frame_number), img_name_MSE)
                    img_name_FR = os.path.join(save_dir, vid_name, str(frame_number), img_name_FR)
                    img_name_FR_CLIP = os.path.join(save_dir, vid_name, str(frame_number), img_name_FR_CLIP)
                    cv2.imwrite(img_name_MSE, cam_image_MSE)
                    cv2.imwrite(img_name_FR, cam_image_FR)
                    cv2.imwrite(img_name_FR_CLIP, cam_image_FR_CLIP)
                    print(f"Generated {img_name_MSE} for the input video")