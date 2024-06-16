import argparse
from collections import OrderedDict
import datetime
from functools import partial
import json
import os
from pathlib import Path
import time

from einops import rearrange
import kornia as K
import numpy as np
import pandas as pd
import scipy
from sklearn.metrics import top_k_accuracy_score, confusion_matrix
import torch
import torch.cuda.amp as amp
from torch.distributed.optim import ZeroRedundancyOptimizer
from timm.data.loader import MultiEpochsDataLoader
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import ModelEmaV2, accuracy, get_state_dict
from sklearn.metrics import f1_score
import wandb

import torchvision
from pytorchvideo.transforms import RandAugment, Normalize

# find the path to the current file
current_path = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.dirname(current_path)
# add parent path to the system path
import sys
sys.path.append(parent_path)

from FILS.data.transforms import Permute, TemporalCrop, SpatialCrop
from FILS.data.classification_dataset import VideoClsDataset_FRIL, multiple_samples_collate
from FILS.data.clip_dataset import VideoClassyDataset
import FILS.models.model_FRIL as model_FRIL
from FILS.optim.layer_decay import LayerDecayValueAssigner
from FILS.optim.lion import Lion
from FILS.optim.schedulers import cosine_scheduler
import FILS.utils.distributed as dist_utils
from FILS.utils.meters import AverageMeter, ProgressMeter
from FILS.utils.misc import check_loss_nan, interpolate_pos_embed, generate_label_map, acc_mappping
from FILS.utils.evaluation_ek100cls import get_marginal_indexes, get_mean_accuracy, marginalize


def get_args_parser():
    parser = argparse.ArgumentParser(description='FRIL fine-tune', add_help=False)
    parser.add_argument('--dataset', default='EGTEA', type=str, choices=['ek100_cls', 'EGTEA'])
    parser.add_argument('--root',
                        default=os.path.join(parent_path, 'datasets/EGTEA/cropped_clips'),
                        choices= [
                            os.path.join(parent_path, 'datasets/EK100/EK100_320p_15sec_30fps_libx264'),
                            os.path.join(parent_path, 'datasets/EGTEA/cropped_clips'),
                        ],
                        type=str, help='path to train dataset root')
    parser.add_argument('--root-val',
                        default=os.path.join(parent_path, 'datasets/EGTEA/cropped_clips'),
                        choices= [
                            os.path.join(parent_path, 'datasets/EK100/EK100_320p_15sec_30fps_libx264'),
                            os.path.join(parent_path, 'datasets/EGTEA/cropped_clips'),
                        ],
                        type=str, help='path to val dataset root')
    parser.add_argument('--train-metadata',
                        default=os.path.join(parent_path, 'datasets/EGTEA/train_split1.txt'),
                        choices=[
                            os.path.join(parent_path, 'datasets/EK100/epic-kitchens-100-annotations/EPIC_100_train.csv'),
                            os.path.join(parent_path, 'datasets/EGTEA/train_split1.txt')
                        ],
                        type=str, help='metadata for train split')
    parser.add_argument('--val-metadata',
                        default=os.path.join(parent_path, 'datasets/EGTEA/test_split1.txt'),
                        choices=[
                            os.path.join(parent_path, 'datasets/EK100/epic-kitchens-100-annotations/EPIC_100_validation.csv'),
                            os.path.join(parent_path, 'datasets/EGTEA/test_split1.txt')
                        ],
                        type=str, help='metadata for val split')
    parser.add_argument('--output-dir', default=os.path.join(parent_path, 'results/finetune_FRILS/'), type=str, help='output dir')
    parser.add_argument('--input-size', default=224, type=int, help='input frame size')
    parser.add_argument('--clip-length', default=16, type=int, help='clip length')
    parser.add_argument('--num-clips', default=1, type=int, help='number of clips for testing')
    parser.add_argument('--video-chunk-length', default=15, type=int)
    parser.add_argument('--clip-stride', default=4, type=int, help='clip stride')
    parser.add_argument('--test-num-segment', default=5, type=int, help='number of temporal segments for testing. default is 5.')
    parser.add_argument('--test-num-crop', default=1, type=int, help='number of spatial crops for testing. default is 3.')
    parser.add_argument('--use-pin-memory', action='store_true', dest='use_pin_memory')
    parser.add_argument('--disable-pin-memory', action='store_false', dest='use_pin_memory')
    parser.set_defaults(use_pin_memory=False)
    parser.add_argument('--nb-classes', default=3806, type=int, help='number of classes, EK100: 3806, SSV2: 174')
    # augmentation
    parser.add_argument('--repeated-aug', default=1, type=int)
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    # mixup
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')    
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')
    # model
    parser.add_argument('--model', default='vit_base_patch16_224', type=str)
    parser.add_argument('--channel-last', action='store_true', dest='channel_last')
    parser.add_argument('--disable-channel-last', action='store_false', dest='channel_last')
    parser.set_defaults(channel_last=False)
    parser.add_argument('--grad-checkpointing', action='store_true', dest='use_grad_checkpointing')
    parser.add_argument('--no-grad-checkpointing', action='store_false', dest='use_grad_checkpointing')
    parser.set_defaults(use_grad_checkpointing=True)
    parser.add_argument('--use-flash-attn', action='store_true', dest='use_flash_attn')
    parser.add_argument('--disable-flash-attn', action='store_false', dest='use_flash_attn')
    parser.set_defaults(use_flash_attn=True)
    parser.add_argument('--use-registers', action='store_true', dest='use_registers')
    parser.set_defaults(use_registers=False)
    parser.add_argument('--num-registers', default=8, type=int)
    parser.add_argument('--fc-drop-rate', default=0.0, type=float)
    parser.add_argument('--drop-rate', default=0.0, type=float)
    parser.add_argument('--attn-drop-rate', default=0.0, type=float)
    parser.add_argument('--drop-path-rate', default=0.1, type=float)
    parser.add_argument('--resume', default='', type=str, help='path to resume from')
    # fine-tune
    parser.add_argument('--finetune', default='/home/mona/FRIL/FILS/results/pretrain_FRILS/pretrain_FR_CLIP_vidcaption_vifi_all_SSV2_decoder_head=6__MSE_scale=0__CLIP_scale=1__FR_scale=1__ssvli_iter=1_800_epochs_totalbatch=240_lr=0.00015_CLIP_strategy=patch-average/checkpoint_00780.pt', help='fine-tune path')
    # parser.add_argument('--finetune', default='', help='fine-tune path')
    parser.add_argument('--model-key', default='model|module|state_dict', type=str)
    # model ema
    parser.add_argument('--model-ema', action='store_true', default=False)
    parser.set_defaults(model_ema=True)
    parser.add_argument('--model-ema-decay', type=float, default=0.9999, help='')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')
    # train
    parser.add_argument('--run_name', default='Finetune_FR_CLIP_FRILS_SSV2pretraining_vidcaption_vifi_800_decoder_head=6_all_EGTEA', type=str)
    parser.add_argument('--use-zero', action='store_true', dest='use_zero', help='use ZeRO optimizer')
    parser.add_argument('--no-use-zero', action='store_false', dest='use_zero', help='use ZeRO optimizer')
    parser.set_defaults(use_zero=False)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--warmup-epochs', default=5, type=int)
    parser.add_argument('--start-epoch', default=0, type=int)
    parser.add_argument('--batch-size', default=64, type=int, help='number of samples per-device/per-gpu')
    parser.add_argument('--optimizer', default='adamw', choices=['adamw', 'lion'], type=str)
    parser.add_argument('--lr', default=1.5e-3, type=float)
    parser.add_argument('--layer-decay', type=float, default=0.75)
    parser.add_argument('--lr-start', default=1e-6, type=float, help='initial warmup lr')
    parser.add_argument('--lr-end', default=1e-6, type=float, help='minimum final lr')
    parser.add_argument('--update-freq', default=1, type=int, help='optimizer update frequency (i.e. gradient accumulation steps)')
    parser.add_argument('--wd', default=0.05, type=float)
    parser.add_argument('--wd-end', type=float, default=None)
    parser.add_argument('--betas', default=(0.9, 0.999), nargs=2, type=float)
    parser.add_argument('--eps', default=1e-8, type=float)
    parser.add_argument('--save-freq', default=1, type=int)
    parser.add_argument('--val-freq', default=1, type=int)
    parser.add_argument('--disable-amp', action='store_true', help='disable mixed-precision training (requires more memory and compute)')
    parser.add_argument('--grad-clip-norm', default=None, type=float)
    parser.add_argument('--use-multi-epochs-loader', action='store_true')
    parser.add_argument('--decode-threads', default=1, type=int)
    # system
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers per process')
    parser.add_argument('--evaluate', action='store_true', help='eval only')
    # parser.set_defaults(evaluate=True)
    parser.add_argument('--evaluate-batch-size', default=1, type=int, help='batch size at evaluation')
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('--dist-url', default='env://', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
    return parser


def main(args):
    dist_utils.init_distributed_mode(args)
    dist_utils.random_seed(args.seed, dist_utils.get_rank())
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=> creating model: {}".format(args.model))
    model = getattr(model_FRIL, args.model)(
        pretrained=False,
        num_classes=args.nb_classes,
        fc_drop_rate = args.fc_drop_rate,
        drop_rate = args.drop_rate,
        drop_path_rate=args.drop_path_rate,
        attn_drop_rate=args.attn_drop_rate,
        use_flash_attn=args.use_flash_attn,
        use_checkpoint=args.use_grad_checkpointing,
        channel_last=args.channel_last,
        args=args,
    )
    model.cuda(args.gpu)

    # add scale values to the run name
    args.run_name = args.run_name + "_" + str(args.epochs) + "_epochs_totalbatch=" \
        + str(args.batch_size * dist_utils.get_world_size()) + "_lr=" + str(args.lr) 

    # initialize wandb
    wandb.init(
        project="FRILS_EGTEA",
        group="finetune",
        name=args.run_name,
        config=args,
        )

    # append the run name to the output_dir
    args.output_dir = os.path.join(args.output_dir, args.run_name)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.finetune:
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(args.finetune, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu')
        print("=> Load checkpoint from %s" % args.finetune)
        for model_key in args.model_key.split('|'):
            if model_key in checkpoint:
                checkpoint_model = checkpoint[model_key]
                if list(checkpoint_model.keys())[0].startswith('module.'):
                    renamed_ckpt = {k[7:]: v for k, v in checkpoint_model.items()}
                    checkpoint_model = renamed_ckpt
                print("Load state_dict by model_key = %s" % model_key)
                break
        if checkpoint_model is None:
            checkpoint_model = checkpoint
        for key in ['head.weight', 'head.bias']: ## modify here to remove extra keys
            if key in checkpoint_model and checkpoint_model[key].shape != model.state_dict()[key].shape:
                print("Removing key %s from pretrained checkpoint" % key)
                checkpoint_model.pop(key)

        new_dict = OrderedDict()
        for key in checkpoint_model.keys():
            if key.startswith('backbone.'):
                new_dict[key[9:]] = checkpoint_model[key]
            elif key.startswith('encoder.'):
                if args.use_flash_attn and 'attn.qkv' in key:
                    new_dict[key[8:].replace('attn.qkv', 'attn.Wqkv')] = checkpoint_model[key]
                elif args.use_flash_attn and 'attn.q_bias' in key:
                    q_bias = checkpoint_model[key]
                    v_bias = checkpoint_model[key.replace('attn.q_bias', 'attn.v_bias')]
                    new_dict[key[8:].replace('attn.q_bias', 'attn.Wqkv.bias')] = torch.cat(
                        (q_bias, torch.zeros_like(v_bias), v_bias))
                elif args.use_flash_attn and 'attn.v_bias' in key:
                    continue
                elif args.use_flash_attn and 'attn.proj' in key:
                    new_dict[key[8:].replace('attn.proj', 'attn.out_proj')] = checkpoint_model[key]
                else:
                    new_dict[key[8:]] = checkpoint_model[key]
            else:
                if args.use_flash_attn and 'attn.qkv' in key:
                    new_dict[key.replace('attn.qkv', 'attn.Wqkv')] = checkpoint_model[key]
                elif args.use_flash_attn and 'attn.q_bias' in key:
                    q_bias = checkpoint_model[key]
                    v_bias = checkpoint_model[key.replace('attn.q_bias', 'attn.v_bias')]
                    new_dict[key.replace('attn.q_bias', 'attn.Wqkv.bias')] = torch.cat(
                        (q_bias, torch.zeros_like(v_bias), v_bias))
                elif args.use_flash_attn and 'attn.v_bias' in key:
                    continue
                elif args.use_flash_attn and 'attn.proj' in key:
                    new_dict[key.replace('attn.proj', 'attn.out_proj')] = checkpoint_model[key]
                else:
                    new_dict[key] = checkpoint_model[key]
        checkpoint_model = new_dict

        if 'pos_embed' in checkpoint_model:
            new_pos_embed = interpolate_pos_embed(checkpoint_model['pos_embed'], model, args.clip_length)
            checkpoint_model['pos_embed'] = new_pos_embed
    
        result = model.load_state_dict(checkpoint_model, strict=False)
        print(result)

    model_ema = None
    if args.model_ema:
        model_ema = ModelEmaV2(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else device,
        )
        print("Using EMA with decay = %.8f" % args.model_ema_decay)

    if args.layer_decay < 1.0:
        num_layers = model.get_num_layers()
        ld_assigner = LayerDecayValueAssigner(list(args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)))
    else:
        ld_assigner = None

    # define loss function (criterion) and optimizer
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)
    if mixup_fn is not None:
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    skip_list = {}
    if hasattr(model, "no_weight_decay"):
        skip_list = model.no_weight_decay()
    parameter_group_names = {}
    parameter_group_vars = {}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue  # frozen weights
        if p.ndim < 2 or 'bias' in n or n in skip_list:
            group_name = 'no_decay'
            this_wd = 0.
        else:
            group_name = 'with_decay'
            this_wd = args.wd

        if ld_assigner is not None:
            layer_id = ld_assigner.get_layer_id(n)
            group_name = 'layer_%d_%s' % (layer_id, group_name)

        if group_name not in parameter_group_names:
            if ld_assigner is not None:
                scale = ld_assigner.get_scale(layer_id)
            else:
                scale = 1.

            parameter_group_names[group_name] = {"weight_decay": this_wd, "params": [], "lr_scale": scale}
            parameter_group_vars[group_name] = {"weight_decay": this_wd, "params": [], "lr_scale": scale}

        parameter_group_names[group_name]["params"].append(n)
        parameter_group_vars[group_name]["params"].append(p)

    print("Param groups:", parameter_group_names)
    optim_params = parameter_group_vars.values()

    total_batch_size = args.batch_size * dist_utils.get_world_size()
    args.lr = args.lr * total_batch_size / 256
    args.lr_start = args.lr_start * total_batch_size / 256
    args.lr_end = args.lr_end * total_batch_size / 256
    if args.optimizer == 'adamw':
        opt_fn = torch.optim.AdamW
    elif args.optimizer == 'lion':
        opt_fn = Lion
    else:
        raise ValueError
    if args.use_zero:
        print('Training with ZeroRedundancyOptimizer')
        optimizer = ZeroRedundancyOptimizer(
            optim_params, optimizer_class=opt_fn,
            lr=args.lr, betas=args.betas, eps=args.eps, weight_decay=args.wd
        )
    else:
        optimizer = opt_fn(optim_params, lr=args.lr, betas=args.betas,
                           eps=args.eps, weight_decay=args.wd)
    scaler = amp.GradScaler(enabled=not args.disable_amp)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], bucket_cap_mb=200)

    # optionally resume from a checkpoint (takes precedence over autoresume)
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading resume checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cpu')
            epoch = checkpoint['epoch'] if 'epoch' in checkpoint else 0
            args.start_epoch = epoch
            # remove prefix 'module.' for the keys of the state_dict
            if not args.distributed:
                new_state_dict = OrderedDict()
                for k, v in checkpoint['state_dict'].items():
                    if k.startswith('module.'):
                        k = k.replace('module.', '')
                    new_state_dict[k] = v
            else:
                new_state_dict = checkpoint['state_dict']
            result = model.load_state_dict(new_state_dict, strict=True)
            print(result)
            optimizer.load_state_dict(checkpoint['optimizer']) if 'optimizer' in checkpoint else ()
            scaler.load_state_dict(checkpoint['scaler']) if 'scaler' in checkpoint else ()
            best_acc1 = checkpoint['best_acc1'] if hasattr(checkpoint, 'best_acc1') else 0
            print("=> loaded resume checkpoint '{}' (epoch {})"
                  .format(args.resume, epoch))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        # auto-resume from latest checkpoint in output directory
        latest = os.path.join(args.output_dir, 'checkpoint.pt')
        if os.path.isfile(latest):
            print("=> loading latest checkpoint '{}'".format(latest))
            latest_checkpoint = torch.load(latest, map_location='cpu')
            args.start_epoch = latest_checkpoint['epoch']
            model.load_state_dict(latest_checkpoint['state_dict'])
            optimizer.load_state_dict(latest_checkpoint['optimizer'])
            scaler.load_state_dict(latest_checkpoint['scaler'])
            best_acc1 = latest_checkpoint['best_acc1'] if hasattr(latest_checkpoint, 'best_acc1') else 0
            print("=> loaded latest checkpoint '{}' (epoch {})"
                  .format(latest, latest_checkpoint['epoch']))

    torch.backends.cudnn.benchmark = True

    # build dataset
    if args.dataset == 'ek100_cls':
        _, mapping_vn2act = generate_label_map(args.dataset, root=parent_path)
        # add mamapping_vn2act to args
        args.label_mapping = mapping_vn2act
        args.mapping_act2v = {i: int(vn.split(':')[0]) for (vn, i) in mapping_vn2act.items()}
        args.mapping_act2n = {i: int(vn.split(':')[1]) for (vn, i) in mapping_vn2act.items()}
        args.actions = pd.DataFrame.from_dict({'verb': args.mapping_act2v.values(), 'noun': args.mapping_act2n.values()})
    if args.dataset == "EGTEA":
        _, mapping_vn2act = generate_label_map(args.dataset, root=parent_path)
    num_clips_at_val = args.num_clips
    args.num_clips = 1

    if args.dataset == "EGTEA":
        crop_size = 224 if '336PX' not in args.model else 336
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
                SpatialCrop(crop_size=crop_size, num_crops=args.test_num_crop),
            ])
    
        train_dataset = VideoClassyDataset(
                args.dataset, args.root, args.train_metadata, train_transform,
                is_training=True, label_mapping=mapping_vn2act,
                num_clips=args.num_clips,
                chunk_len=args.video_chunk_length,
                clip_length=args.clip_length, clip_stride=args.clip_stride,
                threads=args.decode_threads,
                rrc_params=(crop_size, (0.5, 1.0)),
            )
        
        val_dataset = VideoClassyDataset(
                args.dataset, args.root_val, args.val_metadata, val_transform,
                is_training=False, label_mapping=mapping_vn2act,
                num_clips=num_clips_at_val,
                chunk_len=args.video_chunk_length,
                clip_length=args.clip_length, clip_stride=args.clip_stride,
                threads=args.decode_threads,
                rrc_params=(crop_size, (0.5, 1.0)),
            )
        
        test_dataset = val_dataset

    else:
        train_dataset = VideoClsDataset_FRIL(
            args.root, args.train_metadata, mode='train', 
            clip_length=args.clip_length, clip_stride=args.clip_stride,
            args=args,
        )

        val_dataset = VideoClsDataset_FRIL(
            args.root_val, args.val_metadata, mode='validation',
            clip_length=args.clip_length, clip_stride=args.clip_stride,
            args=args,
        )

        test_dataset = VideoClsDataset_FRIL(
            args.root_val, args.val_metadata, mode='test',
            clip_length=args.clip_length, clip_stride=args.clip_stride,
            test_num_segment=args.test_num_segment, test_num_crop=args.test_num_crop,
            args=args,
        )

    ###### update train_dataset, val_dataset, and test_dataset with args
    if args.dataset == 'ek100_cls':
        train_dataset.label_mapping = args.label_mapping
        val_dataset.label_mapping = args.label_mapping
        test_dataset.label_mapping = args.label_mapping
    ######

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False)
    else:
        train_sampler, val_sampler, test_sampler = None, None, None

    if args.repeated_aug > 1:
        collate_func = partial(multiple_samples_collate, fold=False)
    else:
        collate_func = None

    if args.use_multi_epochs_loader:
        train_loader = MultiEpochsDataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=args.use_pin_memory, sampler=train_sampler, drop_last=True,
            collate_fn=collate_func,
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=args.use_pin_memory, sampler=train_sampler, drop_last=True,
            collate_fn=collate_func,
        )
    print('len(train_loader) = {}'.format(len(train_loader)))
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False, sampler=val_sampler, drop_last=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.evaluate_batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False, sampler=test_sampler, drop_last=False
    )


    lr_schedule = cosine_scheduler(
        args.lr, args.lr_end, args.epochs, len(train_loader) // args.update_freq,
        warmup_epochs=args.warmup_epochs, start_warmup_value=args.lr_start
    )
    if args.wd_end is None:
        args.wd_end = args.wd
    wd_schedule = cosine_scheduler(args.wd, args.wd_end, args.epochs, len(train_loader) // args.update_freq)


    print(args)

    if args.evaluate:
        _ = test(test_loader, model, args, len(test_dataset))
        return

    print("=> beginning training")
    best_acc1 = 0.
    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
        train_stats = train(
            train_loader, model, criterion, optimizer, 
            scaler, epoch, model_ema, mixup_fn,
            lr_schedule, wd_schedule, args,
        )
        
        # wandb log
        wandb_dict = {}
        for key, value in train_stats.items():
            wandb_dict["train_epoch_"+key] = value
        wandb.log(wandb_dict, step=epoch)
        
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}

        if (epoch + 1) % args.save_freq == 0:
            print("=> saving checkpoint")
            if args.use_zero:
                print('consolidated on rank {} because of ZeRO'.format(args.rank))
                optimizer.consolidate_state_dict(to=args.rank)
            dist_utils.save_on_master({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                    'scaler': scaler.state_dict(),
                    'args': args,
                    'model_ema': get_state_dict(model_ema) if model_ema is not None else None,
                }, False, args.output_dir)
        if (epoch + 1) % args.val_freq == 0:
            print("=> validate")
            val_stats = validate(val_loader, model, epoch, args)
            
            # wandb log
            wandb_dict = {}
            for key, value in val_stats.items():
                wandb_dict["val_epoch_"+key] = value
            wandb.log(wandb_dict, step=epoch)
        
            if best_acc1 < val_stats['Acc@1']:
                best_acc1 = val_stats['Acc@1']
                if args.use_zero:
                    print('consolidated on rank {} because of ZeRO'.format(args.rank))
                    optimizer.consolidate_state_dict(to=args.rank)
                dist_utils.save_on_master({
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                        'scaler': scaler.state_dict(),
                        'best_acc1': best_acc1,
                        'args': args,
                        'model_ema': get_state_dict(model_ema) if model_ema is not None else None,
                    }, True, args.output_dir)
                
            # add val_stats to log_stats
            log_stats = {**log_stats, **{f'val_{k}': v for k, v in val_stats.items()}}        

        if dist_utils.is_main_process():
            with open(os.path.join(args.output_dir, 'log.txt'), 'a') as f:
                f.write(json.dumps(log_stats) + '\n')
                
    print("=> testing")
    # clear cuda cache
    torch.cuda.empty_cache()
    
    test_stats = test(val_loader, model, args, len(val_loader)*val_loader.batch_size)
    # test_stats = validate(test_loader, model, epoch, args)
    
    # wandb log
    wandb_dict = {}
    for key, value in test_stats.items():
        wandb_dict["test_epoch_"+key] = value
    wandb.log(wandb_dict, step=epoch)
    
    log_stats = {**{f'test_{k}': v for k, v in test_stats.items()}}
    
    if dist_utils.is_main_process():
        with open(os.path.join(args.output_dir, 'log.txt'), 'a') as f:
            f.write(json.dumps(log_stats) + '\n')
        
def train(train_loader, model, criterion, optimizer,
          scaler, epoch, model_ema, mixup_fn,
          lr_schedule, wd_schedule, args):
    batch_time = AverageMeter('Time', ':6.2f')
    data_time = AverageMeter('Data', ':6.2f')
    model_time = AverageMeter('Model', ':6.2f')
    mem = AverageMeter('Mem (GB)', ':6.1f')
    metric_names = ['loss', 'Acc@1', 'Acc@5', 'Noun Acc@1', 'Verb Acc@1']
    iters_per_epoch = len(train_loader) // args.update_freq
    metrics = OrderedDict([(name, AverageMeter(name, ':.2e')) for name in metric_names])
    progress = ProgressMeter(
        iters_per_epoch,
        [batch_time, data_time, model_time, mem, *metrics.values()],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()

    for data_iter, inputs in enumerate(train_loader):
        optim_iter = data_iter // args.update_freq

        # measure data loading time
        if args.verbose:
            print('Time to train: {}'.format(datetime.datetime.now()))
        data_time.update(time.time() - end)

        # update weight decay and learning rate according to their schedule
        it = iters_per_epoch * epoch + optim_iter  # global training iteration
        for k, param_group in enumerate(optimizer.param_groups):
            if lr_schedule is not None:
                param_group['lr'] = lr_schedule[it] * param_group["lr_scale"]
            if wd_schedule is not None and param_group['weight_decay'] > 0:
                param_group['weight_decay'] = wd_schedule[it]

        videos = inputs[0].cuda(args.gpu, non_blocking=True)
        targets = inputs[1].cuda(args.gpu, non_blocking=True)
        org_targets = targets.clone()

        if mixup_fn is not None:
            videos, targets = mixup_fn(videos, targets)

        optimizer.zero_grad()

        tic = time.time()
        # compute output
        with amp.autocast(enabled=not args.disable_amp):
            outputs = model(videos)
            loss = criterion(outputs, targets)
            loss /= args.update_freq

        check_loss_nan(loss)
        scaler.scale(loss).backward()

        if (data_iter + 1) % args.update_freq != 0:
            continue

        # compute gradient and do SGD step
        if args.grad_clip_norm is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
        scaler.step(optimizer)
        scaler.update()
        model.zero_grad(set_to_none=True)
        if model_ema is not None:
            model_ema.update(model)

        # torch.cuda.empty_cache()
        model_time.update(time.time() - tic)

        metrics['loss'].update(loss.item(), args.batch_size)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        outputs = torch.softmax(outputs, dim=1)
        acc1, acc5 = accuracy(outputs, org_targets, topk=(1, 5))
        metrics['Acc@1'].update(acc1.item(), videos.size(0))
        metrics['Acc@5'].update(acc5.item(), videos.size(0))
        if args.dataset == 'ek100_cls':
            vi = get_marginal_indexes(args.actions, 'verb')
            ni = get_marginal_indexes(args.actions, 'noun')
            verb_scores = torch.tensor(marginalize(outputs.detach().cpu().numpy(), vi)).cuda(args.gpu, non_blocking=True)
            noun_scores = torch.tensor(marginalize(outputs.detach().cpu().numpy(), ni)).cuda(args.gpu, non_blocking=True)
            target_to_verb = torch.tensor([args.mapping_act2v[a] for a in org_targets.tolist()]).cuda(args.gpu, non_blocking=True)
            target_to_noun = torch.tensor([args.mapping_act2n[a] for a in org_targets.tolist()]).cuda(args.gpu, non_blocking=True)
            acc1_verb, _ = accuracy(verb_scores, target_to_verb, topk=(1, 5))
            acc1_noun, _ = accuracy(noun_scores, target_to_noun, topk=(1, 5))
            metrics['Verb Acc@1'].update(acc1_verb.item(), videos.size(0))
            metrics['Noun Acc@1'].update(acc1_noun.item(), videos.size(0))
        else:
            metrics['Verb Acc@1'].update(0, videos.size(0))
            metrics['Noun Acc@1'].update(0, videos.size(0))

        mem.update(torch.cuda.max_memory_allocated() // 1e9)

        if optim_iter % args.print_freq == 0:
            if args.verbose:
                print('Time to print: {}'.format(datetime.datetime.now()))
            progress.display(optim_iter)
    progress.synchronize()
    return {**{k: v.avg for k, v in metrics.items()},
            'lr': optimizer.param_groups[0]['lr']}


def validate(val_loader, model, epoch, args):
    criterion = torch.nn.CrossEntropyLoss()
    batch_time = AverageMeter('Time', ':6.2f')
    data_time = AverageMeter('Data', ':6.2f')
    model_time = AverageMeter('Model', ':6.2f')
    mem = AverageMeter('Mem (GB)', ':6.1f')
    metric_names = ['loss', 'Acc@1', 'Acc@5', 'f1_score', 'Noun Acc@1', 'Verb Acc@1']
    metrics = OrderedDict([(name, AverageMeter(name, ':.2e')) for name in metric_names])
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, data_time, model_time, mem, *metrics.values()],
        prefix="Epoch: [{}]".format(epoch))

    # switch to eval mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for data_iter, inputs in enumerate(val_loader):
            data_time.update(time.time() - end)

            if isinstance(inputs[0], list):
                logit_allcrops = []
                for crop in inputs[0]:
                    videos = crop.cuda(args.gpu, non_blocking=True)
                    

                    tic = time.time()
                    # compute output
                    with amp.autocast(enabled=not args.disable_amp):
                        outputs = model(videos)
                        

                    logit_allcrops.append(outputs)

                    # torch.cuda.empty_cache()
                    model_time.update(time.time() - tic)

                logit_allcrops = torch.stack(logit_allcrops, dim=0)
                logit = torch.mean(logit_allcrops, dim=0)
                logit = torch.softmax(logit, dim=1)
                targets = inputs[1].cuda(args.gpu, non_blocking=True)
                loss = criterion(logit, targets)
                acc1, acc5 = accuracy(logit, targets, topk=(1, 5))
                f1score = f1_score(targets.detach().cpu(), torch.argmax(logit, 1).detach().cpu(), average="micro")
                metrics['Acc@1'].update(acc1.item(), videos.size(0))
                metrics['Acc@5'].update(acc5.item(), videos.size(0))
                metrics['loss'].update(loss.item(), args.batch_size)
                metrics['f1_score'].update(f1score, args.batch_size)
                metrics['Verb Acc@1'].update(0, videos.size(0))
                metrics['Noun Acc@1'].update(0, videos.size(0))

            else:

                videos = inputs[0].cuda(args.gpu, non_blocking=True)
                targets = inputs[1].cuda(args.gpu, non_blocking=True)

                tic = time.time()
                # compute output
                with amp.autocast(enabled=not args.disable_amp):
                    outputs = model(videos)
                    loss = criterion(outputs, targets)

                outputs = torch.softmax(outputs, dim=1)
                acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
                f1score = f1_score(targets.detach().cpu(), torch.argmax(outputs, 1).detach().cpu(), average="micro")
                metrics['Acc@1'].update(acc1.item(), videos.size(0))
                metrics['Acc@5'].update(acc5.item(), videos.size(0))
                if args.dataset == 'ek100_cls':
                    vi = get_marginal_indexes(args.actions, 'verb')
                    ni = get_marginal_indexes(args.actions, 'noun')
                    verb_scores = torch.tensor(marginalize(outputs.detach().cpu().numpy(), vi)).cuda(args.gpu, non_blocking=True)
                    noun_scores = torch.tensor(marginalize(outputs.detach().cpu().numpy(), ni)).cuda(args.gpu, non_blocking=True)
                    target_to_verb = torch.tensor([args.mapping_act2v[a] for a in targets.tolist()]).cuda(args.gpu, non_blocking=True)
                    target_to_noun = torch.tensor([args.mapping_act2n[a] for a in targets.tolist()]).cuda(args.gpu, non_blocking=True)
                    acc1_verb, _ = accuracy(verb_scores, target_to_verb, topk=(1, 5))
                    acc1_noun, _ = accuracy(noun_scores, target_to_noun, topk=(1, 5))
                    metrics['Verb Acc@1'].update(acc1_verb.item(), videos.size(0))
                    metrics['Noun Acc@1'].update(acc1_noun.item(), videos.size(0))
                else:
                    metrics['Verb Acc@1'].update(0, videos.size(0))
                    metrics['Noun Acc@1'].update(0, videos.size(0))

                # torch.cuda.empty_cache()
                model_time.update(time.time() - tic)

                metrics['loss'].update(loss.item(), args.batch_size)
                metrics['f1_score'].update(f1score, args.batch_size)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            mem.update(torch.cuda.max_memory_allocated() // 1e9)

            if data_iter % args.print_freq == 0:
                progress.display(data_iter)
    progress.synchronize()
    return {k: v.avg for k, v in metrics.items()}


def test(test_loader, model, args, num_videos):
    criterion = torch.nn.CrossEntropyLoss()
    batch_time = AverageMeter('Time', ':6.2f')
    data_time = AverageMeter('Data', ':6.2f')
    model_time = AverageMeter('Model', ':6.2f')
    mem = AverageMeter('Mem (GB)', ':6.1f')
    metric_names = ['loss', 'Acc@1', 'Acc@5', 'f1_score', 'Noun Acc@1', 'Verb Acc@1']
    metrics = OrderedDict([(name, AverageMeter(name, ':.2e')) for name in metric_names])
    progress = ProgressMeter(
        len(test_loader),
        [batch_time, data_time, model_time, mem, *metrics.values()],
        prefix="Testing: ")

    # switch to eval mode
    model.eval()

    all_logits = [[] for _ in range(args.world_size)]
    all_probs = [[] for _ in range(args.world_size)]
    all_targets = [[] for _ in range(args.world_size)]
    total_num = 0
    with torch.no_grad():
        end = time.time()
        for data_iter, inputs in enumerate(test_loader):
            data_time.update(time.time() - end)

            videos = inputs[0].cuda(args.gpu, non_blocking=True)
            targets = inputs[1].cuda(args.gpu, non_blocking=True)
            this_batch_size = videos.shape[0]

            tic = time.time()
            # compute output
            with amp.autocast(enabled=not args.disable_amp):
                # # for single clip
                # targets_repeated = targets
                # for multiple clips
                targets_repeated = torch.repeat_interleave(targets, videos.shape[1])
                videos = rearrange(videos, 'b n t c h w -> (b n) t c h w')

                logits = model(videos)
                loss = criterion(logits, targets_repeated)

            acc1, acc5 = accuracy(logits, targets_repeated, topk=(1, 5))
            f1score = f1_score(targets_repeated.detach().cpu(), torch.argmax(logits, 1).detach().cpu(), average="micro")

            if args.dataset == 'ek100_cls':
                vi = get_marginal_indexes(args.actions, 'verb')
                ni = get_marginal_indexes(args.actions, 'noun')
                verb_scores = marginalize(torch.softmax(logits, dim=1).detach().cpu().numpy(), vi)
                verb_scores = torch.from_numpy(verb_scores).cuda(args.gpu, non_blocking=True)
                noun_scores = marginalize(torch.softmax(logits, dim=1).detach().cpu().numpy(), ni)
                noun_scores = torch.from_numpy(noun_scores).cuda(args.gpu, non_blocking=True)
                target_to_verb = np.array([args.mapping_act2v[a] for a in targets_repeated.tolist()])
                target_to_verb = torch.from_numpy(target_to_verb).cuda(args.gpu, non_blocking=True)
                target_to_noun = np.array([args.mapping_act2n[a] for a in targets_repeated.tolist()])
                target_to_noun = torch.from_numpy(target_to_noun).cuda(args.gpu, non_blocking=True)
                acc1_verb, _ = accuracy(verb_scores, target_to_verb, topk=(1, 5))
                acc1_noun, _ = accuracy(noun_scores, target_to_noun, topk=(1, 5))
            
            output_dict = acc_mappping(args, {'acc1': acc1, 'acc5': acc5, 'verb_acc1': acc1_verb, 'noun_acc1': acc1_noun, 'f1score': f1score})
            acc1, acc5, acc1_verb, acc1_noun, f1score = output_dict['acc1'], output_dict['acc5'], output_dict['verb_acc1'], output_dict['noun_acc1'], output_dict['f1score']

            logits = rearrange(logits, '(b n) k -> b n k', b=this_batch_size)
            probs = torch.softmax(logits, dim=2)
            gathered_logits = [torch.zeros_like(logits) for _ in range(args.world_size)]
            gathered_probs = [torch.zeros_like(probs) for _ in range(args.world_size)]
            gathered_targets = [torch.zeros_like(targets) for _ in range(args.world_size)]
            if args.distributed:
                torch.distributed.all_gather(gathered_logits, logits)
                torch.distributed.all_gather(gathered_probs, probs)
                torch.distributed.all_gather(gathered_targets, targets)
                for j in range(args.world_size):
                    all_logits[j].append(gathered_logits[j].detach().cpu())
                    all_probs[j].append(gathered_probs[j].detach().cpu())
                    all_targets[j].append(gathered_targets[j].detach().cpu())
            else:
                all_logits[0].append(logits.detach().cpu())
                all_probs[0].append(probs.detach().cpu())
                all_targets[0].append(targets.detach().cpu())

            # torch.cuda.empty_cache()
            model_time.update(time.time() - tic)

            metrics['loss'].update(loss.item(), this_batch_size)
            metrics['Acc@1'].update(acc1.item(), this_batch_size)
            metrics['Acc@5'].update(acc5.item(), this_batch_size)
            metrics['f1_score'].update(f1score, this_batch_size)
            metrics['Noun Acc@1'].update(acc1_noun.item(), this_batch_size)
            metrics['Verb Acc@1'].update(acc1_verb.item(), this_batch_size)
            total_num += logits.shape[0] * args.world_size

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            mem.update(torch.cuda.max_memory_allocated() // 1e9)

            if data_iter % args.print_freq == 0:
                progress.display(data_iter)
    progress.synchronize()
    for j in range(args.world_size):
        all_logits[j] = torch.cat(all_logits[j], dim=0).numpy()
        all_probs[j] = torch.cat(all_probs[j], dim=0).numpy()
        all_targets[j] = torch.cat(all_targets[j], dim=0).numpy()
    all_logits_reorg, all_probs_reorg, all_targets_reorg = [], [], []
    for i in range(total_num):
        all_logits_reorg.append(all_logits[i % args.world_size][i // args.world_size])
        all_probs_reorg.append(all_probs[i % args.world_size][i // args.world_size])
        all_targets_reorg.append(all_targets[i % args.world_size][i // args.world_size])
    all_logits = np.stack(all_logits_reorg, axis=0)
    all_probs = np.stack(all_probs_reorg, axis=0)
    all_targets = np.stack(all_targets_reorg, axis=0)
    all_logits = all_logits[:num_videos, :].mean(axis=1)
    all_probs = all_probs[:num_videos, :].mean(axis=1)
    all_targets = all_targets[:num_videos, ]

    for s, all_preds in zip(['logits', ' probs'], [all_logits, all_probs]):
        if s == 'logits': all_preds = scipy.special.softmax(all_preds, axis=1)

        if args.dataset == 'ek100_cls':
            vi = get_marginal_indexes(args.actions, 'verb')
            ni = get_marginal_indexes(args.actions, 'noun')
            verb_scores = marginalize(all_preds, vi)
            noun_scores = marginalize(all_preds, ni)
            target_to_verb = np.array([args.mapping_act2v[a] for a in all_targets.tolist()])
            target_to_noun = np.array([args.mapping_act2n[a] for a in all_targets.tolist()])
            cm = confusion_matrix(target_to_verb, verb_scores.argmax(axis=1))
            _, acc = get_mean_accuracy(cm)
            output_dict = acc_mappping(args, {'verb_acc1': acc})
            acc = output_dict['verb_acc1']
            print('Verb Acc@1: {:.3f}'.format(acc))
            # metrics['Verb Acc@1'] = acc
            cm = confusion_matrix(target_to_noun, noun_scores.argmax(axis=1))
            _, acc = get_mean_accuracy(cm)
            output_dict = acc_mappping(args, {'noun_acc1': acc})
            acc = output_dict['noun_acc1']
            print('Noun Acc@1: {:.3f}'.format(acc))
            # metrics['Noun Acc@1'] = acc


    # for Epic-Kitchens that does not have all classes in test set
    if args.dataset == 'ek100_cls':
        unique_targets = np.unique(all_targets)
        # filter out classes that are not in unique_targets in all_logits and all_probs
        all_logits = all_logits[:, unique_targets]
        all_probs = all_probs[:, unique_targets]
        # create a mapping and reset all_targets from 0 to len(unique_targets)
        mapping = {k: v for v, k in enumerate(unique_targets)}
        # print mapping
        print("Mapping: ", mapping)
        all_targets = np.array([mapping[t] for t in all_targets])
        num_classes = len(unique_targets)
    else:
        num_classes = args.nb_classes

    
    for s, all_preds in zip(['logits', ' probs'], [all_logits, all_probs]):
        if s == 'logits': all_preds = scipy.special.softmax(all_preds, axis=1)
            
        acc1 = top_k_accuracy_score(all_targets, all_preds, k=1, labels=np.arange(0, num_classes))
        acc5 = top_k_accuracy_score(all_targets, all_preds, k=5, labels=np.arange(0, num_classes))
        output_dict = acc_mappping(args, {'acc1': acc1, 'acc5': acc5})
        acc1, acc5 = output_dict['acc1'], output_dict['acc5']
        dataset = 'EK100' if args.dataset == 'ek100_cls' else 'EGTEA'
        print('[Average {s}] {dataset} * Acc@1 {top1:.3f} Acc@5 {top5:.3f}'.format(s=s, dataset=dataset, top1=acc1, top5=acc5))
        cm = confusion_matrix(all_targets, all_preds.argmax(axis=1))
        mean_acc, acc = get_mean_accuracy(cm)
        # print('Mean Acc. = {:.3f}, Top-1 Acc. = {:.3f}'.format(mean_acc, acc))

    return {k: v.avg for k, v in metrics.items()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser('FILS training and evaluation', parents=[get_args_parser()])
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
    wandb.finish()
