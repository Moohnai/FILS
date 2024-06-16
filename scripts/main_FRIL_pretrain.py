import argparse
from collections import OrderedDict
import copy
import datetime
import json
import math
import os
from pathlib import Path
import time
import numpy as np
import wandb


from einops import rearrange
import pandas as pd
import kornia as K
import torch
import torch.nn.functional as F
import torch.cuda.amp as amp
from torch.distributed.optim import ZeroRedundancyOptimizer
import torchvision.transforms._transforms_video as transforms_video
import torchvision
from timm.data.loader import MultiEpochsDataLoader

# find the path to the current file
current_path = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.dirname(current_path)
# add parent path to the system path
import sys
sys.path.append(parent_path)

from avion.data.clip_dataset import get_pretrain_dataset_FRIL
from avion.data.kinetics_dataset import KineticsDataset
from torchvision.transforms import v2
from avion.data.transforms import GroupMultiScaleCrop, Permute, TubeMaskingGeneratorGPU, Permute_BB, TubeMaskingGeneratorCross
import avion.models.model_FRIL as model_FRIL
from avion.optim.lion import Lion
from avion.optim.schedulers import cosine_scheduler, cyclic_decay_cosine_scheduler
from avion.losses.losses import ClipLoss, Feature_Reconstruction_Loss
import avion.utils.distributed as dist_utils
from avion.utils.meters import AverageMeter, ProgressMeter
from avion.utils.misc import check_loss_nan, generate_label_map, get_grad_norm_


def get_args_parser():
    parser = argparse.ArgumentParser(description='FRIL pretrain', add_help=False)
    parser.add_argument('--dataset', default='ek100_cls', type=str, choices=['ek100_cls', 'ssv2'])
    parser.add_argument('--root',
                        default=os.path.join(parent_path, 'datasets/EK100/EK100_320p_15sec_30fps_libx264'),
                        type=str, help='path to train dataset root',
                        choices=[
                            os.path.join(parent_path, 'datasets/EK100/EK100_320p_15sec_30fps_libx264'),
                            '/mnt/welles/scratch/datasets/SSV2/mp4_videos',
                            ]
                        )
    parser.add_argument('--train-metadata', type=str,
                        default=os.path.join(parent_path, 'datasets/EK100/epic-kitchens-100-annotations/EPIC_100_train.csv'),
                        choices=[
                            os.path.join(parent_path, 'datasets/EK100/epic-kitchens-100-annotations/EPIC_100_train.csv'),
                            os.path.join(parent_path, 'datasets/SSV2/annotation/train.csv'),
                            ],
                        )
    parser.add_argument('--val-metadata', type=str,
                        default=os.path.join(parent_path, 'datasets/EK100/epic-kitchens-100-annotations/EPIC_100_validation.csv'),
                        choices=[
                            os.path.join(parent_path, 'datasets/EK100/epic-kitchens-100-annotations/EPIC_100_validation.csv'),
                            os.path.join(parent_path, 'datasets/ssv2/val.csv'),
                            ],
                        )
    parser.add_argument('--output-dir', default=os.path.join(parent_path, 'results/pretrain_FRILS/'), type=str, help='output dir')
    parser.add_argument('--input-size', default=224, type=int, help='input frame size')
    parser.add_argument('--clip-length', default=16, type=int, help='clip length')
    parser.add_argument('--num-clips', default=1, type=int, help='number of clips for testing')
    parser.add_argument('--video-chunk-length', default=15, type=int)
    parser.add_argument('--clip-stride', default=4, type=int, help='clip stride')
    parser.add_argument('--mask-ratio', default=0.9, type=float, help='mask ratio')
    parser.add_argument('--fused-decode-crop', action='store_true', dest='fused_decode_crop')
    parser.add_argument('--no-fused-decode-crop', action='store_false', dest='fused_decode_crop')
    parser.set_defaults(fused_decode_crop=False)
    parser.add_argument('--decode-threads', default=1, type=int)
    parser.add_argument('--use-pin-memory', action='store_true', dest='use_pin_memory')
    parser.add_argument('--disable-pin-memory', action='store_false', dest='use_pin_memory')
    parser.set_defaults(use_pin_memory=False)
    # model
    parser.add_argument('--model', default='FRILS_VITB16', type=str) # FRILS_VITB16  FRILSCross_VITB32
    parser.add_argument('--channel-last', action='store_true', dest='channel_last')
    parser.add_argument('--disable-channel-last', action='store_false', dest='channel_last')
    parser.set_defaults(channel_last=False)
    parser.add_argument('--decoder-depth', default=6, type=int, help='decoder depth')#4
    parser.add_argument('--grad-checkpointing', action='store_true', dest='use_grad_checkpointing')
    parser.add_argument('--no-grad-checkpointing', action='store_false', dest='use_grad_checkpointing')
    parser.set_defaults(use_grad_checkpointing=True)
    parser.add_argument('--use-flash-attn-at-encoder', action='store_true', dest='use_flash_attn_at_encoder')
    parser.add_argument('--disable-flash-attn-at-encoder', action='store_false', dest='use_flash_attn_at_encoder')
    parser.set_defaults(use_flash_attn_at_encoder=True)
    parser.add_argument('--use-flash-attn-at-decoder', action='store_true', dest='use_flash_attn_at_decoder')
    parser.add_argument('--disable-flash-attn-at-decoder', action='store_false', dest='use_flash_attn_at_decoder')
    parser.set_defaults(use_flash_attn_at_decoder=True)
    parser.add_argument('--drop-path-rate', default=0., type=float)
    parser.add_argument('--resume', default='', type=str, help='path to resume from')
    parser.add_argument('--pretrain-path', 
                        default='', #os.path.join(parent_path, 'results/vit_b_16-laion400m_e32-55e67d44.pt')
                        type=str, help='path to pretrain model')
    parser.add_argument('--normalize-target', action='store_true', dest='normalize_target')
    parser.add_argument('--no-normalize-target', action='store_false', dest='normalize_target')
    parser.set_defaults(normalize_target=True)
    # train
    parser.add_argument('--run_name', default='pretrain_ActCLIP_vidcaption_vifi_all_EK100', type=str)
    parser.add_argument('--use-zero', action='store_true', dest='use_zero', help='use ZeRO optimizer')
    parser.add_argument('--no-use-zero', action='store_false', dest='use_zero', help='use ZeRO optimizer')
    parser.set_defaults(use_zero=False)
    parser.add_argument('--epochs', default=800, type=int)
    parser.add_argument('--warmup-epochs', default=20, type=int)
    parser.add_argument('--start-epoch', default=0, type=int)
    parser.add_argument('--batch-size', default=60, type=int, help='number of samples per-device/per-gpu')
    parser.add_argument('--optimizer', default='adamw', choices=['adamw', 'lion'], type=str)
    parser.add_argument('--lr', default=1.5e-4, type=float) # 1.5e-4 #best for epic:1.2e-4
    parser.add_argument('--fix-lr', action='store_true', help='disable cosine lr decay if set True')
    parser.add_argument('--lr-start', default=1e-6, type=float, help='initial warmup lr')
    parser.add_argument('--lr-end', default=1e-5, type=float, help='minimum final lr')
    parser.add_argument('--update-freq', default=1, type=int, help='optimizer update frequency (i.e. gradient accumulation steps)')
    parser.add_argument('--wd', default=0.05, type=float)
    parser.add_argument('--betas', default=(0.9, 0.95), nargs=2, type=float)
    parser.add_argument('--eps', default=1e-8, type=float)
    parser.add_argument('--save-freq', default=20, type=int)
    parser.add_argument('--disable-amp', action='store_true', help='disable mixed-precision training (requires more memory and compute)')
    parser.set_defaults(disable_amp=False)
    parser.add_argument('--grad-clip-norm', default=None, type=float)
    parser.add_argument('--use-multi-epochs-loader', action='store_true')
    parser.add_argument('--motion_box_path', 
                        default='/mnt/welles/scratch/datasets/Epic-kitchen/EPIC-KITCHENS/EPIC_100_action_recognition/EPIC_100_BB_smooth_train.json', 
                        type=str, help='path to motion box json file',
                        choices=[
                            '/mnt/welles/scratch/datasets/Epic-kitchen/EPIC-KITCHENS/EPIC_100_action_recognition/EPIC_100_BB_smooth_train.json',
                            '/mnt/welles/scratch/datasets/SSV2/Unsupervised_BB_SSV2_train.json',
                            ])
    parser.add_argument('--embedded_text_path', 
                        default=os.path.join(parent_path, "datasets/EK100/vifi_full_epic_train_video_caption_text_dict.pt"), 
                        help='path to embedded text',
                        choices=[
                            os.path.join(parent_path, "datasets/EK100/vifi_full_epic_train_video_caption_text_dict.pt"),
                            os.path.join(parent_path, "datasets/SSV2/vifi_full_SSV2_train_video_caption_text_dict.pt"),
                        ]
                        )
    parser.add_argument('--MSE_scale', default=0, type=float, help='the weight of MSE loss')
    parser.add_argument('--CLIP_scale', default=1, type=float, help='the weight of clip loss')
    parser.add_argument('--FR_scale', default=0, type=float, help='the weight of feature reconstruction loss')
    parser.add_argument('--CLIP-strategy', default='patch-average', type=str, help='the strategy of CLIP', choices=['patch', 'average', 'patch-average'])
    parser.add_argument('--patch_iter', default='1', type=int, help='the number of iterations for patch-wise clip loss')
    parser.add_argument('--ema', type=float, nargs=2, default=[0.996, 1.0], metavar='M',
                        help='EMA momentum schedule (default: 0.996 1.0)')
    parser.add_argument('--ipe_scale', type=float, default=1.0, metavar='M',
                        help='Inverse proportionality constant for EMA momentum schedule (default: 1.0)')
    # system
    parser.add_argument('--print-freq', default=20, type=int, help='print frequency')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',#8
                        help='number of data loading workers per process')
    parser.add_argument('--evaluate', action='store_true', help='eval only')
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

    print("=> creating model: {}".format(args.model))
    model = getattr(model_FRIL, args.model)(
        pretrained=False,
        drop_path_rate=args.drop_path_rate,
        decoder_depth=args.decoder_depth,
        use_flash_attn_at_encoder=args.use_flash_attn_at_encoder,
        use_flash_attn_at_decoder=args.use_flash_attn_at_decoder,
        use_checkpoint=args.use_grad_checkpointing,
        channel_last=args.channel_last,
    )
    model.cuda(args.gpu)
    teacher_model = copy.deepcopy(model)

    # add scale values to the run name
    args.run_name = args.run_name +"_decoder_head="+ str(args.decoder_depth)+ "__MSE_scale=" + str(args.MSE_scale) + "__CLIP_scale=" \
        + str(args.CLIP_scale) + "__FR_scale=" + str(args.FR_scale) + "__ssvli_iter=" + str(args.patch_iter) \
            + "_" + str(args.epochs) + "_epochs_totalbatch=" + str(args.batch_size * dist_utils.get_world_size()) \
                + "_lr=" + str(args.lr) 
    if args.CLIP_scale > 0:
        args.run_name = args.run_name + "_CLIP_strategy=" + args.CLIP_strategy

    if args.CLIP_strategy == 'patch-average':
        args.patch_iter = 1

    if args.pretrain_path != '':
        args.run_name += ' pre-pretrain'

    # initialize wandb
    wandb.init(
        project="FRILS_EK100",
        group="pretrained",
        name=args.run_name,
        config=args,
        )
    
    # append the run name to the output_dir
    args.output_dir = os.path.join(args.output_dir, args.run_name)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    patch_size = model.encoder.patch_embed.patch_size
    args.window_size = (args.clip_length // 2, args.input_size // patch_size[0], args.input_size // patch_size[1])
    args.patch_size = patch_size
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], bucket_cap_mb=200)

    # define loss function (criterion) and optimizer
    MSE_criterion = torch.nn.MSELoss().cuda(args.gpu)
    Clip_criterion = ClipLoss().cuda(args.gpu)
    FR_criterion = Feature_Reconstruction_Loss().cuda(args.gpu)

    n_wd, n_non_wd = [], []
    p_wd, p_non_wd = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue  # frozen weights
        if (p.ndim < 2 or 'bias' in n or
            'ln' in n or 'bn' in n or
            'pos_embed' in n or 'positional_embedding' in n
        ):
            n_non_wd.append(n)
            p_non_wd.append(p)
        else:
            n_wd.append(n)
            p_wd.append(p)

    print('parameters without wd:', n_non_wd)
    print('parameters with wd:', n_wd)
    optim_params = [{"params": p_wd, "weight_decay": args.wd},
                    {"params": p_non_wd, "weight_decay": 0}]

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

    # optionally start from an another pretrained checkpoint
    if args.pretrain_path:
        if os.path.isfile(args.pretrain_path):
            print("=> loading pretrain checkpoint '{}'".format(args.pretrain_path))
            state_dict = torch.load(args.pretrain_path, map_location='cpu')
            new_dict = OrderedDict()
            for key in state_dict.keys():
                # if not args.distributed:
                #     # remove 'module' prefix
                #     new_dict[key.replace('module.', '')] = state_dict[key]

                # # remove 'encoder.' prefix
                # new_dict[key.replace('encoder.', '')] = state_dict[key]

                if key.startswith('visual.transformer.resblocks.'):
                    new_key = key.replace('visual.transformer.resblocks', 'encoder.blocks')
                    new_key = new_key.replace('in_proj_weight', 'Wqkv.weight')
                    new_key = new_key.replace('in_proj_bias', 'Wqkv.bias')
                    new_key = new_key.replace('ln_1', 'norm2')
                    new_key = new_key.replace('c_fc', 'fc1')
                    new_key = new_key.replace('c_proj', 'fc2')
                    new_key = new_key.replace('visual.ln_post', 'encoder.norm')
                    new_dict[new_key] = state_dict[key]
            missing_keys, unexpected_keys = model.load_state_dict(new_dict, strict=False)
            print("=> loaded resume checkpoint '{}'"
                  .format(args.pretrain_path))
            print("missing_keys: ", missing_keys)
            print("unexpected_keys: ", unexpected_keys)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # optionally resume from a checkpoint (takes precedence over autoresume)
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading resume checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cpu')
            epoch = checkpoint['epoch'] if 'epoch' in checkpoint else 0
            args.start_epoch = epoch
            result = model.load_state_dict(checkpoint['state_dict'], strict=False)
            print(result)
            optimizer.load_state_dict(checkpoint['optimizer']) if 'optimizer' in checkpoint else ()
            scaler.load_state_dict(checkpoint['scaler']) if 'scaler' in checkpoint else ()
            best_acc1 = checkpoint['best_acc1'] if 'best_acc1' in checkpoint else 0.
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
            best_acc1 = latest_checkpoint['best_acc1'] if 'best_acc1' in latest_checkpoint else 0.
            print("=> loaded latest checkpoint '{}' (epoch {})"
                  .format(latest, latest_checkpoint['epoch']))

    torch.backends.cudnn.benchmark = True

    # data loading
    mean, std = [0.485 * 255, 0.456 * 255, 0.406 * 255], [0.229 * 255, 0.224 * 255, 0.225 * 255]
    normalize = K.enhance.Normalize(mean=mean, std=std)

    ####################
    crop_size = 336 if args.model.endswith("_336PX") else 224

    if args.fused_decode_crop:
        base_train_transform_ls = [
            # Permute([3, 0, 1, 2]),
            # transforms_video.NormalizeVideo(mean=mean, std=std),
        ]
        gpu_train_transform_ls = [K.enhance.Normalize(mean=mean, std=std)]
        base_val_transform_ls = [
            # Permute([3, 0, 1, 2]),
            # torchvision.transforms.Resize(crop_size),
        ]
        gpu_val_transform_ls = [K.enhance.Normalize(mean=mean, std=std)]
    else:
        # base_train_transform_ls = [
        #     Permute([3, 0, 1, 2]),
        #     torchvision.transforms.RandomResizedCrop(crop_size, scale=(0.5, 1.0)),
        #     transforms_video.NormalizeVideo(mean=mean, std=std),
        # ]
        base_train_transform_ls = [
            Permute_BB([0, 3, 1, 2]),
            v2.RandomResizedCrop(crop_size, scale=(0.5, 1.0), antialias=True),
            v2.RandomHorizontalFlip(0.5),
            Permute_BB([1, 0, 2, 3]),
        ]
        gpu_train_transform_ls = []
        # base_val_transform_ls = [
        #     Permute([3, 0, 1, 2]),
        #     torchvision.transforms.Resize(crop_size),
        #     torchvision.transforms.CenterCrop(crop_size),
        #     transforms_video.NormalizeVideo(mean=mean, std=std),
        # ]
        base_val_transform_ls = [
            Permute_BB([0, 3, 1, 2]),
            v2.Resize(crop_size, antialias=True),
            v2.CenterCrop(crop_size),
            Permute_BB([1, 0, 2, 3]),
        ]
        gpu_val_transform_ls = []
    # train_transform = torchvision.transforms.Compose(base_train_transform_ls)
    train_transform = v2.Compose(base_train_transform_ls)
    train_transform_gpu = torch.nn.Sequential(*gpu_train_transform_ls)
    val_transform = torchvision.transforms.Compose(base_val_transform_ls)
    val_transform_gpu = torch.nn.Sequential(*gpu_val_transform_ls)

    # build dataset
    if args.dataset == 'ek100_cls':
        _, mapping_vn2act = generate_label_map(args.dataset, root=parent_path)
        if args.dataset == 'ek100_cls':
            args.mapping_act2v = {i: int(vn.split(':')[0]) for (vn, i) in mapping_vn2act.items()}
            args.mapping_act2n = {i: int(vn.split(':')[1]) for (vn, i) in mapping_vn2act.items()}
            args.actions = pd.DataFrame.from_dict({'verb': args.mapping_act2v.values(), 'noun': args.mapping_act2n.values()})
    else:
        mapping_vn2act = None
    num_clips_at_val = args.num_clips
    args.num_clips = 1
    train_dataset = get_pretrain_dataset_FRIL(
        train_transform, crop_size, args, subset='train', label_mapping=mapping_vn2act,
    )
    args.num_clips = num_clips_at_val
    # val_dataset = get_downstream_dataset_FRIL(
    #     val_transform, crop_size, args, subset='val', label_mapping=mapping_vn2act,
    # )

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        # val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None
    if args.use_multi_epochs_loader:
        train_loader = MultiEpochsDataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=False, sampler=train_sampler, drop_last=True
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            collate_fn=None,
            num_workers=args.workers, pin_memory=False, sampler=train_sampler, drop_last=True
        )   
    ####################


    print('len(train_loader) = {}'.format(len(train_loader)))

    if args.fix_lr:
        lr_schedule = None
    else:
        lr_schedule = cosine_scheduler(
            args.lr, args.lr_end, args.epochs, len(train_loader) // args.update_freq,
            warmup_epochs=args.warmup_epochs, start_warmup_value=args.lr_start
        )
        # lr_schedule = cyclic_decay_cosine_scheduler(
        #     args.lr, args.lr_end, args.epochs, len(train_loader) // args.update_freq,
        #     warmup_epochs=args.warmup_epochs, start_warmup_value=args.lr_start,
        #     decay_coef=0.9, warmup_decay_coef=0.8, cycle_epochs=[800],
        # )

    # -- momentum schedule
    ipe = len(train_loader)
    momentum_scheduler = (args.ema[0] + i*(args.ema[1]-args.ema[0])/(ipe*args.epochs*args.ipe_scale)
                          for i in range(int(ipe*args.epochs*args.ipe_scale)+1))

    print(args)

    print("=> beginning training")
    best_acc1 = 0.
    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)

        # measure epoch time
        epoch_start_time = time.time()

        train_stats = train(
            train_loader, 
            normalize, 
            model, 
            teacher_model,
            MSE_criterion, 
            Clip_criterion,
            FR_criterion,
            optimizer, 
            scaler, 
            epoch, 
            lr_schedule, 
            args)
        
        epoch_time = time.time() - epoch_start_time
        print('Epoch time: {}'.format(datetime.timedelta(seconds=epoch_time)))

        # wandb log
        wandb_dict = {}
        for key, value in train_stats.items():
            wandb_dict["train_epoch_"+key] = value
        wandb.log(wandb_dict, step=epoch)

        # momentum update of target encoder
        with torch.no_grad():
            m = next(momentum_scheduler)
            for param_q, param_k in zip(model.parameters(), teacher_model.parameters()):
                param_k.data.mul_(m).add_((1.-m) * param_q.detach().data)


        if (epoch + 1) % args.save_freq == 0:
            print("=> saving checkpoint")
            if args.use_zero:
                print('consolidated on rank {} because of ZeRO'.format(args.rank))
                optimizer.consolidate_state_dict(0)
            dist_utils.save_on_master_v2({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict() if dist_utils.is_main_process() else None,
                    'scaler': scaler.state_dict(),
                    'args': args,
                }, epoch + 1, args.output_dir)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}

        if dist_utils.is_main_process():
            with open(os.path.join(args.output_dir, 'log.txt'), 'a') as f:
                f.write(json.dumps(log_stats) + '\n')
        
def train(
        train_loader, 
        normalize, 
        model, 
        teacher_model,
        MSE_criterion,
        Clip_criterion,
        FR_criterion,
        optimizer, 
        scaler, 
        epoch, 
        lr_schedule, 
        args,
    ):
    batch_time = AverageMeter('Time', ':6.2f')
    data_time = AverageMeter('Data', ':6.2f')
    model_time = AverageMeter('Model', ':6.2f')
    mem = AverageMeter('Mem (GB)', ':6.1f')
    metric_names = ['loss', 'mse_loss', 'clip_loss', 'fr_loss', 'total_loss', 'clip_acc', 'loss_scale']
    if args.grad_clip_norm is not None:
        metric_names.append('grad_norm')
    iters_per_epoch = len(train_loader) // args.update_freq
    metrics = OrderedDict([(name, AverageMeter(name, ':.2e')) for name in metric_names])
    progress = ProgressMeter(
        iters_per_epoch,
        [batch_time, data_time, model_time, mem, *metrics.values()],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()

    for data_iter, batch in enumerate(train_loader):
        optim_iter = data_iter // args.update_freq

        inputs, label, motion_patch_yab, text_embed = batch

        # cast text_embed to cuda
        text_embed = text_embed.cuda(args.gpu, non_blocking=True)#.squeeze(1)

        # measure data loading time
        if args.verbose:
            print('Time to train: {}'.format(datetime.datetime.now()))
        data_time.update(time.time() - end)

        # update weight decay and learning rate according to their schedule
        it = iters_per_epoch * epoch + optim_iter  # global training iteration
        for k, param_group in enumerate(optimizer.param_groups):
            if lr_schedule is not None:
                param_group['lr'] = lr_schedule[it]

        videos = inputs.cuda(args.gpu, non_blocking=True)
        if args.fused_decode_crop:
            videos = videos.permute(0, 4, 1, 2, 3)

        if args.model == 'FRILSCross_VITB32':
            bool_masked_pos, ids_restore, ids_keep = TubeMaskingGeneratorCross(videos.shape[0], args.window_size, args.mask_ratio, 1.0, device=args.gpu)()
        else:
            bool_masked_pos = TubeMaskingGeneratorGPU(videos.shape[0], args.window_size, args.mask_ratio, device=args.gpu)().flatten(1).to(torch.bool)


        if args.normalize_target:
            videos_squeeze = rearrange(videos, 'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2) c', p0=2, p1=args.patch_size[0], p2=args.patch_size[1])
            videos_norm = (videos_squeeze - videos_squeeze.mean(dim=-2, keepdim=True)
                ) / (videos_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
            videos_patch = rearrange(videos_norm, 'b n p c -> b n (p c)')
        else:
            videos_patch = rearrange(videos, 'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2 c)', p0=2, p1=args.patch_size[0], p2=args.patch_size[1])

        B, _, C = videos_patch.shape
        targets = videos_patch[bool_masked_pos].reshape(B, -1, C)

        videos = normalize(videos)

        optimizer.zero_grad()

        tic = time.time()
        # compute output
        with amp.autocast(enabled=not args.disable_amp):
            
            if args.model == 'FRILSCross_VITB32':
                outputs, embedded_patches, mapped_embedded_patches, pred_features, \
                _, mapped_masked_pred_features, logit_scale = model(videos, bool_masked_pos, ids_restore, ids_keep)

                with torch.no_grad():
                    _, _, _, _, mapped_masked_embedded_patches, _, _ = teacher_model(videos, bool_masked_pos, ids_restore, ids_keep)
                    mapped_masked_embedded_patches = F.layer_norm(mapped_masked_embedded_patches, (mapped_masked_embedded_patches.size(-1),))
            else:
                outputs, embedded_patches, mapped_embedded_patches, pred_features, \
                _, mapped_masked_pred_features, logit_scale = model(videos, bool_masked_pos)

                with torch.no_grad():
                    _, _, _, _, mapped_masked_embedded_patches, _, _ = teacher_model(videos, bool_masked_pos)
                    mapped_masked_embedded_patches = F.layer_norm(mapped_masked_embedded_patches, (mapped_masked_embedded_patches.size(-1),))  # normalize over feature-dim



            loss_MSE = MSE_criterion(outputs, target=targets)
            

            ####################
            if args.CLIP_strategy == 'patch' or args.CLIP_strategy == 'patch-average':
                patch_wise_clip_loss = 0
                # repeat the motion_patch_yabs for 8 times
                motion_patch_yabs = motion_patch_yab.repeat(1,8)
                patch_wise_clip_acc_list = []
                for i in range (0, args.patch_iter):
                    # find one element indexes in motion_patch_yabs
                    random_index = []
                    vid_embed = []
                    x, y = torch.where(motion_patch_yabs==1)
                    for j in range(B):
                        x_loc = torch.where(x==j)[0].numpy()
                        # shuffle list x_loc
                        np.random.shuffle(x_loc)
                        # randomly select one element from the list
                        if len(x_loc) > 0:
                            if args.CLIP_strategy == 'patch':
                                random_index.append([j, y[x_loc[0]]])
                            elif args.CLIP_strategy == 'patch-average':
                                random_index.append([ [j] + y[x_loc].tolist()])
                                vid_embed.append(mapped_embedded_patches[j, y[x_loc], :].mean(dim=0))
                        else:
                            if args.CLIP_strategy == 'patch':
                                random_index.append([j, np.random.randint(0, 1536)])
                            elif args.CLIP_strategy == 'patch-average':
                                random_index.append([ [j] + np.random.randint(0, 1536, size=8).tolist() ])
                                vid_embed.append(mapped_embedded_patches[j, np.array(list(range(1536))), :].mean(dim=0))

                    if args.CLIP_strategy == 'patch':
                        random_index_patch = torch.tensor(random_index)

                    # get the random index for each video
                    if args.CLIP_strategy == 'patch':
                        video_embed = mapped_embedded_patches[random_index_patch[:,0], random_index_patch[:,1], :] # for patch-wise
                    elif args.CLIP_strategy == 'patch-average':
                        video_embed = torch.stack(vid_embed, dim=0)
                    # video_embed = embedded_patches.mean(dim=1) # for average patch

                    ############################ inside bbox average
                    # # randomly select one element from the list
                    #     if len(x_loc) > 0:
                    #         vid_embed.append(mapped_embedded_patch[j, y[x_loc], :].mean(dim=0))
                    #     else:
                    #         vid_embed.append(mapped_embedded_patch[j, np.array(list(range(1536))), :].mean(dim=0))

                    # # get the random index for each video
                    # video_embed = torch.stack(vid_embed, dim=0)
                    ############################

                    clip_loss = Clip_criterion(video_embed, text_embed, logit_scale)
                    patch_wise_clip_loss = patch_wise_clip_loss + clip_loss['loss']
                    patch_wise_clip_acc_list.append(clip_loss['clip_acc'])
                ####################
                
                patch_wise_clip_loss = patch_wise_clip_loss / args.patch_iter
            elif args.CLIP_strategy == 'average':
                video_embed = mapped_embedded_patches.mean(dim=1)
                clip_loss = Clip_criterion(video_embed, text_embed, logit_scale)
                patch_wise_clip_loss = clip_loss['loss']
                patch_wise_clip_acc_list = [clip_loss['clip_acc']]

            FR_loss = FR_criterion(mapped_masked_embedded_patches, mapped_masked_pred_features)['loss']
                
            loss = loss_MSE * args.MSE_scale + patch_wise_clip_loss * args.CLIP_scale + FR_loss * args.FR_scale
            loss /= args.update_freq

        check_loss_nan(loss)
        scaler.scale(loss).backward()

        if (data_iter + 1) % args.update_freq != 0:
            continue

        # # get grad norm
        # grad_norm = get_grad_norm_(model.parameters() if not args.disable_amp else model.module.parameters(), norm_type=2)
        # # if it's torch(inf) replace it with 0 otherwise convert it to float
        # if torch.isinf(grad_norm):
        #     grad_norm = torch.tensor(0.)

        # compute gradient and do SGD step
        if args.grad_clip_norm is not None:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)

            if torch.isinf(grad_norm) or torch.isnan(grad_norm):
                grad_norm = torch.tensor(0.)

        scaler.step(optimizer)
        scaler.update()
        model.zero_grad(set_to_none=True)

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            if args.distributed:
                model.module.logit_scale.clamp_(0, math.log(100))
            else:
                model.logit_scale.clamp_(0, math.log(100))

        # get loss scale
        loss_scale_value = scaler.state_dict()["scale"]

        # torch.cuda.empty_cache()
        model_time.update(time.time() - tic)

        metrics['loss'].update(loss.item(), args.batch_size)
        metrics['mse_loss'].update(loss_MSE.item(), args.batch_size)
        metrics['clip_loss'].update(patch_wise_clip_loss.item(), args.batch_size)
        metrics['fr_loss'].update(FR_loss.item(), args.batch_size)
        metrics['total_loss'].update(loss.item(), args.batch_size)
        metrics['clip_acc'].update((sum(patch_wise_clip_acc_list)/len(patch_wise_clip_acc_list)).item(), args.batch_size)
        if args.grad_clip_norm is not None:
            metrics['grad_norm'].update(grad_norm.item(), args.batch_size)
        metrics['loss_scale'].update(loss_scale_value, args.batch_size)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        mem.update(torch.cuda.max_memory_allocated() // 1e9)

        if optim_iter % args.print_freq == 0:
            if args.verbose:
                print('Time to print: {}'.format(datetime.datetime.now()))
            progress.display(optim_iter)
    progress.synchronize()
    return {**{k: v.avg for k, v in metrics.items()},
            'lr': optimizer.param_groups[0]['lr']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser('AVION training and evaluation', parents=[get_args_parser()])
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
    wandb.finish()
