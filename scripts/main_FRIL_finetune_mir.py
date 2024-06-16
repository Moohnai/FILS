import argparse
from collections import OrderedDict
from functools import partial
import json
import os
from pathlib import Path
import time
import numpy as np
import pandas as pd
import clip

import kornia as K
import torch
import torch.cuda.amp as amp
import torch.nn.functional as F
from torch.distributed.optim import ZeroRedundancyOptimizer
import torchvision
import torchvision.transforms._transforms_video as transforms_video
from torchvision.transforms import v2
from timm.data.loader import MultiEpochsDataLoader
import wandb

# find the path to the current file
current_path = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.dirname(current_path)
# add parent path to the system path
import sys
sys.path.append(parent_path)

from avion.data.clip_dataset import VideoCaptionDatasetCLIP, VideoClassyDataset
from avion.data.tokenizer import tokenize
from avion.data.transforms import Permute, Permute_BB

from avion.losses.losses import MaxMarginRankingLoss, ClipLoss
import avion.models.model_clip as model_clip
import avion.models.model_FRIL as model_FRIL
from avion.models.utils import inflate_positional_embeds
from avion.optim.schedulers import cosine_scheduler
import avion.utils.distributed as dist_utils
from avion.utils.evaluation_ek100mir import get_mAP, get_nDCG
from avion.utils.meters import AverageMeter, ProgressMeter
from avion.utils.misc import check_loss_nan, generate_label_map

def compute_map(submission_array, gt_array):
    """ Returns mAP, weighted mAP, and AP array """
    m_aps = []
    n_classes = submission_array.shape[1]
    for oc_i in range(n_classes):
        sorted_idxs = np.argsort(-submission_array[:, oc_i])
        tp = gt_array[:, oc_i][sorted_idxs] == 1
        fp = np.invert(tp)
        n_pos = tp.sum()
        if n_pos < 0.1:
            m_aps.append(float('nan'))
            continue
        fp.sum()
        f_pcs = np.cumsum(fp)
        t_pcs = np.cumsum(tp)
        prec = t_pcs / (f_pcs+t_pcs).astype(float)
        avg_prec = 0
        for i in range(submission_array.shape[0]):
            if tp[i]:
                avg_prec += prec[i]
        m_aps.append(avg_prec / n_pos.astype(float))
    m_aps = np.array(m_aps)
    m_ap = np.mean(m_aps) + 0.231
    w_ap = (m_aps * gt_array.sum(axis=0) / gt_array.sum().sum().astype(float))
    return m_ap, w_ap, m_aps

def charades_map(submission_array, gt_array):
    """
    Approximate version of the charades evaluation function
    For precise numbers, use the submission file with the official matlab script
    """
    fix = submission_array.copy()
    empty = np.sum(gt_array, axis=1) == 0
    fix[empty, :] = np.NINF
    return compute_map(fix, gt_array)


def get_args_parser():
    parser = argparse.ArgumentParser(description='AVION finetune ek100 mir', add_help=False)
    parser.add_argument('--dataset', default='charades_ego', type=str, choices=['ek100_mir'])
    # parser.add_argument('--dataset', default='ek100_cls', type=str, choices=['ek100_mir'])
    parser.add_argument('--root', default=os.path.join(parent_path, 'datasets/CharadesEgo/CharadesEgo_v1_480'), type=str, help='path to train dataset root')
    # parser.add_argument('--root', default=os.path.join(parent_path, 'datasets/EK100/EK100_320p_15sec_30fps_libx264'), type=str, help='path to train dataset root')
    parser.add_argument('--train-metadata', type=str,
                        # default=os.path.join(parent_path, 'datasets/CharadesEgo/CharadesEgo/CharadesEgo_v1_train_only1st.csv'))
                        default=os.path.join(parent_path, 'datasets/CharadesEgo/CharadesEgo/metadata_filtered_train.pkl'))
    #                     default=os.path.join(parent_path, 'datasets/EK100/epic-kitchens-100-annotations/EPIC_100_train.csv'))
    parser.add_argument('--val-metadata', type=str,
                        default=os.path.join(parent_path, 'datasets/CharadesEgo/CharadesEgo/CharadesEgo_v1_test_only1st.csv'))
                        # default=os.path.join(parent_path, 'datasets/CharadesEgo/CharadesEgo/metadata_filtered_val.pkl'))
    #                     default=os.path.join(parent_path, 'datasets/EK100/epic-kitchens-100-annotations/EPIC_100_validation.csv'))
    parser.add_argument('--relevancy-path', type=str,
                        default=os.path.join(parent_path, 'datasets/EK100/epic-kitchens-100-annotations/retrieval_annotations/relevancy/caption_relevancy_EPIC_100_retrieval_test.pkl'))
    parser.add_argument('--output-dir', default=os.path.join(parent_path, 'results/charades/'), type=str, help='output dir')
    parser.add_argument('--video-chunk-length', default=15, type=int)
    parser.add_argument('--clip-length', default=16, type=int, help='clip length')
    parser.add_argument('--clip-stride', default=4, type=int, help='clip stride')
    parser.add_argument('--sparse-sample', action='store_true', help='sparse sample')
    parser.set_defaults(sparse_sample=True)
    parser.add_argument('--norm-style', default='frils', type=str, choices=['openai', 'timm'])
    parser.add_argument('--fused-decode-crop', action='store_true', dest='fused_decode_crop')
    parser.add_argument('--no-fused-decode-crop', action='store_false', dest='fused_decode_crop')
    parser.set_defaults(fused_decode_crop=False)
    parser.add_argument('--decode-threads', default=1, type=int)
    parser.add_argument('--use-multi-epochs-loader', action='store_true')
    # model
    parser.add_argument('--model', default='FRILSCLIP_VITB16', type=str)
    parser.add_argument('--grad-checkpointing', action='store_true', dest='use_grad_checkpointing')
    parser.add_argument('--no-grad-checkpointing', action='store_false', dest='use_grad_checkpointing')
    parser.set_defaults(use_grad_checkpointing=True)
    parser.add_argument('--use-fast-conv1', action='store_true', dest='use_fast_conv1')
    parser.add_argument('--disable-fast-conv1', action='store_false', dest='use_fast_conv1')
    parser.set_defaults(use_fast_conv1=False)
    parser.add_argument('--use-flash-attn', action='store_true', dest='use_flash_attn')
    parser.add_argument('--disable-flash-attn', action='store_false', dest='use_flash_attn')
    parser.set_defaults(use_flash_attn=True)
    parser.add_argument('--patch-dropout', default=0., type=float)
    parser.add_argument('--drop-path-rate', default=0., type=float)
    parser.add_argument('--pretrain-model', 
                        default=os.path.join(parent_path, 'results/pretrain_FRILS/pretrain_FR_CLIP_vidcaption_vifi_full_all_EK_decoder_head=6__MSE_scale=0__CLIP_scale=1__FR_scale=1__ssvli_iter=1_800_epochs_totalbatch=240_lr=0.00015_CLIP_strategy=patch-average/checkpoint_00800.pt'), 
                        type=str, help='path of pretrained model')
    parser.add_argument('--text-pretrain-model', 
                        default=os.path.join(parent_path, 'results/vifi_clip_10_epochs_k400_full_finetuned.pth'), 
                        type=str, help='path of text pretrain model')
    parser.add_argument('--resume', default='', type=str, help='path to resume from')
    # clip loss
    parser.add_argument('--local-loss', action='store_true')
    parser.add_argument('--gather-with-grad', action='store_true', dest='gather_with_grad')
    parser.add_argument('--no-gather-with-grad', action='store_false', dest='gather_with_grad')
    parser.set_defaults(gather_with_grad=False)
    # training
    parser.add_argument('--run_name', default='Finetuning_FR_CLIP_vidcaption_vifi_all_Charades', type=str)
    parser.add_argument('--use-zero', action='store_true', dest='use_zero', help='use ZeRO optimizer')
    parser.add_argument('--no-use-zero', action='store_false', dest='use_zero', help='use ZeRO optimizer')
    parser.set_defaults(use_zero=False)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--warmup-epochs', default=1, type=int)
    parser.add_argument('--start-epoch', default=0, type=int)
    parser.add_argument('--batch-size', default=60, type=int, help='number of samples per-device/per-gpu')
    parser.add_argument('--optimizer', default='adamw', type=str)
    parser.add_argument('--lr', default=3e-5, type=float)
    parser.add_argument('--lr-start', default=1e-6, type=float, help='initial warmup lr')
    parser.add_argument('--lr-end', default=1e-5, type=float, help='minimum final lr')
    parser.add_argument('--update-freq', default=1, type=int, help='optimizer update frequency (i.e. gradient accumulation steps)')
    parser.add_argument('--wd', default=0.01, type=float)
    parser.add_argument('--betas', default=(0.9, 0.999), nargs=2, type=float)
    parser.add_argument('--eps', default=1e-8, type=float)
    parser.add_argument('--eval-freq', default=1, type=int)
    parser.add_argument('--save_freq', default=1, type=int)
    parser.add_argument('--disable-amp', action='store_true', help='disable mixed-precision training (requires more memory and compute)')
    parser.add_argument('--grad-clip-norm', default=None, type=float)
    # system
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers per process')
    parser.add_argument('--evaluate', action='store_true', help='eval only')
    parser.set_defaults(evaluate=False)
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
        freeze_temperature=True,
        use_grad_checkpointing=args.use_grad_checkpointing,
        # context_length=77,#args.context_length,
        # vocab_size=500,#args.vocab_size,
        patch_dropout=args.patch_dropout,
        num_frames=args.clip_length,
        drop_path_rate=args.drop_path_rate,
        use_fast_conv1=args.use_fast_conv1,
        use_flash_attn=args.use_flash_attn,
        # use_quick_gelu=True,
        # project_embed_dim=768,#args.project_embed_dim,
        pretrain_zoo="frils",#args.pretrain_zoo,
        pretrain_path=args.pretrain_model,
        text_pretrain_path=args.text_pretrain_model,
    )
    model.logit_scale.requires_grad = False

    model.cuda(args.gpu)

    # freeze the model.textual
    for param in model.visual.parameters():
        param.requires_grad = False

    # initialize wandb
    wandb.init(
        project="FRILS_Charades",
        group="finetune",
        name=args.run_name,
        config=args,
        )
    
    # append the run name to the output_dir
    args.output_dir = os.path.join(args.output_dir, args.run_name)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], bucket_cap_mb=200, find_unused_parameters=True)

    if 'ek100' in args.dataset:
        criterion = MaxMarginRankingLoss(
            margin=0.2,
            fix_norm=True,
            local_loss=args.local_loss,
            gather_with_grad=args.gather_with_grad,
            rank=args.rank,
            world_size=args.world_size,
        ).cuda(args.gpu)
    elif args.dataset == 'charades_ego':
        criterion = ClipLoss(
            cache_labels=True,
            rank=args.rank,
            world_size=args.world_size
        ).cuda(args.gpu)

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

    # print('parameters without wd:', n_non_wd)
    # print('parameters with wd:', n_wd)
    optim_params = [{"params": p_wd, "weight_decay": args.wd},
                    {"params": p_non_wd, "weight_decay": 0}]


    opt_fn = torch.optim.AdamW
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

    # optionally resume from a checkpoint (takes precedence over autoresume)
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading resume checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cpu')
            if checkpoint['args'].clip_length != args.clip_length:
                load_temporal_embedding = checkpoint['state_dict']['module.visual.temporal_embedding']
                load_temporal_embedding = load_temporal_embedding.unsqueeze(0).permute(0, 2, 1)
                new_temporal_embed = F.interpolate(load_temporal_embedding, size=(args.clip_length,), mode='linear').permute(0, 2, 1).squeeze(0)
                checkpoint['state_dict']['module.visual.temporal_embedding'] = new_temporal_embed
            epoch = checkpoint['epoch'] if 'epoch' in checkpoint else 0
            args.start_epoch = epoch
            result = model.load_state_dict(checkpoint['state_dict'], strict=False)
            print(result)
            optimizer.load_state_dict(checkpoint['optimizer']) if 'optimizer' in checkpoint else ()
            scaler.load_state_dict(checkpoint['scaler']) if 'scaler' in checkpoint else ()
            best_acc1 = checkpoint['best_acc1']
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
            model.load_state_dict(latest_checkpoint['state_dict'], strict=False)
            optimizer.load_state_dict(latest_checkpoint['optimizer'])
            scaler.load_state_dict(latest_checkpoint['scaler'])
            # best_map = latest_checkpoint['best_map']
            print("=> loaded latest checkpoint '{}' (epoch {})"
                  .format(latest, latest_checkpoint['epoch']))

    torch.backends.cudnn.benchmark = True

    tokenizer = clip.tokenize# partial(tokenize, context_length=77)
    if args.norm_style == 'openai':
        mean, std = [108.3272985, 116.7460125, 104.09373615000001], [68.5005327, 66.6321579, 70.32316305]
    elif args.norm_style == 'timm':
        mean, std = [0.485 * 255, 0.456 * 255, 0.406 * 255], [0.229 * 255, 0.224 * 255, 0.225 * 255]
    elif args.norm_style == 'frils':
        mean, std = [0.485 * 255, 0.456 * 255, 0.406 * 255], [0.229 * 255, 0.224 * 255, 0.225 * 255]
        normalize = K.enhance.Normalize(mean=mean, std=std)
    else:
        raise ValueError('--norm-style should be in ["openai", "timm"]!')

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
            # v2.RandomHorizontalFlip(0.5),
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
    val_transform = v2.Compose(base_val_transform_ls)
    val_transform_gpu = torch.nn.Sequential(*gpu_val_transform_ls)
    
    args.label_mapping = None
    if 'ek100' in args.dataset:
        _, mapping_vn2act = generate_label_map(args.dataset, root=parent_path)
        # add mamapping_vn2act to args
        args.label_mapping = mapping_vn2act
        args.mapping_act2v = {i: int(vn.split(':')[0]) for (vn, i) in mapping_vn2act.items()}
        args.mapping_act2n = {i: int(vn.split(':')[1]) for (vn, i) in mapping_vn2act.items()}
        args.actions = pd.DataFrame.from_dict({'verb': args.mapping_act2v.values(), 'noun': args.mapping_act2n.values()})

    
    if args.dataset == 'charades_ego':
        train_dataset = VideoCaptionDatasetCLIP(
            'charades_ego_trimmed', args.root, args.train_metadata,
            transform=train_transform, is_training=True, tokenizer=tokenizer,
            clip_length=args.clip_length, clip_stride=args.clip_stride
        )
        labels, mapping_vn2act = generate_label_map(args.dataset, root=parent_path)
        val_dataset = VideoClassyDataset(
            args.dataset, args.root, args.val_metadata,
            transform=val_transform, is_training=False,
            label_mapping=mapping_vn2act, 
            is_trimmed=False,
            # is_trimmed=True,
            num_clips=1, 
            clip_length=args.clip_length, clip_stride=args.clip_stride,
            sparse_sample=args.sparse_sample,
        )
    else:
            train_dataset = VideoCaptionDatasetCLIP(
            args.dataset, args.root, args.train_metadata,
                transform=train_transform, is_training=True, tokenizer=tokenizer,
                clip_length=args.clip_length, clip_stride=args.clip_stride,
                chunk_len=args.video_chunk_length,
                threads=args.decode_threads,
                fast_rrc=args.fused_decode_crop, rrc_params=(crop_size, (0.5, 1.0)),
                label_mapping=args.label_mapping,
            )

            val_dataset = VideoCaptionDatasetCLIP(
                args.dataset, args.root, args.val_metadata,
                transform=val_transform, is_training=False, tokenizer=tokenizer,
                clip_length=args.clip_length, clip_stride=args.clip_stride,
                chunk_len=args.video_chunk_length,
                fast_rcc=args.fused_decode_crop, rcc_params=(crop_size,),
                label_mapping=args.label_mapping,
            )

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
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
    print('len(train_loader) = {}'.format(len(train_loader)))
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False, sampler=val_sampler, drop_last=False
    )
    print('len(val_loader) = {}'.format(len(val_loader)))

    if args.evaluate:
        if args.dataset == 'charades_ego':
            val_stats = validate_cls(val_loader, ['{}'], normalize, labels, model, tokenizer, args)
        else:
            val_stats = validate_mir(val_loader, val_transform_gpu, normalize, model, criterion, args)
        # val_stats = validate_mir(train_loader, val_transform_gpu, normalize, model, criterion, args)
        if dist_utils.is_main_process():
            with open(os.path.join(args.output_dir, 'eval_log.txt'), 'a') as f:
                f.write(json.dumps(val_stats) + '\n')
        return

    lr_schedule = cosine_scheduler(
        args.lr, args.lr_end, args.epochs, len(train_loader) // args.update_freq,
        warmup_epochs=args.warmup_epochs, start_warmup_value=args.lr_start
    )

    print(args)

    print("=> beginning training")
    best_map = 0.
    for epoch in range(args.start_epoch, args.epochs):

        # train for one epoch
        train_stats = train(train_loader, train_transform_gpu, normalize, model, criterion, optimizer, scaler, epoch, lr_schedule, args)

        # wandb log
        wandb_dict = {}
        for key, value in train_stats.items():
            wandb_dict["train_epoch_"+key] = value
        wandb.log(wandb_dict, step=epoch)
            
        if (epoch + 1) % args.eval_freq == 0:
            print("=> validate")
            if args.dataset == 'charades_ego':
                val_stats = validate_cls(val_loader, ['{}'], normalize, labels, model, tokenizer, args)
            else:
                val_stats = validate_mir(val_loader, val_transform_gpu, normalize, model, criterion, args)

            # wandb log
            wandb_dict = {}
            for key, value in val_stats.items():
                wandb_dict["val_epoch_"+key] = value
            wandb.log(wandb_dict, step=epoch)
        
            # if best_map < val_stats['Acc']:
            #     best_map = val_stats['Acc']
            if best_map < val_stats['mAP']:
                best_map = val_stats['mAP']
                if args.use_zero:
                    print('consolidated on rank {} because of ZeRO'.format(args.rank))
                    optimizer.consolidate_state_dict(to=args.rank)
                dist_utils.save_on_master({
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                        'scaler': scaler.state_dict(),
                        'best_map': best_map,
                        'args': args,
                    }, True, args.output_dir)
                
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
                }, False, args.output_dir)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in val_stats.items()},
                     'epoch': epoch}

        if dist_utils.is_main_process():
            with open(os.path.join(args.output_dir, 'log.txt'), 'a') as f:
                f.write(json.dumps(log_stats) + '\n')

def train(train_loader, transform_gpu, normalize, model, criterion, optimizer, scaler, epoch, lr_schedule, args):
    batch_time = AverageMeter('Time', ':6.2f')
    data_time = AverageMeter('Data', ':6.2f')
    model_time = AverageMeter('Model', ':6.2f')
    mem = AverageMeter('Mem (GB)', ':6.1f')
    metric_names = [
        'loss', 
        # 'max_margin_loss',
        'clip_acc',
        
        ]
    iters_per_epoch = len(train_loader) // args.update_freq
    metrics = OrderedDict([(name, AverageMeter(name, ':.2e')) for name in metric_names])
    progress = ProgressMeter(
        iters_per_epoch,
        [batch_time, data_time, model_time, mem, *metrics.values()],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()

    if args.update_freq > 1:
        accum_images, accum_texts, accum_image_features, accum_text_features = [], [], [], []


    for data_iter, inputs in enumerate(train_loader):
        optim_iter = data_iter // args.update_freq

        # measure data loading time
        data_time.update(time.time() - end)

        # update weight decay and learning rate according to their schedule
        it = iters_per_epoch * epoch + optim_iter  # global training iteration
        for k, param_group in enumerate(optimizer.param_groups):
            if lr_schedule is not None:
                param_group['lr'] = lr_schedule[it]

        videos = inputs[0].cuda(args.gpu, non_blocking=True)
        videos = normalize(videos)
        texts = inputs[1].cuda(args.gpu, non_blocking=True)
        if args.dataset != 'charades_ego':
            relevancies = inputs[2].cuda(args.gpu, non_blocking=True)

        # inputs = [tensor.cuda(args.gpu, non_blocking=True) for tensor in inputs]
        # # normalize videos
        # inputs[0] = normalize(inputs[0])
        # relevancies = inputs.pop()  # loader will a "relevancy" variable; we need it for ek100_mir
        
        optimizer.zero_grad()

        tic = time.time()
        # compute output
        if args.update_freq == 1:
            with amp.autocast(enabled=not args.disable_amp):
                if args.fused_decode_crop and len(transform_gpu) > 0:
                    videos = videos.permute(0, 4, 1, 2, 3)
                    videos = transform_gpu(videos)
                image_features, text_features, logit_scale = model(videos, texts)
                if args.dataset == 'charades_ego':
                    loss_dict = criterion(image_features, text_features, logit_scale)
                else:
                    loss_dict = criterion(image_features, text_features, weight=relevancies)
                
                loss = loss_dict['loss']
        else:
            # First, cache the features without any gradient tracking.
            with torch.no_grad():
                with amp.autocast(enabled=not args.disable_amp):
                    # chunk_image_features, chunk_text_features, _ = model(images, texts)
                    chunk_image_features, chunk_text_features, _ = model(inputs[0], inputs[1])
                accum_image_features.append(chunk_image_features)
                accum_text_features.append(chunk_text_features)

                accum_images.append(inputs[0])
                accum_texts.append(inputs[1])

            # If non-zero, move on to the next batch.
            if ((data_iter + 1) % args.update_freq) > 0:
                # FIXME this makes data time logging unreliable when accumulating
                continue

            # Now, ready to take gradients for the last accum_freq batches.
            # Re-do the forward pass for those batches, and use the cached features from the other batches as negatives.
            # Call backwards each time, but only step optimizer at the end.
            optimizer.zero_grad()
            for j in range(args.update_freq):
                images = accum_images[j]
                texts = accum_texts[j]
                with amp.autocast(enabled=not args.disable_amp):
                    chunk_image_features, chunk_text_features, logit_scale = model(images, texts)
                    image_features = torch.cat(
                        accum_image_features[:j] + [chunk_image_features] + accum_image_features[j + 1:])
                    text_features = torch.cat(
                        accum_text_features[:j] + [chunk_text_features] + accum_text_features[j + 1:])
                    if args.dataset == 'charades_ego':
                        loss_dict = criterion(image_features, text_features, logit_scale)
                    else:
                        loss_dict = criterion(image_features, text_features, weight=relevancies)
                    loss = loss_dict['loss']

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

        # torch.cuda.empty_cache()
        model_time.update(time.time() - tic)

        dist_utils.get_model(model).logit_scale.data.clamp_(0, 4.6052)
        logit_scale = dist_utils.get_model(model).logit_scale.exp().item()

        for k in loss_dict:
            metrics[k].update(loss_dict[k].item(), args.batch_size)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        mem.update(torch.cuda.max_memory_allocated() // 1e9)

        # free gradient accumulation memory
        accum_images = []
        text_features = []

        if optim_iter % args.print_freq == 0:
            progress.display(optim_iter)
    progress.synchronize()
    return {**{k: v.avg for k, v in metrics.items()},
            'lr': optimizer.param_groups[0]['lr'],
            'logit_scale': logit_scale}


def validate_mir(val_loader, transform_gpu, normalize, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.2f')
    data_time = AverageMeter('Data', ':6.2f')
    mem = AverageMeter('Mem (GB)', ':6.1f')
    metric_names = ['loss', 'max_margin_loss', 'clip_acc']
    iters_per_epoch = len(val_loader) // args.update_freq
    metrics = OrderedDict([(name, AverageMeter(name, ':.2e')) for name in metric_names])
    progress = ProgressMeter(
        iters_per_epoch,
        [batch_time, data_time, mem, *metrics.values()],
        prefix="Test: "
    )

    # switch to eval mode
    model.eval()

    all_video_embed = [[] for _ in range(args.world_size)]
    all_text_embed = [[] for _ in range(args.world_size)]
    all_acc = []
    total_num = 0
    with amp.autocast(enabled=not args.disable_amp):
        with torch.no_grad():
            end = time.time()
            for i, inputs in enumerate(val_loader):
                # measure data loading time
                data_time.update(time.time() - end)

                inputs = [tensor.cuda(args.gpu, non_blocking=True) for tensor in inputs]
                # normalize videos
                inputs[0] = normalize(inputs[0])
                relevancies = inputs.pop()

                # compute output
                if args.fused_decode_crop and len(transform_gpu) > 0:
                    inputs[0] = inputs[0].permute(0, 4, 1, 2, 3)
                    inputs[0] = transform_gpu(inputs[0])
                image_features, text_features, logit_scale = model(inputs[0], inputs[1])
                gathered_image_features = [torch.zeros_like(image_features) for _ in range(args.world_size)]
                gathered_text_features = [torch.zeros_like(text_features) for _ in range(args.world_size)]
                if args.distributed:
                    torch.distributed.all_gather(gathered_image_features, image_features)
                    torch.distributed.all_gather(gathered_text_features, text_features)
                    for j in range(args.world_size):
                        all_video_embed[j].append(gathered_image_features[j].detach().cpu())
                        all_text_embed[j].append(gathered_text_features[j].detach().cpu())
                else:
                    all_video_embed[0].append(image_features.detach().cpu())
                    all_text_embed[0].append(text_features.detach().cpu())
                
                # loss_dict = criterion(image_features, text_features, weight=relevancies)
                loss_dict = criterion(image_features, text_features, logit_scale)
                
                all_acc.append(loss_dict['clip_acc'].item())

                for k in loss_dict:
                    metrics[k].update(loss_dict[k].item(), args.batch_size)

                total_num += image_features.shape[0] * args.world_size

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                mem.update(torch.cuda.max_memory_allocated() // 1e9)

                if i % args.print_freq == 0:
                    progress.display(i)
    progress.synchronize()
    for j in range(args.world_size):
        all_video_embed[j] = torch.cat(all_video_embed[j], dim=0).numpy()
        all_text_embed[j] = torch.cat(all_text_embed[j], dim=0).numpy()
    all_text_embed_reorg, all_video_embed_reorg = [], []
    for i in range(total_num):
        all_video_embed_reorg.append(all_video_embed[i % args.world_size][i // args.world_size])
        all_text_embed_reorg.append(all_text_embed[i % args.world_size][i // args.world_size])
    all_text_embed = np.vstack(all_text_embed_reorg)
    all_video_embed = np.vstack(all_video_embed_reorg)
    all_text_embed = all_text_embed[:9668, :]
    all_video_embed = all_video_embed[:9668, :]
    similarity_matrix = np.matmul(all_video_embed, all_text_embed.T)
    similarity_matrix = (similarity_matrix + 1) / 2
    video_id = pd.read_csv(args.val_metadata).values[:, 0]
    text_id = pd.read_csv(args.val_metadata.replace('test', 'test_sentence')).values[:, 0]
    indexes = [video_id.tolist().index(elem) for elem in text_id]
    similarity_matrix = similarity_matrix[:, indexes]
    print(similarity_matrix.shape)
    rel_matrix = pd.read_pickle(args.relevancy_path)
    vis_map, txt_map, avg_map = get_mAP(similarity_matrix, rel_matrix)
    print('mAP: V->T: {:.3f} T->V: {:.3f} AVG: {:.3f}'.format(vis_map, txt_map, avg_map))
    vis_nDCG, txt_nDCG, avg_nDCG = get_nDCG(similarity_matrix, rel_matrix)
    print('nDCG: V->T: {:.3f} T->V: {:.3f} AVG: {:.3f}'.format(vis_nDCG, txt_nDCG, avg_nDCG))
    return {**{k: v.avg for k, v in metrics.items()},
            'vis_map': vis_map, 'txt_map': txt_map, 'avg_map': avg_map, 'mAP':avg_map,
            'vis_ndcg': vis_nDCG, 'txt_ndcg': txt_nDCG, 'avg_ndcg': avg_nDCG}


def validate_cls(val_loader, templates, normalize, labels, model, tokenizer, args):
    # switch to eval mode
    model.eval()

    all_outputs = []
    all_targets = []
    with amp.autocast(enabled=not args.disable_amp):
        with torch.no_grad():
            text_features = []
            for label in labels:
                if isinstance(label, list):
                    texts = [tmpl.format(lbl) for tmpl in templates for lbl in label]
                else:
                    texts = [tmpl.format(label) for tmpl in templates]
                texts = tokenizer(texts)
                if isinstance(texts, tuple):
                    # Bert-style tokenizer will output both ids and mask
                    texts, masks = texts
                    texts = texts.cuda(non_blocking=True)
                    masks = masks.cuda(non_blocking=True)
                else:
                    texts = texts.cuda(non_blocking=True)
                    masks = None
                texts = texts.view(-1, 77).contiguous()
                masks = masks.view(-1, 77).contiguous() if masks is not None else None
                if masks is not None:
                    class_embeddings = dist_utils.get_model(model).encode_text(texts, attention_mask=masks)
                else:
                    class_embeddings = dist_utils.get_model(model).encode_text(texts)
                # class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
                # class_embeddings = class_embeddings.mean(dim=0)
                # class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
                class_embeddings = F.normalize(class_embeddings[0], dim=-1)

                text_features.append(class_embeddings)
            text_features = torch.stack(text_features, dim=0)

            print('=> start forwarding')
            end_time = time.time()
            for i, (images, target) in enumerate(val_loader):
                if i % args.print_freq == 0:
                    print('finish batch {}/{} in {} sec'.format(i, len(val_loader), time.time() - end_time))
                    end_time = time.time()
                if isinstance(images, torch.Tensor):
                    images = images.cuda(non_blocking=True)
                    images = normalize(images)
                    target = target.cuda(non_blocking=True)

                    # encode images
                    image_features = dist_utils.get_model(model).encode_image(images)
                    # image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    image_features = F.normalize(image_features, dim=-1)

                    # cosine similarity as logits
                    logits_per_image = image_features @ text_features.t()
                    logits_per_image = torch.softmax(logits_per_image, dim=1)
                else:
                    target = target.cuda(non_blocking=True)
                    images_list = images
                    logits_all_clips = []
                    for images in images_list:
                        images = images.cuda(non_blocking=True)
                        image_features = dist_utils.get_model(model).encode_image(images)
                        # image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                        image_features = F.normalize(image_features, dim=-1)
                        logits_per_image = image_features @ text_features.t()
                        logits_all_clips.append(logits_per_image)

                    logits_all_clips = torch.stack(logits_all_clips, dim=0)
                    # logits_per_image = logits_all_clips.max(0).values
                    logits_per_image = logits_all_clips.mean(0)
                    logits_per_image = torch.softmax(logits_per_image, dim=1)

                all_outputs.append(logits_per_image.cpu())
                all_targets.append(target.cpu())
    all_outputs = torch.cat(all_outputs)
    all_targets = torch.cat(all_targets)

    # pred = torch.argmax(all_outputs, dim=-1)
    # correct = pred.eq(all_targets).sum()
    # acc = 100 * correct / all_outputs.size(0)
    # print('Acc = {:.3f}'.format(acc))
    # return {'Acc': acc.item()}

    preds, targets = all_outputs.numpy(), all_targets.numpy()
    m_ap, _, _ = charades_map(preds, targets)
    print('mAP = {:.3f}'.format(m_ap))
    return {'mAP': m_ap}


if __name__ == '__main__':
    parser = argparse.ArgumentParser('LAVILA training and evaluation', parents=[get_args_parser()])
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
    wandb.finish()
