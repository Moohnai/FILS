import csv
import math
import os
import sys
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from torch import inf, Tensor
from torch.utils._foreach_utils import _group_tensors_by_device_and_dtype, _has_foreach_support


def check_loss_nan(loss):
    if not math.isfinite(loss.item()):
        print("Loss is {}, stopping training".format(loss.item()))
        sys.exit(1)


def interpolate_pos_embed(old_pos_embed, model, num_frames):
    embedding_size = old_pos_embed.shape[-1] # channel dim
    num_patches = model.patch_embed.num_patches #
    num_extra_tokens = model.pos_embed.shape[-2] - num_patches # 0/1

    # height (== width) for the checkpoint position embedding
    orig_size = int(((old_pos_embed.shape[-2] - num_extra_tokens)//(num_frames // model.patch_embed.tubelet_size)) ** 0.5)
    # height (== width) for the new position embedding
    new_size = int((num_patches // (num_frames // model.patch_embed.tubelet_size) )** 0.5)
    # class_token and dist_token are kept unchanged
    if orig_size != new_size:
        print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
        extra_tokens = old_pos_embed[:, :num_extra_tokens]
        # only the position tokens are interpolated
        pos_tokens = old_pos_embed[:, num_extra_tokens:]
        # B, L, C -> BT, H, W, C -> BT, C, H, W
        pos_tokens = pos_tokens.reshape(-1, num_frames // model.patch_embed.tubelet_size, orig_size, orig_size, embedding_size)
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
        # BT, C, H, W -> BT, H, W, C ->  B, T, H, W, C
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(-1, num_frames // model.patch_embed.tubelet_size, new_size, new_size, embedding_size)
        pos_tokens = pos_tokens.flatten(1, 3) # B, L, C
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        return new_pos_embed
    else:
        print('Skipping interpolation')
        return old_pos_embed


def generate_label_map(dataset, root=''):
    if dataset == 'ek100_cls':
        print("Preprocess ek100 action label space")
        vn_list = []
        mapping_vn2narration = {}
        for f in [
            os.path.join(root, 'datasets/EK100/epic-kitchens-100-annotations/EPIC_100_train.csv'),
            os.path.join(root, 'datasets/EK100/epic-kitchens-100-annotations/EPIC_100_validation.csv'),
        ]:
            csv_reader = csv.reader(open(f))
            _ = next(csv_reader)  # skip the header
            for row in csv_reader:
                vn = '{}:{}'.format(int(row[10]), int(row[12]))
                narration = row[8]
                if vn not in vn_list:
                    vn_list.append(vn)
                if vn not in mapping_vn2narration:
                    mapping_vn2narration[vn] = [narration]
                else:
                    mapping_vn2narration[vn].append(narration)
                # mapping_vn2narration[vn] = [narration]
        vn_list = sorted(vn_list)
        print('# of action= {}'.format(len(vn_list)))
        mapping_vn2act = {vn: i for i, vn in enumerate(vn_list)}
        labels = [list(set(mapping_vn2narration[vn_list[i]])) for i in range(len(mapping_vn2act))]
        print(labels[:5])
    elif dataset == 'charades_ego':
        print("=> preprocessing charades_ego action label space")
        vn_list = []
        labels = []
        with open(os.path.join(root, 'datasets/CharadesEgo/CharadesEgo/Charades_v1_classes.txt')) as f:
            csv_reader = csv.reader(f)
            for row in csv_reader:
                vn = row[0][:4]
                vn_list.append(vn)
                narration = row[0][5:]
                labels.append(narration)
        mapping_vn2act = {vn: i for i, vn in enumerate(vn_list)}
        print(labels[:5])
    elif dataset.lower() == 'egtea':
        print("=> preprocessing egtea action label space")
        labels = []
        with open(os.path.join(root, 'datasets/EGTEA/action_idx.txt')) as f:
            for row in f:
                row = row.strip()
                narration = ' '.join(row.split(' ')[:-1])
                labels.append(narration.replace('_', ' ').lower())
                # labels.append(narration)
        mapping_vn2act = {label: i for i, label in enumerate(labels)}
        print(len(labels), labels[:5])
    else:
        raise NotImplementedError
    return labels, mapping_vn2act

def get_grad_norm_(parameters, norm_type: float = 2.0, foreach: Optional[bool] = None) -> torch.Tensor:
    # if isinstance(parameters, torch.Tensor):
    #     parameters = [parameters]
    # parameters = [p for p in parameters if p.grad is not None]
    # norm_type = float(norm_type)
    # if len(parameters) == 0:
    #     return torch.tensor(0.)
    # device = parameters[0].grad.device
    # if norm_type == inf:
    #     total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    # else:
    #     total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    # return total_norm

    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    grads = [p.grad for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(grads) == 0:
        return torch.tensor(0.)
    first_device = grads[0].device
    grouped_grads: Dict[Tuple[torch.device, torch.dtype], List[List[Tensor]]] \
        = _group_tensors_by_device_and_dtype([[g.detach() for g in grads]])  # type: ignore[assignment]

    if norm_type == inf:
        norms = [torch.linalg.vector_norm(g.detach(), inf).to(first_device) for g in grads]
        total_norm = norms[0] if len(norms) == 1 else torch.max(torch.stack(norms))
    else:
        norms = []
        for ((device, _), ([grads], _)) in grouped_grads.items():  # type: ignore[assignment]
            if (foreach is None or foreach) and _has_foreach_support(grads, device=device):
                norms.extend(torch._foreach_norm(grads, norm_type))
            elif foreach:
                raise RuntimeError(f'foreach=True was passed, but can\'t use the foreach API on {device.type} tensors')
            else:
                norms.extend([torch.linalg.vector_norm(g, norm_type) for g in grads])

        total_norm = torch.linalg.vector_norm(torch.stack([norm.to(first_device) for norm in norms]), norm_type)

    return total_norm
