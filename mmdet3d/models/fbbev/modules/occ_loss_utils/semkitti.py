import torch
from torch.profiler import profile, ProfilerActivity
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from mmcv.runner import BaseModule, force_fp32
from torch.cuda.amp import autocast

semantic_kitti_class_frequencies = np.array(
    [
        5.41773033e09,
        1.57835390e07,
        1.25136000e05,
        1.18809000e05,
        6.46799000e05,
        8.21951000e05,
        2.62978000e05,
        2.83696000e05,
        2.04750000e05,
        6.16887030e07,
        4.50296100e06,
        4.48836500e07,
        2.26992300e06,
        5.68402180e07,
        1.57196520e07,
        1.58442623e08,
        2.06162300e06,
        3.69705220e07,
        1.15198800e06,
        3.34146000e05,
    ]
)

kitti_class_names = [
    "empty",
    "car",
    "bicycle",
    "motorcycle",
    "truck",
    "other-vehicle",
    "person",
    "bicyclist",
    "motorcyclist",
    "road",
    "parking",
    "sidewalk",
    "other-ground",
    "building",
    "fence",
    "vegetation",
    "trunk",
    "terrain",
    "pole",
    "traffic-sign",
]



def inverse_sigmoid(x, sign='A'):
    x = x.to(torch.float32)
    while x >= 1-1e-5:
        x = x - 1e-5

    while x< 1e-5:
        x = x + 1e-5

    return -torch.log((1 / x) - 1)

def KL_sep(p, target):
    """
    KL divergence on nonzeros classes
    """
    nonzeros = target != 0
    nonzero_p = p[nonzeros]
    kl_term = F.kl_div(torch.log(nonzero_p), target[nonzeros], reduction="sum")
    return kl_term


def geo_scal_loss(pred, ssc_target, ignore_index=255, non_empty_idx=0, binary=False):
    # pred = pred.to('cpu')
    # ssc_target = ssc_target.to('cpu')
    # Get softmax probabilities
    # pred = F.softmax(pred, dim=1)
    # Compute empty and nonempty probabilities
    if binary:
        nonempty_probs = pred
        empty_probs = 1 - nonempty_probs
    else:
        pred = F.softmax(pred, dim=1)
        empty_probs = pred[:, non_empty_idx]
        nonempty_probs = 1 - empty_probs

    # Remove unknown voxels
    mask = ssc_target != ignore_index
    nonempty_target = ssc_target != non_empty_idx
    nonempty_target = nonempty_target[mask].float()
    nonempty_probs = nonempty_probs[mask]
    empty_probs = empty_probs[mask]

    eps = 1e-5
    intersection = (nonempty_target * nonempty_probs).sum()
    precision = intersection / (nonempty_probs.sum()+eps)
    recall = intersection / (nonempty_target.sum()+eps)
    spec = ((1 - nonempty_target) * (empty_probs)).sum() / ((1 - nonempty_target).sum()+eps)
    output = (
            F.binary_cross_entropy_with_logits(inverse_sigmoid(precision, 'A'), torch.ones_like(precision))
            + F.binary_cross_entropy_with_logits(inverse_sigmoid(recall, 'B'), torch.ones_like(recall))
            + F.binary_cross_entropy_with_logits(inverse_sigmoid(spec, 'C'), torch.ones_like(spec))
        )

    # output = output.to('cuda')
    with autocast(False):
        return output



def sem_scal_loss(pred, ssc_target, ignore_index=255):
    # Get softmax probabilities
    with autocast(False):
        pred = F.softmax(pred, dim=1)
        # pred = pred.to('cpu')
        # ssc_target = ssc_target.to('cpu')
        loss = 0
        count = 0
        mask = ssc_target != ignore_index
        n_classes = pred.shape[1]
        begin = 1 if n_classes == 18 else 0
        for i in range(begin, n_classes):   
            # Get probability of class i
            p = pred[:, i]  

            # Remove unknown voxels
            target_ori = ssc_target
            p = p[mask]
            target = ssc_target[mask]   

            completion_target = torch.ones_like(target)
            completion_target[target != i] = 0
            completion_target_ori = torch.ones_like(target_ori).float()
            completion_target_ori[target_ori != i] = 0
            completion_target_sum = torch.sum(completion_target)

            if completion_target_sum.item() > 0:
                count += 1.0
                nominator = torch.sum(p * completion_target)
                loss_class = 0
                
                p_sum = torch.sum(p)
                if p_sum.item() > 0:
                    precision = nominator / (p_sum + 1e-5)
                    loss_precision = F.binary_cross_entropy_with_logits(
                            inverse_sigmoid(precision, 'D'), torch.ones_like(precision)
                        )
                    loss_class += loss_precision

                if completion_target_sum.item() > 0:
                    recall = nominator / (completion_target_sum +1e-5)
                    # loss_recall = F.binary_cross_entropy(recall, torch.ones_like(recall))

                    loss_recall = F.binary_cross_entropy_with_logits(inverse_sigmoid(recall, 'E'), torch.ones_like(recall))
                    loss_class += loss_recall

                reverse_sum = torch.sum(1 - completion_target)
                if reverse_sum.item() > 0:
                    specificity = torch.sum((1 - p) * (1 - completion_target)) / (
                        reverse_sum +  1e-5
                    )

                    loss_specificity = F.binary_cross_entropy_with_logits(
                            inverse_sigmoid(specificity, 'F'), torch.ones_like(specificity)
                        )
                    loss_class += loss_specificity
                loss += loss_class
                # print(i, loss_class, loss_recall, loss_specificity)
        l = loss/count
        # l = l.to('cuda')
        if torch.isnan(l):
            from IPython import embed
            embed()
            exit()
        return l


def CE_ssc_loss(pred, target, class_weights=None, ignore_index=255):
    """
    :param: prediction: the predicted tensor, must be [BS, C, ...]
    """

    criterion = nn.CrossEntropyLoss(
        weight=class_weights, ignore_index=ignore_index, reduction="mean"
    )
    # from IPython import embed
    # embed()
    # exit()
    with autocast(False):
        loss = criterion(pred, target.long())

    return loss

def BCE_ssc_loss(pred, target, class_weights=None, ignore_index=255):
    """
    :param: prediction: the predicted tensor, must be [BS, C, ...]
    """

    # Target: bs, D, W, H
    gt = target
    gt[gt != 0] = 1
    criterion = nn.BCELoss()

    with autocast(False):
        loss = criterion(pred.unsqueeze(1), gt.unsqueeze(1).float())

    return loss

def vel_loss(pred, gt):
    with autocast(False):
        return F.l1_loss(pred, gt)


def cos_sim_loss(pred):
    cos_sim_list = []
    overlap1, overlap2 = pred
    
    for i, (feat1, feat2) in enumerate(zip(overlap1, overlap2)):
        cos_sim = F.cosine_similarity(feat1, feat2, dim=-1)
        loss = (1 - cos_sim).mean()
        cos_sim_list.append(loss)

    total_loss = torch.stack(cos_sim_list).mean()
    return total_loss

    # pred: Ncam, bs, N, C
    # idx: [bs][Ncam][N][3]
    # camera_pairs = [(0, 1), (1, 2), (2, 3), (3, 4), (5, 0)]
    # pair1 = [0, 2, 4]
    # pair2 = [1, 3, 5]
    # total_loss = 0.0
    # max_len = pred.shape[2]
    # index1 = torch.empty(3, max_len, 3).cuda()
    # index2 = torch.empty(3, max_len, 3).cuda()
    # feat_pair1 = pred[pair1]
    # feat_pair2 = pred[pair2]

    # _, bs, _, C = pred.shape
    # voxel_map = []

    # for i, indices in enumerate(idx):
    #     # NCam, Maxlen, C
    #     feat1 = feat_pair1[:, i]
    #     feat2 = feat_pair2[:, i]

    #     # NCam, Maxlen, 3
    #     for j, (cam1, cam2) in enumerate(zip(pair1, pair2)):
    #         index1[j, :len(indices[cam1])] = indices[cam1]
    #         index2[j, :len(indices[cam2])] = indices[cam2]

        # index1_expanded = index1.unsqueeze(2)  # (N, 1, 3)
        # index2_expanded = index2.unsqueeze(1)  # (1, N, 3)

        # common_mask = (index1_expanded == index2_expanded).all(dim=-1)
        # print(common_mask.shape)
        # overlap_indices = torch.nonzero(common_mask, as_tuple=False)
        # print(overlap_indices.shape)
        # overlap = index1[overlap_indices[:, 0]]
        # print(overlap.shape)
        # assert False

        # set1 = set(map(tuple, index1.view(-1, 3).cpu().numpy()))
        # set2 = set(map(tuple, index2.view(-1, 3).cpu().numpy()))

        # overlap = set1.intersection(set2)
        # # N, 3
        # overlap = torch.tensor(list(overlap)).cuda()

        # N, _ = overlap.shape
        # # N, C
        # overlap1 = torch.empty(N, C).cuda()
        # overlap2 = torch.empty(N, C).cuda()

        # for i, coord in enumerate(overlap):
        #     if not torch.equal(coord, torch.tensor([0, 0, 0]).cuda()):

        #         indice1 = torch.nonzero((index1[..., 0] == coord[0]) & (index1[..., 1] == coord[1]) & (index1[..., 2] == coord[2]))
        #         indice2 = torch.nonzero((index2[..., 0] == coord[0]) & (index2[..., 1] == coord[1]) & (index2[..., 2] == coord[2]))
    
        #         overlap1[i] = feat1[indice1[0][0], indice1[0][1]]
        #         overlap2[i] = feat2[indice2[0][0], indice2[0][1]]

