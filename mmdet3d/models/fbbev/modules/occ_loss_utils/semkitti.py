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

def chamfer_distance_loss(pred, gt):
    # gt: bs, D, W, H
    # Expand dims for pairwise distance computation
    
    bs = gt.shape[0]
    
    
    # batch_indices = gt[:, 0]
    # gt_indices = gt[:, 1:]
    
    # gt_expand = [gt_indices[batch_indices == i].unsqueeze(0) for i in range(bs)] # bs[1, N_ture, 3]

    pred_expand = pred.unsqueeze(2)  # (bs, Npred, 1, 3)
    # Pairwise L2 distances
    total_dist = 0
    

    for i in range(bs):
        gt_expand = torch.nonzero(gt[i]).float().to('cuda') # Ntrue, 3
        
        num_pred = pred_expand.shape[1]
        num_gt = gt_expand.shape[1]
        # pred: Npred, 1, 3
        pred_to_gt = 0
        gt_to_pred = 0 

        for j in range(num_pred):
            target_pred = pred_expand[i][j].unsqueeze(0) # 1, 1, 3
            distances = torch.cdist(target_pred, gt_expand.unsqueeze(0), p=1)  # Npred, Ntrue
            # For each point in pred, find closest point in gt
            pred_to_gt = pred_to_gt + torch.min(distances, dim=-1)[0]  # (Npred)

        for k in range(num_gt):
            target_gt = gt_expand.unsqueeze(1)[k].unsqueeze(0) # 1, 1, 3
            distances = torch.cdist(pred_expand[i], target_gt, p=1) # Npred, Ntrue
            gt_to_pred = gt_to_pred + torch.min(distances, dim=0)[0]  # (Npred)

        chamfer_dist = pred_to_gt/num_pred + gt_to_pred/num_gt
        total_dist = total_dist + chamfer_dist
    
    total_dist = total_dist / bs
    return total_dist


def hard_feature_query_alignment_loss(features, queries, mask):
    features = F.normalize(features, dim=-1) # bsNcam, HW, C
    queries = F.normalize(queries, dim=-1) # bsNcam, N, C
    mask = ~mask.unsqueeze(-1) # bsNcam, N

    similarity_scores = torch.matmul(queries, features.transpose(-1, -2)) # bsNcam, N, HW
    similarity_scores = (similarity_scores + 1) / 2
    similarity_scores = similarity_scores * mask

    max_indices = similarity_scores.argmax(dim=-1)  # bsNcam, N

    selected_features = torch.gather(
        features, dim=1,
        index=max_indices.unsqueeze(-1).expand(-1, -1, features.size(-1))
    ) # bsNcam, N, C
    
    selected_features = selected_features * mask
    queries = queries * mask

    loss = F.mse_loss(selected_features, queries)
    # print(f'hard_align_loss: {loss}')
    return loss

def feature_query_reconstruction_loss(features, queries):
    features = F.normalize(features, dim=-1) # bsNcam, HW, C
    queries = F.normalize(queries, dim=-1) # bsNcam, N, C

    similarity_scores = torch.matmul(features, queries.transpose(-1, -2)) # bsNcam, HW, N

    attention_weights = F.softmax(similarity_scores, dim=-1) # bsNcam, HW, N 

    reconstructed_features = torch.matmul(attention_weights, queries) # bsNcam, HW, C

    align_loss = F.mse_loss(reconstructed_features, features)
    return align_loss

def query_diversity_loss(queries, mask):
    queries = F.normalize(queries, dim=-1) # bsNcam, N, C
    mask = ~mask.unsqueeze(-1) # bsNcam, N
    dot_products = torch.matmul(queries, queries.transpose(-1, -2)) # bsNcam, N, N

    identity = torch.eye(dot_products.size(-1), device=dot_products.device)
    dot_products = dot_products - identity.unsqueeze(0) # bsNcam, N, N # -1~1 range
    dot_products = (dot_products + 1) / 2
    dot_products = dot_products * mask # bsNcam, N, N

    diversity_loss = dot_products.abs().mean()
    # print(f'diversity_loss: {diversity_loss}')
    return diversity_loss

def cos_sim_loss(pred):
    cos_sim_list = []
    overlap1, overlap2 = pred
    
    for i, (feat1, feat2) in enumerate(zip(overlap1, overlap2)):
        cos_sim = F.cosine_similarity(feat1, feat2, dim=-1)
        loss = (1 - cos_sim).mean()
        cos_sim_list.append(loss)

    total_loss = torch.stack(cos_sim_list).mean()
    return total_loss


def radius_ce_loss(gt, preds, pred_pixel_coords, cam_params):
    # gt: bs, Ncam, ori_H, ori_W (gt depth for image)
    # pred: bs*Ncam, N_queries, 100
    # pred_pixel_coords: bs*Ncam, N_queries, 2

    radius_range = [2, 42, 0.4] # 100
    radius_channels = preds.shape[-1]

    bs, Ncam, orig_H, orig_W = gt.shape
    gt_downsample = get_downsampled_gt_depth(16, gt) #  bs*Ncam, W, H
    _, W, H = gt_downsample.shape
    gt_downsample = gt_downsample.contiguous().view(bs, Ncam, W, H)

    assert (orig_W // W) == (orig_H // H)

    u = torch.linspace(0, W - 1, W, device=gt.device)
    v = torch.linspace(0, H - 1, H, device=gt.device)

    u, v = torch.meshgrid(u, v, indexing='ij')
    img_grid = torch.stack([u, v], dim=-1).flatten(0, 1)
    img_grid = img_grid[None, None, :].repeat(bs, Ncam, 1, 1)
        
    # img_grid: bs, Ncam, WH, 2
    img_grid[..., 0] *= (orig_W/W)
    img_grid[..., 1] *= (orig_H/H)

    gt_downsample = gt_downsample.flatten(2, 3).unsqueeze(-1)
    img_grid = torch.cat([img_grid, gt_downsample], dim=-1) # bs, Ncam, WH, 3

    gt_rad = gt_to_spherical(img_grid, cam_params) # bs, Ncam, WH
    gt_rad = gt_rad.reshape(bs, Ncam, W, H).flatten(0, 1) # bs*Ncam, W, H
    gt_rad[gt_rad > 45] = 45

    ## mapping value d to k categories: gt_rad = bs*Ncam, W, H -> bs*Ncam, W, H, depth_channels
    gt_rad = torch.log(gt_rad) - torch.log(torch.tensor(radius_range[0]).float())
    gt_rad = gt_rad * (radius_channels - 1) / torch.log(torch.tensor(radius_range[1] - 1.).float() / radius_range[0])
    gt_rad = gt_rad + 1.

    gt_rad = torch.where((gt_rad < radius_channels + 1) & (gt_rad >= 0.0), gt_rad, torch.zeros_like(gt_rad))
    gt_rad = F.one_hot(gt_rad.long(), num_classes=radius_channels + 1)[..., 1:] # bs*Ncam, W, H, depth_channels
    ###
    
    ## mapping corresponding gt values
    gt_rad_queries = torch.zeros_like(preds, device=preds.device) # bs*Ncam, N_queries, 100
    pred_pixel_coords = pred_pixel_coords.permute(1, 0, 2).long() # N_queries, bs*Ncam, 2

    for i in range(gt_rad_queries.shape[0]):
        gt_rad_queries[i] = gt_rad[i, pred_pixel_coords[..., i, 0], pred_pixel_coords[..., i, 1]]
    ####

    ### gt_rad_queries, pred: bs*Ncam, N_queries, 100
    pred_rad = preds
    pred_rad = pred_rad.contiguous().view(-1, radius_channels)
    gt_rad_queries = gt_rad_queries.contiguous().view(-1, radius_channels)
    
    fg_mask = torch.max(gt_rad_queries, dim=1).values > 0.0
    gt_rad_queries = gt_rad_queries[fg_mask]
    
    pred_rad = pred_rad[fg_mask]
    with autocast(enabled=False):
        radius_loss = F.binary_cross_entropy(
            pred_rad,
            gt_rad_queries,
            reduction='none',
        ).sum() / max(1.0, fg_mask.sum())

    return radius_loss
    

def get_downsampled_gt_depth(downsample, gt_depths):
    """
    Input:
        gt_depths: [B, N, H, W]
    Output:
        gt_depths: [B*N, h, w, 1]
    """
    # if self.downsample == 8 and self.se_depth_map:
    #    downsample = 16 
    B, N, H, W = gt_depths.shape
    gt_depths = gt_depths.view(B * N, H // downsample,
                                downsample, W // downsample,
                                downsample, 1)
    gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()
    gt_depths = gt_depths.view(-1, downsample * downsample)
    gt_depths_tmp = torch.where(gt_depths == 0.0,
                                1e5 * torch.ones_like(gt_depths),
                                gt_depths)
    gt_depths = torch.min(gt_depths_tmp, dim=-1).values
    gt_depths = gt_depths.view(B * N, H // downsample,
                                W // downsample)
    
    gt_depths = gt_depths.permute(0, 2, 1) # bs*Ncam, W, H

    return gt_depths.float() # bs*Ncam, WH, 1

def gt_to_spherical(img_coords, cam_params, mode=None):
    """
    Args:
        u, v: Pixel coordinates in the image (bs, ncam, H, W)
        intrinsic: Camera intrinsic matrix (bs, ncam, 3, 3)
        rotation: Camera rotation matrix (bs, ncam, 3, 3)
        trans: Camera translation vector (bs, ncam, 3)
        post_rotation: Post-rotation matrix (bs, ncam, 3, 3)
        post_trans: Post-translation vector (bs, ncam, 3)
        bda: BDA rotation matrix (bs, 3, 3)

    Returns:
        theta: Azimuth angles (bs, ncam, H, W)
        phi: Elevation angles (bs, ncam, H, W)
    """
    rots, trans, intrins, post_rots, post_trans, bda = cam_params

    bs, Ncam, N, _ = img_coords.shape # bs, Ncam, WH, 3

    img_coords -= post_trans.view(bs, Ncam, 1, 3)
    img_coords = post_rots.inverse().view(bs, Ncam, 1, 3, 3).matmul(img_coords.unsqueeze(-1)).squeeze(-1)

    img_coords = intrins.inverse().view(bs, Ncam, 1, 3, 3).matmul(img_coords.unsqueeze(-1)).squeeze(-1)
    
    img_coords = rots.view(bs, Ncam, 1, 3, 3).matmul(img_coords.unsqueeze(-1)).squeeze(-1)
    img_coords += trans.view(bs, Ncam, 1, 3)

    final_coords = bda.view(bs, 1, 1, 3, 3).matmul(img_coords.unsqueeze(-1)).squeeze(-1)

    D, W, H = final_coords[..., 0], final_coords[..., 1], final_coords[..., 2]

    rad = torch.sqrt(D**2 + W**2 + H**2) # bs, Ncam, WH

    return rad