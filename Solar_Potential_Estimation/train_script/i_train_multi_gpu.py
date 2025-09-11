import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
# nn.SyncBatchNorm = nn.BatchNorm2d

from torch.nn.parallel import DataParallel
from torch.utils.data import DataLoader
from lib.models.seg_hrnet_ocr import HighResolutionNet
from datasets.inria import INRIADataset
from collections import OrderedDict
from sklearn.metrics import precision_score, recall_score, f1_score

import os
import math
import datetime
import matplotlib.pyplot as plt
import random
import argparse

import pprint
from lib.config import config
from lib.utils.utils import create_logger
from yacs.config import CfgNode as CN
from torch.cuda.amp.grad_scaler import GradScaler
from torch.cuda.amp.autocast_mode import autocast
import warnings

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

def setup_ddp():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method='env://')
    return local_rank

warnings.filterwarnings(
    "ignore",
    message="The default behavior for interpolate/upsample with float scale_factor changed in 1.6.0*",
    category=UserWarning
)



# Boundary Loss
class BoundaryLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.laplacian = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], 
                                     dtype=torch.float32).view(1, 1, 3, 3)
    
    def forward(self, pred, target):
        # Calculate edges
        pred_edges = F.conv2d(pred, self.laplacian.to(pred.device), padding=1)
        target_edges = F.conv2d(target, self.laplacian.to(target.device), padding=1)
        
        # Calculate boundary-aware loss
        return F.mse_loss(pred_edges, target_edges)
    
    
# class ObstructionEdgeLoss(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.laplacian = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], 
#                                      dtype=torch.float32).view(1, 1, 3, 3)
    
#     def forward(self, pred, target):
#         # Calculate edges for obstructions
#         pred_edges = F.conv2d(pred, self.laplacian.to(pred.device), padding=1)
#         target_edges = F.conv2d(target, self.laplacian.to(target.device), padding=1)
#         return F.mse_loss(pred_edges, target_edges)

# class BoundaryEdgeLoss(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
#                                    dtype=torch.float32).view(1, 1, 3, 3)
#         self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
#                                    dtype=torch.float32).view(1, 1, 3, 3)
    
#     def forward(self, pred, target):
#         # Calculate Sobel edges for roof boundaries
#         pred_gx = F.conv2d(pred, self.sobel_x.to(pred.device), padding=1)
#         pred_gy = F.conv2d(pred, self.sobel_y.to(pred.device), padding=1)
#         pred_edges = torch.sqrt(pred_gx**2 + pred_gy**2 + 1e-6)
        
#         target_gx = F.conv2d(target, self.sobel_x.to(target.device), padding=1)
#         target_gy = F.conv2d(target, self.sobel_y.to(target.device), padding=1)
#         target_edges = torch.sqrt(target_gx**2 + target_gy**2 + 1e-6)
        
#         return F.l1_loss(pred_edges, target_edges)
    

# Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        # pred: [B, 1, 512, 512] (logits)
        # target: [B, 1, 512, 512] (binary)
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        p = torch.sigmoid(pred)
        pt = p * target + (1 - p) * (1 - target)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        focal_loss = alpha_t * (1 - pt).pow(self.gamma) * bce
        return focal_loss.mean() # scalar tensor
    
# Dice Loss
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        # pred: [B, 1, 512, 512] (logits)
        # target: [B, 1, 512, 512] (binary)
        pred = torch.sigmoid(pred)  # Convert logits to probabilities
        intersection = (pred * target).sum(dim=(2, 3))  # [B, C]
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))  # [B, C]
        dice = (2. * intersection + self.smooth) / (union + self.smooth)  # [B, C]
        return 1 - dice.mean()  # Average over batch and channels #scalar tensor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/inria_hrnet_ocr.yaml')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('opts', nargs=argparse.REMAINDER)
    parser.add_argument('--gpu', type=int, default=0)
    return parser.parse_args()


def get_alpha(current_epoch, max_epoch, base_alpha):
    if current_epoch is None:  # Validation mode
        return base_alpha  
    # Gradually reduce focus
    return base_alpha * (0.5 + 0.5 * (current_epoch/max_epoch))    # Gradually increase focus


# Initialize losses with class-specific parameters
focal_roof = FocalLoss(alpha=0.75, gamma=2)  # Roof: 24% positive
# focal_obs = FocalLoss(alpha=0.9, gamma=3)    # Obstruction: 4% positive
dice_loss = DiceLoss(smooth=1e-6)

# Loss of a specific channel(roof_channel or obstruction_channel)
def channel_specific_loss(pred, target, epoch, max_epoch):
    # pred: [B, 2, 512, 512]
    # target: [B, 2, 512, 512]
    
    pred_channel = pred # Size: [B, 1, 512, 512]
    target_channel = target # Size: [B, 1, 512, 512]

    # is_obstruction = (channel_idx == 1)
    # Get dynamic alpha for combined loss(focal + dice) calculation for each channel
    # alpha specific what weight should be of focal loss and what weight should be of dice loss
    base_alpha = 0.6
    alpha = get_alpha(epoch, max_epoch, base_alpha)

    # Select appropriate focal loss
    focal = focal_roof

    # calculating focal loss
    f1 = focal(pred_channel, target_channel)
    d1 = dice_loss(pred_channel, target_channel)

    return alpha * f1 + (1 - alpha) * d1 # scalar tensor

# obs_edge_loss = ObstructionEdgeLoss()
# boundary_edge_loss = BoundaryEdgeLoss()
# Loss for a batch
def compute_batch_loss(outputs, masks, epoch, max_epoch, is_validation):
    # outputs: dict {'main_out': tensor, 'aux_out': tensor} tensor size: [B, 2, 512, 512]
    # masks: [B, 2, 512, 512]
    # Dynamic weights

    # Add input validation
    assert not torch.isnan(masks).any()


    roof_weight = 1.0 
    if is_validation:
        # obs_weight = 2.0
        boundary_weight = 0.3 
        # obs_edge_w = 0.2
        # hard_w = 0.5
        # boundary_edge_w = 0.3
    else:
        # obs_weight = 2.5 if epoch < 15 else 2.0  # Reduce later
        boundary_weight = 0.3 * (epoch / 25) if epoch < 25 else 0.3  # Ramp up
        # --- weights for boundary_edge_loss , obs_edge_loss and hard loss
        # Ramp obs_edge loss from 0.0 to 0.2 over first 10 epochs
        # obs_edge_w = min(0.2, 0.02 * epoch)
        # hard_w = min(0.5, 0.05 * epoch)
        # boundary_edge_w = min(0.3, 0.03 * epoch)

    aux_weight = 0.4  # Keep aux head but with lower weight

    main_out = outputs['main_out'] # size: [B, 2, 512, 512]
    aux_out = outputs['aux_out'] # size: [B, 2, 512, 512]

    # --- Main Output Loss ---
    main_roof_loss = channel_specific_loss(main_out, masks, epoch, max_epoch) * roof_weight
    # main_obs_loss = channel_specific_loss(main_out, masks, 1, epoch, max_epoch) * obs_weight #heavy weighting for obs
 

    # --- Auxillary Output Loss ---
    aux_roof_loss = channel_specific_loss(aux_out, masks, epoch, max_epoch) * aux_weight
    # aux_obs_loss = channel_specific_loss(aux_out, masks, 1, epoch, max_epoch) * aux_weight *0.5 #heavy weighting for obs


    # Calculate boundary loss only for roofs
    roof_target = masks
    boundary_target = F.conv2d(roof_target, 
                             BoundaryLoss().laplacian.to(roof_target.device), 
                             padding=1).abs()
    boundary_loss = F.mse_loss(outputs['boundary'], boundary_target) * boundary_weight
    # Weighted

    # # Weight obstructions near boundaries more heavily
    # boundary_weight_map = 1.0 + 3.0 * boundary_target  # [B,1,H,W]
    # boundary_weight_map = torch.clamp(boundary_weight_map, min=1.0, max=4.0)
    # obs_loss_weighted = (F.binary_cross_entropy_with_logits(main_out[:, 1:2], masks[:, 1:2], reduction='none') * boundary_weight_map).mean()

    # main_obs_loss +=  obs_loss_weighted #################################################
    # Edge attention regularization (prevent over-smoothing)
    edge_reg = torch.mean(outputs['edge_attention']) * 0.1



    # --- weights for boundary_edge_loss , obs_edge_loss and hard loss
    # Ramp obs_edge loss from 0.0 to 0.2 over first 10 epochs
    # obs_edge_w = min(0.2, 0.02 * epoch)
    # hard_w = min(0.5, 0.05 * epoch)
    # boundary_edge_w = min(0.3, 0.03 * epoch)


    # # Add focal loss for hard examples (high-confidence errors)
    # pred_probs = torch.sigmoid(main_out[:, 1:2])
    # hard_mask =((pred_probs > 0.7) & (masks[:, 1:2] == 0)) | ((pred_probs < 0.3) & (masks[:, 1:2] == 1))
    # hard_loss = F.binary_cross_entropy_with_logits(main_out[:, 1:2][hard_mask], masks[:, 1:2][hard_mask]) * hard_w # Weighted


    # # Obstruction Edge Loss (only on obstruction pixels)
    # obs_edge_target = masks[:, 1:2] * F.conv2d(
    #     masks[:, 1:2], 
    #     obs_edge_loss.laplacian.to(masks.device), 
    #     padding=1
    # ).abs()  # Edges of GT obstructions
    # obs_edge_loss_val = obs_edge_loss(outputs['obs_boundary'], obs_edge_target) * obs_edge_w

    # Boundary Edge Loss (Sobel-based, only on roofs)
    # roof_mask = masks[:, 0:1]
    # boundary_edge_target = roof_mask * torch.sqrt(
    #     F.conv2d(roof_mask, boundary_edge_loss.sobel_x.to(masks.device), padding=1)**2 +
    #     F.conv2d(roof_mask, boundary_edge_loss.sobel_y.to(masks.device), padding=1)**2
    # )
    # boundary_edge_loss_val = boundary_edge_loss(outputs['boundary_edge'], boundary_edge_target) * boundary_edge_w


    # # Add structural similarity loss for roof continuity
    # ssim_loss = 1 - SSIM(outputs['main_out'][:,0:1], masks[:,0:1], window_size=11) * 0.3


    # --- Total Loss ---
    total_loss = (main_roof_loss) +  boundary_loss + (aux_roof_loss) + edge_reg # scalar tensor


    # print(
    # f"Main Obs Loss: {main_obs_loss.item():.4f} | "
    # f"Weighted Obs Loss: {obs_loss_weighted.item():.4f} | "
    # f"Obs Edge Loss: {obs_edge_loss_val.item():.4f}"
    # )

    return {
        'total_loss': total_loss,
        'main_roof': main_roof_loss,
        # 'main_obs': main_obs_loss,
        'aux_roof': aux_roof_loss,
        # 'aux_obs': aux_obs_loss,
        'boundary': boundary_loss,
        'edge_reg': edge_reg,
        # 'obs_edge': obs_edge_loss_val,  # New
        # 'boundary_edge': boundary_edge_loss_val  # New
    }

# logging total gradient norm
def log_total_grad_norm(model, epoch):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    with open("progress/i_logger.txt", "a") as f:
        f.write(f"\n----------------------------------- EPOCH: {epoch} -----------------------------------\n")
        f.write(f"\nTotal Gradient Norm: {total_norm:.4f}\n")


# logging per layer gradient norm
def log_per_layer_grad_norm(model):
    print("\nPer-layer gradient norms:\n")
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.data.norm(2).item()
            with open("progress/i_logger.txt", "a") as f:
                f.write(f"\n{name:40s} : {grad_norm:.4f}\n")


# Detecting vanishing or exploding gradients
def check_gradients(model, threshold_explode=100.0, threshold_vanish=1e-4):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.data.norm(2).item()
            if grad_norm > threshold_explode:
                with open("progress/i_logger.txt", "a") as f:
                    f.write(f"\nExploding gradient in {name}: {grad_norm:.4f}\n")
            elif grad_norm < threshold_vanish:
                with open("progress/i_logger.txt", "a") as f:
                    f.write(f"\nVanishing gradient in {name}: {grad_norm:.4e}\n")


def initialSetup():
    
    # 1. Loading configuration file
    args = parse_args()
    local_rank = setup_ddp()
    cfg = CN(new_allowed=True)
    cfg.merge_from_file("configs/inria_hrnet_ocr.yaml")

    print(cfg)
    print("Learning rate:", cfg.SOLVER.BASE_LR)
    print("Backbone:", cfg.MODEL.PRETRAINED)

    # 2. Setting the cuda device(selecting gpu)
    # device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    # torch.cuda.set_device(device)

    return cfg, local_rank



# Metrics Calculation Function
def calculate_metrics(pred, target, eps=1e-6):
    # pred: [B, 1, H, W] (logits)
    # target: [B, 1, H, W] (binary)

    pred_bin = (torch.sigmoid(pred) > 0.5).float()
    target_bin = (target > 0.5).float()

    # Flatten tensors
    pred_flat = pred_bin.view(-1).cpu().numpy()
    target_flat = target_bin.view(-1).cpu().numpy()
    
    # Avoid division by zero
    if target_flat.sum() == 0:
        return {
            'iou': float('nan'),
            'f1': float('nan'),
            'precision': float('nan'),
            'recall': float('nan'),
            'accuracy': float('nan')
        }
    
    # Calculate metrics
    intersection = (pred_bin * target_bin).sum()
    union = (pred_bin + target_bin - pred_bin * target_bin).sum()
    iou = (intersection + eps) / (union + eps)
    
    accuracy = (pred_bin == target_bin).float().mean()
    
    # Use sklearn for stable calculation
    precision = precision_score(target_flat, pred_flat, zero_division=0)
    recall = recall_score(target_flat, pred_flat, zero_division=0)
    f1 = f1_score(target_flat, pred_flat, zero_division=0)
    
    return {
        'iou': iou.item(),
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'accuracy': accuracy.item()
    }



# Training function
def train_epoch(model, train_loader, optimizer, scheduler, epoch, cfg, local_rank):

    ## This is because I'm using 'DataPatallel' (using multiple gpus at once to train)
    scaler = GradScaler()
    model.train() 

    loss = {}   
    train_metrics = {
        'roof': {'loss': 0.0, 'iou': 0.0, 'f1': 0.0, 'precision': 0.0, 'recall': 0.0, 'accuracy': 0.0},
        # 'obs': {'loss': 0, 'iou': 0, 'f1': 0, 'precision': 0.0, 'recall': 0.0, 'accuracy': 0.0}
    }
    num_samples = 0
    # 1. Training over each batch
    for batch_idx, batch in enumerate(train_loader):
        print(f"\nEpoch[{epoch}/{cfg.SOLVER.MAX_EPOCHES-1}], Batch[{batch_idx}/{len(train_loader)-1}]\n")

        # 1. Getting images and masks of the current batch and loading them on cuda device
        # Each batch is a dictionary which has 'image' and 'mask' as keys for images and masks resp.
        images = batch['image'].to(local_rank, non_blocking=True) # Size: [B, 3, 512, 512]
        masks = batch['mask'].to(local_rank, non_blocking=True) # Size: [B, 2, 512, 512]

        # 2. Making any previous gradient zero
        optimizer.zero_grad()

        ## Using 'autocast' because of 'DataParallel'
        with autocast():

            # 3. Forward Pass
            # 'outputs' is a dictionary which has 'main_output'([B, 2, 128, 128]) and 'aux_out'([B, 2, 65, 65]) as key      or main output and auxillary output from ocd head resp.
            outputs = model(images)

            # 4. Upsampling 'outputs' to bring it to size of mask[B, 2, 512, 512]
            outputs_upsampled = {
                'main_out': F.interpolate(outputs['main_out'], size=(512, 512), mode='bilinear', align_corners=False),
                'aux_out': F.interpolate(outputs['aux_out'], size=(512, 512), mode='bilinear', align_corners=False),
                'boundary': F.interpolate(outputs['boundary'], size=(512, 512), mode='bilinear', align_corners=False),
                'edge_attention': F.interpolate(outputs['edge_attention'], size=(512, 512), mode='bilinear', align_corners=False),
                # 'obs_boundary': F.interpolate(outputs['obs_boundary'], size=(512, 512), mode='bilinear', align_corners=False),
                # 'boundary_edge': F.interpolate(outputs['boundary_edge'], size=(512, 512), mode='bilinear', align_corners=False)
            }


            # # with torch.no_grad():
            # main_out = outputs_upsampled['main_out']
            # print("Main output stats: mean =", main_out.mean().item(), "std =", main_out.std().item())
            # pred = torch.sigmoid(main_out)
            # print("Main output sigmoid stats: min =", pred.min().item(), "max =", pred.max().item())

            # 5. Calculating batch loss
            loss = compute_batch_loss(outputs_upsampled, masks, epoch, cfg.SOLVER.MAX_EPOCHES, is_validation=False)

            ### NOTE: if i want to visualize the predicted masks then i can filter masks for obstructions on roof after loss calculation only

        scaler.scale(loss['total_loss']).backward()

        # 6. Logging gradient tracks
        # log_total_grad_norm(model, epoch)
        # log_per_layer_grad_norm(model)
        # check_gradients(model)

        # 7. Add norm clip if needed
        scaler.unscale_(optimizer)
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scheduler.step() 
        scaler.update()

        # 8. Training metrics calculation
        # --- METRICS CALCULATION ---
        with torch.no_grad():
            # Roof metrics
            roof_metrics = calculate_metrics(
                outputs_upsampled['main_out'], 
                masks
            )

            # # Obstruction metrics
            # obs_metrics = calculate_metrics(
            #     outputs_upsampled['main_out'][:, 1:2], 
            #     masks[:, 1:2]
            # )

        # Accumulate metrics
        batch_size = images.size(0)
        num_samples += batch_size
        for channel in ['roof']:
            metrics = roof_metrics
            train_metrics[channel]['loss'] += loss[f'main_{channel}'].item() * batch_size
            for k in metrics:
                train_metrics[channel][k] += metrics[k] * batch_size
        
        # 9. Printing Training metrics per batch
        # Log batch progress
        if batch_idx % 2 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx}/{len(train_loader)-1}] '
                f" Roof IoU: {roof_metrics['iou']:.4f} "
                # f" Obs IoU: {obs_metrics['iou']:.4f}"
                # f" Obs F1: {obs_metrics['f1']:.4f}")
            )
            
    # Average metrics
    for channel in ['roof']:
        for k in train_metrics[channel]:
            train_metrics[channel][k] /= num_samples 
            

    return train_metrics, loss

    

def validation(model, val_loader, epoch, max_epoch, local_rank):

    model.eval()
    val_metrics = {
        'roof': {'loss': 0.0, 'iou': 0.0, 'f1': 0.0, 'precision': 0.0, 'recall': 0.0, 'accuracy': 0.0},
        # 'obs': {'loss': 0.0, 'iou': 0.0, 'f1': 0.0, 'precision': 0.0, 'recall': 0.0, 'accuracy': 0.0}
    }
    num_samples = 0

    with torch.no_grad(): 
        for val_batch_idx, batch in enumerate(val_loader):

            # 1. Loading images and masks 
            images = batch['image'].to(local_rank, non_blocking=True)
            masks = batch['mask'].to(local_rank, non_blocking=True)

            # 2. Edge case recheck 
            # invalid = (masks[:,0].sum(dim=[1,2]) == 0) & (masks[:,1].sum(dim=[1,2]) > 0)
            # if invalid.any():
            #     masks[invalid, 1] = 0  # Clear batch-wise
            
            # 3. Forward Pass only for validation
            outputs = model(images)

            # 4. Upsampling 'outputs' to bring it to size of mask[B, 2, 512, 512]
            outputs_upsampled = {
                'main_out': F.interpolate(outputs['main_out'], size=(512, 512), mode='bilinear', align_corners=False),
                'aux_out': F.interpolate(outputs['aux_out'], size=(512, 512), mode='bilinear', align_corners=False),
                'boundary': F.interpolate(outputs['boundary'], size=(512, 512), mode='bilinear', align_corners=False),
                'edge_attention': F.interpolate(outputs['edge_attention'], size=(512, 512), mode='bilinear', align_corners=False),
                # 'obs_boundary': F.interpolate(outputs['obs_boundary'], size=(512, 512), mode='bilinear', align_corners=False),
                # 'boundary_edge': F.interpolate(outputs['boundary_edge'], size=(512, 512), mode='bilinear', align_corners=False)
            }

            # 5. Validation Metrics 
            loss = compute_batch_loss(outputs_upsampled, masks, epoch=None, max_epoch=None, is_validation=True)

            ### NOTE: compute metrics for obstructions with filters obstructions
            # 6. --- PREDICTION FILTERING ---
            # roof_mask = (torch.sigmoid(outputs_upsampled['main_out'][:, 0:1]) > 0.5).float()


            # obs_mask = (torch.sigmoid(outputs_upsampled['main_out'][:, 1:2]) > 0.5).float()
            # obs_mask = F.max_pool2d(obs_mask, kernel_size=3, stride=1, padding=1)  # Dilate
            # obs_mask = F.avg_pool2d(obs_mask, kernel_size=5, stride=1, padding=2)  # Erode
            # obs_mask = (obs_mask > 0.5).float() * roof_mask  # Final cleanup

            # obs_mask = (torch.sigmoid(outputs_upsampled['main_out'][:, 1:2]) > 0.5).float() * roof_mask

            # 7. Calculating mertics
            # Calculate metrics on filtered obstructions
            roof_metrics = calculate_metrics(
                outputs_upsampled['main_out'], 
                masks
            )

            # obs_metrics = calculate_metrics(
            #     obs_mask,  # Use filtered predictions
            #     masks[:, 1:2]
            # )

            # val_metrics['obs_edge'] = obs_edge_loss_val.item()
            # val_metrics['boundary_edge'] = boundary_edge_loss_val.item()

            
            # Debug snippet (add to validation)
            # plt.imshow(boundary_weight_map[0,0].cpu().numpy(), cmap='jet')  # Red = high weight
            # plt.show()

            # Accumulate
            batch_size = images.size(0)
            num_samples += batch_size

            if val_batch_idx % 2 == 0:
                print(f'Epoch: [{epoch}/{max_epoch}]  Val Batch: [{val_batch_idx}/{len(val_loader)}] '
                    f" Roof IoU: {roof_metrics['iou']:.4f}"
                    # f" Obs IoU: {obs_metrics['iou']:.4f}"
                    # f" Obs F1: {obs_metrics['f1']:.4f}")
                )
            
            for channel in ['roof']:
                metrics = roof_metrics
                val_metrics[channel]['loss'] += loss[f'main_{channel}'].item() * batch_size
                for k in ['iou', 'f1', 'precision', 'recall', 'accuracy']:
                    val_metrics[channel][k] += metrics[k] * batch_size
     
    # Average metrics
    for channel in ['roof']:
        for k in val_metrics[channel]:
            val_metrics[channel][k] /= num_samples

    return val_metrics



def configure_phase(model, epoch, optimizer, cfg):
    # Phase 1 (0-5): Warmup (heads only)
    if epoch < 5:
        for name, param in model.named_parameters():
            # Freeze backbone and boundary heads
            if any(k in name for k in ["stem", "conv1", "conv2", "bn1", "bn2", "layer1", "stage1", "transition1", "stage2", "transition2", "stage3", "transition3", "stage4"]):
                param.requires_grad = False
            if 'boundary' in name or 'edge' in name:
                param.requires_grad = False
            # Freeze ALL OCR components
            if 'ocr' in name:  # This covers conv3x3_ocr, ocr_gather_head, ocr_distri_head
                param.requires_grad = False
            # Keep main heads trainable
            if any(k in name for k in ['cls_head', 'aux_head']):
                param.requires_grad = True
    
    # Phase 2 (5-15):
    elif 5 <= epoch < 15:
        for name, param in model.named_parameters():
            # Freeze backbone and boundary heads
            if any(k in name for k in ["stem", "conv1", "conv2", "bn1", "bn2", "layer1", "stage1", "transition1", "stage2", "transition2"]):
                param.requires_grad = True

            if any(k in name for k in ["stage3", "transition3", "stage4"]):
                param.requires_grad = False

            if 'boundary' in name or 'edge' in name:
                param.requires_grad = False

            # Keep ALL OCR components frozen
            if 'ocr' in name:
                param.requires_grad = False
        
            # Always train these
            if any(k in name for k in ['cls_head', 'aux_head']):
                param.requires_grad = True

    
    # Phase 3 (15-25): 
    elif 15 <= epoch < 25:

        for name, param in model.named_parameters():

            # Freeze early stages again
            if any(k in name for k in ["stem", "conv1", "conv2", "bn1", "bn2", "layer1", "stage1", "transition1"]):
                param.requires_grad = False

            if any(k in name for k in ["stage2", "transition2", "stage3", "transition3", "stage4"]):
                param.requires_grad = True

            # Unfreeze OCR modules (all components)
            if 'ocr' in name:
                param.requires_grad = True
            # UnFreeze boundary heads
            if 'boundary' in name or 'edge' in name:
                param.requires_grad = True

            # Always train these
            if any(k in name for k in ['cls_head', 'aux_head']):
                param.requires_grad = True

    # Phase 4 (25-35): 
    elif 25 <= epoch < 35:
        for name, param in model.named_parameters():
            if any(k in name for k in ["stem", "conv1", "conv2", "bn1", "bn2", "layer1", "stage1", "transition1"]):
                param.requires_grad = True
            if any(k in name for k in ["stage4"]):
                param.requires_grad = False
    
    # Phase 5 (35+): Full fine-tuning
    else:
        for name, param in model.named_parameters():
                param.requires_grad = True


# At the START of each epoch (before training loop):
def update_learning_rates(optimizer, epoch):
    base_lr = 3e-4
    lrs = []
    # Phase 1 (0-5): Warmup
    if epoch < 5:
        lrs = [0, 0, 0, base_lr]  # [backbone, boundary, ocr, heads]
    
    # Phase 2 (5-15): Boundary
    elif 5 <= epoch < 15:
        lrs = [1e-5*(epoch-4), 0, 0, 2e-4]
    
    # Phase 3 (15-25): Obstruction
    elif 15 <= epoch < 25:
        lrs = [1e-4, 1e-4, 2e-5, 1e-4]

    # Phase 4 (25-35): Obstruction
    elif 25 <= epoch < 35:
        lrs = [5e-5, 5e-5, 1e-5, 5e-5]
    
    # Phase 5 (35+): Fine-tune
    elif epoch >= 35:
        lrs = [2e-5, 2e-5, 1e-5, 2e-5]
    
    # Apply to optimizer
    for i, group in enumerate(optimizer.param_groups):
        group['lr'] = max(lrs[i], 1e-5)


def is_main_process():
    return dist.get_rank() == 0


def reinit_classifier(model):
    for m in [model.module.cls_head, model.module.aux_head[-1]]:
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def load_pretrained_weights(model, pretrained_path):
    checkpoint = torch.load(pretrained_path, map_location='cpu')
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = 'module.' + k  # add module. prefix
        new_state_dict[new_key] = v

    model_dict = model.state_dict()
    matched_weights = {}
    skipped = []

    for k, v in new_state_dict.items():
        if k in model_dict:
            if v.shape == model_dict[k].shape:
                matched_weights[k] = v
            else:
                skipped.append((k, v.shape, model_dict[k].shape))
        else:
            skipped.append((k, v.shape, None))

    model_dict.update(matched_weights)

    new_state_dict = OrderedDict()
    for k, v in model_dict.items():
        new_key = 'module.' + k  # add module. prefix
        new_state_dict[new_key] = v
    
    model.load_state_dict(new_state_dict, strict=False)

    print(f"Loaded {len(matched_weights)} layers from pretrained weights.")
    print(f"Skipped {len(skipped)} layers due to shape or name mismatch:")
    for name, pre_shape, model_shape in skipped:
        print(f" - {name}: pretrained {pre_shape} vs model {model_shape}")



def main():


    ## Running initial setup
    cfg, local_rank = initialSetup()
    
    ### Training code begins...................

    # 1. Initialize Model
    model = HighResolutionNet(cfg)
    # 3. Transfering model to cuda device(s)(GPU(s))
    model = model.to(local_rank)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)  # Optional but helps with BN across GPUs
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)
    
    # 2. Load Pretrained weights
    pretrained_path = cfg.MODEL.PRETRAINED
    load_pretrained_weights(model, pretrained_path)
    reinit_classifier(model)
    # checkpoint = torch.load(pretrained_path)
    # model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    ## Looping because the pretrained weights are not wrapped in DataParallel so it is needed to add 'module.' before its keys
    # new_state_dict = OrderedDict()
    # for k, v in checkpoint.items():
    #     new_key = 'module.' + k  # add module. prefix
    #     new_state_dict[new_key] = v
    # missing, unexpected = model.load_state_dict(new_state_dict, strict=True)

    # print("Missing keys = ", missing)
    # print("Unexpected keys = ", unexpected)
    

    # 4. Initializing Optimizer
    param_groups = [
        {'params': [p for n,p in model.module.named_parameters() if any(k in n for k in ["stem", "conv1", "conv2", "bn1", "bn2", "layer1", "stage1", "transition1", "stage2", "transition2", "stage3", "transition3", "stage4"])]},
        {'params': [p for n,p in model.module.named_parameters() if 'boundary' in n or 'edge' in n]},
        {'params': [p for n,p in model.module.named_parameters() if 'ocr' in n]},
        {'params': [p for n,p in model.module.named_parameters() if 'cls_head' in n or 'aux_head' in n]}
    ]
    optimizer = torch.optim.SGD(
        param_groups,
        lr=cfg.SOLVER.BASE_LR,  # Default LR (will be overwritten)
        momentum=0.9,
        weight_decay=cfg.SOLVER.WEIGHT_DECAY
    )
    
    
    # 5. Initializing learning rate scheduler
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=5,             # Restart every 5 epochs
        T_mult=2,          # Double cycle length after each restart
        eta_min=1e-5       # Minimum LR
    )


    # 6. Specifying Dataset
    train_dataset = INRIADataset(cfg.DATASET.ROOT, split='train')
    val_dataset = INRIADataset(cfg.DATASET.ROOT, split='val')

    # 7. Dataset loaders
    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset,
                              batch_size=cfg.SOLVER.BATCH_SIZE,
                            #   shuffle=True,
                              sampler=train_sampler,
                              num_workers=8,
                              drop_last=True,
                              pin_memory=True)
    
    val_sampler = DistributedSampler(val_dataset, shuffle=False)
    val_loader = DataLoader(val_dataset,
                            batch_size=cfg.SOLVER.BATCH_SIZE,
                            shuffle=False,
                            sampler=val_sampler,
                            num_workers=8,
                            drop_last=True,
                            pin_memory=True)
    
    
    # 8. Opening file to log training metrics and gradient tracks
    with open("progress/i_accuracy.txt", "a") as f:
        f.write(f"\nDATE and TIME: {datetime.datetime.now()}\n")

    with open("progress/i_logger.txt", "a") as l:
        l.write(f"\nDATE and TIME: {datetime.datetime.now()}\n")

    # 9. Training Loop starts here
    start_epoch = 0
    best_iou = 0.0
    patience = 15
    no_improvement = 0

    checkpoint_path = 'i_outputs/xyz.pth'  # or your checkpoint file
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1  # Start from next epoch
        best_iou = checkpoint['best_iou']
        phase = checkpoint['phase']
        print(f"Resuming training from epoch {start_epoch} and phase {phase}")

    # Training loop with phases
    for epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCHES):
        train_sampler.set_epoch(epoch)

        trainable_count = sum(p.requires_grad for p in model.parameters())
        print(f"Epoch {epoch}: {trainable_count} trainable parameters")


        configure_phase(model, epoch, optimizer, cfg)
        update_learning_rates(optimizer, epoch)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable params = {trainable_params}, LR = {optimizer.param_groups[0]['lr']}, {optimizer.param_groups[1]['lr']}, {optimizer.param_groups[2]['lr']}, {optimizer.param_groups[3]['lr']}\n")

        if epoch == 5 or epoch == 15 or epoch == 25:
            print("Trainable OCR params at epoch", epoch)
            for name, param in model.named_parameters():
                if 'ocr' in name and param.requires_grad:
                    print(name)

        # 1. Training 
        train_metrics, train_loss = train_epoch(model, train_loader, optimizer, scheduler, epoch, cfg, local_rank)

        # 2. Validation
        val_metrics = validation(model, val_loader, epoch, cfg.SOLVER.MAX_EPOCHES, local_rank)

        # 3. Save best model
        current_iou = (val_metrics['roof']['iou'])

        # 4. Adjusting learning rate after each epoch
        scheduler.step()  # Update learning rate based on validation performance

        if is_main_process() and ((epoch+1) % 5 == 0):
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_iou': best_iou,
                'phase': 'phase1' if epoch < 15 else 'phase2',  # Explicit phase tracking
            }, f'i_outputs/epoch_{epoch}.pth')

        if current_iou > best_iou:
            best_iou = current_iou
            no_improvement = 0  
        else:
            no_improvement += 1
            if no_improvement >= patience:                
                print("Early stopping triggered")
                print(f"no_imporvement: {no_improvement}")
                break


        # 5. Logging
        with open("progress/i_accuracy.txt", "a") as f:     
            f.write(f"\nEpoch {epoch} Summary:\n")
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            f.write(f"Trainable params = {trainable_params}, LR = {optimizer.param_groups[0]['lr']}\n")
            for channel in ['roof']:
                f.write(f"{channel.upper()}:\n")
                f.write(f"  Train Loss: {train_loss[f'main_{channel}'].item():.4f}")
                f.write(f"  Val Loss: {val_metrics[channel]['loss']:.4f}")
                f.write(f"  Train accuracy: {train_metrics[channel]['accuracy']:.4f}")
                f.write(f"  Val accuracy: {val_metrics[channel]['accuracy']:.4f}")
                f.write(f"  IoU: {val_metrics[channel]['iou']:.4f}")
                f.write(f"  F1: {val_metrics[channel]['f1']:.4f}")
                f.write(f"  Precision/Recall: {val_metrics[channel]['precision']:.4f}/{val_metrics[channel]['recall']:.4f} \n")
            f.write(f"\n\n")       

if __name__ == '__main__':
    
    main()
