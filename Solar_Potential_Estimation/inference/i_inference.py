import torch
import numpy as np
from PIL import Image
import cv2
from cv2 import drawContours
import matplotlib.pyplot as plt
from torchvision import transforms
import albumentations as A
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Change import path to use model directory
from model.seg_hrnet_ocr import HighResolutionNet
from datasets.inria import INRIADataset  # For consistent preprocessing
import argparse
from yacs.config import CfgNode as CN
from collections import OrderedDict
# Comment out pydensecrf imports
# import pydensecrf.densecrf as dcrf
# from pydensecrf.utils import unary_from_softmax


main_dir = f'test_crowd'

# Comment out apply_crf function
"""
def apply_crf(image, prob_map):
    # The input should be the raw probability map (not thresholded)
    # print(prob_map.shape[0])
    # print(prob_map.shape[1])
    # h, w = prob_map.shape[0], prob_map.shape[1]

    if prob_map.ndim == 3:
        _, h, w = prob_map.shape
    elif prob_map.ndim == 2:
        h, w = prob_map.shape
    else:
        raise ValueError(f"Unexpected prob_map shape: {prob_map.shape}")
    
    
    # Create CRF model
    d = dcrf.DenseCRF2D(w, h, 2)  # width, height, nlabels
    
    # Unary potential
    U = unary_from_softmax(prob_map)
    d.setUnaryEnergy(U)
    
    # Add color-independent term
    # d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseGaussian(sxy=5, compat=6)
    
    # Add color-dependent term
    # d.addPairwiseBilateral(sxy=20, srgb=3, rgbim=image, compat=10)
    d.addPairwiseBilateral(sxy=30, srgb=10, rgbim=image, compat=20)
    
    # Inference
    # Q = d.inference(5)
    Q = d.inference(10)
    map_result = np.argmax(Q, axis=0).reshape((h, w))
    
    return map_result
"""



def load_model(checkpoint_path, cfg):
    """Load trained model with proper state dict handling"""
    model = HighResolutionNet(cfg)
    
    # Handle DataParallel wrapping
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['model_state_dict']
    if all(k.startswith('module.') for k in state_dict.keys()):
        # Model was saved with DataParallel
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # Remove 'module.' prefix
            new_state_dict[name] = v
        state_dict = new_state_dict
    
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model.to(device)


def preprocess_image(image_path, img_size=512):
    """Match training preprocessing exactly"""
    transform = A.Compose([
            # A.HorizontalFlip(p=0.5),
            # A.VerticalFlip(p=0.5),
            # A.RandomRotate90(p=0.5),  
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    image = np.array(Image.open(image_path).convert('RGB')).astype(np.uint8)
    transformed = transform(image=image)
    img = transformed['image']
    return img.unsqueeze(0) # Size: [1, 3, 512, 512]


def filter_small_obstructions(mask, min_size=50):
    """Remove small isolated obstruction predictions"""
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8)
    
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < min_size:
            mask[labels == i] = 0
            
    return mask


# def post_process(mask):
#     # Convert to uint8
#     mask = (mask * 255).astype(np.uint8)
    
#     # Apply morphological closing to fill small holes
#     kernel = np.ones((3,3), np.uint8)
#     mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
#     # Apply morphological opening to remove tiny specks
#     mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
#     return (mask > 127).astype(np.float32)


def edge_aware_smooth(mask, image):
    import cv2
    mask_uint8 = (mask * 255).astype(np.uint8)
    image_uint8 = image.astype(np.uint8)

    # Guided filter (edge-preserving smoothing)
    smoothed = cv2.ximgproc.guidedFilter(guide=image_uint8, src=mask_uint8, radius=4, eps=1e-2)
    binary = (smoothed > 127).astype(np.float32)
    return binary



def refine_contours(mask, min_area=100):
    mask = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    refined = np.zeros_like(mask)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= min_area:
            drawContours(refined, [cnt], contourIdx=-1, color=(255, 0, 0), thickness=-1)
    return (refined > 127).astype(np.float32)



def predict(model, image_tensor, filename):
    """Run inference with proper filtering"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        outputs = model(image_tensor.to(device))
        # Upsample to original size
        main_out = F.interpolate(outputs['main_out'], 
                                size=(512, 512), 
                                mode='bilinear',
                                align_corners=False)
        
        # Get probabilities
        roof_prob = torch.sigmoid(main_out).cpu().numpy()  # [1,H,W]
        
        # Get original image for CRF
        org_img = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        org_img = (org_img * 255).astype(np.uint8)

        roof_prob_np = roof_prob.squeeze()  # shape (512, 512)
        roof_prob_np[roof_prob_np < 0.3] = 0 

        # Replace CRF with simple thresholding
        # roof_input = np.stack([1 - roof_prob_np, roof_prob_np], axis=0)  # shape (2, 512, 512)  
        # Apply CRF
        # roof_crf = apply_crf(org_img, roof_input)
        
        # Simple thresholding instead of CRF
        roof_mask = (roof_prob_np > 0.5).astype(np.float32)

        # Final masks
        # roof_mask = post_process((roof_crf > 0)).astype(np.float32)
        # roof_mask = (roof_crf > 0).astype(np.float32)
        # roof_mask = roof_mask * (roof_prob_np > 0.5).astype(np.float32)
        
        # Apply guided filtering
        roof_mask = edge_aware_smooth(roof_mask, org_img)

        # Clean up and sharpen contours
        roof_mask = refine_contours(roof_mask, min_area=30)

        # name = os.path.basename(filename)
        # np.save(f'{main_dir}/pred_bin/{os.path.splitext(name[0])}.npy', roof_mask)

        # # Threshold and filter obstructions
        # roof_mask = (roof_prob > 0.5).float()
        # obs_mask = (obs_prob > 0.5).float() * roof_mask
        
    return {
        'roof_prob': roof_prob,  # [H,W]
        'roof_mask': roof_mask,
        'org_image': image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    }

def visualize(results, original_image=None, save_path=None):
    """Create a 3-panel visualization"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    
    # Original image (if provided)
    # transform = A.Compose([
    #         A.HorizontalFlip(p=0.5),
    #         A.RandomRotate90(p=0.5)
    #     ])
    # transformed = transform(image=original_image)
    if original_image is not None:
        # ax1.imshow(Image.fromarray(original_image).convert('RGB'))
        ax1.imshow(original_image)
    ax1.set_title('Original Image')
    
    # Roof prediction
    ax2.imshow(results['roof_mask'], cmap='gray')
    ax2.set_title('Roof Prediction')
    
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    else:
        plt.show()
    plt.close()


def calculate_metrics(pred_mask, gt_mask):
    """Compute metrics for single image"""
    intersection = (pred_mask * gt_mask).sum()
    union = (pred_mask + gt_mask - pred_mask * gt_mask).sum()
    iou = intersection / (union + 1e-6)
    accuracy = (pred_mask == gt_mask).mean()
    return {'iou': iou, 'accuracy': accuracy}



def batch_inference(model, image_dir, output_dir, cfg):
    """Process all images in a directory"""
    os.makedirs(output_dir, exist_ok=True)
    
    for img_file in os.listdir(image_dir):
        if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
            
        # Process image
        img_path = os.path.join(image_dir, img_file)
        image = preprocess_image(img_path)
        results = predict(model, image, img_path)
        
        # Save visualizations
        orig_img = np.array(Image.open(img_path).resize((512, 512)))
        save_path = os.path.join(output_dir, f'{img_file}')
        visualize(results, orig_img, save_path)
        
        # Save raw masks
        np.save(os.path.join(output_dir, f'{os.path.splitext(img_file)[0]}.npy'), 
                results['roof_mask'])




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/inria_hrnet_ocr.yaml')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--image', type=str, help='Single image path')
    parser.add_argument('--dir', type=str, help=f'{main_dir}/images')
    parser.add_argument('--output', type=str, default=f'{main_dir}/pred')
    args = parser.parse_args()

    # Load config and model
    cfg = CN(new_allowed=True)
    # Use the provided config file path
    cfg.merge_from_file(args.cfg)
    model = load_model(args.checkpoint, cfg)
 
    # Run inference
    if args.image:
        image = preprocess_image(args.image)
        name = os.path.basename(args.image)
        file_name = os.path.splitext(name)[0]
        results = predict(model, image, file_name)
        orig_img = np.array(Image.open(args.image))
        visualize(results, orig_img, os.path.join(args.output, file_name + '_' + '.png'))
        
        # Save the mask as .npy file
        np.save(os.path.join(args.output, f'{file_name}.npy'), results['roof_mask'])
    elif args.dir:
        batch_inference(model, args.dir, args.output, cfg)



# # Single image
# CUDA_VISIBLE_DEVICES=0   python inference/i_inference.py    --checkpoint i_outputs/epoch_99.pth  --cfg configs/inria_hrnet_ocr.yaml   --image india_dataset/twoChannels_in/val/images/25_6.png 

# Directory
# CUDA_VISIBLE_DEVICES=0   python inference/i_inference.py    --checkpoint i_outputs/epoch_99.pth  --cfg configs/inria_hrnet_ocr.yaml   --dir max  --output max_output

# Directory
# CUDA_VISIBLE_DEVICES=0   python inference/i_inference.py    --checkpoint i_outputs/epoch_99.pth  --cfg configs/inria_hrnet_ocr.yaml   --dir max/RID  --output max_output/
