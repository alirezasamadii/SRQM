#reference : https://github.com/AakashKT/pytorch-recurrent-ae-siggraph17/blob/master/losses.py
import torch
import numpy as np
from torch.nn import functional as func
from skimage.metrics import structural_similarity as ssim
from torch.nn.functional import mse_loss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import torch
import torch.nn.functional as F

import torch
import torch.nn.functional as F
def LoG(img):
	weight = [
		[0, 0, 1, 0, 0],
		[0, 1, 2, 1, 0],
		[1, 2, -16, 2, 1],
		[0, 1, 2, 1, 0],
		[0, 0, 1, 0, 0]
	]
	weight = np.array(weight)

	weight_np = np.zeros((1, 1, 5, 5))
	weight_np[0, 0, :, :] = weight
	weight_np = np.repeat(weight_np, img.shape[1], axis=1)
	weight_np = np.repeat(weight_np, img.shape[0], axis=0)

	weight = torch.from_numpy(weight_np).type(torch.FloatTensor).to('cuda:0')

	return func.conv2d(img, weight, padding=1)

def HFEN(output, target):
	return torch.sum(torch.pow(LoG(output) - LoG(target), 2)) / torch.sum(torch.pow(LoG(target), 2))


	

def SSIM(output,ground_truth): 
    """
    Computes the average Structural Similarity Index (SSIM) over entire slices of the subject tensor.

    Parameters:
    output (torch.Tensor): The output tensor with shape [d, h, w].
    ground_truth (torch.Tensor): The ground truth tensor with shape [d, h, w].

    Returns:
    float: The average SSIM value over all slices.

    Note:
    - The function assumes that the input tensors are 3D with dimensions [depth, height, width].
    - If you want to evaluate SSIM on a single slice, you need to unsqueeze the slice to match the required dimensions.
    - The `data_range` parameter for SSIM computation is set as the maximum value in the ground truth tensor.

    Example:
    >>> output = torch.rand(10, 256, 256)  # Example output tensor
    >>> ground_truth = torch.rand(10, 256, 256)  # Example ground truth tensor
    >>> ssim_value = SSIM(output, ground_truth)
    >>> print(ssim_value)
    """
    output_np = output.detach().cpu().numpy()
    ground_truth_np = ground_truth.detach().cpu().numpy()
    data_range=np.max(ground_truth_np)-np.min(ground_truth_np)  #should it be max of any posible input?
   
    ssim_values = [ssim(output_np[i], ground_truth_np[i], data_range=data_range) for i in range(ground_truth.shape[0])]
    avg_ssim = sum(ssim_values) / len(ssim_values)
    return avg_ssim
import torch

def compute_psnr(image_3d: torch.Tensor, reference_3d: torch.Tensor) -> list:
    """
    Computes PSNR for each slice along the depth dimension (D) of a 3D image.
    
    Args:
        image_3d (torch.Tensor): The 3D image of shape (D, H, W).
        reference_3d (torch.Tensor): The reference 3D image of the same shape (D, H, W).
    
    Returns:
        list: PSNR values for each slice along the depth (D), with length D.
    """
    assert image_3d.shape == reference_3d.shape, "Image and reference must have the same shape"

    psnr_values = []
    for i in range(image_3d.shape[0]):  # Iterate over depth dimension (D)
        mse = torch.mean((image_3d[i] - reference_3d[i]) ** 2)
        if mse == 0:
            psnr = float('inf')  # If MSE is zero, PSNR is infinite
        else:
            max_pixel = torch.max(reference_3d).item()
            psnr = 10 * torch.log10((max_pixel ** 2) / mse)
        psnr_values.append(psnr)
        
        avg_psnr = sum(psnr_values) / len(psnr_values)
        
    
    return avg_psnr