import torch
import SimpleITK as sitk
import os
import numpy as np
import shutil
#import dicom2nifti


import numpy as np
import random
from math import exp

import torch.nn.functional as F
from torch.autograd import Variable

seed = 42
random.seed(seed)
np.random.seed(seed)

def number_of_features_per_level(init_channel_number, num_levels):
    return [init_channel_number * 2 ** k for k in range(num_levels)]

def rot_tensor(tensor):
    #rotates a 2d tensor for visulization in radilogical orientation
    rotated_tensor = torch.transpose(tensor, -1, -2).flip(1).flip(0) 
    return rotated_tensor


def compute_zscore(tensor):
    '''computes zscore of a tensor [could be a slice or volume]
     IF IT HAS SHAPE [C,D,H,W] COMPUTES IT PER VOLUME OF CHANNEL'''
    if tensor.ndimension()<=3:
        mean=torch.mean(tensor)    #calculation of Z-score with some mathematical tricks to avoid overflow
        std=torch.std(tensor)
        tensor = (tensor-mean) / std
    if tensor.ndimension()==4:
        t_list = []
        for c in range(tensor.size(0)):
            t=tensor[c]
            mean=torch.mean(t)    #calculation of Z-score with some mathematical tricks to avoid overflow
            std=torch.std(t)
            z_scored_t = (t-mean) / std
            t_list.append(z_scored_t)
        tensor = torch.stack(t_list,dim=0)
    return tensor
    
import torch

def pearson_correlation_loss_per_slice(output, target, bg_mask):
    """
    Calculates the Pearson Correlation Loss between the predicted tensor and the target tensor 
    across the `h` and `w` dimensions, and sums the losses across the `b` and `c` dimensions.
    The Pearson loss for each slice is multiplied by (N / 2), where N is the number of non-zero elements 
    in the slice extracted from the background mask.

    Parameters:
    - output (torch.Tensor): The predicted tensor.
        Shape: [c,b, h, w]
    - target (torch.Tensor): The target tensor.
        Shape: c,b, h, w]
    - bg_mask (torch.Tensor): The background mask tensor.
        Shape: [c,b, h, w], where background is 0 and the object is non-zero.

    Returns:
    - Sum of weighted Pearson Correlation Losses across `b` and `c` dimensions.
    """
    # Calculate the mean across the h and w dimensions for each b and c
    mean_x = torch.mean(output, dim=(2, 3), keepdim=True)
    mean_y = torch.mean(target, dim=(2, 3), keepdim=True)
    
    # Subtract the mean from the original tensors
    vx = output - mean_x
    vy = target - mean_y
    
    # Compute the numerator and denominators for Pearson correlation
    numerator = torch.sum(vx * vy, dim=(2, 3))
    
    eps = 1e-4
    denominator = torch.sqrt(torch.sum(vx ** 2, dim=(2, 3)) * torch.sum(vy ** 2, dim=(2, 3)) + eps)
    
    # Calculate the Pearson correlation coefficient for each b and c
    correlation = numerator / denominator
    
    # Compute the Pearson correlation squared and log loss
    if False:
        correlation_squared = correlation ** 2
        logged_corr = torch.log(1 - correlation_squared + eps)
        
        # Calculate the number of non-zero elements (N) in the background mask for each slice (c,b)
        #N = torch.sum(bg_mask, dim=(2, 3))  # Sum over the h and w dimensions, resulting in shape [c,b]
        
        # Multiply the logged correlation by (N / 2)
        #weighted_logged_corr = logged_corr * (N / 2)  
        weighted_logged_corr= logged_corr
        # Sum up the weighted losses across the batch and channel dimensions
        total_loss = torch.mean(weighted_logged_corr)
        
        return total_loss
    else:
        total_loss = 1- torch.mean(correlation)
        
        return total_loss



def scale_tensor_between_zero_and_one(array):
    '''
    Scale the values of a  pytorch tensor between 0 and 1.

    Parameters:
    - array : Input  pytorch tensor to be scaled.

    Returns:
    - torch.Tensor: Scaled tensor with values between 0 and 1.

    WARNING:
    This scale tensor with any size , which means min and max values could potentially be from different slices within one volume. 
    So min,max normalization is done on the entire volume(if input is volume)

    '''
    if array.ndimension()<4:
        array_min = array.min().item()
        array_max = array.max().item()
        scaled_array = (array - array_min) / (array_max - array_min)

    if array.ndimension()==4:
        arrays = []
        for c in range(array.size(0)):
            array_min = array[c].min()
            array_max = array[c].max()
            scaled_array = (array[c] - array_min) / (array_max - array_min)
            arrays.append(scaled_array)
        scaled_array = torch.stack(arrays,dim=0)

            
    return scaled_array
def scale_qmri_channels(qmri, mode):
    """
    Scales channels of a QMRI tensor based on the specified mode.
    
    Args:
        qmri (torch.Tensor): Input tensor of shape [c, d, h, w] where c >= 3.
        mode (str): The mode of scaling; either 'divide' or 'multiply'.
    
    Returns:
        torch.Tensor: Scaled QMRI tensor.
    
    Raises:
        ValueError: If mode is not 'divide' or 'multiply'.
        ValueError: If input tensor does not have at least 3 channels.
    """
    # Define scaling constants
    scaling_factors = {
        0: 100,    # Constant value for channel c=0
        1: 1000,   # Constant value for channel c=1
        2: 1000    # Constant value for channel c=2
    }
    
    # Validate input tensor shape
    if qmri.dim() != 4 or qmri.size(0) < 3:
        raise ValueError("Input tensor must be of shape [c, d, h, w] with at least 3 channels.")
    
    # Validate mode
    if mode not in ["divide", "multiply"]:
        raise ValueError("Mode must be either 'divide' or 'multiply'.")
    
    # Scale the tensor based on the mode, avoiding in-place operations
    scaled_qmri = qmri.clone()  # Clone the tensor to avoid in-place modification
    for channel, scale in scaling_factors.items():
        if mode == "divide":
            scaled_qmri[channel, :, :, :] = scaled_qmri[channel, :, :, :] / scale
        elif mode == "multiply":
            scaled_qmri[channel, :, :, :] = scaled_qmri[channel, :, :, :] * scale
    
    return scaled_qmri




def standardize_nonzero_per_slice_per_channel(tensor):
    """
    Standardizes the tensor values using Z-score normalization, but only for non-zero values,
    independently for each channel and each slice.

    Args:
        tensor (torch.Tensor): Input tensor of shape [c, d, h, w].
    
    Returns:
        torch.Tensor: Standardized tensor where non-zero values are normalized, zeros are left unchanged.
    """
    # Ensure the tensor is a float tensor for precision
    tensor = tensor.float()
    
    # Get the dimensions of the tensor
    c, d, h, w = tensor.shape
    
    # Initialize a tensor to hold the standardized values
    standardized_tensor = tensor.clone()
    
    # Iterate over each channel
    for channel in range(c):
        # Iterate over each slice
        for slice_d in range(d):
            # Extract the slice for the current channel and slice
            slice_data = tensor[channel, slice_d, :, :]
            
            # Mask for non-zero values
            nonzero_mask = slice_data != 0
            
            # Calculate mean and std of non-zero values
            nonzero_values = slice_data[nonzero_mask]
            if nonzero_values.numel() == 0:  # Check if there are no non-zero values
                continue  # Skip if there are no non-zero values
            
            mean = nonzero_values.mean()
            std = nonzero_values.std()
            
            # Avoid division by zero in case std is zero
            if std == 0:
                continue  # Skip standardization if std is zero
            
            # Standardize only non-zero values
            standardized_tensor[channel, slice_d, :, :][nonzero_mask] = (slice_data[nonzero_mask] - mean) / std
    
    return standardized_tensor


def min_max_normalize_per_slice_per_channel(tensor):
    """
    Min-Max normalize a 4D tensor [c, d, h, w] per slice (d) and per channel (c).
    Args:
        tensor (torch.Tensor): Input tensor of shape [c, d, h, w].
    Returns:
        torch.Tensor: Min-Max normalized tensor.
    """
    # Ensure the tensor is a float tensor
    tensor = tensor.float()
    
    # Get the shape of the tensor
    c, d, h, w = tensor.shape
    
    # Prepare an output tensor of the same shape
    normalized_tensor = torch.empty_like(tensor)
    
    # Iterate over each channel and slice
    for channel in range(c):
        for slice_d in range(d):
            # Extract the slice for the current channel and slice_d
            slice_data = tensor[channel, slice_d, :, :]
            
            # Compute min and max for the slice
            min_val = slice_data.min()
            max_val = slice_data.max()
            
            # Avoid division by zero if all values are the same
            if max_val - min_val == 0:
                normalized_tensor[channel, slice_d, :, :] = 0
            else:
                # Normalize the slice
                normalized_tensor[channel, slice_d, :, :] = (slice_data - min_val) / (max_val - min_val)
    
    return normalized_tensor



def synthesize_weighted_image(modality, Qmaps): 
    
    '''
    Synthesizes a weighted image based on the specified modality.

    Parameters:
    - modality (str): The imaging modality, which can be either 'T1W' or 'T2FLAIR'.
    - Qmaps (torch.Tensor): A PyTorch tensor of size [B, C, D, H, W], where:
        - B: Batch size (always 1 to avoid padding, tensors are concatenated over the D dimension).
        - C: Channel dimension, with the following mappings:
            - C=0 corresponds to Proton Density (PD), which will be scaled back by multiplying by 100.
            - C=1 corresponds to T1 in seconds, which will be scaled back by multiplying by 1000.
            - C=2 corresponds to T2 in deci-seconds, which will be scaled back by multiplying by 100.
        - D: Number of slices (dimensions), for more information, check the collate_function.
        - H, W: Height and width dimensions.

    Returns:
    - Synthesized weighted image based on the specified modality.

    Note: The function takes the input Qmaps tensor and performs scaling operations to obtain PD, T1, and T2 components.
    Note: The scanniing seqeunce and scanning params such as TE,TI or TR are assumedd to be known.
    TODO: input based on parametrs
    '''
    epsilon = 10e-15
    PD=Qmaps[0,:,:,:]
    T1=Qmaps[1,:,:,:] + epsilon
    if Qmaps.size(0) == 3:
       T2=Qmaps[2,:,:,:] + epsilon
  

    if modality=="T2FLAIR": #Scanning sequence : spin echo inversion recovery s=k ρ (1-2exp(-TI/T1)+exp(-TR/T1)) exp(-TE/T2) 
        TI=3100.0 
        TE=90.0 
        TR=15000.0      #Repetition between inversion pulses
        signal=PD*(1-2*torch.exp(-TI/(T1))+torch.exp(-(TR)/(T1)))*torch.exp(-TE/(T2))
        return signal

    if modality=="T1W": #scanning sequence is Spin echo :S = k ρ (1-exp(-TR/T1)) exp(-TE/T2)
        TE=10.0
        TR=650.0   #with 4500=tr it is mostly sensitive to PD! chosing TR value between 500-1500! 
        signal = PD*(1 -  torch.exp(-(TR) / (T1)))*torch.exp(-TE /( T2))
        return signal
    else:
        ValueError("Modality is undifined, use T1W OR T2FLAIR. Check spelling or Capital letters!")
    
def synthesize_weighted_image_per_batch(modality, Qmaps):
   #input [b,c,h,w]
    Qmaps_ = Qmaps.permute(1,0,2,3) #c,d,h,w

    signal = synthesize_weighted_image(modality, Qmaps_).permute(1,0,2,3) #b,ch,w
    return signal

def standardize_per_slice(tensor):
    # Calculate mean and standard deviation per slice (D ) for an input tensor of size [D,H,W]
    mean_per_slice = torch.mean(tensor, dim=(1, 2), keepdim=True)
    std_per_slice = torch.std(tensor, dim=(1, 2), keepdim=True)

    # Standardize each slice
    standardized_array = (tensor - mean_per_slice) / std_per_slice

    return standardized_array

import torch
import torch.fft

def lower_resolution_image_4d(image, crop_fraction,sigma):
    """
    Reduce the resolution of a 4D image tensor by keeping only the central part of k-space for each channel and slice.
    
    Parameters:
    - image (torch.Tensor): A 4D tensor representing the image, with shape (c, d, h, w).
    - crop_fraction (float): Fraction of k-space to keep around the center. E.g., 0.5 means keeping 50% of k-space.
    
    Returns:
    - lr_image (torch.Tensor): The lower resolution image after reconstruction, with the same shape as the input.
    """
    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Ensure image is a 4D tensor and move it to the appropriate device
    if image.ndimension() != 4:
        raise ValueError("Input image must be a 4D tensor with shape (c, d, h, w).")
    
    image = image.to(device)

    c, d, h, w = image.shape
    real_noise = torch.normal(mean=0.0, std=sigma, size=(c, d, h, w), device=device)
    imag_noise = torch.normal(mean=0.0, std=sigma, size=(c, d, h, w), device=device)
    noisy_image = image + real_noise + 1j * imag_noise

    # Compute the 2D Fourier Transform for the entire batch
    k_space = torch.fft.fft2(noisy_image, dim=(-2, -1))
    
    # Shift zero frequency component to the center
    k_space_shifted = torch.fft.fftshift(k_space, dim=(-2, -1))
    
    # Create a mask to keep only the central part of k-space
    mask = torch.zeros((c, d, h, w), dtype=torch.bool, device=device)
    #mask = torch.zeros((c, d, h, w), dtype=torch.complex64, device=device)

    
    center_h, center_w = h // 2, w // 2
    crop_h, crop_w = int(center_h * crop_fraction), int(center_w * crop_fraction)
    
    mask[:, :, center_h - crop_h:center_h + crop_h, center_w - crop_w:center_w + crop_w] = True
    
    # Apply the mask to shifted k-space
    k_space_filtered = k_space_shifted * mask
    
    # Shift back before applying inverse FFT
    k_space_filtered_shifted_back = torch.fft.ifftshift(k_space_filtered, dim=(-2, -1))
    
    # Compute the inverse 2D Fourier Transform to get the lower resolution image
    #lr_image = torch.fft.ifft2(k_space_filtered_shifted_back, dim=(-2, -1)).real
    lr_image = torch.abs(torch.fft.ifft2(k_space_filtered_shifted_back, dim=(-2, -1)))
    
    bg_mask = (image != 0)
    masked_lr_image= lr_image*bg_mask
    return masked_lr_image

    
def plot_results(slice_index, target, LRQM, outputs, output, crop_size, plot_diff=False, rotate=True):
    from plot_with_recommendation.relaxationColorMap import relaxationColorMap, create_own_cmap
    import numpy as np
    import matplotlib.pyplot as plt
    import torch
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    # Determine the number of channels to plot
    num_channels = outputs.size(0)
    
    if output == "show":
        dpi = 100  # typical display DPI
    if output == "save":
        dpi = 300
    height, width = crop_size, crop_size  # or data.shape if not cropped
    figsize = (width * (3 + int(plot_diff)) / dpi, height * num_channels / dpi)

    # Create a figure and axis for each subplot
    i = 0
    if plot_diff:
        i += 1
    fig, axs = plt.subplots(num_channels, 3 + i, figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor('black')
    plt.subplots_adjust(wspace=0, hspace=0)  # Remove space between rows and columns

    # Helper function to plot a single image with improved data handling
    def plot_image(ax, data, cmap, vmin, vmax, alpha, show_colorbar=False):
        # Calculate crop parameters based on the actual data dimensions
        total_height, total_width = data.shape[-2], data.shape[-1]
        
        # Ensure crop_size doesn't exceed image dimensions
        actual_crop_size = min(crop_size, total_height, total_width)
        
        crop_start_h = (total_height - actual_crop_size) // 2
        crop_end_h = crop_start_h + actual_crop_size
        crop_start_w = (total_width - actual_crop_size) // 2  
        crop_end_w = crop_start_w + actual_crop_size
        
        # Apply cropping
        cropped_data = data[crop_start_h:crop_end_h, crop_start_w:crop_end_w]
        
        # Ensure we have valid data after cropping
        if cropped_data.numel() == 0:
            print(f"Warning: Empty tensor after cropping. Using full image.")
            cropped_data = data
        
        # Safety check for constant or problematic data
        if cropped_data.numel() > 0:
            data_min, data_max = cropped_data.min().item(), cropped_data.max().item()
            
            # Handle constant data gracefully
            if abs(data_max - data_min) < 1e-6:  # Essentially constant
                if data_min == 0:
                    # All zeros - create small range around zero
                    actual_vmin, actual_vmax = -0.1, 0.1
                else:
                    # Constant non-zero value - create small range around it
                    actual_vmin = data_min * 0.99 if data_min > 0 else data_min * 1.01
                    actual_vmax = data_min * 1.01 if data_min > 0 else data_min * 0.99
            else:
                # Use provided limits, but fall back to data range if they're problematic
                if vmin is not None and vmax is not None and vmin != vmax:
                    actual_vmin, actual_vmax = vmin, vmax
                else:
                    actual_vmin, actual_vmax = data_min, data_max
        else:
            # Fallback for empty data
            actual_vmin, actual_vmax = 0, 1
        
        im = ax.imshow(cropped_data.numpy(), cmap=cmap, vmin=actual_vmin, vmax=actual_vmax, 
                      alpha=alpha, interpolation=None)
        ax.axis('off')
        ax.set_aspect('equal')  # Set aspect ratio to be equal

        if show_colorbar:
            # Create an inset for the color bar
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            cbar = fig.colorbar(im, cax=cax)
            return cbar
        return None

    # Column labels
    column_labels = ['GT', 'Pred', 'LR']
    if plot_diff:
        column_labels.append('DIFF')
    
    # Color bar labels
    color_bar_labels = ['a.u.', 'ms', 'ms']

    # Plot GT, LR, and Pred images for each channel
    for i in range(num_channels):
        if i == 0:
            data_target = target[i, slice_index].detach()
            data_pred = outputs[i, slice_index].detach()
            data_lr = LRQM[i, slice_index].detach()
            loLev = 1.0
            upLev = 160
            relaxation_type = 'T1'
        elif i == 1:
            data_target = target[i, slice_index].detach()
            data_pred = outputs[i, slice_index].detach()
            data_lr = LRQM[i, slice_index].detach()
            loLev = 1.0
            upLev = 3000
            relaxation_type = 'T1'
        elif i == 2:
            data_target = target[i, slice_index].detach()
            data_pred = outputs[i, slice_index].detach()
            data_lr = LRQM[i, slice_index].detach()
            loLev = 1.0
            upLev = 500
            relaxation_type = 'T2'

        # Validate slice_index
        if slice_index >= data_target.shape[0]:
            raise ValueError(f"slice_index {slice_index} is out of bounds for tensor with {data_target.shape[0]} slices")

        # Call the relaxationColorMap function
        imClip_target, cmap_data = relaxationColorMap(relaxation_type, data_target, loLev=loLev, upLev=upLev)
        imClip_pred, cmap_data = relaxationColorMap(relaxation_type, data_pred, loLev=loLev, upLev=upLev)
        imClip_lr, cmap_data = relaxationColorMap(relaxation_type, data_lr, loLev=loLev, upLev=upLev)
      
        cmap = create_own_cmap(cmap_data)
        if i == 0:
            cmap = 'copper'
            loLev = 0

        # Plot Ground Truth, Predictions, and Low Resolution
        if rotate and 'rot_tensor' in globals():
            plot_image(axs[i, 0], rot_tensor(imClip_target), cmap, loLev, upLev, alpha=1, show_colorbar=False)
            plot_image(axs[i, 1], rot_tensor(imClip_pred), cmap, loLev, upLev, alpha=1, show_colorbar=False)
            plot_image(axs[i, 2], rot_tensor(imClip_lr), cmap, loLev, upLev, alpha=1, show_colorbar=False)
            if plot_diff: 
                diff_data = rot_tensor(data_pred - data_target)
                plot_image(axs[i, 3], diff_data, "coolwarm", -upLev/10, upLev/10, alpha=1, show_colorbar=False)
        else:
            # Use data without rotation if rot_tensor is not available or rotate=False
            plot_image(axs[i, 0], imClip_target, cmap, loLev, upLev, alpha=1, show_colorbar=False)
            plot_image(axs[i, 1], imClip_pred, cmap, loLev, upLev, alpha=1, show_colorbar=False)
            plot_image(axs[i, 2], imClip_lr, cmap, loLev, upLev, alpha=1, show_colorbar=False)
            if plot_diff: 
                diff_data = data_pred - data_target
                plot_image(axs[i, 3], diff_data, "coolwarm", -upLev/10, upLev/10, alpha=1, show_colorbar=False)

    # Add column labels
    num_cols = 3 + int(plot_diff)
    for j, label in enumerate(column_labels):
        fig.text((j + 0.5) / num_cols, 1.02, label, ha='center', va='center', fontsize=12, color='white')

    # Add row labels for color bars
    row_label = ["PD", "T1", "T2"]
    for i in range(num_channels):
        fig.text(-0.03, (num_channels - i - 0.5) / num_channels, f'{row_label[i]}', 
                ha='center', va='center', rotation=90, fontsize=12, color='white')

    # Adjust layout to prevent overlap of titles
    plt.tight_layout(pad=0.0, rect=[0, 0, 1, 1])  # Adjust padding to 0

    if output == "show":
        plt.show()
    elif output == "save":
        fig.savefig(f"output_plot_slice_{slice_index}.png", dpi=300, bbox_inches='tight', 
                   pad_inches=0, facecolor='black')
        plt.close(fig)  # Close figure to free memory
    else:
        raise ValueError("Invalid output mode. Choose either 'show' or 'save'.")
    

import os
import pydicom
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, IntSlider
import ipympl
import os
import pydicom
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, IntSlider
import ipympl

# Define your function
def plot_results_interactive(target, LRQM, outputs, single_channel_backgound_mask):
    # Apply the mask and convert to numpy array
    images = (outputs * single_channel_backgound_mask.expand(3, -1, -1, -1)).detach().numpy()
    LRQM =  (LRQM * single_channel_backgound_mask.expand(3, -1, -1, -1)).detach().numpy() 
    target=target.detach().numpy()

    # Helper function to rotate tensor (if needed, adapt accordingly)
    def rot_tensor(tensor):
        return np.rot90(tensor, k=3)  # Adjust rotation as necessary

    # Titles and color ranges for each channel
    titles = ['PD', 'T1', 'T2']
    cmaps = ['rainbow', 'rainbow', 'rainbow']
    vmin_values = [0, 0, 0]
    vmax_values = [160, 4000, 1000]

    # Plot an image and display pixel value on click
    def plot_image(slice_index):
        num_channels = outputs.size(0)

        # Create a figure and axis for each subplot
        fig, axs = plt.subplots(5, num_channels, figsize=(12, 20))
        fig.patch.set_facecolor('black')

        # Helper function to plot a single image
        def plot_single_image(ax, data, title, cmap, vmin, vmax):
            im = ax.imshow(rot_tensor(data), cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(title)
            fig.colorbar(im, ax=ax)
# create an array of all images u wanna show and array of titles and then in below call them
        # Plot GT, LR, Pred, and differences for each channel
        for i in range(num_channels):
            # Plot Ground Truth
            plot_single_image(axs[0, i], target[i][slice_index], f'GT {titles[i]}', cmaps[i], vmin_values[i], vmax_values[i])



            # Plot Predictions
            plot_single_image(axs[1, i], images[i][slice_index], f'Pred {titles[i]}', cmaps[i], vmin_values[i], vmax_values[i])

            # Plot Absolute Difference GT vs LR
            abs_diff_lr = np.abs(target[i][slice_index] - LRQM[i][slice_index])
            plot_single_image(axs[2, i], abs_diff_lr, f'Abs Diff GT vs LR {titles[i]}', cmaps[i], 0, vmax_values[i]/2)

            # Plot Absolute Difference GT vs Pred
            abs_diff_pred = np.abs(target[i][slice_index] - images[i][slice_index])
            plot_single_image(axs[3, i], abs_diff_pred, f'Abs Diff GT vs Pred {titles[i]}', cmaps[i], 0, vmax_values[i]/2)

            # Plot Low Resolution
            plot_single_image(axs[4, i], LRQM[i][slice_index], f'LR {titles[i]}', cmaps[i], vmin_values[i], vmax_values[i])

        # Adjust layout to prevent overlap of titles
        plt.tight_layout()

        # Click event to display pixel value
        def onclick(event):  
            if event.inaxes:  #  isn't x and y swapped ? due to rotate image ?   print out x and y values to make sure with index u take
                x, y = int(event.xdata), int(event.ydata)
                for row in range(5):
                    for col in range(num_channels):
                        ax = axs[row, col]
                        if row == 0:
                            pixel_value = target[col][slice_index][y, x]
                            ax.set_title(f'GT {titles[col]} Pixel value: {pixel_value:.2f}')
                        
                        elif row == 1:
                            pixel_value = images[col][slice_index][y, x]
                            ax.set_title(f'Pred {titles[col]} Pixel value: {pixel_value:.2f}')
                        elif row == 2:
                            pixel_value = np.abs(target[col][slice_index][y, x] - LRQM[col][slice_index][y, x])
                            ax.set_title(f'Abs Diff GT vs LR {titles[col]} Pixel value: {pixel_value:.2f}')
                        elif row == 3:
                            pixel_value = np.abs(target[col][slice_index][y, x] - images[col][slice_index][y, x])
                            ax.set_title(f'Abs Diff GT vs Pred {titles[col]} Pixel value: {pixel_value:.2f}')

                        elif row == 4:
                            pixel_value = LRQM[col][slice_index][y, x]
                            ax.set_title(f'LR {titles[col]} Pixel value: {pixel_value:.2f}')
                fig.canvas.draw()

        fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show()

    # Create a slider to navigate through slices
    interact(plot_image, slice_index=IntSlider(min=0, max=images.shape[1] - 1, step=1, value=0))

# Example usage (you need to provide appropriate tensors for target, LRQM, outputs, and single_channel_backgound_mask)
# plot_results_interactive(target, LRQM, outputs, single_channel_backgound_mask)


def plot_results_horizontal(slice_index, target, LRQM, outputs):
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    
    # Determine the number of channels to plot
    num_channels = outputs.size(0)
    
    # Create a figure and axis for each subplot
    fig, axs = plt.subplots(2, 3, figsize=(18, 8)) if num_channels == 2 else plt.subplots(2, num_channels, figsize=(18, 8))
    
    # Helper function to plot a single image
    def plot_image(ax, data, title, cmap, vmin, vmax):
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        return im

    # Titles and color ranges for each channel
    titles = ['PD', 'T1', 'T2']
    cmaps = ['rainbow', 'rainbow', 'rainbow']
    vmin_values = [0, 0, 0]
    vmax_values = [160, 4000, 1000]

    # Plot GT and LR images for each channel
    for i in range(num_channels):
        # Plot Ground Truth
        im = plot_image(axs[0, i], target[i][slice_index].numpy(), f'GT {titles[i]}', cmaps[i], vmin_values[i], vmax_values[i])
        
        # Plot Low Resolution
        im = plot_image(axs[0, i + num_channels], LRQM[i][slice_index].numpy(), f'LR {titles[i]}', cmaps[i], vmin_values[i], vmax_values[i])
        
        # Plot Predictions
        im = plot_image(axs[1, i], outputs[i][slice_index].detach().numpy(), f'Pred {titles[i]}', cmaps[i], vmin_values[i], vmax_values[i])
        
        # Create a single horizontal colorbar for each type of image
        divider = make_axes_locatable(axs[1, i])
        cax = divider.append_axes("bottom", size="5%", pad=0.3)
        fig.colorbar(im, cax=cax, orientation='horizontal')

    # Adjust layout to prevent overlap of titles
    plt.tight_layout()
    plt.show()


    
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def create_window_3D(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t())
    _3D_window = _1D_window.mm(_2D_window.reshape(1, -1)).reshape(window_size, window_size, window_size).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_3D_window.expand(channel, 1, window_size, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
    
def _ssim_3D(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv3d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv3d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)

    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv3d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv3d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv3d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
    


class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel


        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)
    
    
class SSIM3D(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM3D, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window_3D(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window_3D(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel


        return _ssim_3D(img1, img2, window, self.window_size, channel, self.size_average)

    
def ssim(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)

def ssim3D(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _, _) = img1.size()
    window = create_window_3D(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim_3D(img1, img2, window, window_size, channel, size_average)
    
    
def collate_3d_to_2d(batch): 
    '''Takes a batch<list> of length Batch_size=b B*(C,D,H,W) and returns a batch of size [B*D,C,H,W]
    note that batch[i] is a tuple contating three elements (HRQM,HRWI,LRQM)'''
    """
   
    """
    
    B = len(batch)
   
    # Initialize lists to store concatenated tensors
    concat_tensor_1 = []
    concat_tensor_2 = []

    for i in range(B):
        # Get the two parts from the current batch element
        part_1, part_2 = batch[i]
        # Reshape and concatenate the first part
        concat_tensor_1.append(part_1.permute(1, 0, 2, 3))

        # Reshape and concatenate the second part
        concat_tensor_2.append(part_2.permute(1, 0, 2, 3))

  

    # Concatenate along the batch dimension
    concatenated_tensor_1 = torch.cat(concat_tensor_1, dim=0)
    concatenated_tensor_2 = torch.cat(concat_tensor_2, dim=0)

    return concatenated_tensor_1, concatenated_tensor_2


def sigmoid(input_tensor):
    
    """
    Scale a PyTorch tensor using the sigmoid function.
    
    Args:
    input_tensor (torch.Tensor): Input tensor to be scaled.
    
    Returns:
    scaled_tensor (torch.Tensor): Scaled tensor using sigmoid function.
    """
    return 1 / (1 + torch.exp(-input_tensor))

def reverse_sigmoid(scaled_tensor):
    """
    Reverse the scaling applied using the sigmoid function.
    
    Args:
    scaled_tensor (torch.Tensor): Scaled tensor to be reversed.
    
    Returns:
    original_tensor (torch.Tensor): Original tensor before scaling.
    """
    return -torch.log(1 / scaled_tensor - 1)

    import torch

    
def zscore_non_background(input_tensor, mask_tensor):
    # Ensure the mask is binary (0 or 1)
    mask_tensor = mask_tensor.float()

    # Calculate the mean and std over the non-background (where mask == 1) for each slice along d
    masked_sum = torch.sum(input_tensor * mask_tensor, dim=(1, 2), keepdim=True)
    masked_count = torch.sum(mask_tensor, dim=(1, 2), keepdim=True)

    # Handle the case where there might be no non-background pixels in some slices
    mean = masked_sum / (masked_count + 1e-10)

    # Calculate variance and std (masked std)
    variance = torch.sum(((input_tensor - mean) * mask_tensor) ** 2, dim=(1, 2), keepdim=True) / (masked_count + 1e-10)
    std = torch.sqrt(variance + 1e-10)  # add a small number to avoid division by zero

    # Z-score normalization (subtract mean and divide by std)
    zscored_tensor = (input_tensor - mean) / std

    # Zero out the background in the z-scored tensor
    zscored_tensor = zscored_tensor * mask_tensor

    return zscored_tensor
import pickle
def open_pickle(dir):
    with open(dir, 'rb') as file:
        data = pickle.load(file)
    return data


def plot_images(images, slice_index,title,cmap="gray"):
    """plots image with c channels input is a tensor of size c,d,h,w"""
    fig, axes = plt.subplots(1, images.size(0), figsize=(images.size(0)*3, 3))  # 1 row, 6 columns
    for i in range(images.size(0)):
        axes[i].imshow(images[i, slice_index, :, :], cmap=cmap)  # Display the image
        axes[i].axis('off')  # Turn off the axis
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

import zipfile
import os

def unzip_file(zip_path):
    # Get the directory where the ZIP file is located
    extract_dir = os.path.dirname(zip_path)
    
    # Open the ZIP file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Extract all the contents into the same directory
        zip_ref.extractall(extract_dir)

    print(f"Files extracted to: {extract_dir}")


import os
import pydicom
import torch
import numpy as np
from torchvision import transforms

def load_dicom_series_as_tensor(directory,ends_with_dcm):
    # Get list of files in the directory
    
    
    # Sort the files (assuming filenames represent correct order)
    if ends_with_dcm==True:
        dicom_files = [os.path.join(directory, f) for f in os.listdir(directory) if  f.endswith('.dcm')]
        dicom_files.sort()
    else:
        dicom_files = [os.path.join(directory, f) for f in os.listdir(directory) if not f.startswith('.')]
        dicom_files.sort(key=lambda x: int(x.split('.')[-1]))

    tensors = []
    transform = transforms.Compose([transforms.ToTensor()])  # Transform to convert images to Tensor
    
    for dicom_file in dicom_files:
        # Read DICOM file
        dicom = pydicom.dcmread(dicom_file)
        # Get pixel array
        image_array = dicom.pixel_array.astype(np.int64)
    
        # Convert to PyTorch tensor
        image_tensor = transform(image_array)
        tensors.append(image_tensor)

    # Stack tensors into a single tensor if needed, else return as a list
    tensor_stack = torch.stack(tensors)

    return tensor_stack



def compute_psnr(img1, img2):
    """
    Compute PSNR between two images with an adaptive max pixel value.
    """
    mse = F.mse_loss(img1, img2, reduction='mean')
    if mse == 0:
        return torch.tensor(float('inf'))  # Avoid log(0)
    max_pixel = max(img1.max(), img2.max()).item()
    return 10 * torch.log10(max_pixel ** 2 / mse)

def SPGR(maps, TR=6.616, TE=2.624, alpha=12):
    """
    Compute the SPGR signal intensity for a given input tensor.

    Parameters:
    - maps (torch.Tensor): Input tensor of size (C, D, H, W), where:
        - maps[0, :, :, :] represents M0 (equilibrium magnetization)
        - maps[1, :, :, :] represents T1 (longitudinal relaxation time)
        - maps[2, :, :, :] represents T2* (effective transverse relaxation time)
    - alpha (float or torch.Tensor): Flip angle in degrees.
    - TR (float or torch.Tensor): Repetition time in milliseconds, default value from SPGR BRAVO.
    - TE (float or torch.Tensor): Echo time in milliseconds, default value from SPGR BRAVO.

    Returns:
    - torch.Tensor: Signal intensity tensor of size (D, H, W)
    """
    
    # Ensure input tensor is non-negative and avoid instability
    tensor = torch.relu(maps)

    # Get device from input tensor
    device = tensor.device

    # Ensure alpha, TR, and TE are tensors and on the same device as maps
    TR = TR if isinstance(TR, torch.Tensor) else torch.tensor(TR, dtype=torch.float32, device=device)
    TE = TE if isinstance(TE, torch.Tensor) else torch.tensor(TE, dtype=torch.float32, device=device)
    alpha = alpha if isinstance(alpha, torch.Tensor) else torch.tensor(alpha, dtype=torch.float32, device=device)

    # Extract parameters from the input tensor
    M0 = tensor[0, :, :, :]  # Equilibrium magnetization
    T1 = tensor[1, :, :, :]+0.1 # Longitudinal relaxation time
    T2_star = tensor[2, :, :, :]+0.01  # Effective transverse relaxation time

    # Convert alpha to radians
    alpha_rad = torch.deg2rad(alpha)


    # Compute E1 and E2
    E1 = torch.exp(-TR / T1)
    E3 = torch.exp(-TE / T2_star)

    # Compute the SPGR signal intensity
    numerator = (1 - E1) * torch.sin(alpha_rad)
    denominator = 1 - E1 * torch.cos(alpha_rad)
    signal = M0 * (numerator / denominator) * E3

    return signal
def FSPGR(maps, TR=6.616, TE=2.624, alpha=12, TP=450.0, td=0.0):
    """
    Compute the FSPGR signal intensity after preparation when td=0.
    
    Parameters:
    - maps (torch.Tensor): (3, D, H, W) with [M0, T1, T2*]
    - TR (float): Repetition time in ms
    - TE (float): Echo time in ms
    - alpha (float): Flip angle in degrees
    - TP (float): Time from inversion to acquisition in ms
    - td (float): Delay before inversion in seconds (default 0)
    
    Returns:
    - torch.Tensor: Signal (D, H, W)
    """
    # Ensure non-negativity
    maps = torch.relu(maps)
    device = maps.device

    M0 = maps[0]
    T1 = maps[1] + 0.1  # Avoid div by zero
    T2_star = maps[2] + 0.01

    alpha_rad = torch.deg2rad(torch.tensor(alpha, dtype=torch.float32, device=device))
    TR = torch.tensor(TR, dtype=torch.float32, device=device)
    TE = torch.tensor(TE, dtype=torch.float32, device=device)
    TP = torch.tensor(TP, dtype=torch.float32, device=device)
    td = torch.tensor(td, dtype=torch.float32, device=device)

    # Convert TR, TE from ms to s
    #TR_s = TR / 1000
    #TE_s = TE / 1000

    # Steady-state SPGR longitudinal signal
    E1 = torch.exp(-TR / T1)
    E2 = torch.exp(-TE / T2_star)

    numerator = 1 - E1
    denominator = 1 - E1 * torch.cos(alpha_rad)
    signal_z_eq = M0 * (numerator / denominator) * E2

    # Recovery terms after inversion
    if td.item() == 0.0:
        signal = -signal_z_eq + M0 * (1 - torch.exp(-TP / T1))
    else:
        A = - signal_z_eq * torch.exp(-td / T1)
        B = M0 * (1 - torch.exp(-td / T1)) * torch.exp(-TP / T1)
        C = M0 * (1 - torch.exp(-TP / T1))
        signal = A + B + C

    return signal




import torch

def SpinEcho(maps, TR=5211, TE=146.076):
    """
    Compute the T2-weighted signal intensity for a given input tensor. (T2 PROPELLER)

    Parameters:
    - maps (torch.Tensor): Input tensor of size (C, D, H, W), where:
        - maps[0, :, :, :] represents M0 (equilibrium magnetization)
        - maps[1, :, :, :] represents T1 (longitudinal relaxation time)
        - maps[2, :, :, :] represents T2 (effective transverse relaxation time)
    - TR (float or torch.Tensor): Repetition time in milliseconds, default is from T2 PROPELLER.
    - TE (float or torch.Tensor): Echo time in milliseconds, default is from T2 PROPELLER.

    Returns:
    - torch.Tensor: Signal intensity tensor of size (D, H, W)
    """

    # Ensure maps is non-negative and avoid instability
    tensor = torch.relu(maps)

    # Get device from input tensor
    device = tensor.device

    # Ensure TR and TE are tensors and move them to the correct device
    TR = TR if isinstance(TR, torch.Tensor) else torch.tensor(TR, dtype=torch.float32, device=device)
    TE = TE if isinstance(TE, torch.Tensor) else torch.tensor(TE, dtype=torch.float32, device=device)

    # Extract parameters from the input tensor
    M0 = tensor[0, :, :, :]  # Equilibrium magnetization
    T1 = tensor[1, :, :, :] +0.1 # Longitudinal relaxation time
    T2 = tensor[2, :, :, :]+0.01  # Effective transverse relaxation time

    # Compute the Spin Echo signal intensity
    signal = M0 * (1 - torch.exp(-TR / T1)) * torch.exp(-TE / T2)

    return signal








def InversionRecovery(maps, TR=6000, TE=100.512, TI=1803):
    """
    Compute the Inversion Recovery signal intensity for a given input tensor (Sag CUBE FLAIR).

    Parameters:
    - maps (torch.Tensor): Input tensor of size (C, D, H, W), where:
        - maps[0, :, :, :] represents M0 (equilibrium magnetization)
        - maps[1, :, :, :] represents T1 (longitudinal relaxation time)
        - maps[2, :, :, :] represents T2 (effective transverse relaxation time)
    - TR (float or torch.Tensor): Repetition time in milliseconds, default is from SAG CUBE FLAIR.
    - TE (float or torch.Tensor): Echo time in milliseconds, default is from SAG CUBE FLAIR.
    - TI (float or torch.Tensor): Inversion time in milliseconds, default is from SAG CUBE FLAIR.

    Returns:
    - torch.Tensor: Signal intensity tensor of size (D, H, W)
    """

    # Ensure maps is non-negative and avoid instability
    tensor = torch.relu(maps)

    # Get device from input tensor
    device = tensor.device

    # Ensure TR, TE, and TI are tensors and move them to the correct device
    TR = TR if isinstance(TR, torch.Tensor) else torch.tensor(TR, dtype=torch.float32, device=device)
    TE = TE if isinstance(TE, torch.Tensor) else torch.tensor(TE, dtype=torch.float32, device=device)
    TI = TI if isinstance(TI, torch.Tensor) else torch.tensor(TI, dtype=torch.float32, device=device)

    # Extract parameters from the input tensor
    M0 = tensor[0, :, :, :]  # Equilibrium magnetization
    T1 = tensor[1, :, :, :]+0.001  # Longitudinal relaxation time
    T2 = tensor[2, :, :, :] +0.001 # Effective transverse relaxation time

    # Compute the Inversion Recovery signal intensity
    signal =  (1 - 2 * torch.exp(-TI / T1) + torch.exp(-TR / T1)) * torch.exp(-TE / T2)
    # for some regions , 1 - 2 * torch.exp(-TI / T1) + torch.exp(-TR / T1) may become negative resulting in negative signal intesity, so
    # I simply relu this signal
    signal_relued = torch.relu(signal)

    return signal_relued
def z_score_per_slice(tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute Z-score normalization per slice along the depth (d) dimension of a 3D tensor (d, h, w).
    """
    mean = tensor.mean(dim=(1, 2), keepdim=True)  # Compute mean per slice
    std = tensor.std(dim=(1, 2), keepdim=True)    # Compute std per slice
    std = torch.where(std == 0, torch.ones_like(std), std)
    return (tensor - mean) / std


def SpinEhowithFlipAngle(maps, TR=750, TE=10, flip_angle_deg=90):
    """
    Compute the Spin Echo signal intensity with flip angle effects using the Ernst equation.

    Parameters:
    - maps (torch.Tensor): Input tensor of size (C, D, H, W), where:
        - maps[0, :, :, :] represents M0 (proton density)
        - maps[1, :, :, :] represents T1 (longitudinal relaxation time, ms)
        - maps[2, :, :, :] represents T2 (transverse relaxation time, ms)
    - TR: Repetition time (ms)
    - TE: Echo time (ms)
    - flip_angle_deg: Flip angle in degrees

    Returns:
    - torch.Tensor: Simulated signal of size (D, H, W)
    """

    # Ensure maps is non-negative (ReLU for stability)
    tensor = torch.relu(maps)
    flip_angle_deg = 180-flip_angle_deg

    # Get device
    device = tensor.device

    # Convert TR and TE to tensors on the same device
    TR = torch.tensor(TR, dtype=torch.float32, device=device)
    TE = torch.tensor(TE, dtype=torch.float32, device=device)

    # Extract parameters
    PD = tensor[0, :, :, :]  # M0 / proton density
    T1 = tensor[1, :, :, :] + 0.1  # Avoid division by zero
    T2 = tensor[2, :, :, :] + 0.01

    # Convert flip angle to radians as a tensor on the same device
    alpha_rad = torch.deg2rad(torch.tensor(flip_angle_deg, dtype=torch.float32, device=device))

    # Compute Ernst equation
    E1 = torch.exp(-TR / T1)
    numerator = torch.sin(alpha_rad) * (1 - E1)
    denominator = 1 - torch.cos(alpha_rad) * E1
    Mxy = numerator / (denominator + 1e-8)  # add epsilon for stability

    # Final spin echo signal
    signal = PD * Mxy * torch.exp(-TE / T2)

    return signal
import os
import torch
import pydicom
import numpy as np

def natural_sort_key(path):
    """Sort key that handles numbers in filenames correctly"""
    import re
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', os.path.basename(path))]

def load_series(directory, keyword):
    # First try filtering by keyword in filename
    candidates = [os.path.join(directory, f) for f in os.listdir(directory)
                  if f.lower().endswith('.dcm') and keyword.lower() in f.lower()]
    
    if not candidates:
        # Fallback: look into SeriesDescription if keyword not in filename
        for f in os.listdir(directory):
            if f.lower().endswith('.dcm'):
                path = os.path.join(directory, f)
                try:
                    dcm = pydicom.dcmread(path, stop_before_pixels=True)
                    if keyword.lower() in str(dcm.SeriesDescription).lower():
                        candidates.append(path)
                except Exception:
                    continue

    if not candidates:
        raise ValueError(f"No DICOM files found for keyword '{keyword}' in '{directory}'.")

    # Read and collect with InstanceNumber
    slices = []
    for path in candidates:
        try:
            dcm = pydicom.dcmread(path)
            instance_number = int(getattr(dcm, "InstanceNumber", 0))
            slices.append((instance_number, dcm.pixel_array))
        except Exception as e:
            print(f"Warning: Failed to read {path}: {e}")
    
    if not slices:
        raise ValueError(f"No readable DICOM slices found for keyword '{keyword}'.")

    # Sort by InstanceNumber to preserve anatomical order
    slices.sort(key=lambda x: x[0])
    volume = np.stack([s[1] for s in slices], axis=0)  # shape: [D, H, W]

    # Cast to float32 for PyTorch compatibility
    return torch.from_numpy(volume.astype(np.float32))


def load_magic_maps_as_tensor(directory):
    pd_tensor = load_series(directory, "PD")
    t1_tensor = load_series(directory, "T1")
    t2_tensor = load_series(directory, "T2")

    # Sanity check: all must have the same shape
    if not (pd_tensor.shape == t1_tensor.shape == t2_tensor.shape):
        raise ValueError("PD, T1, and T2 volumes must have the same shape.")

    stacked = torch.stack([pd_tensor, t1_tensor, t2_tensor], dim=0)  # shape: [3, D, H, W]
    return stacked


def clip_tensor(tensor):
    # Check if the tensor is 4D and has 3 channels (C = 3)
    if tensor.ndimension() == 4 and tensor.shape[1] == 3:
        # Define the clipping values for each channel
        clip_vals = {0: 160, 1: 4300, 2: 2000}
        
        # Iterate over channels and apply clipping
        for c in range(3):
            # Clip the values for each channel
            tensor[:, c, :, :] = torch.clamp(tensor[:, c, :, :], max=clip_vals[c])
            
        return tensor
    else:
        raise ValueError("Input tensor must be 4D with 3 channels (C=3)")
    





def plot_results_with_roi(slice_index, target, LRQM, outputs, output, crop_size, plot_diff=False, rotate=True, ROI=None, zoom_factor=2, offset=(0,0), interactive=False, show_qMRI_metrics=False, show_weighted_image_metrics=None, weighted_images=None):
    """
    Visualize quantitative MRI results with optional region of interest (ROI) zoom and difference maps.
    Now includes HFEN, SSIM, and PSNR metrics computation and display on each image.
    
    This function plots side-by-side comparisons of ground truth, model predictions, and low-resolution
    quantitative maps for a specified slice and multiple parameter channels (e.g., PD, T1, T2). It supports
    optional display of difference maps and inset or side-by-side zoom-ins of a specified ROI. The figure
    layout and resolution adapt based on the ROI size and output mode (show or save).
    
    Parameters:
    -----------
    slice_index : int
        Index of the slice to visualize from the 4D tensors (channels, slices, height, width).
        If interactive=True, this serves as the initial slice to display.
    
    target : torch.Tensor
        Ground truth quantitative maps tensor with shape [C, S, H, W].
    
    LRQM : torch.Tensor
        Low-resolution quantitative maps tensor with shape [C, S, H, W].
    
    outputs : torch.Tensor
        Model-predicted quantitative maps tensor with shape [C, S, H, W].
    
    output : str
        Output mode, either "show" to display the plot interactively or "save" to save the figure to disk.
        Note: Interactive mode (interactive=True) only works with output="show".
    
    crop_size : int
        Size of the square crop extracted from the center of the images for display. The function will
        center-crop all images to this size for visualization. If crop_size is larger than or equal to
        the actual image size, the full image will be displayed without cropping. This helps focus on
        the central region and standardize the display size across different image dimensions.
        Example: crop_size=300 will show a 300x300 pixel region from the center of each image.
    
    plot_diff : bool, optional (default=False)
        Whether to include a difference map (prediction - ground truth) in the visualization.
        The difference map uses a 'coolwarm' colormap with symmetric limits around zero.
    
    rotate : bool, optional (default=True)
        Whether to rotate images 90 degrees counterclockwise for display orientation.
        This is commonly needed to match standard medical imaging display conventions.
    
    ROI : tuple or None, optional (default=None)
        Region of interest specified as (x, y, width, height) in pixel coordinates relative to the
        original (uncropped) image. If provided, the function will:
        - Draw a yellow rectangle around the ROI on ground truth and predicted images
        - Show either an inset zoom (for small ROIs < 100x100 pixels) or side-by-side zoom view
        
        The ROI coordinates are:
        - x: left edge position (pixels from left)  
        - y: top edge position (pixels from top)
        - width: ROI width in pixels
        - height: ROI height in pixels
        
        Example: ROI=(250, 200, 85, 40) defines a region starting at (250,200) with size 85x40 pixels.
    
    zoom_factor : float, optional (default=2)
        Magnification factor for the inset zoom of the ROI when the ROI is small (< 100x100 pixels).
        This controls how large the inset appears within each image axis. Higher values create larger
        insets but may overlap with the main image content. The inset size is calculated as:
        inset_size = base_size (0.3) * zoom_factor, capped at 90% of the axis size.
        
        Examples:
        - zoom_factor=1: Small inset (30% of axis)
        - zoom_factor=2: Medium inset (60% of axis) 
        - zoom_factor=3: Large inset (90% of axis, maximum)
    
    offset : tuple of floats, optional (default=(0, 0))
        (x, y) offsets in pixels controlling which part of the ROI region to extract and display.
        This parameter only affects small ROI insets, not side-by-side ROI displays.
        
        The offset allows you to shift the ROI sampling window within the defined ROI bounds:
        - offset=(0, 0): Extract ROI starting from the top-left corner of the defined ROI
        - offset=(10, 5): Extract ROI starting 10 pixels right and 5 pixels down from ROI top-left
        
        Note: The inset is always positioned at the bottom-left corner of each displayed image,
        regardless of the offset value. The offset only controls what part of the ROI is shown
        in that bottom-left inset.
        
        Examples:
        - offset=(0, 0): Show ROI from its defined top-left corner in bottom-left inset
        - offset=(20, 10): Show ROI shifted by (20,10) pixels in bottom-left inset
    
    interactive : bool, optional (default=False)
        Enable interactive slice scrolling in Jupyter notebooks using ipywidgets. When True:
        - Creates a slider widget to scroll through all available slices
        - Updates the plot in real-time as you move the slider
        - Only works with output="show" and in Jupyter notebook environments
        - Automatically detects the number of available slices from the tensor dimensions
        - Ignores the slice_index parameter and uses it only as the initial display slice
        
        Requires: pip install ipywidgets
        
        Interactive mode features:
        - Smooth slice navigation with immediate visual feedback
        - Metrics update in real-time for each slice (if show_metrics=True)
        - All other parameters (ROI, crop_size, etc.) remain active
        - Keyboard shortcuts: Arrow keys or clicking slider
    
    show_qMRI_metrics : bool, optional (default=False)
        Control whether to display metrics (HFEN, SSIM, PSNR) on the qMRI images (PD, T1, T2 rows):
        - True: Shows metrics in bold white text on top-right corner of Predicted and LR images
        - False: qMRI images display without any metric annotations
        
        When show_qMRI_metrics=True:
        - Predicted images show metrics compared to ground truth
        - LR images show baseline metrics compared to ground truth
        - Ground truth and difference images never show metrics regardless of this setting
        - Metrics are still computed and printed to console regardless of this setting
    
    show_weighted_image_metrics : bool or None, optional (default=None)
        Control whether to display metrics on the weighted image rows:
        - True: Shows metrics computed per row using first column as ground truth
        - False: Weighted image rows display without metric annotations
        - None: Defaults to True if weighted_images is provided, False otherwise
        
        When show_weighted_image_metrics=True:
        - First column in each weighted image row serves as ground truth
        - Metrics are computed between ground truth and columns 2, 3, 4 in that row
        - Results displayed on columns 2, 3, 4 respectively
    
    weighted_images : torch.Tensor or None, optional (default=None)
        Additional tensor with shape [n, 4, D, H, W] containing weighted images to display.
        If provided, adds n additional rows at the bottom of the figure, with each row showing
        4 images from the corresponding weighted_images[i, :, slice_index, :, :].
        The slice_index parameter is used to select the slice along the D dimension.
        
        IMPORTANT: Weighted images will be normalized using z-score normalization (mean=0, std=1) 
        before metrics computation and visualization for consistent comparison and display.
        
        Layout when weighted_images is provided:
        - Rows 1-3: Standard PD/T1/T2 with empty first column, then target/pred/LR
        - Rows 4 to (3+n): weighted_images[0-n, 0-3] for the selected slice
    
    ROI Display Logic:
    ------------------
    The function determines ROI display mode based on ROI size:
    
    1. **Small ROI** (width < 100 AND height < 100 pixels):
       - Shows as inset overlay on main images
       - Inset size controlled by zoom_factor  
       - Inset position controlled by offset parameter
       - Uses 1 column per image type
    
    2. **Large ROI** (width >= 100 OR height >= 100 pixels):
       - Shows as separate side-by-side images
       - Uses 2 columns per image type (main + ROI)
       - offset parameter is ignored
       - ROI images show the exact ROI region without additional magnification
    
    3. **No ROI** (ROI=None):
       - Standard display without ROI visualization
       - Uses 1 column per image type
    
    Metrics Display:
    ----------------
    For each channel (PD, T1, T2), the function computes and displays:
    - **HFEN** (High Frequency Error Norm): Measures preservation of high-frequency details
    - **SSIM** (Structural Similarity Index): Measures structural similarity (0-1, higher is better)  
    
    Metrics are always computed and printed to the console. When show_metrics=True, they are also
    displayed in bold white text on the top-right corner of:
    - Predicted images: metrics vs ground truth
    - LR images: metrics vs ground truth (baseline comparison)
    - Ground truth and difference images: no metrics displayed regardless of show_metrics setting
    
    Example usage:
    --------------
    # Basic usage with weighted images
    plot_results_with_roi(slice_index=5, target=gt_tensor, LRQM=lr_tensor, 
                         outputs=pred_tensor, output="show", crop_size=300,
                         weighted_images=weight_tensor)  # shape [n, 4, D, H, W]
    
    # Interactive mode with weighted images
    plot_results_with_roi(slice_index=5, target=gt_tensor, LRQM=lr_tensor, 
                         outputs=pred_tensor, output="show", crop_size=300, 
                         interactive=True, weighted_images=weight_tensor)
    """
    
    # Import dependencies with error handling
    try:
        from plot_with_recommendation.relaxationColorMap import relaxationColorMap, create_own_cmap
    except ImportError:
        print("Warning: relaxationColorMap not found. Using default colormaps.")
        relaxationColorMap = None
        create_own_cmap = None
    
    import numpy as np
    import matplotlib.pyplot as plt
    import torch
    import matplotlib.patches as patches
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import peak_signal_noise_ratio as psnr
    import torch.nn.functional as F
    
    # Interactive mode imports
    if interactive:
        try:
            from ipywidgets import interact, IntSlider, Layout
            from IPython.display import display, clear_output
            import ipywidgets as widgets
        except ImportError:
            print("Warning: ipywidgets not found. Install with 'pip install ipywidgets' for interactive mode.")
            print("Falling back to non-interactive mode.")
            interactive = False
    
    # Input validation
    if output not in ["show", "save"]:
        raise ValueError("Invalid output mode. Choose either 'show' or 'save'.")
    
    if interactive and output != "show":
        print("Warning: Interactive mode only works with output='show'. Setting output to 'show'.")
        output = "show"
    
    if not isinstance(target, torch.Tensor) or not isinstance(LRQM, torch.Tensor) or not isinstance(outputs, torch.Tensor):
        raise TypeError("target, LRQM, and outputs must be torch.Tensor objects")
    
    if target.dim() != 4 or LRQM.dim() != 4 or outputs.dim() != 4:
        raise ValueError("Input tensors must be 4D (channels, slices, height, width)")
    
    if ROI is not None and len(ROI) != 4:
        raise ValueError("ROI must be a tuple of 4 values (x, y, width, height)")
    
    # Validate weighted_images if provided
    if weighted_images is not None:
        if not isinstance(weighted_images, torch.Tensor):
            raise TypeError("weighted_images must be a torch.Tensor object")
        if weighted_images.dim() != 5:
            raise ValueError("weighted_images must be 5D (n, 4, D, H, W)")
        if weighted_images.size(2) <= slice_index:
            raise ValueError(f"slice_index ({slice_index}) exceeds weighted_images depth dimension ({weighted_images.size(2)})")
    
    # Set default for show_weighted_image_metrics
    if show_weighted_image_metrics is None:
        show_weighted_image_metrics = weighted_images is not None

    # Get tensor dimensions
    num_channels = outputs.size(0)
    num_slices = outputs.size(1)
    total_size = outputs.size(2)
    
    # Get weighted images dimensions
    num_weighted_rows = 0
    if weighted_images is not None:
        num_weighted_rows = weighted_images.size(0)
    
    # Validate slice_index
    if slice_index < 0 or slice_index >= num_slices:
        slice_index = min(max(0, slice_index), num_slices - 1)
        print(f"Warning: slice_index adjusted to valid range [0, {num_slices-1}]. Using slice {slice_index}.")
    
    # If interactive mode is requested, create the interactive widget
    if interactive:
        def plot_slice(slice_idx):
            """Internal function to plot a specific slice"""
            clear_output(wait=True)
            plot_results_with_roi(
                slice_index=slice_idx, 
                target=target, 
                LRQM=LRQM, 
                outputs=outputs, 
                output="show", 
                crop_size=crop_size,
                plot_diff=plot_diff, 
                rotate=rotate, 
                ROI=ROI, 
                zoom_factor=zoom_factor, 
                offset=offset,
                interactive=False,  # Avoid recursion
                show_qMRI_metrics=show_qMRI_metrics,  # Pass through metric flags
                show_weighted_image_metrics=show_weighted_image_metrics,
                weighted_images=weighted_images  # Pass through weighted_images
            )
        
        # Create slider widget
        slice_slider = IntSlider(
            value=slice_index,
            min=0,
            max=num_slices - 1,
            step=1,
            description=f'Slice:',
            style={'description_width': 'initial'},
            layout=Layout(width='80%'),
            continuous_update=True
        )
        
        print(f"Interactive mode: Scroll through {num_slices} slices")
        print("Use the slider below or arrow keys to navigate slices")
        
        # Create interactive widget
        interactive_plot = interact(plot_slice, slice_idx=slice_slider)
        return interactive_plot
    
    # Continue with regular (non-interactive) plotting...

    def rot_tensor(t):
        """Rotate tensor 90 degrees counterclockwise"""
        return torch.rot90(t, k=1, dims=[0, 1])

    def normalize_tensor(tensor):
        """Normalize tensor using z-score normalization (mean=0, std=1)"""
        tensor_mean = tensor.mean()
        tensor_std = tensor.std()
        if tensor_std > 0:
            return (tensor - tensor_mean) / tensor_std
        else:
            return tensor - tensor_mean  # If std=0, just center at mean=0

    def compute_hfen(img1, img2):
        """
        Compute High Frequency Error Norm (HFEN) between two images.
        HFEN = ||LoG(img1) - LoG(img2)||_F / ||LoG(img1)||_F
        where LoG is the Laplacian of Gaussian filter.
        """
        try:
            # Convert to numpy if tensors
            if isinstance(img1, torch.Tensor):
                img1_np = img1.detach().cpu().numpy().astype(np.float64)
            else:
                img1_np = img1.astype(np.float64)
            
            if isinstance(img2, torch.Tensor):
                img2_np = img2.detach().cpu().numpy().astype(np.float64)
            else:
                img2_np = img2.astype(np.float64)
            
            # Apply Laplacian of Gaussian filter
            from scipy import ndimage
            from scipy.ndimage import gaussian_laplace
            
            # Use sigma=1.5 for LoG filter (common choice)
            log_img1 = gaussian_laplace(img1_np, sigma=1.5)
            log_img2 = gaussian_laplace(img2_np, sigma=1.5)
            
            # Compute HFEN
            numerator = np.linalg.norm(log_img1 - log_img2, 'fro')
            denominator = np.linalg.norm(log_img1, 'fro')
            
            # Avoid division by zero
            if denominator == 0:
                return 0.0
            
            hfen = numerator / denominator
            return hfen
        except Exception as e:
            print(f"Error computing HFEN: {e}")
            return 0.0

    def compute_metrics(img_gt, img_pred, img_lr, data_range):
        """
        Compute HFEN and SSIM metrics for predicted and LR images against ground truth.
        """
        metrics = {}
        
        try:
            # Convert tensors to numpy arrays
            if isinstance(img_gt, torch.Tensor):
                gt_np = img_gt.detach().cpu().numpy().astype(np.float64)
            else:
                gt_np = img_gt.astype(np.float64)
                
            if isinstance(img_pred, torch.Tensor):
                pred_np = img_pred.detach().cpu().numpy().astype(np.float64)
            else:
                pred_np = img_pred.astype(np.float64)
                
            if isinstance(img_lr, torch.Tensor):
                lr_np = img_lr.detach().cpu().numpy().astype(np.float64)
            else:
                lr_np = img_lr.astype(np.float64)
            
            # Ensure images have the same shape
            min_h = min(gt_np.shape[0], pred_np.shape[0], lr_np.shape[0])
            min_w = min(gt_np.shape[1], pred_np.shape[1], lr_np.shape[1])
            
            gt_np = gt_np[:min_h, :min_w]
            pred_np = pred_np[:min_h, :min_w]
            lr_np = lr_np[:min_h, :min_w]
            
            # Compute metrics for predicted vs ground truth
            metrics['pred_hfen'] = compute_hfen(gt_np, pred_np)
            metrics['pred_ssim'] = ssim(gt_np, pred_np, data_range=data_range)
            
            # Compute metrics for LR vs ground truth
            metrics['lr_hfen'] = compute_hfen(gt_np, lr_np)
            metrics['lr_ssim'] = ssim(gt_np, lr_np, data_range=data_range)
            
        except Exception as e:
            print(f"Error computing metrics: {e}")
            # Return default values if computation fails
            metrics = {
                'pred_hfen': 0.0, 'pred_ssim': 0.0,
                'lr_hfen': 0.0, 'lr_ssim': 0.0
            }
        
        return metrics

    # Configuration
    total_rows = num_channels + num_weighted_rows  # Total rows including weighted images
    
    # Safe cropping logic
    if crop_size >= total_size:
        # If crop_size is larger than image, use full image
        crop_start = 0
        crop_end = total_size
        actual_crop_size = total_size
        print(f"Warning: crop_size ({crop_size}) >= image_size ({total_size}). Using full image.")
    else:
        # Center crop
        crop_start = (total_size - crop_size) // 2
        crop_end = crop_start + crop_size
        actual_crop_size = crop_size
    
    print(f"Image size: {total_size}x{total_size}, Crop: {crop_start}:{crop_end} (size: {actual_crop_size})")

    # Compute metrics for each channel (only if show_qMRI_metrics is True)
    channel_metrics = []
    if show_qMRI_metrics:
        print("\nComputing qMRI metrics...")
        channel_names = ["PD", "T1", "T2"]
        
        for i in range(num_channels):
            # Get data for current channel
            data_target = target[i, slice_index].detach().cpu()
            data_pred = outputs[i, slice_index].detach().cpu()
            data_lr = LRQM[i, slice_index].detach().cpu()
            
            # Set data range for each channel
            if i == 0:  # PD
                data_range = 160.0
            elif i == 1:  # T1
                data_range = 2000.0
            elif i == 2:  # T2
                data_range = 500.0
            else:
                data_range = 1.0
            
            # Compute metrics
            metrics = compute_metrics(data_target, data_pred, data_lr, data_range)
            channel_metrics.append(metrics)
            
            # Print metrics
            channel_name = channel_names[i] if i < len(channel_names) else f"Ch{i}"
            print(f"\n{channel_name} Channel Metrics:")
            print(f"  Predicted vs GT - HFEN: {metrics['pred_hfen']:.4f}, SSIM: {metrics['pred_ssim']:.4f}")
            print(f"  LR vs GT       - HFEN: {metrics['lr_hfen']:.4f}, SSIM: {metrics['lr_ssim']:.4f}")
    
    # Compute metrics for weighted images (only if show_weighted_image_metrics is True)
    weighted_metrics = []
    if show_weighted_image_metrics and weighted_images is not None:
        print(f"\nComputing weighted image metrics...")
        
        for w_row in range(num_weighted_rows):
            # Get ground truth (first column) for this weighted image row and normalize it
            gt_weighted_raw = weighted_images[w_row, 0, slice_index].detach().cpu()
            gt_weighted = normalize_tensor(gt_weighted_raw)
            
            # Set data range based on the ground truth image statistics (z-score normalized)
            # For z-score normalized data, we use a reasonable range around the mean
            data_range = 6.0  # Covers ~99.7% of data (±3 standard deviations)
            
            row_metrics = []
            # Compute metrics for columns 1, 2, 3 against column 0 (ground truth)
            for w_col in range(1, 4):
                pred_weighted_raw = weighted_images[w_row, w_col, slice_index].detach().cpu()
                pred_weighted = normalize_tensor(pred_weighted_raw)
                
                # Compute metrics using normalized ground truth as reference
                metrics = {}
                try:
                    # Convert to numpy
                    gt_np = gt_weighted.detach().cpu().numpy().astype(np.float64)
                    pred_np = pred_weighted.detach().cpu().numpy().astype(np.float64)
                    
                    # Ensure same shape
                    min_h = min(gt_np.shape[0], pred_np.shape[0])
                    min_w = min(gt_np.shape[1], pred_np.shape[1])
                    gt_np = gt_np[:min_h, :min_w]
                    pred_np = pred_np[:min_h, :min_w]
                    
                    # Compute metrics
                    metrics['hfen'] = compute_hfen(gt_np, pred_np)
                    metrics['ssim'] = ssim(gt_np, pred_np, data_range=data_range)
                    
                except Exception as e:
                    print(f"Error computing metrics for weighted image [{w_row}, {w_col}]: {e}")
                    metrics = {'hfen': 0.0, 'ssim': 0.0}
                
                row_metrics.append(metrics)
                
                # Print metrics
                print(f"  W{w_row} Col{w_col} vs GT - HFEN: {metrics['hfen']:.4f}, SSIM: {metrics['ssim']:.4f}")
            
            weighted_metrics.append(row_metrics)

    # DPI and figure configuration
    dpi = 300 if output == "save" else 300
    
    roi_is_small = ROI is not None and ROI[2] < 100 and ROI[3] < 100
    cols_per_image = 1 if roi_is_small else (2 if ROI is not None else 1)
    
    # Calculate total columns based on weighted_images presence
    if weighted_images is not None:
        # With weighted images: empty column + 3 main columns (or 4 with diff)
        num_main_cols = 4 + int(plot_diff)  # 1 empty + 3 main + optional diff
    else:
        # Without weighted images: just the main columns
        num_main_cols = 3 + int(plot_diff)
    
    total_cols = num_main_cols * cols_per_image
    
    # Increase figure height to accommodate metrics text and weighted images
    base_height = actual_crop_size * total_rows / dpi
    figsize = (actual_crop_size * total_cols / dpi, base_height)

    fig, axs = plt.subplots(total_rows, total_cols, figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor('black')
    
    # Ensure axs is always 2D array
    if total_rows == 1:
        axs = axs.reshape(1, -1)
    elif total_cols == 1:
        axs = axs.reshape(-1, 1)
    
    plt.subplots_adjust(wspace=0, hspace=0)  # Standard spacing

    def plot_image(ax, data, cmap, vmin, vmax, alpha=1.0, draw_rect=False, metrics_text=None):
        """Plot image with optional ROI rectangle and metrics text"""
        try:
            # Safe cropping with bounds checking
            h, w = data.shape[-2:]  # Get actual data dimensions
            
            # Ensure crop bounds are valid
            safe_crop_start = max(0, min(crop_start, h - 1, w - 1))
            safe_crop_end_h = min(h, crop_end)
            safe_crop_end_w = min(w, crop_end)
            
            # Extract crop region
            data_crop = data[safe_crop_start:safe_crop_end_h, safe_crop_start:safe_crop_end_w]
            
            # Handle empty crops
            if data_crop.numel() == 0:
                print(f"Warning: Empty crop region. Using full image.")
                data_crop = data
            
            # Convert to numpy if it's a tensor
            if isinstance(data_crop, torch.Tensor):
                data_np = data_crop.detach().cpu().numpy()
            else:
                data_np = data_crop
            
            im = ax.imshow(data_np, cmap=cmap, vmin=vmin, vmax=vmax, alpha=alpha, interpolation=None)
            ax.axis('off')
            ax.set_aspect('equal')

            # Add metrics text to top right corner
            if metrics_text is not None:
                # Check if we should display based on metric type and flags
                should_display = False
                if 'qMRI_' in str(metrics_text) and show_qMRI_metrics:
                    should_display = True
                    metrics_text = metrics_text.replace('qMRI_', '')  # Remove flag
                elif 'weighted_' in str(metrics_text) and show_weighted_image_metrics:
                    should_display = True
                    metrics_text = metrics_text.replace('weighted_', '')  # Remove flag
                
                if should_display:
                    ax.text(0.98, 0.98, metrics_text, transform=ax.transAxes, 
                           fontsize=5, color='white', ha='right', va='top', weight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7,edgecolor = 'none'))

            if draw_rect and ROI is not None:
                x, y, w, h = ROI
                # Adjust ROI coordinates for cropped view
                rect_x = x - safe_crop_start
                rect_y = y - safe_crop_start
                
                # Only draw rectangle if it's visible in the crop
                if (rect_x + w > 0 and rect_y + h > 0 and 
                    rect_x < data_crop.shape[1] and rect_y < data_crop.shape[0]):
                    
                    # Clip rectangle to crop boundaries
                    rect_x = max(0, rect_x)
                    rect_y = max(0, rect_y)
                    rect_w = min(w, data_crop.shape[1] - rect_x)
                    rect_h = min(h, data_crop.shape[0] - rect_y)
                    
                    if rect_w > 0 and rect_h > 0:
                        rect = patches.Rectangle((rect_x, rect_y), rect_w, rect_h, 
                                               linewidth=0.5, edgecolor='yellow', facecolor='none')
                        ax.add_patch(rect)
            return im
        except Exception as e:
            print(f"Error in plot_image: {e}")
            print(f"Data shape: {data.shape}, Crop: {crop_start}:{crop_end}")
            ax.text(0.5, 0.5, 'Error', ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
            return None
    
    def plot_empty_axis(ax):
        """Plot an empty axis (black background, no content)"""
        ax.set_facecolor('black')
        ax.axis('off')
        
    def plot_roi_inset(ax, data, cmap, vmin, vmax, alpha=1.0):
        """Plot ROI as inset always positioned at bottom-left of the image"""
        if ROI is None:
            return
            
        try:
            x, y, w, h = ROI
            
            # Apply offset to ROI coordinates (offset controls what part of ROI to show)
            roi_x = x + offset[0]
            roi_y = y + offset[1]
            
            # Ensure ROI bounds are within image after offset
            roi_x = max(0, min(roi_x, total_size - w))
            roi_y = max(0, min(roi_y, total_size - h))
            roi_w = min(w, total_size - roi_x)
            roi_h = min(h, total_size - roi_y)
            
            # Extract ROI region
            roi_crop = data[int(roi_y):int(roi_y + roi_h), int(roi_x):int(roi_x + roi_w)]
            
            if isinstance(roi_crop, torch.Tensor):
                roi_np = roi_crop.detach().cpu().numpy()
            else:
                roi_np = roi_crop
    
            # Calculate inset size based on zoom_factor
            base_size = 0.25  # Base size for inset (25% of axis)
            inset_size = min(base_size * zoom_factor, 0.9)  # Cap at 90%
    
            # Always position inset at true bottom-left corner
            inset_left = 0.02   # 2% from left edge
            inset_bottom = 0.02  # 2% from bottom edge
            
            # Create inset axes with absolute positioning (left, bottom, width, height)
            axin = ax.inset_axes([inset_left, inset_bottom, inset_size, inset_size])
    
            axin.imshow(roi_np, cmap=cmap, vmin=vmin, vmax=vmax, alpha=alpha, interpolation=None)
            axin.axis('off')
            axin.set_aspect('equal')
            
            # Add a thin yellow border rectangle around the inset to match ROI rectangle color
            # Use image coordinates that span the full displayed area
            h, w = roi_np.shape
            border_rect = patches.Rectangle((-0.5, -0.5), w, h, 
                                          linewidth=1, edgecolor='yellow', facecolor='none')
            axin.add_patch(border_rect)
                
        except Exception as e:
            print(f"Error in plot_roi_inset: {e}")

    def plot_roi_side(ax, data, cmap, vmin, vmax, alpha=1.0):
        """Plot ROI in separate axis"""
        if ROI is None:
            return
            
        try:
            x, y, w, h = ROI
            # Ensure ROI bounds are within image
            x = max(0, min(x, total_size - w))
            y = max(0, min(y, total_size - h))
            w = min(w, total_size - x)
            h = min(h, total_size - y)
            
            roi_crop = data[y:y + h, x:x + w]
            
            if isinstance(roi_crop, torch.Tensor):
                roi_np = roi_crop.detach().cpu().numpy()
            else:
                roi_np = roi_crop
                
            ax.imshow(roi_np, cmap=cmap, vmin=vmin, vmax=vmax, alpha=alpha, interpolation=None)
            ax.axis('off')
            ax.set_aspect('equal')
        except Exception as e:
            print(f"Error in plot_roi_side: {e}")
            ax.text(0.5, 0.5, 'ROI Error', ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')

    # Column and row labels
    if weighted_images is not None:
        column_labels = ['', 'GT', 'Pred']  # Empty first column for weighted images layout
        if plot_diff:
            column_labels.append('DIFF')
        column_labels.append('LR')
    else:
        column_labels = ['GT', 'Pred']
        if plot_diff:
            column_labels.append('DIFF')
        column_labels.append('LR')

    row_labels = ["PD", "T1", "T2"]
    
    # Add labels for weighted image rows
    if weighted_images is not None:
        for i in range(num_weighted_rows):
            row_labels.append(f"W{i}")

    # Main plotting loop - First plot the standard PD/T1/T2 rows
    for i in range(num_channels):
        # Set parameters for each channel
        if i == 0:  # PD
            loLev, upLev = 0.0, 160
            relaxation_type = 'T1'
            default_cmap = 'copper'
        elif i == 1:  # T1
            loLev, upLev = 1.0, 2000
            relaxation_type = 'T1'
            default_cmap = 'viridis'
        elif i == 2:  # T2
            loLev, upLev = 1.0, 500
            relaxation_type = 'T2'
            default_cmap = 'plasma'
        else:
            loLev, upLev = 0.0, 1.0
            relaxation_type = 'T1'
            default_cmap = 'gray'

        # Get data for current channel
        data_target = target[i, slice_index].detach().cpu()
        data_pred = outputs[i, slice_index].detach().cpu()
        data_lr = LRQM[i, slice_index].detach().cpu()

        # Apply custom colormap if available
        if relaxationColorMap is not None and create_own_cmap is not None:
            try:
                imClip_target, cmap_data = relaxationColorMap(relaxation_type, data_target, loLev=loLev, upLev=upLev)
                imClip_pred, _ = relaxationColorMap(relaxation_type, data_pred, loLev=loLev, upLev=upLev)
                imClip_lr, _ = relaxationColorMap(relaxation_type, data_lr, loLev=loLev, upLev=upLev)
                cmap = create_own_cmap(cmap_data)
            except Exception as e:
                print(f"Warning: Error with custom colormap for channel {i}: {e}. Using default.")
                imClip_target, imClip_pred, imClip_lr = data_target, data_pred, data_lr
                cmap = default_cmap
        else:
            imClip_target, imClip_pred, imClip_lr = data_target, data_pred, data_lr
            cmap = default_cmap

        # Override for PD channel
        if i == 0:
            cmap = 'copper'
            loLev = 0

        # Prepare images list
        if weighted_images is not None:
            # With weighted images: empty, target, pred, [diff], lr
            images = [None, imClip_target, imClip_pred]  # None for empty column
            if plot_diff:
                diff_image = data_pred - data_target
                images.append(diff_image)
            images.append(imClip_lr)
        else:
            # Without weighted images: target, pred, [diff], lr
            images = [imClip_target, imClip_pred]
            if plot_diff:
                diff_image = data_pred - data_target
                images.append(diff_image)
            images.append(imClip_lr)

        # Plot each image in the row
        for j, img in enumerate(images):
            base_col = j * cols_per_image
            
            if base_col >= axs.shape[1]:
                print(f"Warning: Column index {base_col} exceeds axis array bounds")
                continue
                
            ax_main = axs[i, base_col]
            
            # Handle empty column (first column when weighted_images is provided)
            if img is None:
                plot_empty_axis(ax_main)
                continue
            
            img_proc = rot_tensor(img) if rotate else img
            
            # FIXED: Determine if ROI rectangle should be drawn
            # Draw ROI rectangle on ALL images except empty columns and difference maps
            if weighted_images is not None:
                # With weighted images: draw on GT(j=1), Pred(j=2), LR(last), but not empty(j=0) or diff
                if plot_diff:
                    draw_rect = ROI is not None and j in [1, 2, 4]  # GT, Pred, LR (skip empty and diff)
                else:
                    draw_rect = ROI is not None and j in [1, 2, 3]  # GT, Pred, LR (skip empty)
            else:
                # Without weighted images: draw on GT(j=0), Pred(j=1), LR(last), but not diff
                if plot_diff:
                    draw_rect = ROI is not None and j in [0, 1, 3]  # GT, Pred, LR (skip diff)
                else:
                    draw_rect = ROI is not None and j in [0, 1, 2]  # GT, Pred, LR
            
            # Set colormap and limits for difference images
            if ((weighted_images is not None and j == 3 and plot_diff) or 
                (weighted_images is None and j == 2 and plot_diff)):  # Difference image
                use_cmap = 'coolwarm'
                use_vmin = -upLev / 10
                use_vmax = upLev / 10
                metrics_text = None  # No metrics for difference images
            else:
                use_cmap = cmap
                use_vmin = loLev
                use_vmax = upLev
                
                # Set metrics text for each image (only if show_qMRI_metrics=True)
                if show_qMRI_metrics and len(channel_metrics) > i:
                    if weighted_images is not None:
                        if j == 2:  # Predicted image (3rd column)
                            metrics = channel_metrics[i]
                            metrics_text = f"HFEN: {metrics['pred_hfen']:.3f}\nSSIM: {metrics['pred_ssim']:.3f}"
                            metrics_text = f"qMRI_{metrics_text}"  # Add flag for display logic
                        elif j == len(images) - 1:  # LR image (last column)
                            metrics = channel_metrics[i]
                            metrics_text = f"HFEN: {metrics['lr_hfen']:.3f}\nSSIM: {metrics['lr_ssim']:.3f}"
                            metrics_text = f"qMRI_{metrics_text}"  # Add flag for display logic
                        else:  # Ground truth or empty
                            metrics_text = None
                    else:
                        if j == 1:  # Predicted image
                            metrics = channel_metrics[i]
                            metrics_text = f"HFEN: {metrics['pred_hfen']:.3f}\nSSIM: {metrics['pred_ssim']:.3f}"
                            metrics_text = f"qMRI_{metrics_text}"  # Add flag for display logic
                        elif j == len(images) - 1:  # LR image (last image)
                            metrics = channel_metrics[i]
                            metrics_text = f"HFEN: {metrics['lr_hfen']:.3f}\nSSIM: {metrics['lr_ssim']:.3f}"
                            metrics_text = f"qMRI_{metrics_text}"  # Add flag for display logic
                        else:  # Ground truth image
                            metrics_text = None
                else:
                    metrics_text = None  # No metrics displayed when show_qMRI_metrics=False
            
            plot_image(ax_main, img_proc, use_cmap, use_vmin, use_vmax, alpha=1.0, 
                      draw_rect=draw_rect, metrics_text=metrics_text)

            # Add ROI visualization if specified
            if ROI is not None:
                img_roi = rot_tensor(img) if rotate else img
                if roi_is_small:
                    plot_roi_inset(ax_main, img_roi, use_cmap, use_vmin, use_vmax, alpha=1.0)
                else:
                    if base_col + 1 < axs.shape[1]:
                        ax_roi = axs[i, base_col + 1]
                        plot_roi_side(ax_roi, img_roi, use_cmap, use_vmin, use_vmax, alpha=1.0)

    # Plot weighted images rows if provided
    if weighted_images is not None:
        print(f"\nPlotting {num_weighted_rows} weighted image rows...")
        
        for w_row in range(num_weighted_rows):
            row_idx = num_channels + w_row  # Row index in the subplot grid
            
            # Plot all 4 weighted images for this row
            for w_col in range(4):
                base_col = w_col * cols_per_image
                
                if base_col >= axs.shape[1]:
                    print(f"Warning: Column index {base_col} exceeds axis array bounds for weighted images")
                    continue
                
                ax_main = axs[row_idx, base_col]
                
                try:
                    # Get weighted image data for current row, column, and slice
                    weighted_data_raw = weighted_images[w_row, w_col, slice_index].detach().cpu()
                    
                    # FIXED: Normalize weighted images using z-score normalization before processing
                    weighted_data = normalize_tensor(weighted_data_raw)
                    weighted_proc = rot_tensor(weighted_data) if rotate else weighted_data
                    
                    # Use grayscale colormap with dynamic range for z-score normalized data
                    use_cmap = 'gray'
                    # Set visualization range starting from 0 to +3 standard deviations
                    use_vmin = 0.0
                    use_vmax = 3.0
                    
                    # FIXED: Draw ROI rectangles on ALL weighted image columns
                    draw_rect = ROI is not None
                    
                    # Set metrics text for weighted images (only if show_weighted_image_metrics=True)
                    metrics_text = None
                    if show_weighted_image_metrics and w_col > 0:  # Columns 1, 2, 3 (not GT column 0)
                        metrics_idx = w_col - 1  # Convert to 0-based index for metrics array
                        if w_row < len(weighted_metrics) and metrics_idx < len(weighted_metrics[w_row]):
                            metrics = weighted_metrics[w_row][metrics_idx]
                            metrics_text = f"HFEN: {metrics['hfen']:.3f}\nSSIM: {metrics['ssim']:.3f}"
                            metrics_text = f"weighted_{metrics_text}"  # Add flag for display logic
                    
                    plot_image(ax_main, weighted_proc, use_cmap, use_vmin, use_vmax, alpha=1.0, 
                              draw_rect=draw_rect, metrics_text=metrics_text)
                    
                    # Add ROI visualization if specified
                    if ROI is not None:
                        weighted_roi = rot_tensor(weighted_data) if rotate else weighted_data
                        if roi_is_small:
                            plot_roi_inset(ax_main, weighted_roi, use_cmap, use_vmin, use_vmax, alpha=1.0)
                        else:
                            if base_col + 1 < axs.shape[1]:
                                ax_roi = axs[row_idx, base_col + 1]
                                plot_roi_side(ax_roi, weighted_roi, use_cmap, use_vmin, use_vmax, alpha=1.0)
                
                except Exception as e:
                    print(f"Error plotting weighted image [{w_row}, {w_col}]: {e}")
                    ax_main.text(0.5, 0.5, f'W[{w_row},{w_col}]\nError', ha='center', va='center', 
                                transform=ax_main.transAxes, color='white')
                    ax_main.set_facecolor('black')
                    ax_main.axis('off')

    # Add labels
    try:
        for j, label in enumerate(column_labels):
            center = (j * cols_per_image + 0.5 * cols_per_image) / total_cols
            fig.text(center, 1.02, label, ha='center', va='center', fontsize=12, color='white')

        for i in range(len(row_labels)):
            if i < total_rows:
                fig.text(-0.03, (total_rows - i - 0.5) / total_rows, f'{row_labels[i]}', 
                        ha='center', va='center', rotation=90, fontsize=12, color='white')
    except Exception as e:
        print(f"Warning: Error adding labels: {e}")

    fig.tight_layout(pad=0.0, rect=[0, 0, 1, 1])

    try:
        if output == "show":
            plt.show()
        elif output == "save":
            filename = f"output_plot_slice_{slice_index}_with_metrics.png"
            fig.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0.1, facecolor='black')
            print(f"Plot saved as {filename}")
    except Exception as e:
        print(f"Error displaying/saving plot: {e}")
    finally:
        plt.close(fig)  # Prevent memory leaks