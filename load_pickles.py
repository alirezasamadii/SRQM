import os
import pickle
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
from unet3d.utils import synthesize_weighted_image
import nibabel as nib

# Provided functions and variables
means = [35.47540283203125, 20.562599182128906]
stds = [21.34469223022461, 14.580052375793457]
T1W_MAXVALUE= max([103.32, 103.46, 103.92, 104.049995, 107.259995, 107.27, 107.89, 108.34, 109.45, 109.47, 109.59, 109.77, 109.97, 110.21, 110.759995, 110.829994, 111.03, 111.64, 111.7, 111.84, 112.159996, 113.06, 113.939995, 114.259995, 114.43, 114.63, 114.82, 115.11, 115.409996, 115.52, 115.71, 115.869995, 116.17, 116.409996, 116.63, 117.02, 117.04, 117.119995, 117.27, 117.479996, 117.86, 118.1, 118.299995, 118.939995, 119.39, 119.479996, 120.45, 120.549995, 120.63, 120.64, 120.899994, 121.11, 121.54, 121.68, 121.96999, 122.39, 122.59, 123.09, 123.399994, 123.56, 123.75, 123.82, 124.24, 124.29, 124.82, 124.909996, 125.61, 125.82, 126.93, 127.21, 127.32, 127.45, 128.55, 128.70999, 129.12, 129.72, 130.41, 130.61, 132.05, 132.84999, 133.16, 133.18, 133.27, 134.31, 136.84])
T2FLAIR_MAXVALUE = max( [64.67, 67.64, 68.88, 68.99, 69.799995, 70.939995, 71.159996, 71.32, 71.42, 72.04, 72.6, 72.71, 72.88, 72.97, 73.24, 73.39, 73.43, 73.51, 73.85, 74.2, 74.5, 74.57, 74.82, 75.07, 75.68, 75.72, 76.11, 76.36, 76.369995, 76.67, 76.88, 77.43, 77.88, 78.15, 78.299995, 78.64, 78.93, 79.38, 79.549995, 79.58, 79.7, 79.82, 79.96, 80.54, 80.63, 80.86, 80.92, 81.09, 81.2, 81.24, 81.54, 81.7, 81.74, 82.0, 82.009995, 82.09, 82.17, 82.31, 82.549995, 82.619995, 83.299995, 83.33, 83.63, 84.509995, 84.56, 84.939995, 85.15, 85.39, 85.56, 86.0, 86.229996, 86.4, 86.46, 87.03, 87.229996, 87.29, 87.93, 88.63, 88.729996, 89.17, 89.57, 90.61, 90.619995, 91.02, 91.1, 95.53])
max_values = [T1W_MAXVALUE, T2FLAIR_MAXVALUE]
central_patch_size = (512, 512)

def custom_relu(tensor, min_value, channel):
    """
    Applies a ReLU-like operation to the specified channel of the tensor,
    setting a minimum value for non-zero elements.

    Parameters:
    tensor (torch.Tensor): The input tensor with shape (c, d, h, w).
    min_value (float): The minimum value to enforce on the specified channel.
    channel (int): The channel index to apply the operation.

    Returns:
    torch.Tensor: The modified tensor with the ReLU operation applied to the specified channel.
    """
    # Ensure the tensor is in float type to avoid issues with operations
    tensor = tensor.float()

    # Apply the custom ReLU operation only to the specified channel
    selected_channel = tensor[channel]
    
    # Apply the ReLU-like operation
    selected_channel = torch.where(selected_channel > 0, torch.max(selected_channel, torch.tensor(min_value)), selected_channel)

    # Assign the modified channel back to the tensor
    tensor[channel] = selected_channel

    return tensor
    
def scale_tensors(tensor_list, max_values):
    #CHECK IF ACTUALLY MAX VALUE IS WHAT IS HARD CODED ABOVE
    scaled_tensors = []
    for tensor in tensor_list:
        tensor = torch.tensor(tensor) if not isinstance(tensor, torch.Tensor) else tensor
        c, d, h, w = tensor.shape
        scaled_tensor = torch.empty_like(tensor)
        for channel in range(c):
            channel_data = tensor[channel]
            max_value = max_values[channel]
            scaled_channel_data = (channel_data / max_value) * 10.0
            #scaled_channel_data = torch.clamp(scaled_channel_data, 0, 10)
            scaled_tensor[channel] = scaled_channel_data
        scaled_tensors.append(scaled_tensor)
    return scaled_tensors

def get_central_patch(tensor, target_size):
    c, d, h, w = tensor.shape
    target_h, target_w = target_size
    start_x = (w - target_w) // 2
    start_y = (h - target_h) // 2
    central_patch = tensor[:, :, start_y:start_y + target_h, start_x:start_x + target_w]
    return central_patch

class CustomDataset(Dataset): 
    def __init__(self, directory, downsampling_mask_size):
        self.directory = directory
        self.downsampling_mask_size = downsampling_mask_size
       
        self.data = self.load_data()
        
    def __len__(self):
        # Assuming N is the number of samples in the dataset
        return self.data['hrwi'].shape[1]

    def __getitem__(self, idx):
       
        hrqmri = self.data['magic_hrqmri'][:, idx, :, :]  # [c_hrqmri, h, w]
        hrwi = self.data['hrwi'][:, idx, :, :]           # [c_hrwi, h, w]
        lrqmri = self.data['lrqmri'][:, idx, :, :]  
        lrmcwi = self.data['lrmcwi'][:, idx, :, :]
        brain_mask = self.data['brain_masks'][:, idx, :, :]
        #return hrqmri*brain_mask.expand(hrqmri.shape[0],-1,-1), hrwi*brain_mask.expand(hrwi.shape[0],-1,-1), lrqmri*brain_mask.expand(lrqmri.shape[0],-1,-1) , lrmcwi*brain_mask.expand(lrmcwi.shape[0],-1,-1)
        return hrqmri, hrwi, lrqmri , lrmcwi,brain_mask
    def load_data(self):
        data = {}
        lrqmri_list = []
        lrmcwi_list=[]
        hrwi_list = []
        magic_hrqmri_list = []
        locations = []
        brain_mask_tensor_4d = []
       
        for file_path in tqdm(sorted(os.listdir(os.path.join(self.directory, 'magic_maps'))), desc="processing magic_hrqmri"):
            fp = os.path.join(self.directory, 'magic_maps', file_path)
            with open(fp, 'rb') as f:
                magic_hrqmri = pickle.load(f).squeeze(0)
                magic_hrqmri = get_central_patch(magic_hrqmri, central_patch_size)
                magic_hrqmri_list.append(magic_hrqmri)
                locations.append(fp)       
        magic_hrqmri = torch.cat(magic_hrqmri_list, dim=1)  # [c_hrqmri, N, h, w]
        data['magic_hrqmri'] = magic_hrqmri
   
       

        # Load and process hrwi
        for file_path in tqdm(sorted(os.listdir(os.path.join(self.directory, 'syn_conventional_images'))), desc="processing syn_conventional_images"):
            fp = os.path.join(self.directory, 'syn_conventional_images', file_path)
            with open(fp, 'rb') as f:
                hrwi = pickle.load(f).squeeze(0)/100
                hrwi = get_central_patch(hrwi, central_patch_size)
                hrwi_list.append(hrwi)
        #hrwi_list = scale_tensors(hrwi_list, max_values)

        hrwi = torch.cat(hrwi_list, dim=1)  # [c_hrwi, N, h, w]

        data['hrwi'] = hrwi

        # Load and process lrqmri
       
        for file_path in tqdm(sorted(os.listdir(os.path.join(self.directory, 'downsampled_magic_maps', 'mask_size_' + str(self.downsampling_mask_size)))), desc="processing lrqmri"):
            fp = os.path.join(self.directory, 'downsampled_magic_maps', 'mask_size_' + str(self.downsampling_mask_size), file_path)
            with open(fp, 'rb') as f:
                lrqmri = pickle.load(f).squeeze(0)
                lrqmri = get_central_patch(lrqmri, central_patch_size)
                lrqmri_list.append(lrqmri)
        lrqmri = torch.cat(lrqmri_list, dim=1)  # [c_lrqmri, N, h, w]
        data['lrqmri'] = lrqmri

        
        for file_path in tqdm(sorted(os.listdir(os.path.join(self.directory, 'downsampled_mcwi', 'mask_size_' + str(self.downsampling_mask_size)))), desc="processing lr mcwi"):
            fp = os.path.join(self.directory, 'downsampled_mcwi', 'mask_size_' + str(self.downsampling_mask_size), file_path)
            with open(fp, 'rb') as f:
                lrmcwi = pickle.load(f).squeeze(0)
                lrmcwi = get_central_patch(lrmcwi, central_patch_size)
                lrmcwi_list.append(lrmcwi)
        lrmcwi = torch.cat(lrmcwi_list, dim=1)  # [c_lrqmri, N, h, w]
        data['lrmcwi'] = lrmcwi

        for file_path in tqdm(sorted(os.listdir(os.path.join(self.directory, 'brain_masks'))), desc="processing brain_masks"):
            fp = os.path.join(self.directory, 'brain_masks', file_path)
            brain_mask = nib.load(fp)
            # Get the image data as a NumPy array
            brain_mask_data = brain_mask.get_fdata()
            brain_mask_tensor_4d.append( torch.from_numpy(brain_mask_data).float().permute(2,0,1).unsqueeze(0))
       

        all_brain_masks = torch.cat(brain_mask_tensor_4d, dim=1)  # [1, N, h, w]

        data['brain_masks'] = all_brain_masks



        

        return data

