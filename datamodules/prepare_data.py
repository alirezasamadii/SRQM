import os
import random
import numpy as np
import pickle
import torch
import nibabel as nib
import glob
from unet3d.utils import  lower_resolution_image_4d
from unet3d import mle_relaxometry
from tqdm import tqdm
import pydicom

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def read_dicom_header_as_dict(dicom_path):
    # Read the DICOM file
    dicom_data = pydicom.dcmread(dicom_path)
    
    # Convert the DICOM header to a dictionary
    dicom_header_dict = {}
    
    # Iterate over the elements in the DICOM dataset
    for element in dicom_data:
        # Skip sequence items to avoid nested structures
        if element.VR != 'SQ':  # 'SQ' means Sequence
            dicom_header_dict[element.name] = element.value
    
    return dicom_header_dict

def return_patient_ids(directory_path): 
    """
    Extract patient IDs from a directory containing studies with specific subfolders.

    Parameters:
    - directory_path (str): Path to the directory containing studies.

    Returns:
    - List[str]: List of patient IDs(i.e: folders) for studies containing specific subfolders (SYMAPS, T1W, T2FLAIR).

    Description:
    This function takes a directory path as input and extracts patient IDs for studies that meet the following criteria:
    1. Contain a subfolder with the name "400_SYMAPS".
    2. Contain a subfolder with the name "T1W".
    3. Contain a subfolder with the name "FLAIR".

    Example:
    >>> patient_ids = return_patient_ids('/path/to/studies_directory')
    >>> print(patient_ids)
    ['patient1', 'patient2', ...]
    """
    
    
    patient_ids = [
    folder_name 
    for folder_name in os.listdir(directory_path) 
    if any("400_SYMAPS" in subfolder_name for subfolder_name in os.listdir(os.path.join(directory_path, folder_name)) if os.path.isdir(os.path.join(directory_path, folder_name, subfolder_name)))
    and  any("T1W" in subfolder_name for subfolder_name in os.listdir(os.path.join(directory_path, folder_name)) if os.path.isdir(os.path.join(directory_path, folder_name, subfolder_name)))
    and  any("FLAIR" in subfolder_name for subfolder_name in os.listdir(os.path.join(directory_path, folder_name)) if os.path.isdir(os.path.join(directory_path, folder_name, subfolder_name)))
]
    
    return patient_ids

def load_tensor(patient_id): 
    root_dir = os.path.join('/data/scratch/r098956/SRQM/FullGliomaPatientsDataset/Processed SynthMR', patient_id) 
    for folder_name in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder_name)
        
            # Check if the folder contains the name "SYMAPS"
        if "SYMAPS" in folder_name:
            #print("Full name of the folder: ", folder_name)
            for modality in ["PD", "T1", "T2"]:
                Q_niftii_image_path = glob.glob(os.path.join(folder_path, modality ,'*.nii.gz' ))[0]
                nii_Q_img_modality = nib.load(Q_niftii_image_path)
                raw_Q_data_modality = nii_Q_img_modality.get_fdata()

                if modality== "PD":
                    data_Q_PD=raw_Q_data_modality 
                if modality=="T1":
                    data_Q_T1=raw_Q_data_modality 
                if modality=="T2":
                    data_Q_T2=raw_Q_data_modality  
            HRQPD = torch.Tensor(data_Q_PD)
            HRQT1 = torch.Tensor(data_Q_T1)
            HRQT2 = torch.Tensor(data_Q_T2)
            magic_HRQM=torch.stack((HRQPD,HRQT1,HRQT2),dim=0).permute(0,3,1,2) #HRQM from dataset 
            sample_dicom_image = glob.glob(os.path.join(folder_path, modality ,'*.dcm' ))[0]
            patient_data = read_dicom_header_as_dict(sample_dicom_image)

        if "T1W" in folder_name:
            
            Q_niftii_image_path = glob.glob(os.path.join(folder_path ,'*.nii.gz' ))[0]
            nii_Q_img_modality = nib.load(Q_niftii_image_path)
            raw_T1W_data_modality = nii_Q_img_modality.get_fdata()
            T1W = torch.Tensor(raw_T1W_data_modality)

        if "FLAIR" in folder_name:
            
            Q_niftii_image_path = glob.glob(os.path.join(folder_path ,'*.nii.gz' ))[0]
            nii_Q_img_modality = nib.load(Q_niftii_image_path)
            raw_T2FLAIR_data_modality = nii_Q_img_modality.get_fdata()
            T2FLAIR = torch.Tensor(raw_T2FLAIR_data_modality)

    HRWI=torch.stack((T1W,T2FLAIR),dim=0).permute(0,3,1,2) 
    return magic_HRQM,HRWI,patient_data


def downsample(params,mask_size,magic_HRQM):
    repetition_times_list = params.get('repetition_times_list')
    echo_times_list = params.get('echo_times_list')
    t1_rlaxometry_steps = params.get('t1_rlaxometry_steps')
    t1_rlaxometry_lr = params.get('t1_rlaxometry_lr')
    t2_rlaxometry_steps = params.get('t2_rlaxometry_steps')
    t2_relaxometry_lr = params.get('t2_relaxometry_lr')

    single_channel_bg_mask = (magic_HRQM[0].unsqueeze(0) != 0)
    single_channel_bg_mask = single_channel_bg_mask.to(device)

    T1W_from_fwd_model= mle_relaxometry.spin_echo(magic_HRQM,repetition_times_list=repetition_times_list)    
    lr_t1w_images=lower_resolution_image_4d(T1W_from_fwd_model,mask_size,sigma=10) 
    lr_T1map=mle_relaxometry.estimate_qMRI_maps(lr_t1w_images, repetition_times_list, relaxometry_mode='T1_spinEcho',num_iterations=t1_rlaxometry_steps,learning_rate=t1_rlaxometry_lr,single_channel_bg_mask=single_channel_bg_mask)
    

    pd_t2=torch.stack((magic_HRQM[0],magic_HRQM[2]),dim=0)
    T2w_from_fwd_model= mle_relaxometry.fse(pd_t2,echo_times_list)  
    lr_t2w_images=lower_resolution_image_4d(T2w_from_fwd_model,mask_size,sigma=10)
    lr_T2map=mle_relaxometry.estimate_qMRI_maps(lr_t2w_images, echo_times_list, relaxometry_mode='T2_fse',num_iterations=t2_rlaxometry_steps,learning_rate=t2_relaxometry_lr,single_channel_bg_mask=single_channel_bg_mask)[1].unsqueeze(0)
    
    LRQM=torch.cat((lr_T1map,lr_T2map),dim=0)*(single_channel_bg_mask.expand(3,-1,-1,-1))
    MCWI= torch.cat((lr_t1w_images,lr_t2w_images),dim=0)*(single_channel_bg_mask.expand(12,-1,-1,-1))
    return LRQM,MCWI




def save_data( patient_ids,mask_sizes,params,output_dir):
    #output_dir="/data/scratch/r098956/final_dataset/"
    i=0
    for patient_id in patient_ids:
        i+=1
        print("___ Processing and Saving pickled tensor data for patient #",str(i), " out of total ", str(len(patient_ids)))
        hrqm,hrwi,patient_data = load_tensor(patient_id)

        hrqm_dir = os.path.join(output_dir, 'magic_maps')
        os.makedirs(hrqm_dir, exist_ok=True)
        hrqm_path = os.path.join(hrqm_dir, f'magic_HRQM{i}.pkl')
        with open(hrqm_path, "wb") as f:
            pickle.dump(hrqm.detach().cpu(), f)

        patient_data_dir = os.path.join(output_dir, 'patient_data')
        os.makedirs(patient_data_dir, exist_ok=True)
        patient_data_path = os.path.join(patient_data_dir, f'patient_data{i}.pkl')
        with open(patient_data_path, "wb") as f:
            pickle.dump(patient_data, f)
            

        
        hrwi_dir = os.path.join(output_dir, 'weighted_images')
        os.makedirs(hrwi_dir, exist_ok=True)
        hrwi_path = os.path.join(hrwi_dir, f'hrwi{i}.pkl')
        with open(hrwi_path, "wb") as f:
            pickle.dump(hrwi, f)
        
        j=0
        for mask_size in mask_sizes:
            j+=1
            print("___ Generating LR qMRI for mask ",str(j),"out of ",str(len(mask_sizes)))
            lrqm,mcwi = downsample(params,mask_size,hrqm)

            lrqm_dir = os.path.join(output_dir, 'downsampled_magic_maps', f'mask_size_{mask_size}')
            mcwi_dir = os.path.join(output_dir, 'downsampled_mcwi', f'mask_size_{mask_size}')

            os.makedirs(lrqm_dir, exist_ok=True)
            os.makedirs(mcwi_dir, exist_ok=True)

            lrqm_path = os.path.join(lrqm_dir, f'mask_size_{mask_size}_{i}.pkl')
            mcwi_path = os.path.join(mcwi_dir, f'mask_size_{mask_size}_{i}.pkl')
            with open(lrqm_path, "wb") as f:
                pickle.dump(lrqm.detach().cpu(), f)
            with open(mcwi_path, "wb") as f:
                pickle.dump(mcwi.detach().cpu(), f)