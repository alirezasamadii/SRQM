import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from unet3d.model import UNet3D, UNet2D,ResidualUNet2D
from unet3d.resnet import ResNet50
import json
from load_pickles import CustomDataset
from unet3d.train_model import train_model
import os
import tempfile
import os
tempfile.tempdir = '/data/scratch/r098956/tmp/'  # or another safe temp location

#from unet3d.simple_resnet import SimpleResNet

# Set random seed for reproducibility
seed = 42
torch.manual_seed(seed)

# Function to read JSON configuration
def read_config(filename):
    with open(filename, 'r') as f:
        return json.load(f)

# Load configuration
config = read_config('/trinity/home/r098956/SRQM/config.json')

# Device setup: use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Extract parameters from the configuration file
downsampling_mask_size = config['downsampling_mask_size']
mask_size_deviation = config['mask_size_deviation']
batch_size = config["batch_size"]
patch_size = config["patch_size"]
guide_modality = config["guide_modality"]
HR_loss_modality = config["HR_loss_modality"]
learning_rate = config['learning_rate']
num_epochs = config['num_epochs']
save_interval = config['save_interval']
experiment_id = config['experiment']['experiment_id']
description = config['experiment']['description']
dataset_path = config['other_parameters']["processed_dataset_dir"]
in_channels = config['model']["input_channels"]
output_channels = config['model']["output_channels"]
model_type = config['model']['model_type']
HR_loss_weight = config['HR_loss_weight']
LR_t1w_loss_weight = config['LR_t1w_loss_weight']
LR_t2w_loss_weight = config['LR_t2w_loss_weight']
HR_loss_mode = config['HR_loss_mode']
LR_loss_mode = config['LR_loss_mode']
params = config['parameters']
load_checkpoint = config['load_checkpoint']
if "Free_qMRI" in HR_loss_modality:
    LR_t2w_loss_weight = 0
    LR_t1w_loss_weight = 0
# Experiment description
experiment_description = (
    f"downsampling mask size = {downsampling_mask_size} {experiment_id} description: {description}"
)

# Convert guide_modality list to a string with underscores
guide_modality_str = '_'.join(guide_modality)
if len(guide_modality)==0:
    guide_modality_str = "null"
HR_loss_modality_str = '_'.join(HR_loss_modality)
if len(HR_loss_modality_str)==0:
    HR_loss_modality_str = "null"
# Format the file path for saving the model
file_path = (
    f'/data/scratch/r098956/SRQM/logs/saved_models/PHIREQfinal/{experiment_id}_{model_type}_'
    f'res{downsampling_mask_size}_G{guide_modality_str}_HRL{HR_loss_modality}_HRw{HR_loss_weight}_'
    f'bs{batch_size}_LRt1wL{LR_t1w_loss_weight}_LRt2wL{LR_t2w_loss_weight}_LRm{LR_loss_mode}_HRm{HR_loss_mode}.pth'
)

# Setup TensorBoard writer
writer = SummaryWriter(
    f'/data/scratch/r098956/SRQM/logs/code_logs/tensorboard_logs/PHIREQfinal/{experiment_id}_'
    f'{model_type}_res{downsampling_mask_size}_G{guide_modality_str}_HRL{HR_loss_modality}_'
    f'HRw{HR_loss_weight}_bs{batch_size}_LRt1wL{LR_t1w_loss_weight}_LRt2wL{LR_t2w_loss_weight}_LRm{LR_loss_mode}_HRm{HR_loss_mode}'
)

writer.add_text('experiment_description', experiment_description)

# Dataset setup
custom_dataset = CustomDataset(dataset_path, downsampling_mask_size)
dataset_size = len(custom_dataset)
val_size = int(0.3 * dataset_size)  # = 0.3 results in 4 validation of total 18 patients
train_size = dataset_size - val_size
train_dataset, val_dataset = random_split(custom_dataset, [train_size, val_size])

# DataLoader setup
train_loader = DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2
)
val_loader = DataLoader(
    dataset=val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=2
)

# Adjust in_channels based on guide_modality
if "null" in guide_modality:
    pass
if "T1W" in guide_modality or "T1W_BRAVO" in guide_modality:
    in_channels += 1
if "T2W" in guide_modality:
    in_channels += 1



# Model setup

if model_type == "3DUNET":
    model = UNet3D(in_channels=in_channels, out_channels=output_channels, is_segmentation=False)
elif model_type == "2DUNET":
    model = UNet2D(in_channels=in_channels, out_channels=output_channels, is_segmentation=False)
elif model_type=="ResidualUNet2D":
    model = ResidualUNet2D(in_channels=in_channels, out_channels=output_channels, is_segmentation=False)
elif model_type=="ResNet50":
    model = ResNet50(in_channels=in_channels, out_channels=output_channels, num_blocks=5 , upscale_factor=1)

        
        

# Move model and criterion to the device

if load_checkpoint == 1 and os.path.exists(file_path):
    model.load_state_dict(torch.load(file_path, map_location=device))
    print(f"Model loaded successfully from {file_path}, training from checkpoint is continuing.")
else:
    print("No valid checkpoint found or loading disabled, starting training from scratch.")

model.to(device)
#criterion = nn.L1Loss().to(device)
criterion=nn.MSELoss().to(device)




# Optimizer setup
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)
print(f"Training a {model_type} with {in_channels} input channels of which three channels are reserved for qMRI. Guide is {guide_modality_str} in the input and HR {HR_loss_modality_str} is/are synthesized from the maps with {HR_loss_weight} as the weight for HR loss computation and LR t1w Loss weight is {LR_t1w_loss_weight}, LR t2w Loss weight is {LR_t2w_loss_weight}")
# Start training
prof = torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA
    ],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=1, repeat=1),
    on_trace_ready=torch.profiler.tensorboard_trace_handler(
    f'/data/scratch/r098956/SRQM/logs/code_logs/tensorboard_logs/PHIREQfinal/{experiment_id}_'
    f'{model_type}_res{downsampling_mask_size}_G{guide_modality_str}_HRL{HR_loss_modality}_'
    f'HRw{HR_loss_weight}_bs{batch_size}_LRt1wL{LR_t1w_loss_weight}_LRt2wL{LR_t2w_loss_weight}_LRm{LR_loss_mode}_HRm{HR_loss_mode}'),
    record_shapes=True,
    with_stack=True
)

train_model(
    model, train_loader, val_loader, criterion,optimizer, num_epochs, save_interval, 
    device, writer, file_path, params, guide_modality, HR_loss_modality,downsampling_mask_size, 
    model_type, HR_loss_weight,LR_t1w_loss_weight,LR_t2w_loss_weight,prof,HR_loss_mode,LR_loss_mode
)
