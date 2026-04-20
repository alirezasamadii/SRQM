
import torch
from tqdm import tqdm
import pickle
from unet3d import mle_relaxometry
from unet3d.utils import (
    frequency_domain_downsampler_tensor,
    synthesize_weighted_image,pearson_correlation_loss)

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, save_interval, device, writer, file_path, guide_modality,params):
    
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode

        for batch_idx, (target,mle_hrqm, HRWI, LRQM) in enumerate(tqdm(train_loader)):
      
            with open('/data/scratch/r098956/LR_HR_QMRI/hrwi/hrwi'+str(batch_idx)+'.pkl', "wb") as f:
                pickle.dump(HRWI.detach().cpu(), f)
            with open('/data/scratch/r098956/LR_HR_QMRI/lrqmri/lrqmri'+str(batch_idx)+'.pkl', "wb") as f:
                pickle.dump(LRQM.detach().cpu(), f)
            with open('/data/scratch/r098956/LR_HR_QMRI/magic_hrqmri/magic_hrqmri'+str(batch_idx)+'.pkl', "wb") as f:
                pickle.dump(target.detach().cpu(), f)
            with open('/data/scratch/r098956/LR_HR_QMRI/mle_hrqmri/mle_hrqmri'+str(batch_idx)+'.pkl', "wb") as f:
                pickle.dump(mle_hrqm.detach().cpu(), f)
      