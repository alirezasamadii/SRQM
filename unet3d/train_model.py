
import torch
from tqdm import tqdm
from unet3d import mle_relaxometry
from unet3d.utils import (lower_resolution_image_4d, synthesize_weighted_image,scale_qmri_channels,pearson_correlation_loss_per_slice,FSPGR,InversionRecovery,SpinEcho,z_score_per_slice,SpinEhowithFlipAngle)
import torch
import torch.fft
import math
import torch.profiler

def clip_channels_with_atan(x):
    """
    Clips each channel of the tensor using arctangent transformation and scaling to the given min and max values.
    
    Args:
        x (torch.Tensor): Input tensor of shape [b, c, d, h].
    
    Returns:
        torch.Tensor: Tensor with clipped values based on the specified min and max values for each channel.
    
    Raises:
        ValueError: If tensor does not have 4 dimensions or does not have at least 3 channels.
    """
    # Validate input tensor shape
    if x.dim() != 4 or x.size(1) < 3:
        raise ValueError("Input tensor must have shape [b, c, d, h] with at least 3 channels.")
    
    # Define min and max values for each channel
    min_vals = torch.tensor([1.0, 100.0, 10.0], device=x.device).view(1, 3, 1, 1)  # Shape: [1, 3, 1, 1]
    max_vals = torch.tensor([160.0, 4300.0, 2000.0], device=x.device).view(1, 3, 1, 1)  # Shape: [1, 3, 1, 1]

    # Apply arctangent function to each channel (element-wise)
    x = torch.arctan(x) / (torch.pi / 2)  # Normalize to range (-1, 1)
    x = 0.5 * (x + 1)                    # Scale to range (0, 1)
    
    # Scale and shift to the specified min and max values
    x = min_vals + x * (max_vals - min_vals)
    
    return x

def predict_maps(model, HRWI, LRQM, guide_modality, model_type):
    """
    Predicts quantitative maps using a given neural network model, guided by high-resolution images if specified.

    Parameters:
    -----------
    model : torch.nn.Module
        The neural network model to use for prediction, either a 2D or 3D U-Net, or another supported model.
    HRWI : torch.Tensor or None
        High-resolution weighted images tensor of size [B, C, H, W], where:
        - C=0 corresponds to T1-weighted images (T1W) FSE (not used)
        - C=2 corresponds to T1-weighted images (T1W) BRAVO
        - C=1 corresponds to T2W-PROPELLAR images

        Can be None if `guide_modality` is an empty list or not applicable.
    LRQM : torch.Tensor
        Low-resolution quantitative maps tensor of size [B, C, H, W], where:
        - C=0 corresponds to PD (Proton Density)
        - C=1 corresponds to T1
        - C=2 corresponds to T2
    guide_modality : list of str
        A list of strings indicating the guide modalities to use. Can include , []"T1W_BRAVO","T2W_PROPELLAR","T2W_FLAIR"] OR ["T1W"], ["T2FLAIR"], ["T1W", "T2FLAIR"], or [] or other combinations
    model_type : str
        The type of model to use, either "2DUNET", "3DUNET", "ResidualUNet2D", or "ResNet50".

    Returns:
    --------
    torch.Tensor
        The predicted maps tensor of size [B, C, H, W], with the background masked out.
    """

    # Permute LRQM tensor to [C, B, H, W], scale the channels, then permute back to [B, C, H, W]
    LRQM_permuted = LRQM.permute(1, 0, 2, 3)  # [C, B, H, W]
    LRQM_scaled = scale_qmri_channels(LRQM_permuted, "divide").permute(1, 0, 2, 3)  # [B, C, H, W]

    # Determine inputs based on guide modalities
    if guide_modality==[] or 'null' in guide_modality:
        inputs = LRQM_scaled.clone()  # Ensures that LRQM_scaled is not modified in place
        #inputs = LRQM.clone()

    else:
        selected_hrwi = []
        if "T1W" in guide_modality:
            print("T1W_FSE as guide is selected")
            HRWI_selected_T1Wfse = HRWI[:, 0]# Select T1W
            HRWI_selected_zscored_T1Wfse = z_score_per_slice(HRWI_selected_T1Wfse).unsqueeze(1)  
            selected_hrwi.append(HRWI_selected_zscored_T1Wfse)


        if "T1W_BRAVO" in guide_modality:
            print("BRAVO as guide is selected")
            HRWI_selected_T1WBRAVO = HRWI[:, 2]# Select T1W
            HRWI_selected_zscored_T1WBRAVO = z_score_per_slice(HRWI_selected_T1WBRAVO).unsqueeze(1)  
            selected_hrwi.append(HRWI_selected_zscored_T1WBRAVO)

        if "T2W" in guide_modality:
            print("T2W_PROPELLAR as guide is selected")
            HRWI_selected_T2W = HRWI[:, 1]  # Select T2W
            HRWI_selected_zscored_T2W = z_score_per_slice(HRWI_selected_T2W).unsqueeze(1)  
            selected_hrwi.append(HRWI_selected_zscored_T2W)

        HRWI_selected = torch.cat(selected_hrwi, dim=1)  # Concatenate selected high-resolution images
        
        if "Free_qMRI" not in guide_modality:
            inputs = torch.cat([LRQM_scaled, HRWI_selected], dim=1)
        if "Free_qMRI" in guide_modality:
            inputs = torch.cat([LRQM_scaled*0, HRWI_selected], dim=1)
        #inputs = torch.cat([LRQM, HRWI_selected], dim=1)

    # Generate predictions based on the model type
    if model_type in ["2DUNET", "ResidualUNet2D","ResNet50"]:
        outputs = model(inputs)   # [B, C, H, W]
       
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    # Rescale, clip, and apply mask to the outputs
    LRQM_rescaled = scale_qmri_channels(LRQM_scaled.permute(1,0,2,3), "multiply").permute(1, 0, 2, 3)  # [B, C, H, W]
    outputs_rescaled = scale_qmri_channels(outputs.permute(1,0,2,3), "multiply").permute(1, 0, 2, 3)  # [B, C, H, W]

    #outputs_clipped = clip_channels_with_atan(outputs_rescaled)

   

    #bg = HRWI[:,1].unsqueeze(1) 
    #single_channel_backgound_mask = (bg != 0)
    #single_channel_backgound_mask=single_channel_backgound_mask.float() #b.,1,h,w



    return outputs_rescaled   #b,c,h,w


def predict_weighted_image(HR_loss_modality, outputs):
    """
    Predicts weighted images from quantitative maps based on the specified high-resolution (HR) loss modalities.

    Parameters:
    -----------
    HR_loss_modality : list of str
        A list of strings indicating the weighted image modalities to synthesize for loss computation.
        Supported values include: "T1W", "T2FLAIR". The order in this list determines the order of output tensors.
     
    outputs : torch.Tensor
        Low-resolution quantitative maps tensor of size [B, C, H, W], where:
        - C=0 corresponds to PD (Proton Density)
        - C=1 corresponds to T1
        - C=2 corresponds to T2

    Returns:
    --------
    List[torch.Tensor]
        A list of tensors, where each tensor represents a synthesized weighted image corresponding to the modalities
        specified in `HR_loss_modality`. Each tensor has a size of [B, H, W], and the order of tensors matches
        the order in `HR_loss_modality`.
    """

    # Permute outputs to [C, B, H, W] for processing
    outputs_permuted = outputs.permute(1, 0, 2, 3)
    
    # Initialize list to store synthesized weighted images
    pred_weighted_images = []
  # watchout : order of these matters: the idea is that weighted images similar to what is available in dataset in synthesized anyway. but during loss computation only relvant ones are used.
    if True:
        #pred_image_SpinEcho = SpinEcho(outputs_permuted,TR=700,TE= 10)
        pred_image_SpinEcho = SpinEhowithFlipAngle(outputs_permuted, TR=750, TE=10, flip_angle_deg=65)
        pred_weighted_images.append(pred_image_SpinEcho)

        

    if True:
        pred_image_SpinEcho_T2W= SpinEhowithFlipAngle(outputs_permuted, TR=5211, TE=146.076,flip_angle_deg=160)
        pred_weighted_images.append(pred_image_SpinEcho_T2W)

    if True:
        pred_image_T1WBRAVO= FSPGR(outputs_permuted)
        pred_weighted_images.append(pred_image_T1WBRAVO)
    
    # If no modalities were provided, return an empty list
    if not pred_weighted_images:
        return None
    
    # Stack the predicted weighted images and permute to [B, N, H, W], where N is the number of modalities
    pred_weighted_images = torch.stack(pred_weighted_images, dim=0).permute(1, 0, 2, 3)

    return pred_weighted_images



def compute_loss(masked_syn_weighted_image_with_guide_seq, HRWI, target, repetition_times_list, echo_times_list, single_channel_backgound_mask, outputs, criterion, mask_size,HR_loss_modality,HR_loss_mode,LR_loss_mode):
    """
    Compute various losses (low resolution T1W, T2W, and high-resolution loss).
    """
    # Helper Functions
    def apply_mask(data, mask):
        """Apply a mask to the data."""
        return data * mask.expand_as(data)
    
    def lower_res_with_mask(data, mask, mask_size, sigma=0):
        """Apply lower resolution conversion and mask."""
        low_res = lower_resolution_image_4d(data, mask_size, sigma=sigma)
        return apply_mask(low_res, mask)
    
    def calculate_hr_loss(syn_pred, hrwi, mask,HR_loss_mode,criterion):
        """Calculate high-resolution loss."""
        if HR_loss_mode == "PCL":
            return pearson_correlation_loss_per_slice(syn_pred, hrwi, mask)
        if HR_loss_mode == "MSE":
            return criterion(syn_pred, hrwi)
        

    # Permute inputs to the required format
    target = target.permute(1, 0, 2, 3)
    HRWI = HRWI.permute(1, 0, 2, 3)
    single_channel_backgound_mask = single_channel_backgound_mask.permute(1, 0, 2, 3)
    outputs = outputs.permute(1, 0, 2, 3)
    masked_syn_weighted_image_with_guide_seq = masked_syn_weighted_image_with_guide_seq.permute(1, 0, 2, 3)


    pred_T1W_from_fwd_model = mle_relaxometry.spin_echo(outputs, repetition_times_list)
    masked_pred_T1W = apply_mask(pred_T1W_from_fwd_model, single_channel_backgound_mask)
    masked_lr_pred_T1W = lower_res_with_mask(masked_pred_T1W, single_channel_backgound_mask, mask_size)

    o_pd_t2 = torch.stack([outputs[0], outputs[2]], dim=0)
    pred_T2W_from_fwd_model = mle_relaxometry.fse(o_pd_t2, echo_times_list)
    masked_pred_T2W = apply_mask(pred_T2W_from_fwd_model, single_channel_backgound_mask)
    masked_lr_pred_T2W = lower_res_with_mask(masked_pred_T2W, single_channel_backgound_mask, mask_size)

    LR_t1w_losses = []
    LR_t2w_losses = []
    if LR_loss_mode=='PCL':
        for i in range(0, 6):
            LR_t1w_losses.append(calculate_hr_loss(target[i].unsqueeze(0), masked_lr_pred_T1W[i].unsqueeze(0), single_channel_backgound_mask,"PCL",criterion))
        for i in range(0, 6):
            LR_t2w_losses.append(calculate_hr_loss(target[i+6].unsqueeze(0), masked_lr_pred_T2W[i].unsqueeze(0), single_channel_backgound_mask,'PCL',criterion))
    if LR_loss_mode=="MSE":
        for i in range(0, 6):
            LR_t1w_losses.append(criterion(target[i], masked_lr_pred_T1W[i]))
        for i in range(0, 6):
            LR_t2w_losses.append(criterion(target[6+i],masked_lr_pred_T2W[i]))
    LR_t1w_losses_sum = sum(LR_t1w_losses) 
    LR_t2w_losses_sum = sum(LR_t2w_losses) 

    # High-Resolution Loss
  
    hr_t1w_loss,hr_t2w_loss =0,0
    if "T1W" in HR_loss_modality:
        HRWI_selected_zscored_T1Wfse= z_score_per_slice(HRWI[0]).unsqueeze(0)
        hr_t1w_loss = calculate_hr_loss(z_score_per_slice(masked_syn_weighted_image_with_guide_seq[0]).unsqueeze(0), HRWI_selected_zscored_T1Wfse, single_channel_backgound_mask,HR_loss_mode,criterion)

    if "T1W_BRAVO" in HR_loss_modality:
        extra_w_for_bravo = 5
        HRWI_selected_zscored_T1WBRAVO= z_score_per_slice(HRWI[2]).unsqueeze(0)
        hr_t1w_loss = calculate_hr_loss(z_score_per_slice(masked_syn_weighted_image_with_guide_seq[2]).unsqueeze(0), HRWI_selected_zscored_T1WBRAVO, single_channel_backgound_mask,HR_loss_mode,criterion)*extra_w_for_bravo



    
    if "T2W" in HR_loss_modality:
        HRWI_selected_zscored_T2W= z_score_per_slice(HRWI[1]).unsqueeze(0)
        hr_t2w_loss = calculate_hr_loss(z_score_per_slice(masked_syn_weighted_image_with_guide_seq[1]).unsqueeze(0),HRWI_selected_zscored_T2W, single_channel_backgound_mask,HR_loss_mode,criterion)
        
    if "null" in HR_loss_modality:
        hr_t1w_loss,hr_t2w_loss =0,0

    return LR_t1w_losses_sum, LR_t2w_losses_sum,hr_t1w_loss,hr_t2w_loss

    ##########################################################################################################
def train_model(model, train_loader, val_loader, criterion,optimizer, num_epochs, save_interval, device, writer, file_path,params,guide_modality,HR_loss_modality,mask_size,model_type,HR_loss_weight,LR_t1w_loss_weight,LR_t2w_loss_weight,prof,HR_loss_mode,LR_loss_mode):
    prof.start()
    repetition_times_list=params['repetition_times_list']
    echo_times_list=params['echo_times_list']
    torch.autograd.set_detect_anomaly(True)

    for epoch in tqdm(range(num_epochs)):
        print(epoch)
        model.train()
        running_weighted_loss = 0.0
        running_loss_hr_t1w  = 0.0
        running_loss_hr_t2w = 0.0
        running_loss_lr_t2w = 0.0
        running_loss_lr_t1w  = 0.0
  

        total_batches = len(train_loader)

        for b_idx, ( _ ,HRWI, LRQM,lrmcwi,_ ) in enumerate(train_loader): #B,C,H,W
          
            optimizer.zero_grad() 
            _,HRWI , LRQM  ,lrmcwi,_= _ , HRWI.to(device, non_blocking=True) , LRQM.to(device, non_blocking=True),lrmcwi.to(device, non_blocking=True),_
        
            bg = HRWI[:,1].unsqueeze(1) # [b,1,h,w]
            single_channel_backgound_mask_ = (bg != 0)
            single_channel_backgound_mask=single_channel_backgound_mask_.float()
           
            
            outputs = predict_maps(model,HRWI,LRQM,guide_modality,model_type) 
            
            syn_weighted_image_with_guide_seq = predict_weighted_image(HR_loss_modality,outputs)
            masked_syn_weighted_image_with_guide_seq = syn_weighted_image_with_guide_seq*(single_channel_backgound_mask.expand(-1, 3, -1, -1)) if syn_weighted_image_with_guide_seq is not None else None

                                                           
            LR_t1w_losses_sum, LR_t2w_losses_sum,hr_t1w_loss,hr_t2w_loss = compute_loss(masked_syn_weighted_image_with_guide_seq,HRWI,lrmcwi,repetition_times_list,echo_times_list,single_channel_backgound_mask,outputs,criterion,mask_size,HR_loss_modality,HR_loss_mode,LR_loss_mode)
            
            loss_sum = ((hr_t1w_loss+hr_t2w_loss)*HR_loss_weight) + (0.1*LR_t1w_losses_sum*LR_t1w_loss_weight) + (0.1*LR_t2w_losses_sum*LR_t2w_loss_weight) 
            loss_sum.backward()
            optimizer.step()

            running_weighted_loss += loss_sum
            running_loss_hr_t1w += hr_t1w_loss*HR_loss_weight
            running_loss_hr_t2w += hr_t2w_loss*HR_loss_weight
            running_loss_lr_t2w += 0.1*LR_t2w_losses_sum*LR_t2w_loss_weight
            running_loss_lr_t1w += 0.1*LR_t1w_losses_sum*LR_t1w_loss_weight

            prof.step()
        prof.stop()

        average_running_weighted_loss = loss_sum / total_batches
        average_running_loss_hr_t1w = hr_t1w_loss*HR_loss_weight / total_batches
        average_running_loss_hr_t2w = hr_t2w_loss*HR_loss_weight / total_batches

        average_running_loss_lr_t2w = 0.1*LR_t2w_losses_sum*LR_t2w_loss_weight / total_batches
        average_running_loss_lr_t1w = 0.1*LR_t1w_losses_sum*LR_t1w_loss_weight / total_batches



        writer.add_scalar('Train/Loss', average_running_weighted_loss, epoch)
        writer.add_scalar('Train/Loss HR T1W', average_running_loss_hr_t1w, epoch)
        writer.add_scalar('Train/Loss HR T2W', average_running_loss_hr_t2w, epoch)
        writer.add_scalar('Train/Loss LR T2W', average_running_loss_lr_t2w, epoch)
        writer.add_scalar('Train/Loss LR T1W', average_running_loss_lr_t1w, epoch)
    

       
        # Validation phase
        model.eval()
        running_weighted_loss = 0.0
        running_loss_hr_t1w  = 0.0
        running_loss_hr_t2w = 0.0
        running_loss_lr_t2w = 0.0
        running_loss_lr_t1w  = 0.0

        total_val_batches = len(val_loader)
        with torch.no_grad():
            for batch_idx, ( _,HRWI, LRQM ,lrmcwi,brain_mask) in enumerate(val_loader):
                _ ,HRWI , LRQM ,lrmcwi,brain_mask = _ , HRWI.to(device, non_blocking=True) , LRQM.to(device, non_blocking=True),lrmcwi.to(device, non_blocking=True),brain_mask.to(device, non_blocking=True)
                if epoch== 0 and batch_idx==0:
                    #target_sampled= target
                    HRWI_sampled= HRWI
                    LRQM_sampled = LRQM
                
                bg = HRWI[:,1].unsqueeze(1) # [1,d,h,w]
                single_channel_backgound_mask_ = (bg != 0)
                single_channel_backgound_mask=single_channel_backgound_mask_.float()
        
      
                outputs = predict_maps(model,HRWI,LRQM,guide_modality,model_type) 
                syn_weighted_image_with_guide_seq = predict_weighted_image(HR_loss_modality,outputs)

                masked_syn_weighted_image_with_guide_seq = syn_weighted_image_with_guide_seq*(single_channel_backgound_mask.expand(-1, 3, -1, -1)) if syn_weighted_image_with_guide_seq is not None else None
                LR_t1w_losses_sum, LR_t2w_losses_sum, hr_t1w_loss,hr_t2w_loss = compute_loss(masked_syn_weighted_image_with_guide_seq,HRWI,lrmcwi,repetition_times_list,echo_times_list,single_channel_backgound_mask,outputs,criterion,mask_size,HR_loss_modality,HR_loss_mode,LR_loss_mode)
                
                loss_sum = ((hr_t1w_loss+hr_t2w_loss)*HR_loss_weight) + (0.1*LR_t1w_losses_sum*LR_t1w_loss_weight) + (0.1*LR_t2w_losses_sum*LR_t2w_loss_weight) 
  
                running_weighted_loss += loss_sum
                running_loss_hr_t1w += hr_t1w_loss*HR_loss_weight
                running_loss_hr_t2w += hr_t2w_loss*HR_loss_weight
                running_loss_lr_t2w += 0.1*LR_t2w_losses_sum*LR_t2w_loss_weight
                running_loss_lr_t1w += 0.1*LR_t1w_losses_sum*LR_t1w_loss_weight


        average_running_weighted_loss = loss_sum / total_val_batches
        average_running_loss_hr_t1w = hr_t1w_loss*HR_loss_weight / total_val_batches
        average_running_loss_hr_t2w = hr_t2w_loss*HR_loss_weight / total_val_batches
        average_running_loss_lr_t2w = 0.1*LR_t2w_losses_sum*LR_t2w_loss_weight / total_val_batches
        average_running_loss_lr_t1w = 0.1*LR_t1w_losses_sum*LR_t1w_loss_weight / total_val_batches



        writer.add_scalar('val/Loss', average_running_weighted_loss, epoch)
        writer.add_scalar('val/Loss HR T1W', average_running_loss_hr_t1w, epoch)
        writer.add_scalar('val/Loss HR T2W', average_running_loss_hr_t2w, epoch)
        writer.add_scalar('val/Loss LR T2W', average_running_loss_lr_t2w, epoch)
        writer.add_scalar('val/Loss LR T1W', average_running_loss_lr_t1w, epoch)
        # Save model
        if epoch%save_interval==0:
            #outputs_Sampled = predict_maps(model,HRWI_sampled,LRQM_sampled,guide_modality,model_type) 
            #t1_loss =  criterion(outputs_Sampled[:,1],target_sampled[:,1])
            #t2_loss =  criterion(outputs_Sampled[:,2],target_sampled[:,2])
            #pd_loss =  criterion(outputs_Sampled[:,0],target_sampled[:,0])


            #writer.add_scalar('T1 LOSS', t1_loss, epoch)
            #writer.add_scalar('T2 LOSS', t2_loss, epoch)
            #writer.add_scalar('PD LOSS', pd_loss, epoch)
          

            current_device = next(model.parameters()).device
        
            # Move the model to CPU for saving
            model_cpu = model.to('cpu')
            torch.save(model_cpu.state_dict(), file_path)
            model.to(current_device)
        
        # Move the model back to the original device
       
            
    model_cpu = model.to('cpu')
    torch.save(model_cpu.state_dict(), file_path)

    print("Model saved to", file_path)
    writer.close()