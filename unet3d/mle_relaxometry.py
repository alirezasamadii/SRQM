import torch
from tqdm import tqdm
from unet3d.utils import compute_zscore
device = torch.device('cuda' if torch.cuda.is_available()  else 'cpu')
epsilon=0.0001
def initial_params(weighted_image_series,relaxometry_mode,single_channel_bg_mask):
    
    PD_initial_guess=torch.ones_like(weighted_image_series)[0].to(device)

    if relaxometry_mode=='T1_spinEcho':
        T1_initial_guess =torch.ones_like(PD_initial_guess).to(device)*1000
        initial_guess= torch.stack([PD_initial_guess, T1_initial_guess]) 
        return initial_guess
    
    elif relaxometry_mode =='T2_fse':
        T2_initial_guess =torch.ones_like(PD_initial_guess).to(device)*100
        initial_guess=torch.stack([PD_initial_guess, T2_initial_guess])
        return initial_guess

    


   

def Looklocker(kappa, inversion_times):
        predicted_images = []
        for tau in inversion_times:
            weighted_image = torch.abs(torch.abs(kappa[0]) * (1 -   torch.exp(-tau/ torch.abs(kappa[1]))))
            predicted_images.append(weighted_image)
        return torch.stack(predicted_images)
    
    
def fse(kappa, echo_times):
    
    predicted_images = []
    PD = torch.abs(kappa[0])
    T2 =  torch.abs(kappa[1]) +epsilon

    for tau in echo_times:
        weighted_image= torch.abs(PD * ( torch.exp((-tau)/ T2) ) )
        predicted_images.append(weighted_image)

    return torch.stack(predicted_images)
        
def spin_echo(kappa, repetition_times_list):
    
    PD=torch.abs(kappa[0])
    T1= torch.abs(kappa[1])+epsilon
    predicted_images = []
   
  
    for TR in repetition_times_list:
        weighted_image =  torch.abs(PD*(1 -  torch.exp(-(TR) / (T1)))) #spin echo with T2 term equal to one 
        predicted_images.append(weighted_image)
    return torch.stack(predicted_images)
    
    
# Define the negative log likelihood function
def likelihood(modeled_map, weighted_images, taus, relaxometry_mode,single_channel_bg_mask):
    device = torch.device('cuda' if torch.cuda.is_available()  else 'cpu')
    #if relaxometry_mode == 'T1_looklocker':
    #  forward_model = Looklocker
    if relaxometry_mode=='T1_spinEcho':
        forward_model=spin_echo
    elif relaxometry_mode == 'T2_fse':
        forward_model = fse
    elif relaxometry_mode=='T1_looklocker':
        forward_model=Looklocker
    else:
        raise ValueError("only 'T1_looklocker' 'T1_spinEcho' or 'T2' relaxometry modes supported")
    
    single_channel_bg_mask=single_channel_bg_mask.to(device)
    modeled_weighted_images = forward_model(modeled_map, taus)*(single_channel_bg_mask.expand(len(taus),-1,-1,-1))
    error = modeled_weighted_images - weighted_images.to(device)*(single_channel_bg_mask.expand(len(taus),-1,-1,-1))
    likelihood = torch.sum(error ** 2)  
    return likelihood
def apply_mask(params):
    threshold=0.5
    first_channel = params[0]

    # Create a mask for zero values in the first channel(PD)
    background_mask = torch.abs(first_channel) <= threshold

    # Apply the mask to the second channel (T1,or T2)
    second_channel = params[1].clone()
    second_channel[background_mask] = 0

    # Return the final tensor
    params = torch.stack([first_channel, second_channel])

    return params
# Define the optimization function to estimate qMRI maps
def estimate_qMRI_maps(weighted_images, taus, relaxometry_mode,num_iterations,learning_rate,single_channel_bg_mask):
    single_channel_bg_mask=single_channel_bg_mask.to(device)
    initial_guess=initial_params(weighted_images,relaxometry_mode,single_channel_bg_mask)
    # Convert initial_guess to a parameter that requires gradients
    initial_guess = torch.nn.Parameter(initial_guess)
    
    # Create the Adam optimizer
    optimizer = torch.optim.Adam([initial_guess], lr=learning_rate)  
    

    progress_bar = tqdm(total=num_iterations, leave=True)
    for step in range(num_iterations):
        optimizer.zero_grad()
        

        loss = likelihood(initial_guess, weighted_images, taus, relaxometry_mode,single_channel_bg_mask)
        loss.backward()
        optimizer.step()
        if step % 200 == 0:  # Print loss and update progress bar every 200 iterations
            progress_bar.set_description(f"Loss: {loss:.4f}")
            progress_bar.update(200)
    
    # Close the progress bar
    progress_bar.close()
    #initial_guess=apply_mask(initial_guess).clone().detach()
    #print("loss:",loss)
    # Return the estimated qMRI maps
    return torch.abs(initial_guess)
