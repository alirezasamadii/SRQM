import torch
def generate_series_of_weighted_image(Qmaps, settings_dict):
    '''Generates a series of weighted images given the Qmaps and settings.

    Args:
    Qmaps (torch.Tensor): Tensor of size [B, C, D, H, W].
        B: Batch size
        C: Number of channels, where:
           - C=0 corresponds to proton density (PD)
           - C=1 corresponds to T1
           - C=2 corresponds to T2
        D: Depth (number of slices)
        H: Height
        W: Width

    settings_dict (dict): A dictionary with keys as modalities and values as lists of tuples.
        Each key represents a modality:
            - For example, modality "T1W" or modality = "T2FLAIRW" 
        Each value for modality is a list of tuples representing the acquisition settings for respective modality.
            - For spin echo sequence synthesizing T1W: [(TE, TR),(TE, TR)...]
             - or for spin echo inversion recovery synthesizing T2FLAIRW: [(TE, TR, TI)....]

    Returns:
    tuple: A tuple containing tensors of weighted image series.
        - T1W_series: Tensor of size [B, N, D, H, W] representing the T1-weighted image series.
        - T2FLAIR_series: Tensor of size [B, N, D, H, W] representing the T2 FLAIR-weighted image series.
        B: Batch size
        N: Number of images in the series (determined by the number of tuples in settings_dict)

    '''
    PD = Qmaps[:, 0, :, :, :]
    T1 = Qmaps[:, 1, :, :, :]
    T2 = Qmaps[:, 2, :, :, :]

    # Initialize series tensors
    T1W_series = None
    T2FLAIR_series = None

    # Process each modality
    for modality, params in settings_dict.items():
        if modality == "T1W":
            # Generate T1-weighted image series
            T1W_series = torch.stack([PD * (1 - torch.exp(-(param[1]) / T1)) * torch.exp(-param[0] / T2) for param in params], dim=1)

        elif modality == "T2FLAIR":
            # Generate T2 FLAIR-weighted image series
            T2FLAIR_series = torch.stack([PD * (1 - 2 * torch.exp(-param[2] / T1) + torch.exp(-(param[1]) / T1)) * torch.exp(-param[0] / T2) for param in params], dim=1)

    return T1W_series, T2FLAIR_series

   


def frequency_domain_downsampler(tensor,mask_size,mode):
    """
    Downsamples a tensor in the frequency domain by retaining a central portion of its k-space data.

    Args:
        tensor (torch.Tensor): Input tensor of size (B, C, D, H, W) where:
                               - B: batch size
                               - C: number of channels
                               - D: depth
                               - H: height
                               - W: width
        mask_size (float): Determines the fraction of the central region of k-space to retain.
                           For example, a mask_size of 0.25 retains the central inner part of Acceleration factor is approximately (1/mask_size)^2.
        mode (str): Mode of operation, can be 'preprocessing' or 'training':
                    - 'preprocessing': Processing is done on CPU.
                    - 'training': Processing is done on GPU if available, else on CPU.

    Returns:
        torch.Tensor: Downsampled tensor in the frequency domain.

    Raises:
        ValueError: If an invalid mode is provided.

    Example:
        # Downsampling during preprocessing
        downsampled_tensor = frequency_domain_downsampler(input_tensor, 0.25, 'preprocessing')
    """
    if mode=='preprocessing':
        device = torch.device('cpu')
    elif mode=='training':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    
    processed_channels=[]
    for c in range(tensor.size(1)):
        frequency_domain_data = torch.fft.fftn(tensor[:,c,:,:,:])
        frequency_domain_data = torch.fft.fftshift(frequency_domain_data)
        window_size = ( int(frequency_domain_data.shape[-1] * mask_size),
            int(frequency_domain_data.shape[-2] * mask_size),)
        
        center_x, center_y = frequency_domain_data.shape[-1] // 2, frequency_domain_data.shape[-2] // 2
        mask = torch.zeros(frequency_domain_data.shape, dtype=torch.complex64)
        mask[ center_x - window_size[0] // 2 : center_x + window_size[0] // 2,
            center_y - window_size[1] // 2 : center_y + window_size[1] // 2, ] = 1
        mask=mask.to(device)

        filtered_frequency_domain_data = frequency_domain_data * mask
        reconstructed_image = torch.fft.ifftn(torch.fft.ifftshift(filtered_frequency_domain_data)).real
        #relu = torch.nn.ReLU()
        #reconstructed_image=relu(reconstructed_image)
        processed_channels.append(reconstructed_image)
    reconstructed_image = torch.stack(processed_channels, dim=0)
    

    return reconstructed_image.to(device)
    
    
               