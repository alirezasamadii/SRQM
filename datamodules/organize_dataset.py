#this is an adhoc module

def move_files_by_keyword(src_dir, keyword, dest_folder): 
    """
    Organize files in a source directory by moving them to separate folders based on a specified keyword.

    Parameters:
    - src_dir (str): Source directory containing the files to be organized.
    - keyword (str): Keyword used to filter files for moving.
    - dest_folder (str): Destination folder where files matching the keyword will be moved(within source directory).

    Example Usage:
    Suppose you have a test root directory containing different  DICOM images in the same folder of different modalities(T1,T2,PD).
    This function is designed to move each modality to a separate folder based on a specified keyword.
    
    Example:
    >>> move_files_by_keyword('TEST ROOT', 'PD', 'PD')
    
    In this example, files containing the keyword 'PD' in their names from the 'TEST ROOT' directory
    will be moved to a folder named 'PD' within the same directory.
    """
    # Ensure the destination folder exists, create it if not 
    dest_path = os.path.join(src_dir, dest_folder)
    os.makedirs(dest_path, exist_ok=True)

    # List all files in the source directory
    items = os.listdir(src_dir)

    # Iterate through items and move files containing the keyword to the destination folder
    for item in items:
        item_path = os.path.join(src_dir, item)
        if os.path.isfile(item_path) and keyword in item:
            dest_item_path = os.path.join(dest_path, item)
            shutil.move(item_path, dest_item_path)


def process_full_dataset(base_dir):

    """
    Process the MRACE dataset by organizing DICOM files, moving them based on keyword, and converting to NIfTI format.

    Parameters:
    - base_dir (str): Base directory containing MRACE dataset studies.

    Example Usage:
    >>> process_full_dataset('path/to/MRACE_dataset')

    Description:
    This function is designed to process the MRACE dataset, iterating over different studies in the specified 'base_dir'.
    It checks for the existence of a 'SYMAPS_' and uses the 'move_files_by_keyword' function to organize DICOM files
    within each study. After having all DICOM files in their designated places, the function converts them to the NIfTI format,
    making them ready to be used by a custom dataset.

    Note:
    The 'move_files_by_keyword' function is expected to be defined(check above) and available for use within this script.
    """
    # List all items (files and folders) in the base directory
    studies = os.listdir(base_dir)

    # Iterate through items in the base directory
    for item in studies:
        item_path = os.path.join(base_dir, item)

        # Check if the item is a directory
        if os.path.isdir(item_path):
            symaps_folder_found = False

            # List sub-items in the current directory
            sub_items = os.listdir(item_path)

            # Check if any sub-item is a directory containing "SYMAPS" in its name
            for sub_item in sub_items: #each folder in study
                sub_item_path = os.path.join(item_path, sub_item)

                if os.path.isdir(sub_item_path) and "SYMAPS" in sub_item:
                    symaps_folder_found = True
                    move_files_by_keyword(sub_item_path,"PD","PD")
                    move_files_by_keyword(sub_item_path,"T1","T1")
                    move_files_by_keyword(sub_item_path,"T2","T2")
                    dicom2nifti.convert_directory(sub_item_path+"/PD/",sub_item_path+"/PD/")
                    dicom2nifti.convert_directory(sub_item_path+"/T1/",sub_item_path+"/T1/")
                    dicom2nifti.convert_directory(sub_item_path+"/T2/",sub_item_path+"/T2/")

                    # Perform your operation here for the folder containing "SYMAPS"
                    #print("Performing operation on ", sub_item_path)
                if os.path.isdir(sub_item_path) and "FLAIR" in sub_item:
                    dicom2nifti.convert_directory(sub_item_path, sub_item_path)
                if os.path.isdir(sub_item_path) and "T1W" in sub_item:
                    dicom2nifti.convert_directory(sub_item_path, sub_item_path)

            # If "SYMAPS" folder is not found in the current folder, print a message
            if not symaps_folder_found:
                print(item+ " Folder does not have a folder named 'SYMAPS'.")