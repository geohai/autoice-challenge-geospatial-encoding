

from testv3 import test
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from PIL import Image
from matplotlib.colors import Normalize
import xarray as xr
from mpl_toolkits.axes_grid1 import make_axes_locatable
#first run the generate patches and put the dir address in the dir_train_with_icecharts
seed_value = 42

# Set the seed for PyTorch
torch.manual_seed(seed_value)
np.random.seed(seed_value)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed(seed_value)
#     torch.cuda.manual_seed_all(seed_value)


def apply_gradcam(chart_label , target_category = 0):  
    dataset_test , test_loader ,model, test_options, device = test( )
    device = 'cpu'
    model_name = '86.90.ckpt'
    model_path = '{}'.format(model_name)
    #checkpoint = torch.load(model_path)
    #model.load_state_dict(torch.load(model_path)['model_state_dict'])
    model.load_state_dict(torch.load(model_path, map_location=device)['model_state_dict'])

    # state_dict = torch.load(model_path)['model_state_dict']

    # new_state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()}

    # # Load the adjusted state dictionary
    # model.load_state_dict(new_state_dict)
    print("model loaded")
    model.to(device)
    model.eval()
    
    if chart_label == 'SOD':
        num_classes = 6
        
        class_names = {
            0: 'Open water',
            1: 'New Ice',
            2: 'Young ice',
            3: 'Thin FYI',
            4: 'Thick FYI',
            5: 'Old ice'
        }
        fig, axs = plt.subplots(2, 3, figsize=(18, 12))
        
        #target_layer = model.sod_decoder[-1]
        target_layer = model.sod_decoder[1]
    elif chart_label == 'FLOE':
        num_classes = 7
        
        class_names = {
            0: 'Open water',
            1: 'Cake Ice',
            2: 'Small floe',
            3: 'Medium floe',
            4: 'Big floe',
            5: 'Vast floe',
            6: 'Bergs'
        }
        fig, axs = plt.subplots(3, 3, figsize=(30, 30))
        #target_layer = model.floe_decoder[-1]
        target_layer = model.floe_decoder[1]

    else:
        raise ValueError(f"Invalid chart label: {chart_label}")

    # Create a GradCAM explainer
    class SegmentationModelOutputWrapper(torch.nn.Module):
        def __init__(self, model): 
            super(SegmentationModelOutputWrapper, self).__init__()
            self.model = model
            
        def forward(self, x):
            return self.model(x)[chart_label]
    class SemanticSegmentationTarget:
        def __init__(self, category, mask):
            self.category = category
            self.mask = torch.from_numpy(mask)
            # if torch.cuda.is_available():
            #     self.mask = self.mask.cuda()
            
        def __call__(self, model_output):
            return (model_output[self.category, :, : ] * self.mask).sum()
        
    sample_input, sample_target , sample_mask_, sample_name = next(iter(test_loader))
    #sample_input, sample_target = next(iter(test_loader))
    sample_input = sample_input.to(device)
    print("shape of sample input is ", sample_input.shape)
    sample = sample_input[0].cpu().numpy().transpose(1, 2, 0)
    
    #print("name of test file is", sample_name)
    print("labels in this file ", np.unique(sample_target[chart_label].cpu().numpy()))
 

    model_wrapper = SegmentationModelOutputWrapper(model)
    model_wrapper.to(device)
    model_wrapper.eval()

    # Create a GradCAM explainer
    target_layers = [target_layer]
    target_mask = (sample_target[chart_label] == target_category).float().numpy()
    target = [SemanticSegmentationTarget(target_category, target_mask[0])]

    cam = GradCAM (model = model_wrapper,target_layers = target_layers)
    
    # Generate the CAM
    grayscale_cam = cam(input_tensor=sample_input, targets=target)[0, :]
    print("unique values in grayscale cam", np.unique(np.array(grayscale_cam)))
    # Normalize grayscale_cam to be between 0 and 1, handling potential issues
    grayscale_cam_min = np.min(grayscale_cam)
    grayscale_cam_max = np.max(grayscale_cam)
    if grayscale_cam_min == grayscale_cam_max:
        print("Warning: grayscale_cam has constant values")
        grayscale_cam_normalized = np.zeros_like(grayscale_cam)
    else:
        grayscale_cam_normalized = (grayscale_cam - grayscale_cam_min) / (grayscale_cam_max - grayscale_cam_min)
    
    # Replace any NaN or Inf values with 0
    grayscale_cam_normalized = np.nan_to_num(grayscale_cam_normalized, nan=0, posinf=1, neginf=0)

   
    # Display the CAM
    files_test_path = 'F:\data\Test' # change it to your path
    files_test_name = '20200217T102731_cis_prep.nc'
    scene = xr.open_dataset(os.path.join(files_test_path, files_test_name))
    SCENE_VARIABLES = [
                        'nersc_sar_primary', 
                        'nersc_sar_secondary',
                        'sar_incidenceangle']
    channels = [SCENE_VARIABLES[0], SCENE_VARIABLES[1], SCENE_VARIABLES[2]]
    normalized_channels = []
    for channel in channels:
        data = scene[channel].values
        data[data == scene[channel].attrs['variable_fill_value']] = np.nan
        # norm = Normalize(vmin=np.nanquantile(data, q=0.025), vmax=np.nanquantile(data, q=0.975))
        # normalized_data = norm(data)
        normalized_data = (data - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data))
        normalized_data = np.nan_to_num(normalized_data, nan=0.0)
        normalized_channels.append(normalized_data)
    rgb_sample = np.stack(normalized_channels, axis=-1)
    rgb_sample = np.clip(rgb_sample, 0, 1).astype(np.float32)
    
    
    
    cam_image = show_cam_on_image( rgb_sample , grayscale_cam, use_rgb=True)
    
    cam_pil_image = Image.fromarray(cam_image)
    # Save the image
    cam_pil_image.save('gradcam_overlay.png')

    # Create a figure and display the CAM overlay
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cam_image , cmap='jet' , vmin=0 , vmax=1)
    ax.axis('off')
    #ax.set_title('GradCAM Overlay')
    #cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
    #cbar.set_label('Activation Intensity', rotation=270, labelpad=15)

    #ax.set_title('GradCAM Overlay', fontsize=16)  # Increased font size for the title
    cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
    cbar.set_label('Activation Intensity', rotation=270, labelpad=32, fontsize=22)
    cbar.ax.tick_params(labelsize=20)

    plt.tight_layout()
    plt.savefig('gradcam_overlay.png', dpi=300, bbox_inches='tight')
    print("GradCAM overlay with colorbar saved as 'gradcam_overlay.png'")
    plt.close(fig) 

    # Display the raw  CAM
    fig, ax = plt.subplots(figsize=(18, 12))

    # Plot original image
    ax.imshow(rgb_sample, alpha=0.6)
    ax.set_title('Original Image with GradCAM Overlay', fontsize=16)
    ax.axis('off')

    # Plot CAM overlay on the same axis
    cam_plot = ax.imshow(cam_image, cmap='jet', alpha=0.4, vmin=0, vmax=1)
    class_name = class_names[target_category]

    # Add colorbar
    cbar = fig.colorbar(cam_plot, ax=ax, orientation='vertical', fraction=0.035, pad=0.02)
    cbar.set_label('Activation Intensity', rotation=270, labelpad=15)

    # Add main title
    fig.suptitle(f'Original Image vs GradCAM Visualization\nClass: {class_name}', fontsize=20)

    # Adjust layout and save the figure
    plt.tight_layout()
    safe_filename = os.path.basename(sample_name).replace('.nc', '')
    plt.savefig(f'gradcam_original_and_overlay_{safe_filename}.png')
    print(f"Original image and GradCAM overlay saved as 'gradcam_original_and_overlay_{safe_filename}.png'")


if __name__ == '__main__':
    apply_gradcam('SOD',  3)


