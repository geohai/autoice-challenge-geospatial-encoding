from captum.attr import GradientShap
from testv3 import test
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Set the seed for reproducibility
seed_value = 42
torch.manual_seed(seed_value)
np.random.seed(seed_value)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

def sod_floe_analysis(chart_label, batch_size, patch_size):
    dataset_test, test_loader, model, test_options, device = test(chart_label, batch_size, patch_size)
    model_name = '60.27.ckpt'
    model_path = '{}'.format(model_name)
    checkpoint = torch.load(model_path)
    #model.load_state_dict(torch.load(model_path)['model_state_dict'])

    state_dict = torch.load(model_path)['model_state_dict']
    new_state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()}
    model.load_state_dict(new_state_dict)

    print("Model loaded")
    model.to(device)
    model.eval()
    
    save_path = 'captum_plots'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    features_name = ['HH', 'HV', 'Incidence Angle',
                     'Sine of Longitude', 'Cosine of Longitude', 'Sine of Latitude', 'Cosine of Latitude', 'Distance map',
                     'AMSR2 18.7 GHz (h)', 'AMSR2 18.7 GHz (v)', 'AMSR2 36.5 GHz (h)', 'AMSR2 36.5 GHz (v)', 'wind speed (eastward)',
                     'wind speed (northward)', '2-m air temperature', 'skin temperature', 'water vapor', 'cloud liquid water']

    def agg_segmentation_wrapper(inp):
        model_out_dict = model(inp)
        model_out = model_out_dict[chart_label]
        out_max = model_out.argmax(dim=1)
        selected_inds = torch.zeros_like(model_out).scatter_(1, out_max.unsqueeze(1), 1)
        ignore_mask = (out_max != 255).unsqueeze(1).float()
        return (model_out * selected_inds).sum(dim=(2, 3))

    def aggregate_attributions(attributions):
        return attributions.sum(dim=(2, 3)).mean(dim=0).cpu().detach().numpy()

    def clear_cuda():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    if chart_label == 'SOD':
        num_classes = 6
        target_layer = model.sod_decoder[-1]
        class_names = {
            0: 'Open water',
            1: 'New Ice',
            2: 'Young ice',
            3: 'Thin FYI',
            4: 'Thick FYI',
            5: 'Old ice'
        }
        fig, axs = plt.subplots(2, 3, figsize=(18, 12))
    elif chart_label == 'FLOE':
        num_classes = 7
        target_layer = model.floe_decoder[-1]
        class_names = {
            0: 'Open water',
            1: 'Cake Ice',
            2: 'Small floe',
            3: 'Medium floe',
            4: 'Big floe',
            5: 'Vast floe',
            6: 'Bergs'
        }
        fig, axs = plt.subplots(3, 3, figsize=(20, 18))
    else:
        raise ValueError(f"Invalid chart label: {chart_label}")

    axs = axs.flatten()
    num_input_channels = len(features_name)
    height = patch_size
    width = patch_size
    feature_mask = torch.arange(num_input_channels).view(1, num_input_channels, 1, 1).expand(batch_size, -1, height, width)

    avg_gradient_shap = np.zeros((num_classes, num_input_channels))

    print("Epoch length:", len(test_loader))

    for input_data, target_data in test_loader:
        input_data = input_data.to(device)
        print("Input data shape:", input_data.shape)

        for target in range(num_classes):
            gradient_shap = GradientShap(agg_segmentation_wrapper)
            baseline_dist = torch.zeros_like(input_data)
            attributions = gradient_shap.attribute(input_data, target=target, baselines=baseline_dist)
            gradient_shap_agg = aggregate_attributions(attributions)

            # Normalize
            norm = np.linalg.norm(gradient_shap_agg, ord=1)
            normalized_gradient_shap = gradient_shap_agg / norm if norm != 0 else gradient_shap_agg
            avg_gradient_shap[target] += normalized_gradient_shap

        clear_cuda()

    for target in range(num_classes):
        ax = axs[target]
        width = 0.35
        indices = np.arange(len(features_name))

        # Set colors: red for negative and blue for positive
        colors = ['#D9534F' if val < 0 else '#397bb3' for val in avg_gradient_shap[target]]

        ax.barh(indices, avg_gradient_shap[target], width, color=colors)

        ax.set_yticks(indices)
        ax.set_yticklabels(features_name, fontsize=14)
        ax.set_xlabel('Feature Importance', fontsize=18)
        ax.set_title(f'{class_names[target]}', fontsize=20)
        ax.invert_yaxis()
        ax.set_xlim(-1, 1)
        ax.tick_params(axis='x', labelsize=14)
        ax.grid(axis='x', linestyle='--', alpha=0.7)

    if chart_label == 'FLOE':
        for idx in range(num_classes, len(axs)):
            fig.delaxes(axs[idx])

    # Custom legend for color meaning
    red_patch = mpatches.Patch(color='#D9534F', label='Negative Impact')
    blue_patch = mpatches.Patch(color='#397bb3', label='Positive Impact')
    fig.legend(handles=[red_patch, blue_patch], loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=20)

    plt.tight_layout(rect=[0, 0.03, 1, 0.90])
    plt.savefig(f'captum_plots/feature_importance_deeplab_all_targets_{chart_label}_GradientShap.png')

def sic_analysis_over_batch(chart_label, batch_size, patch_size):
    attribution = 'GradientShap'
    dataset_test, test_loader, model, test_options, device = test(chart_label, batch_size, patch_size)
    model_name = '60.27.ckpt'
    model_path = '{}'.format(model_name)
    checkpoint = torch.load(model_path)
    #model.load_state_dict(torch.load(model_path)['model_state_dict'])

    state_dict = torch.load(model_path)['model_state_dict']
    new_state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()}
    model.load_state_dict(new_state_dict)

    model.to(device)
    model.eval()

    save_path = 'captum_plots'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    features_name = ['HH', 'HV', 'Incidence Angle',
                     'Sine of Longitude', 'Cosine of Longitude', 'Sine of Latitude', 'Cosine of Latitude', 'Distance map',
                     'AMSR2 18.7 GHz (h)', 'AMSR2 18.7 GHz (v)', 'AMSR2 36.5 GHz (h)', 'AMSR2 36.5 GHz (v)', 'wind speed (eastward)',
                     'wind speed (northward)', '2-m air temperature', 'skin temperature', 'water vapor', 'cloud liquid water']

    num_input_channels = len(features_name)
    test_loader_iter = iter(test_loader)

    def agg_segmentation_wrapper(inp):
        model_out_dict = model(inp)
        model_out = model_out_dict['SIC'].squeeze(1)
        return model_out.sum(dim=(1, 2))

    def aggregate_attributions(attributions):
        return attributions.sum(dim=(2, 3)).mean(dim=0).cpu().detach().numpy()

    def clear_cuda():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    fig, ax = plt.subplots(figsize=(18, 12))

    num_batches = len(test_loader)
    avg_gradient_shap_scores = 0

    print("Epoch length:", len(test_loader))
    for input_data, _ in test_loader:
        input_data = input_data.to(device)

        gradient_shap = GradientShap(agg_segmentation_wrapper)
        baseline_dist = torch.zeros_like(input_data)
        gs_attr = gradient_shap.attribute(input_data, baselines=baseline_dist)
        gs_aggregated = aggregate_attributions(gs_attr)
        norm_gs = np.linalg.norm(gs_aggregated, ord=1)
        normalized_gs = gs_aggregated / norm_gs if norm_gs != 0 else gs_aggregated
        avg_gradient_shap_scores += normalized_gs

        clear_cuda()

    indices = np.arange(len(features_name))
    width = 0.25

    # Set colors: red for negative and blue for positive
    colors = ['#D9534F' if val < 0 else '#397bb3' for val in avg_gradient_shap_scores]

    ax.barh(indices, avg_gradient_shap_scores, width, color=colors)

    ax.set_yticks(indices)
    ax.set_yticklabels(features_name, fontsize=22)
    ax.set_xlabel('Feature Importance', fontsize=26)
    ax.invert_yaxis()
    ax.set_xlim(-1, 1)
    ax.tick_params(axis='x', labelsize=22)
    ax.grid(axis='x', linestyle='--', alpha=0.7)

    # Custom legend for color meaning
    red_patch = mpatches.Patch(color='#D9534F', label='Negative Impact')
    blue_patch = mpatches.Patch(color='#397bb3', label='Positive Impact')
    fig.legend(handles=[red_patch, blue_patch], loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=2, fontsize=26)

    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.savefig('captum_plots/feature_importance_deeplab_sic_regression_overbatch_GradientShap.png')


if __name__ == '__main__':
    chart_label = 'SIC'
    batch_size = 16
    patch_size = 256
    #sod_floe_analysis(chart_label, batch_size, patch_size)
    sic_analysis_over_batch(chart_label, batch_size, patch_size)

