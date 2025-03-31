import os

import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

import torch

from functions import chart_cbar
from utils import ICE_STRINGS, GROUP_NAMES



def plot_model_predictions(
    output,
    inf_y,
    masks,
    scene_name,
    train_options,
):
  """Plot and save the model predictions and ground truth."""
  os.makedirs('inference', exist_ok=True)
  fig = plt.figure(figsize=(16, 10), constrained_layout=True)
  fig.suptitle(f"Scene: {scene_name}", fontsize=14)
  subfigs = fig.subfigures(nrows=2, ncols=1)
  titles = ['Model Predictions', 'Ground Truth']
  charts = [chart for chart in train_options['charts'] if chart in ['SOD', 'FLOE']]

  for row, subfig in enumerate(subfigs):
    subfig.suptitle(titles[row], fontweight='bold', y=0.98)
    axs = subfig.subplots(nrows=1, ncols=len(charts))
    #axs = subfig.subplots(nrows=1, ncols=3)
    for idx, chart in enumerate(charts):
      ax = axs[idx]
      nmasks = masks[chart].numpy()
      if row == 0:  # plot predictions
        pred_chart = output[chart].astype(float)
        pred_chart = torch.from_numpy(pred_chart)
        pred_chart[nmasks] = np.nan
        ax.imshow(pred_chart,
                  vmin=0,
                  vmax=train_options['n_classes'][chart] - 2,
                  cmap='jet',
                  interpolation='nearest')
      else:  # plot ground truth
        gt_chart = inf_y[chart].squeeze().float()
        gt_chart[nmasks] = np.nan
        ax.imshow(gt_chart,
                  vmin=0,
                  vmax=train_options['n_classes'][chart] - 2,
                  cmap='jet',
                  interpolation='nearest')
      ax.set_xticks([])
      ax.set_yticks([])
      chart_cbar(ax=ax,
                 n_classes=train_options['n_classes'][chart],
                 chart=chart,
                 cmap='jet')

  fig.savefig(f"inference/{scene_name}.png",
              format='png',
              dpi=128,
              bbox_inches="tight")
  plt.close('all')


def plot_scene_confusion_matrix(inf_y,
                                output,
                                masks,
                                train_options,
                                scene_name,
                                normalization='all'):
  """Plot and save the confusion matrices for each chart and each scene."""
  os.makedirs('confusion_matrices', exist_ok=True)
  charts = [chart for chart in train_options['charts'] if chart in ['SOD', 'FLOE']]
  fig, axs = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
  fig.suptitle(f'Confusion Matrices for {scene_name}', fontsize=14)
  

  for idx, chart in enumerate(charts):
    y_true = np.squeeze(inf_y[chart], 0)[~masks[chart]].numpy()
    y_pred = output[chart][~masks[chart]]
    n_classes = train_options['n_classes'][chart]
    cm = confusion_matrix(y_true,
                          y_pred,
                          labels=range(n_classes - 1),
                          normalize=normalization)
    sns.heatmap(cm,
                annot=True,
                fmt='.1f',
                cmap='Blues',
                xticklabels=list(GROUP_NAMES[chart].values()),
                yticklabels=list(GROUP_NAMES[chart].values()),
                ax=axs[idx])
    axs[idx].set_yticklabels(axs[idx].get_yticklabels(), rotation=45)
    axs[idx].set_xticklabels(axs[idx].get_xticklabels(), rotation=45)
    axs[idx].set_title(ICE_STRINGS[chart])
    axs[idx].set_xlabel('Predicted')
    axs[idx].set_ylabel('True')

  plt.savefig(f"confusion_matrices/{scene_name}.png")
  plt.close()


def plot_test_confusion_matrix(inf_ys_flat,
                               outputs_flat,
                               train_options,
                               normalization='all',
                               model_name=None):
  """Plot and save the confusion matrices for the entire test set"""

  os.makedirs('confusion_matrices', exist_ok=True)
  # Filter the charts to include only SOD and FLOE
  charts = [chart for chart in train_options['charts'] if chart in ['SOD', 'FLOE']]
    
  fig, axs = plt.subplots(1, len(charts), figsize=(8 * len(charts), 7), constrained_layout=True)
  fig.suptitle(f'Confusion Matrices for the Test Set', fontsize=14)

  # os.makedirs('confusion_matrices', exist_ok=True)
  # charts = [chart for chart in train_options['charts'] if chart in ['SOD', 'FLOE']]
  # fig, axs = plt.subplots(1, 3, figsize=(24, 7), constrained_layout=True)
  # fig.suptitle(f'Confusion Matrices for the Test Set', fontsize=14)

  for idx, chart in enumerate(charts):
    y_true = inf_ys_flat[chart]
    y_pred = outputs_flat[chart]
    n_classes = train_options['n_classes'][chart]
    cm = confusion_matrix(y_true,
                          y_pred,
                          labels=range(n_classes - 1),
                          normalize=normalization)
    sns.heatmap(cm,
                annot=True,
                fmt='.1f',
                cmap='Blues',
                xticklabels=list(GROUP_NAMES[chart].values()),
                yticklabels=list(GROUP_NAMES[chart].values()),
                ax=axs[idx])
    axs[idx].set_xticklabels(axs[idx].get_xticklabels(), rotation=45)
    axs[idx].set_yticklabels(axs[idx].get_yticklabels(), rotation=45)
    axs[idx].set_title(ICE_STRINGS[chart])
    axs[idx].set_xlabel('Predicted')
    axs[idx].set_ylabel('True')
  if model_name is not None:
    plt.savefig(f"confusion_matrices/test_set_{model_name}.png")
  else:
    plt.savefig(
        f"confusion_matrices/test_set_{train_options['model_name']}.png")
  plt.close()


def plot_class_frequencies(inf_y, train_options, model_name=None):
  """Plot and save the class frequencies"""
  os.makedirs('frequencies', exist_ok=True)
  charts = [chart for chart in train_options['charts'] if chart in ['SOD', 'FLOE']]

  fig, axs = plt.subplots(1, 2, figsize=(24, 7), constrained_layout=True)
  fig.suptitle(f'Class Frequencies for the Test Set', fontsize=14)
  
  for idx, chart in enumerate(charts):
    # Get unique values and their frequencies
    unique_values, frequencies = np.unique(inf_y[chart], return_counts=True)

    # Calculate the total number of samples
    total_samples = len(inf_y[chart])

    # Calculate the percentage of each class
    percentages = (frequencies / total_samples) * 100

    # Plotting the class frequencies as percentages
    axs[idx].bar(unique_values, percentages, color='skyblue')
    axs[idx].set_xlabel('Class')
    axs[idx].set_ylabel('Frequency (%)')
    axs[idx].set_title(ICE_STRINGS[chart])
    axs[idx].set_xticks(unique_values)
    axs[idx].set_ylim(0, 100)
    xtick_labels = [GROUP_NAMES[chart][val] for val in unique_values]
    axs[idx].set_xticklabels(xtick_labels)
    # Add percentage labels above the bars
    for i, percentage in enumerate(percentages):
      axs[idx].text(unique_values[i],
                    percentage + 1,
                    f'{percentage:.2f}%',
                    ha='center',
                    va='bottom')

  if model_name is not None:
    plt.savefig(f"frequencies/test_set_{model_name}.png")
  else:
    plt.savefig(f"frequencies/test_set_{train_options['model_name']}.png")
  plt.close()
