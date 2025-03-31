
from utils import (CHARTS, FLOE_LOOKUP, SCENE_VARIABLES, SIC_LOOKUP,
                   SOD_LOOKUP, colour_str, LOOKUP_NAMES)
from functions import chart_cbar
import matplotlib.pyplot as plt
import copy
import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset
import torch.nn.functional as F
import gc
import json
import os
import time

# load xarray dataset from file
def load_dataset(scene_name):
  file_path = os.path.join(os.environ['AI4ARCTIC_DATA'], scene_name)
  return xr.open_dataset(file_path)


# plot a single chart
def plot_chart(scene):
  fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12, 8))
  for idx, chart in enumerate(CHARTS):
    scene[chart] = scene[chart].astype(
        float
    )  # Convert charts from uint8 to float to enable nans in the arrays,
    # replace chart fill values with nan for better visualization.
    scene[chart].values[scene[chart] ==
                        scene[chart].attrs['chart_fill_value']] = np.nan
    axs[idx].imshow(scene[chart].values,
                    vmin=0,
                    vmax=LOOKUP_NAMES[chart]['n_classes'] - 2,
                    cmap='jet',
                    interpolation='nearest')
    chart_cbar(ax=axs[idx],
               n_classes=LOOKUP_NAMES[chart]['n_classes'],
               chart=chart,
               cmap='jet')  # Creates colorbar with ice class names.

  plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.75, hspace=0)
  fig.savefig(f"inference/{scene_name}.png",
              format='png',
              dpi=128,
              bbox_inches="tight")
  plt.close('all')

  # plot AMSR2 channels from introduction.ipynb
  # There is no mask in these scene variables.
  # fig, axs = plt.subplots(nrows=4,
  #                         ncols=4,
  #                         figsize=(12, 12),
  #                         constrained_layout=True)
  # for idx, amsr2_channel in enumerate(SCENE_VARIABLES):
  #   ax = axs[idx // 4, idx % 4]
  #   ax.set_title(amsr2_channel)
  #   channel = inf_x[:, idx, :, :].squeeze().cpu().numpy()
  #   # channel[nmasks] = np.nan
  #   im = ax.imshow(channel)
  #   ax.set_xticks([])
  #   ax.set_yticks([])
  #   plt.colorbar(im, ax=ax, fraction=0.0485, pad=0.049)

  # fig.suptitle(f"AMSR2 Brightness Temperatures - Scene: {scene_name}",
  #              fontweight='bold')
  # [axs[-1, col].axis('off') for col in range(2, 4)]

  # fig.savefig(f"inference/{scene_name}_AMSR_masked.png",
  #             format='png',
  #             dpi=128,
  #             bbox_inches="tight")
  # plt.close('all')

