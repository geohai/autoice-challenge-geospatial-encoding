# Sea Ice Charting using the AutoICE  Benchmark

This repository contains scripts that can be used to train a  customized multi-task DeepLabV3 model to map sea ice parameters including Sea Ice Concentration (SIC), Stage of Development (SoD) and floe Size (FLOE) using the AutoICE challenge data.

![image](https://github.com/user-attachments/assets/8afa1f60-4197-4e6b-8167-a09ea0ad1b87)
Model architecture. A truncated ResNet-152 is used as backbone, followed by customized ASPP module with average pooling (instead of the default global pooling) to allow for training on sub-images and inference on full-scenes. One ASPP decoder module is used per each target of SIC, SoD, and Floe.

# Getting started
## Using a terminal
Clone this repository to your local machine using your tool of choice. Open the [Anaconda Prompt](https://docs.anaconda.com/anaconda/user-guide/getting-started/) (requires a working [Anaconda](https://www.anaconda.com/) installation):

Then, use the prompt to **navigate to the location of the cloned repository**. Install the [environment](env_exported.yml) using the command:  
`conda env create -f env_exported.yml`

Follow the instructions to activate the new environment:  
`conda activate sea-ice-segment`

We have two environment files: 
- [env_exported](env_exported.yml): the environment exported from  Anaconda's history. This should be enough to replicate the results.
- [env_full](env_full.yml): the full environment installed. This includes more information and might be OS dependent. The experiments were executed using Windows 10 Pro for Workstations, mostly using version 21H2.

## Using the code
Use [`main.py`](main.py) to train the model according to [`configuration file`](config_main.ini). Evaluate the model using  [`test.py`](test.py) that also requires [`configuration file`](config_eval.ini)

