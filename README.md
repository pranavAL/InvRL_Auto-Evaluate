# Learning Reward Functions using Expert Demonstrations for Crane Automation

<p align="center">
  <img width="75%" src="datasets/approach.png">
</p>

Official code repository for the paper 
***Learning Reward Functions using Expert Demonstrations for Crane Automation*** by [Pranav Agarwal](https://pranaval.github.io/).

# Installations
## Simulator
* We use [Vortex Studio 2021a](https://vortexstudio.atlassian.net/wiki/spaces/VSD21A/overview) for collecting the dataset and further training the reinforcement learning policy. 
* All files for installing the simulator is availaible [here](https://drive.google.com/drive/folders/1nCHXmvMzyiH3GqQtNNYV-WuR99p5xqDD). 
* The installation involves downloading all the files and running the **.exe** file, further selcting all the files when prompted.
 * The simulator requires a license which can be requested from [CMLabs](https://www.cm-labs.com/vortex-studio/software/vortex-studio-academic-access/) using their academic access program.

## Code
* Install [Anaconda](https://www.anaconda.com/products/distribution#Downloads).

```
git clone https://github.com/pranavAL/InvRL_Auto-Evaluate
cd InvRL_Auto-Evaluate
conda env create -f environment.yml
conda activate myenv
```
⚠️**WARNING**⚠️ 

All code should run within the specified virtual environment as created above, considering the strict requirements of Vortex. No further packages are required to be installed.

## System Requirements
* OS - Windows 10
* GPU - NVIDIA GeForce GTX 1050 Ti
* NVIDIA Driver - 457.49
* CUDA Version - 11.1

🔴**IMPORTANT**❗🔴

These are strict requirements. Vortex tools used in this project is not supported on Linux and its graphics is currently tested only on the above configurations.