# QuantLab
This project provides the experimental environment used to produce the results reported in the paper *Additive Noise Annealing and Approximation Properties of Quantized Neural Networks* available on [arXiv](https://arxiv.org/abs/1905.10452). If you find this work useful in your research, please cite
```
@inproceedings{spallanzani2019ana,
  author = {Matteo Spallanzani and Lukas Cavigelli and Gian Paolo Leonardi and Marko Bertogna and Luca Benini},
  booktitle = {arXiv preprint arXiv:1905.10452},
  title = {{A}dditive {N}oise {A}nnealing and {A}pproximation {P}roperties of {Q}uantized {N}eural {N}etworks},
  year = {2019}
}

```

## Getting started

### Prerequisites
* We developed and used QuantLab on [Ubuntu 16.04.5 LTS Xenial Xerus (64bit)](http://old-releases.ubuntu.com/releases/16.04.5/).
* QuantLab is based on Python3, and [Anaconda3](https://www.anaconda.com/distribution/) is required.
* We used [NVidia GTX1080 Ti GPUs](https://developer.nvidia.com/cuda-gpus) to accelerate the training of our models (driver version [396.44](https://www.nvidia.com/Download/driverResults.aspx/136950/en-us)). In this case, CUDA and the cuDNN library are needed (we used [CUDA 8.0.61](https://developer.nvidia.com/cuda-toolkit-archive) and [cuDNN 7.1.2](https://developer.nvidia.com/rdp/cudnn-archive) respectively).

### Installing
Navigate to QuantLab's main folder and create the environment using Anaconda:
```
$ conda env create -f quantlab.yml -n quantlab
```
In order to boost performances, QuantLab defines special *hard* folders indicating where to store datasets, training checkpoints and statistics about the experiments.
For example, suppose that the system has two hard drives: a fast but small SSD and a slower but larger HDD.
To accelerate training, the datasets should be stored on the SSD (since they are shared amongst all the experiments); on the other hand, the logs could take more space (every experiment will need its own) but are not so performance-critical.
Supposing that the folders 	`/fast/` and `/slow/` refer to folders on the SSD and HDD respectively, create the folders for data and logs:
```
$ mkdir /fast/QuantLab/
$ mkdir /slow/QuantLab/
```
Then, edit the pointers file `cfg/hard_storage.json` so that datasets and logs will be stored on the appropriate drive.

### Setup
QuantLab main abstractions are the **problem** and the **experiment**.
A **problem** is characterized by a dataset.
When a new problem is added, the user should create an appropriate folder and add a configuration file.
For example, ImageNet comes already configured with the project, but its creation would look like the following:
```
$ cd ~/QuantLab/
$ mkdir ImageNet/
$ touch ImageNet/config.json
```
To install a new dataset, the corresponding folder must be created on the appropriate drive, and the files should be copied there before launching the experiments.
As an example, suppose that ImageNet training and validation sets have already been downloaded to `~/Downloads/ImageNet/train/` and `~/Downloads/ImageNet/val/` respectively.
Then, the user should execute:
```
$ mkdir /fast/QuantLab/ImageNet/
$ mkdir /fast/QuantLab/ImageNet/data/
$ cp -R ~/Downloads/ImageNet/* /fast/QuantLab/ImageNet/data/
```
When launched, QuantLab will automatically create a system of links from the main folder `~/QuantLab` to the appropriate folders.
An **experiment** is characterized by an individual (in this case, a network to be quantized) and a treatment (the set of hyperparameters that define the training algorithm).
The file `config.json` mentioned above provides exactly this information.


## Usage
To use QuantLab, navigate to its main directory and activate the environment:
```
$ conda activate quantlab
```
To reproduce the quantization experiment on, for example, ImageNet, execute:
```
$ (quantlab) python main.py --problem=ImageNet --topology AlexNet
```
When launched, QuantLab assigns a numeric `[ID]` to the experiment and creates a corresponding sub-folder in the problem's `log` folder (in the above example, it would be `~/Quantlab/ImageNet/log/exp[ID]/`).
The program will take periodic snapshots of the parameters and log statistics about the training.
The experiment can also be interrupted and restarted from the last checkpoint:
```
$ (quantlab) python main.py --problem=ImageNet --topology AlexNet --exp_id=[ID] --load=last
```
To monitor the logged statistics, open another shell, navigate to QuantLab's main folder and launch [TensorBoard](https://www.tensorflow.org/guide/summaries_and_tensorboard):
```
$ conda activate quantlab
$ (quantlab) tensorboard --logdir=ImageNet/exp[ID]/ --port=6006
```
