# Mask Generator

## Overview

Generates segmentation masks for AgNOR images using pre-trained U-Net models in the [labelme](https://github.com/maikelronnau/labelme) format.

## Build stand alone executable

Create Anaconda virtual environment:
```console
conda env create -n mskg --file environment_[cpu|gpu].yml
```

Download the pre-trained models and place them in the root directory. The models can be downloaded from the releases page.

Compile the standalone executable:
```console
`pyinstaller mask_generator.spec`
```

The executable will be in the `dist` directory.
