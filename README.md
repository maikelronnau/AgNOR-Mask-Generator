# Mask Generator

## Overview

Generates segmentation masks and annotations in the [labelme](https://github.com/maikelronnau/labelme) format for AgNOR images using pre-trained models.

## Requirements

**CPU**

None.

**GPU**

From source and `exe`:
- `CUDA 11.2`
- `CUDNN 8.1`
- [Microsoft Visual C++ Redistributable for Visual Studio 2015, 2017 and 2019](https://learn.microsoft.com/en-US/cpp/windows/latest-supported-vc-redist?view=msvc-170) (Windows only)

## Build stand alone executable

1. Install [Anaconda](https://www.anaconda.com/).
2. Execute the following commands:

```console
git clone https://github.com/maikelroennau/AgNOR-Mask-Generator.git
cd AgNOR-Mask-Generator
conda env create -n mskg --file environment_gpu.yml
conda activate mskg
```

3. Download the pre-trained models and place them in the root directory of the cloned repository. The models can be downloaded from the [releases page](https://github.com/maikelroennau/AgNOR-Mask-Generator/releases).

4. Compile the standalone executable:

```console
pyinstaller "AgNOR Slide-Image Examiner.spec"`
```

The executable will be in the `dist` directory.

## Usage

Double click on the executable in `dist/` to open the application. To have execution logs saved to a file, run the executable from a command prompt with the argument `-d`. For example:

```console
"AgNOR Slide-Image Examiner.exe" -d
```

To select what GPU to use:

```console
"AgNOR Slide-Image Examiner.exe" --gpu 0
```

Use `--gpu -1` to use CPU.
