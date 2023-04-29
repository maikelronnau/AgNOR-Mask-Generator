# AgNOR Slide-Image Examiner

Examines AgNOR slide-images using a convolutional neural network and produces counts of nuclei and NORs in the `.csv` format.

![User interface of the AgNOR Slide-Image Examiner](program.png)

## Download

You can download built executables (Windows and Linux) from the page (coming soon).

To use GPU acceleration, install:
- `CUDA 11.2` (optional)
- `CUDNN 8.1` (optional)

[Microsoft Visual C++ Redistributable for Visual Studio 2015, 2017 and 2019](https://learn.microsoft.com/en-US/cpp/windows/latest-supported-vc-redist?view=msvc-170) is required for running on Windows.

## Run from source or build a standalone executable

1. Install [Anaconda](https://www.anaconda.com/).
2. Execute the following commands:

```console
git clone https://github.com/maikelroennau/agnor-slide-image-examiner.git
cd agnor-slide-image-examiner
conda env create -n asie --file environment.yml
conda activate asie
```

3. Download the pre-trained models and place them in the root directory of the cloned repository. The models can be downloaded from [this link](https://ufrgscpd-my.sharepoint.com/:f:/g/personal/00330519_ufrgs_br/EnzAQbs3_4FHlbxemScpD9IBVKNpGUbXRH0Oqqw7nFkYGA?e=vRbBpS).

Make sure the model file you want to use matches the `MODEL_PATH` value in the `utils/utils.py` file.

### Run the main script

```console
python agnor_slide_image_examiner.py
```

### Compile the standalone executable:

```console
pyinstaller "AgNOR Slide-Image Examiner.spec"`
```

The executable will be in the `dist` directory.

**Note**: For the executable to work using GPU, `CUDA` and `CUDNN` must be installed in the OS at build time, and not in Anaconda.

## Usage

Double click on the executable in `dist/` to open the application. To have execution logs saved to a file, run the executable from a command prompt with the argument `-d`. For example:

**Note**: Instructions use the executable for Windows but they also apply for Linux, except that the Linux executable does not have the `.exe` extension.

```console
"AgNOR Slide-Image Examiner.exe" -d
```

To select what GPU to use:

```console
"AgNOR Slide-Image Examiner.exe" --gpu 0
```

Use `--gpu -1` to use CPU.

Using a different model other than the embedded model:

```console
"AgNOR Slide-Image Examiner.exe" --model path/to/model.h5
```

## Config file

The program loads some of the menu options from a `config.txt` file. There are tree configuration options:

- `group`: names of groups patients belong to.
- `site`: site from where cells were collected from patients.
- `database`: a file where the program will append all generated records.

To add new entries, use the prefixes `group:`, `site:`, or `database:` followed by the information. Only one entire per line is supported. To remove an entry, either delete the line or comment it out by adding a `#` at the beginning of the line.
