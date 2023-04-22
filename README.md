# AgNOR Slide-Image Examiner

Examines AgNOR -slide-images and produces counts of nuclei and NORs.

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
conda env create -n asim --file environment.yml
conda activate asim
```

3. Download the pre-trained models and place them in the root directory of the cloned repository. The models can be downloaded from [this link](https://ufrgscpd-my.sharepoint.com/:f:/g/personal/00330519_ufrgs_br/EnzAQbs3_4FHlbxemScpD9IBVKNpGUbXRH0Oqqw7nFkYGA?e=vRbBpS).

Make sure the model file you want to use matches the `MODEL_PATH` value in the `utils/utils.py` file.

4. Compile the standalone executable:

```console
pyinstaller "AgNOR Slide-Image Examiner.spec"`
```

The executable will be in the `dist` directory.

**Note**: For the executable to work using GPU, `CUDA` and `CUDNN` must be installed in the OS at build time, and not in Anaconda.

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

Using a different model other than the embedded model:

```console
"AgNOR Slide-Image Examiner.exe" --model path/to/model.h5
```

## Config file

The program loads some of the menu options from a `config.txt` file. There are tree configuration options:

- `group`: names of groups patients belong to.
- `site`: site from where cells were collected from patients.
- `database`: a file where the program will append all generated records.

To add new entries, use the prefixes `group:`, `site:`, or `database:` followed by the information. Only one entire per line is supported. To remove an entry, either delete the line or comment it out by adding a `#` in the beginning of the row.
