# Early-Prediction-of-Sepsis

#### Challenge References can be found in [challenge_papers](./challenge_papers/README.md)

<details>
<summary><b>Git and GitHub Guide: Managing Pull Requests</b> (click to expand)</summary>
<br>

This guide provides a comprehensive overview of using Git with GitHub, focusing on managing pull requests. It is designed to be useful for users across different operating systems, including Windows, Linux, and macOS.

### Setting Up Git

Before diving into pull requests, ensure Git is installed on your system.

#### Windows

1. Download the Git installer from [git-scm.com](https://git-scm.com/).
2. Run the installer and follow the prompts. Include Git Bash if you'd like a Unix-style command line.
3. Verify installation by opening Git Bash or Command Prompt and running `git --version`.

#### Linux

Most Linux distributions include Git. Install it using your package manager.

- For Ubuntu/Debian-based systems:

```bash
sudo apt update
sudo apt install git
```

- For Fedora:

```bash
sudo dnf install git
```

- Verify installation with `git --version`.

#### macOS

Git comes pre-installed on macOS. If not, install it via the Xcode Command Line Tools:

```bash
xcode-select --install
```

Alternatively, use Homebrew:

```bash
brew install git
```

### Configuring Git

Set your username and email address for your commits.

```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

### Cloning a Repository

To work on an existing repository, you first need to clone it.

```bash
git clone <repository-URL>
```

This command creates a local copy of the repository on your machine.

#### Note:
- To use GitHub SSH keys for conveniece feel free to follow this [GitHub guide](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account)

### Creating a New Branch

Before making changes, create a new branch. This keeps your changes separate from the main project until they're ready to be reviewed.

```bash
git checkout -b <branch-name>
```

### Making Changes and Committing

1. Make your changes in the local copy.
2. Use `git add` to stage changes for commit.
   - To stage a specific file: `git add <file-path>`
   - To stage all changes: `git add .` or `git add *`
3. Commit your changes with a message:

```bash
git commit -m "A brief description of the changes"
```

### Pushing Changes to GitHub

After committing your changes, push them to GitHub.

```bash
git push origin <branch-name>
```
or just:
```bash
git push <branch-name>
```

#### Note:
- When first pushing to a newly created branch you need to set origin:

```bash
git push --set-upstream origin <branch-name>
```

### Creating a Pull Request

1. Go to the GitHub page of the repository.
2. Click on "Pull requests" > "New pull request".
3. Select your branch and the branch you want to merge into (usually the development branch).
4. Fill in the pull request details and create it.

### Review and Merge

- The project maintainers will review your pull request. Be ready to make additional changes if requested.
- Once approved, the maintainer can merge your pull request.
- Once merged, the merger (the person who merged the branch) will delete the feature branch unless asked not to.

#### Note:
- Development branch should **never** be deleted.

### Keeping Your Branch Up to Date

Before making more changes or before finalizing your pull request, ensure your branch is up to date with the main branch.

```bash
git checkout main
git pull origin main
git checkout <your-branch>
git merge main
```

This should mostly be done for `development` branch as it is the primary branch used to put all the features together.

```bash
git checkout development
git pull origin development
git checkout <your-branch>
git merge development
```

Further, you can check if there were any updates and their status by using:

```bash
git fetch
```

This will no apply the changes made remotely! It is used to check for any changes in preparation to pull. 


#### Note:
- If you are unsure of the status use: `git status`

</details>

<details>
<summary><b>Working with the Conda Environment</b> (click to expand)</summary>
<br>

## Setting Up the Conda Environment

This project uses a conda environment to manage dependencies. To set up the environment on your local machine, follow these steps:

1. **Install Miniconda or Anaconda**:

   If you haven't already, install Miniconda or Anaconda on your machine. Visit [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/individual) for installation instructions.

2. **Create the Environment**:

   Navigate to the project directory and run the following command to create a conda environment from the `environment.yml` file:

```bash
conda env create -f environment.yml
```

3. **Activate the Environment**:

    Once the environment is created, you can activate it using:

```bash
conda activate myenv
```

Replace `myenv` with the name of the environment specified in the `environment.yml` file.

## Working with the Conda Environment

### Installing Additional Packages

If you need to install additional packages, make sure to activate the environment and use:

```bash
conda install package-name
```

Or, if the package is only available via pip (still check installation guide for the specific package):

```bash
pip install package-name
```

There may be other ways to install a package for example using `conda-forge`  ( `conda install package -c conda-forge` ) so always look for instructions online.

### Updating the Environment

If you've added new packages or made other changes to the environment that you want to share with the team, you can update the `environment.yml` file by running:

```bash
conda env export --from-history > environment.yml
```

**Note:** The yml file contains `prefix` field which relates to the path of the environment **locally**, conda however, doesn't care and besides manually deleting the line there doens't seem to be a way to avoid creating that line when exporting.

**Note:** Use the `--from-history` flag to only include packages you've explicitly installed, avoiding platform-specific packages in the environment file.

### Sharing Changes

After updating the `environment.yml` file, commit and push the changes to the GitHub repository so the team members can update their environments by running:

```bash
conda env update --file environment.yml --prune
```

The `--prune` option removes any dependencies that are no longer needed from the environment.

### Adding conda environment to JupyterLab

To make your conda environment visible to JupyterLab you need to add your environment by creating a kernel spec:

```bash
python -m ipykernel install --user --name YourEnvironmentName --display-name "Display Name"
```

### Running JupyterLab

1. Intall JupyterLab:

```bash
pip3 install jupyter
```

2. Navigate to the notebooks directory:

```bash
cd notebooks
```

3. Run JupyterLab

```bash
jupyter lab
```

</details>

## DATA

Link to the challenge site https://physionet.org/content/challenge-2019/1.0.0/

The data is split into two folders and the data is saved in [.psv](https://docs.amperity.com/reference/format_psv.html) files:

```sh
Dataset/
├── training_setA/
└── training_setB/
```

Combined the data has 42 MB. Training Set A contains 20,336 subjects and Training Set B contains 20,000 subject. Each training data file provides a table with measurements over time. Each column of the table provides a sequence of measurements over time (e.g., heart rate over several hours), where the header of the column describes the measurement. Each row of the table provides a collection of measurements at the same time.

### Columns in each training data file:

<details>
<summary><b>Full table of columns with descriptions</b> (click to expand)</summary>
<br>

| Variable Name       | Description                                                               |
|---------------------|---------------------------------------------------------------------------|
**Vital Signs (columns 1-8)**
| HR                  | Heart rate (beats per minute)                                             |
| O2Sat               | Pulse oximetry (%)                                                        |
| Temp                | Temperature (Deg C)                                                       |
| SBP                 | Systolic BP (mm Hg)                                                       |
| MAP                 | Mean arterial pressure (mm Hg)                                            |
| DBP                 | Diastolic BP (mm Hg)                                                      |
| Resp                | Respiration rate (breaths per minute)                                     |
| EtCO2               | End tidal carbon dioxide (mm Hg)                                          |
**Laboratory Values (columns 9-34)**
| BaseExcess          | Measure of excess bicarbonate (mmol/L)                                    |
| HCO3                | Bicarbonate (mmol/L)                                                      |
| FiO2                | Fraction of inspired oxygen (%)                                           |
| pH                  | N/A                                                                       |
| PaCO2               | Partial pressure of carbon dioxide from arterial blood (mm Hg)            |
| SaO2                | Oxygen saturation from arterial blood (%)                                 |
| AST                 | Aspartate transaminase (IU/L)                                             |
| BUN                 | Blood urea nitrogen (mg/dL)                                               |
| Alkalinephos        | Alkaline phosphatase (IU/L)                                               |
| Calcium             | (mg/dL)                                                                   |
| Chloride            | (mmol/L)                                                                  |
| Creatinine          | (mg/dL)                                                                   |
| Bilirubin_direct    | Bilirubin direct (mg/dL)                                                  |
| Glucose             | Serum glucose (mg/dL)                                                     |
| Lactate             | Lactic acid (mg/dL)                                                       |
| Magnesium           | (mmol/dL)                                                                 |
| Phosphate           | (mg/dL)                                                                   |
| Potassium           | (mmol/L)                                                                  |
| Bilirubin_total     | Total bilirubin (mg/dL)                                                   |
| TroponinI           | Troponin I (ng/mL)                                                        |
| Hct                 | Hematocrit (%)                                                            |
| Hgb                 | Hemoglobin (g/dL)                                                         |
| PTT                 | Partial thromboplastin time (seconds)                                     |
| WBC                 | Leukocyte count (count*10^3/µL)                                           |
| Fibrinogen          | (mg/dL)                                                                   |
| Platelets           | (count*10^3/µL)                                                           |
**Demographics (columns 35-40)**
| Age                 | Years (100 for patients 90 or above)                                      |
| Gender              | Female (0) or Male (1)                                                    |
| Unit1               | Administrative identifier for ICU unit (MICU)                             |
| Unit2               | Administrative identifier for ICU unit (SICU)                             |
| HospAdmTime         | Hours between hospital admit and ICU admit                                |
| ICULOS              | ICU length-of-stay (hours since ICU admit)                                |
**Outcome (column 41)**
| SepsisLabel         | For sepsis patients, SepsisLabel is 1 if t≥tsepsis-6 and 0 if t<tsepsis-6. For non-sepsis patients, SepsisLabel is 0. |

</details>

<!-- Using [Hypertools](https://hypertools.readthedocs.io/en/latest/auto_examples/plot_PPCA.html). -->