# Early-Prediction-of-Sepsis

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

<details>
<summary><b>Challange References:</b> (click to expand)</summary>
<br>

6. **Early Prediction of Sepsis Using Gradient Boosting Decision Trees with Optimal Sample Weighting**

   [link](https://physionet.org/content/challenge-2019/1.0.0/papers/CinC2019-459.pdf)

   - **Team**: Ibrahim Hammoud, IV Ramakrishnan, Mark Henry (Stony Brook University)
   - **Abstract**: The team developed a model for early sepsis prediction using an ensemble of gradient boosting decision trees, trained with weighted binary cross-entropy loss. The model uses a fixed-size feature vector from the last 20 hours of patient data, with imputation mimicking real-time healthcare information. The model was tuned and evaluated through 5-fold cross-validation, achieving a 6th rank out of 78 in the PhysioNet/Computing in Cardiology Challenge 2019.

   #### What the Team Did

   - Proposed a method to train an ensemble of gradient boosting decision trees for early sepsis prediction, focusing on a weighted binary cross-entropy loss to handle the unique challenges of sepsis data.
   - Developed a fixed-size feature vector based on the last 20 hours of data for each patient, incorporating a real-time imputation scheme that simulates the information available to healthcare professionals.
   - Employed 5-fold cross-validation for hyper-parameter tuning and model evaluation, aiming for the maximum utility score on the training set to guide the selection of the evaluation set threshold.

   #### What They Found Useful

   - The use of weighted binary cross-entropy loss was pivotal in handling the imbalance and specificity of the sepsis prediction challenge, allowing for the efficient training of the model.
   - Real-time imputation and fixed-size feature vectors were effective in mimicking the decision-making environment of healthcare professionals, providing a more realistic basis for the model's predictions.
   - Early prediction of sepsis showed potential for significant impact, with the model achieving a notable rank in the challenge, demonstrating the viability of gradient boosting decision trees for this application.

   #### Challenges and Limitations

   - The fixed-size window of 20 hours for feature vectors, while computationally necessary, might have limited the model's ability to utilize more extended historical data potentially beneficial for prediction accuracy.
   - Despite achieving a high rank, the model encountered challenges with a high false positive rate and variance in score distribution among positive patients, indicating room for optimization in threshold setting and score calibration.
   - The heavy reliance on the challenge's utility function for model training and evaluation may have introduced biases or artifacts in prediction behavior, emphasizing the need for further exploration of alternative metrics and methods.

   #### Future Directions

   - Investigating sequence models like LSTMs for their potential to incorporate both short-term and long-term information from real-time signals, addressing the limitations of fixed-size feature vectors.
   - Exploring alternative metrics, scoring functions, and models to improve early prediction tasks, aiming to optimize real-time prediction settings more effectively.
   - Continued examination of the impacts of utility functions on model outputs and prediction timing to refine and enhance early sepsis prediction approaches.

</details>

Using [Hypertools](https://hypertools.readthedocs.io/en/latest/auto_examples/plot_PPCA.html).