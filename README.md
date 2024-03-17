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

1. **The Signature-Based Model for Early Detection of Sepsis From Electronic Health Records in the Intensive Care Unit**

   [link](https://physionet.org/content/challenge-2019/1.0.0/papers/CinC2019-014.pdf)

   - **Team**: James Morrill, Andrey Kormilitzin, Alejo Nevado-Holgado, Sumanth Swaminathan, Sam Howison, Terry Lyons (University of Oxford, Iterex Therapeutics)
   - **Abstract**: Introduced a signature-based regression model for sepsis detection from ICU patient data, achieving the highest utility function score (0.360) and ranking 1st in the PhysioNet Challenge 2019. The model utilizes gradient boosting machines and signature features from patient time-series data to predict sepsis risk at every time interval post-admission.

   #### What the Team Did

   - Developed a new machine learning approach using signature transformation to extract features from time-series physiological data of ICU patients, enhancing prediction accuracy for sepsis onset.
   - Implemented a gradient boosting machine algorithm that leverages both current time-point data and extracted signature features to model sepsis effects longitudinally.
   - Conducted a detailed analysis of various feature sets, including hand-crafted features and signature transformations, to evaluate their predictive power and impact on model performance.
   - Employed stratified 5-fold cross-validation and light gbm for model training and validation, optimizing for a utility score that considers the trade-offs between true positives, false positives, and timely prediction.

   #### What They Found Useful

   - Signature features significantly improved model performance by providing a comprehensive summary of longitudinal physiological measurements, distinguishing between septic and non-septic cases effectively.
   - The inclusion of hand-crafted features, such as ShockIndex and BUN/CR ratios, alongside signature transformations, showcased a systematic improvement in predicting sepsis risk.
   - The model achieved an AUC ROC of 0.868, demonstrating its efficacy in screening for sepsis risk with the ability to predict sepsis cases correctly in 65.3% of instances, often well before the onset.

   #### Challenges and Limitations

   - Despite the model's high utility score and AUC ROC, achieving the desired balance between sensitivity and specificity for clinical application remains a challenge, particularly in predicting sepsis within the crucial 6-hour window prior to onset.
   - The study focuses on the utility function optimization, which might not fully encapsulate the clinical nuances of sepsis prediction and management within the ICU setting.

   #### Future Directions

   - Explore the potential of signature-based models in other clinical prediction tasks, leveraging the method's ability to process complex time-series data effectively.
   - Investigate the integration of more diverse data sources and feature engineering techniques to further enhance the predictive accuracy and timeliness of sepsis detection.
   - Evaluate the model's performance in a real-world clinical setting, focusing on its utility as a decision-support tool for healthcare professionals in the intensive care unit.

2. **A Multi-Task Imputation and Classification Neural Architecture for Early Prediction of Sepsis from Multivariate Clinical Time Series**

   [link](https://physionet.org/content/challenge-2019/1.0.0/papers/CinC2019-110.pdf)

   - **Team**: Yale Chang, Jonathan Rubin, Gregory Boverman, Shruti Vij, Asif Rahman, Annamalai Natarajan, Saman Parvaneh (Philips Research North America, Cambridge, USA)
   - **Abstract**: This work focuses on early sepsis prediction using multivariate clinical time series data. The authors employed a recurrent imputation model (RITS) for handling missing data, followed by a Temporal Convolutional Network (TCN) for prediction. A custom time-dependent weighting approach for error types in the loss function was applied. The model achieved a utility score of 0.328 in the PhysioNet Computing in Cardiology Challenge 2019, placing 9th, and an improved version later reached a utility score of 0.342 in a follow-up event, securing 2nd place.

   #### What the Team Did

   - Developed a multi-task neural architecture combining recurrent imputation for time series (RITS) with Temporal Convolutional Networks (TCN) for early detection of sepsis.
   - Introduced a novel set of features that model the missingness in clinical data, enhancing the prediction model's accuracy.
   - Employed a custom-designed loss function incorporating time-dependent weights to manage different error types, effectively balancing the trade-offs between early, on-time, and late predictions of sepsis.
   - Conducted experiments on a real-world dataset provided by the PhysioNet/Computing in Cardiology Challenge 2019, demonstrating the proposed model's effectiveness in sepsis prediction.

   #### What They Found Useful

   - The RITS approach for imputing missing values significantly outperformed traditional imputation methods, providing a strong foundation for accurate sepsis prediction.
   - The TCN model was chosen for its efficiency in handling long historical sequences and its ability to make predictions at any point during the ICU stay without future data leakage.
   - The custom loss function tailored for the sepsis prediction task played a crucial role in optimizing the model's performance, particularly in minimizing the penalties associated with too early or too late predictions.
   - The combination of RITS-imputed data with TCN, augmented by missingness indicator variables, proved to be highly effective, outperforming other sequence prediction models.

   #### Challenges and Limitations

   - Handling irregularly sampled and missing data points in multivariate clinical time series posed significant challenges, addressed through the RITS model.
   - Balancing predictions to avoid too early or too late detection of sepsis required careful tuning of the loss function, highlighting the complexity of modeling clinical decision-making processes.
   - The variance in test utility scores across different folds indicated the need for ensemble models to improve prediction reliability and reduce variance.

   #### Future Directions

   - Further exploration of ensemble models could potentially lead to higher test utility scores by incorporating a greater variety of prediction models and increasing the number of RITS-TCN models.
   - Investigating model interpretation techniques, especially for black-box models like RITS and TCN, would be valuable for integrating these models into clinical workflows more effectively.
   - Continuous refinement of the loss function to better align with clinical needs and enhance the practical applicability of sepsis prediction models in real-world settings.


3. **Sepsis Prediction in Intensive Care Unit Using Ensemble of XGboost Models**

   [link](https://physionet.org/content/challenge-2019/1.0.0/papers/CinC2019-238.pdf)

   - **Team**: Morteza Zabihi, Serkan Kiranyaz, Moncef Gabbouj (Tampere University, Finland, and Qatar University, Qatar)
   - **Abstract**: This study addresses the challenge of early sepsis prediction in ICU patients by leveraging an ensemble of XGboost models. A novel feature set including patterns of missing values is proposed, which significantly contributes to the predictive performance. The methodology achieved third place in the PhysioNet/Computing in Cardiology Challenge 2019, demonstrating its effectiveness with an overall utility score of 0.339.

   #### What the Team Did

   - Developed an ensemble learning approach using five XGboost models for early sepsis prediction, focusing on ICU patients' clinical data.
   - Extracted 407 features from clinical data, including vital signs, demographic variables, and laboratory values, with a particular emphasis on modeling missingness.
   - Employed a wrapper feature selection algorithm to identify the most clinically relevant features, considering both present and missing data.
   - Achieved robust performance across different hospital datasets, officially ranking as the third team in the PhysioNet Challenge with a utility score of 0.339.

   #### What They Found Useful

   - The introduction of discriminative features to model the patterns of missing values in clinical data, acknowledging that missingness may carry informative signals for sepsis prediction.
   - A comprehensive feature engineering strategy that extracted a wide range of features, including both sliding-window and non-sliding-window based features, to capture the dynamic nature of sepsis.
   - The ensemble approach, combining multiple XGboost models, enhanced the robustness and accuracy of sepsis prediction, outperforming traditional clinical criteria.
   - The study identified that variables related to hospital administration time, temperature, heart rate, and blood pressure were among the top predictors of sepsis, underscoring the clinical relevance of the selected features.

   #### Challenges and Limitations

   - The presence of significant class imbalance between sepsis and non-sepsis observations required careful data balancing techniques to train effective models.
   - Performance variability across different hospital datasets highlighted the challenge of generalizing the predictive model, with a noticeable drop in performance on one of the test sets.
   - The reliance on sophisticated machine learning models and extensive feature engineering may limit the interpretability of the predictive process, an essential aspect for clinical adoption.

   #### Future Directions

   - Further exploration of the role of missing data in clinical prediction models, specifically investigating the informative nature of missingness across various medical conditions.
   - Enhancement of the ensemble model by incorporating advanced machine learning techniques and exploring alternative ensemble strategies to improve prediction accuracy and generalizability.
   - Clinical validation and integration of the proposed predictive model into ICU workflows, aiming to assess its impact on clinical outcomes and sepsis management strategies.

4. **TASP: A Time-Phased Model for Sepsis Prediction**

   [lik](https://physionet.org/content/challenge-2019/1.0.0/papers/CinC2019-049.pdf)

   - **Team**: Xiang Li, Yanni Kang, Xiaoyu Jia, Junmei Wang, Guotong Xie (Ping An Technology, Beijing, China)
   - **Abstract**: The Time-phAsed model for Sepsis Prediction (TASP) is introduced, leveraging the time-dependent nature of sepsis incidence in ICU patients. TASP integrates multiple modeling frameworks corresponding to different ICU stay phases: early, middle, and late stages, employing gradient boosting trees and deep learning to accommodate varying feature sets and missing value imputations tailored to each phase. This model scored 0.415 in cross-validation on the training set and ranked 4th with a score of 0.337 on the full test set of the Physionet/Computing in Cardiology Challenge 2019.

   #### Innovations and Key Findings

   - **Time-Phased Approach**: TASP is designed around the observation that sepsis incidence varies with ICU length of stay, implementing specific models for early (1-9 hours), middle (10-49 hours), and late (50+ hours) stages.
   - **Adaptive Modeling**: Utilizes gradient boosting trees for initial risk assessment and a deep learning framework (RNN) to capture long-term relationships in late-stage sepsis risk, optimizing prediction across different stages of ICU stay.
   - **Feature Engineering and Missing Value Imputation**: Implements various strategies for feature selection and missing value imputation, addressing the challenges of sparse and irregular data inherent in ICU records.
   - **Cross-Validation Performance**: Achieved a 0.415 score through 10-fold cross-validation on the training dataset, with simplified versions of the model attaining scores of 0.420 and 0.419 on the official online test set.

   #### Challenges and Limitations

   - **Model Complexity and Interpretability**: The multi-model approach, while effective, increases complexity and may pose challenges for clinical interpretation and real-time application in diverse ICU settings.
   - **Data-Driven Insights and Generalizability**: Insights gained through data exploration, such as the non-linear relationship between ICU stay length and sepsis incidence, underpin model design but may affect generalizability to other patient populations or clinical conditions.

   #### Future Directions

   - **Refinement of Feature Sets and Objectives**: Further research will focus on optimizing feature sets for each sub-model and aligning the objective function more closely with clinical utility metrics, enhancing both prediction accuracy and clinical relevance.
   - **Enhanced Model Interpretability**: Efforts to improve the interpretability of complex models like TASP are crucial for clinical adoption, with potential exploration of methods to elucidate model predictions and decision-making processes.
   - **Extended Validation and Online Testing**: Additional validation across diverse clinical settings and patient populations will be critical for assessing TASP's generalizability and effectiveness in real-world ICU environments.



5. **Utilizing Informative Missingness for Early Prediction of Sepsis**
   
   [link](https://physionet.org/content/challenge-2019/1.0.0/papers/CinC2019-280.pdf)

   - **Team**: Janmajay Singh, Kentaro Oshiro, Raghava Krishnan, Masahiro Sato, Tomoko Ohkuma, Noriji Kato (Fuji Xerox Co, Ltd, Yokohama, Japan)
   - **Abstract**: This study presents a novel approach to predict sepsis early in ICU patients by leveraging patterns in the missingness of physiological variables. The research introduces an XGBoost model that incorporates informative missingness, resulting in a utility score of 0.337 and securing a 5th place ranking in the challenge.

   #### What the Team Did

   - Developed an XGBoost model for early sepsis prediction, emphasizing the role of informative missingness in physiological data.
   - Explored various model variations with adjustments in hyperparameters, window sizes, and imputation methods to enhance prediction accuracy.
   - Implemented a strategy to represent the missingness of features through masking vectors, aligning with patterns observed in sepsis versus non-sepsis patients.
   - Shifted the sepsis labels to earlier time steps and fine-tuned the classification probability threshold to maximize the utility score.

   #### What They Found Useful

   - Analyzing the missingness patterns (informative missingness) in the data provided critical insights, revealing that certain variables exhibited different observation rates between sepsis and non-sepsis patients.
   - The non-imputation approach, combined with the use of masking vectors for all temporal variables, significantly improved the model's performance.
   - Shifting the sepsis labels to encourage the model to predict sepsis earlier than the actual onset time proved to be an effective strategy for improving utility scores.
   - The best-performing model, which included informative missingness and label shifting, achieved a utility score of 0.337 on the full test set, indicating its potential for early sepsis prediction in clinical settings.

   #### Challenges and Limitations

   - Dealing with a significant class imbalance and the inherent challenges of predicting sepsis, which affects a relatively small percentage of ICU patients.
   - The need to balance between false positives and true positives, especially given the high stakes of early sepsis prediction in terms of patient outcomes.
   - The approach's reliance on the specific characteristics of the dataset, which may limit its generalizability to other clinical settings or patient populations.

   #### Future Directions

   - Further research into the implications of informative missingness across different medical conditions and datasets to validate the approach's efficacy beyond sepsis prediction.
   - Exploration of sequence learning models that can inherently handle temporal data and missing values to possibly improve prediction accuracy.
   - Real-world implementation and validation of the model in clinical settings to assess its practical utility and impact on patient care and outcomes.

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

   ### Notable Mention:

   ### Time-Specific Metalearners for the Early Prediction of Sepsis

   [link](https://physionet.org/content/challenge-2019/1.0.0/papers/CinC2019-029.pdf)

   - **Team**: Marcus Vollmer, Christian F Luz, Philipp Sodmann, Bhanu Sinha, Sven-Olaf Kuhn (University Medicine Greifswald, University of Groningen, University Medical Center Groningen)
   - **Abstract**: Proposed a novel approach to predict sepsis 6 hours prior to onset using time-specific stacked ensembles and a non-specific XGBoost model, trained on ICU data from the 2019 PhysioNet Challenge. Despite the challenges of imprecise and incomplete data, the models demonstrated potential in sepsis prediction with a normalized utility score of 0.394 for the XGBoost model.

   #### What the Team Did

   - Developed time-specific metalearners and a general XGBoost model to predict sepsis in ICU patients 6 hours before onset, leveraging a dataset of 40,336 ICU stays.
   - Employed extensive data cleaning, feature engineering, and rolling window techniques to build robust features from clinical scores (e.g., SOFA, qSOFA, SIRS) and physiological data.
   - Evaluated model performance using task-specific utility functions and assessed variable importance to identify key predictors of sepsis.
   - Conducted a triple data split for training, validation, and testing, optimizing model parameters and threshold selection for binary classification (sepsis/no sepsis).

   #### What They Found Useful

   - Time-specific metalearners allowed for nuanced prediction by adapting to the dynamic clinical landscape and varying sepsis prevalence throughout ICU stays.
   - Feature engineering, particularly the generation of rolling window features and clinical scores, proved crucial in capturing the temporal dynamics of sepsis.
   - The non-specific XGBoost model achieved a notable utility score, demonstrating the effectiveness of machine learning techniques over traditional clinical scores for sepsis prediction.
   - Variables such as ventilation status, white blood cell count, and partial thromboplastin time emerged as significant predictors, highlighting their clinical relevance in early sepsis detection.

   #### Challenges and Limitations

   - Handling the imprecision and incompleteness of ICU data posed significant challenges, necessitating sophisticated data cleaning and imputation strategies.
   - Time-specific metalearners, while promising, exhibited limitations in threshold selection, affecting their overall performance compared to the non-specific model.
   - The study did not participate in the official PhysioNet Challenge, limiting external validation and comparison with other state-of-the-art models.

   #### Future Directions

   - Further research is needed to refine time-specific metalearning approaches, possibly by incorporating more granular temporal analysis and advanced feature engineering techniques.
   - Exploring the integration of additional data sources, such as genetic or immunological markers, could enhance model sensitivity and specificity for sepsis prediction.
   - Deployment and real-world validation of these models in ICU settings are essential steps toward assessing their clinical utility and impact on patient outcomes.


</details>

Using [Hypertools](https://hypertools.readthedocs.io/en/latest/auto_examples/plot_PPCA.html).