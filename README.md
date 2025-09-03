# TripDataset Machine Learning Project

## Project Overview

This project is a machine learning assignment focused on processing the TripDataset dataset, involving data preprocessing, classification, and regression tasks. Core functionalities include data cleaning, outlier detection, feature engineering, model training, and performance evaluation. The goal is to predict travel-related metrics using various machine learning models, such as classification tasks for categorizing and regression tasks for continuous value prediction.

### Main Technology Stack and Dependencies
- **Programming Language**: Python 3.x
- **Main Libraries**:
  - pandas: Data processing
  - numpy: Numerical computations
  - scikit-learn: Machine learning models (classifiers like Random Forest, SVC; regressors like Linear Regression, SVR)
  - Jupyter Notebook: Interactive development
  - Others: matplotlib/seaborn for visualization (inferred from visualization notebooks)

## Installation Guide

### Environment Requirements
- Python 3.8 or higher
- pip package manager
- Optional: Anaconda or Miniconda for virtual environment management

### Step-by-Step Installation Instructions
1. Clone the repository (if applicable) or download the project files.
2. Create a virtual environment (recommended):
   ```
   conda create -n trip-ml python=3.8
   conda activate trip-ml
   ```
   Or using venv:
   ```
   python -m venv env
   source env/bin/activate  # Linux/Mac
   env\Scripts\activate  # Windows
   ```
3. Install dependencies:
   ```
   pip install pandas numpy scikit-learn jupyter matplotlib seaborn
   ```
4. Unzip TripDataset.zip to the project root directory.

## Usage Instructions

### How to Run/Configure the Project
1. Activate the virtual environment.
2. Start Jupyter Notebook:
   ```
   jupyter notebook
   ```
3. Open relevant notebooks such as classification.ipynb or regression.ipynb.

### Usage Examples of Key Features
- **Data Preparation**: Run data_prepare.py or corresponding notebooks (e.g., data_prepare_all_scaled.ipynb) to process data.
- **Model Training**: Train classification models in classification.ipynb; train regression models in regression.ipynb.
- **Visualization**: Use classification_visualization.ipynb to view classification result charts.
- Example command:
  ```python
  import pandas as pd
  df = pd.read_pickle('data/data_processed.pkl')
  # Further processing...
  ```

## Project Structure

- **TripDataset/**: Raw dataset, including Excel files from 2015-2019 and feature descriptions.
- **data/**: Processed data files, such as data_processed.pkl and outlier files.
- **report/**: Report files, including PDF reports, charts, and LaTeX source files.
- **results/**: Model result JSON files, divided into clf (classification) and reg (regression) subdirectories.
- **notebooks**: Such as classification.ipynb (classification tasks), regression.ipynb (regression tasks), data_prepare_*.ipynb (data preparation).
- **scripts**: Such as data_prepare.py (data preparation script), results_reg.py (result processing script).

### Notebooks Description

This project utilizes several Jupyter Notebooks to organize the workflow. Here is a detailed description of each:

- **`classification.ipynb`**: This notebook is dedicated to the classification tasks. It loads the preprocessed data, trains various classification models (e.g., Logistic Regression, SVC, Random Forest), evaluates their performance using metrics like accuracy and F1-score, and saves the results in the `results/clf` directory.

- **`classification_visualization.ipynb`**: This notebook focuses on visualizing the results from the classification models. It generates plots such as confusion matrices, ROC curves, and feature importance charts to provide a deeper understanding of model performance. The generated figures are stored in `report/Figs`.

- **`data_prepare_all_scaled.ipynb`**: This notebook contains the data preparation pipeline that includes feature scaling. It handles data cleaning, missing values, outlier treatment, and applies scaling techniques (e.g., StandardScaler) to the features before saving the processed data.

- **`data_prepare_all_unscaled.ipynb`**: Similar to the scaled version, this notebook prepares the data but without applying feature scaling. This allows for training and evaluating models that do not require scaled data.

- **`data_prepare_augmented.ipynb`**: This notebook implements data augmentation techniques, such as SMOTE, to address class imbalance or increase the diversity of the training set. The augmented data is then used for model training.

- **`regression.ipynb`**: This notebook handles the regression tasks. It loads the data and trains various regression models (e.g., Linear Regression, SVR, Random Forest), evaluates them using metrics like Mean Squared Error (MSE) and R-squared, and stores the results in the `results/reg` directory.

- **`regression_visualization.ipynb`**: This notebook is used to create visualizations for the regression results. It generates plots like residual plots and predicted vs. actual value plots to analyze model performance. The figures are saved in `report/Figs`.

- **`results_read.ipynb`**: This notebook is used to read, parse, and display the model performance results stored in the `results` directory. It provides a convenient way to compare the performance of different models. The results are parsed from JSON files and displayed in a tabular format.

Important Components:
- Notebooks handle core logic: data loading, model training, and evaluation.
- Results directory stores training metrics for analysis.

## Related Documentation

- **[report/report.pdf](report/report.pdf)**: Main project report, including methodology and experimental results.
- **[Supplementaries.pdf](Supplementaries.pdf)**: Assignment supplementary instructions, providing additional guidance.
- **[Topic and Requirements.pdf](Topic and Requirements.pdf)**: Assignment topic description, defining project requirements.

Other Reference Materials:
- scikit-learn Official Documentation: https://scikit-learn.org/stable/
- pandas Documentation: https://pandas.pydata.org/docs/

## License Information

This project uses the MIT License. See the LICENSE file for details (if not present, a standard MIT License can be added).