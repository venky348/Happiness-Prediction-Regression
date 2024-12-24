Happiness Regression Model

Project Overview

This project focuses on analyzing and predicting happiness scores (Life Ladder) using various attributes related to a country’s economic, social, and emotional factors. The model uses linear regression, polynomial regression, and regularization techniques to make predictions based on features such as GDP, social support, freedom of choice, generosity, and perceptions of corruption.

Table of Contents

- Project Overview
- Data Description
- Tech Stack
- Installation
- Usage
- Project Structure
- Results
- Conclusions
- License

Data Description

The dataset contains the following attributes:

- Country name: Name of the country
- Year: Year data was collected (not used for prediction)
- Life ladder: The happiness score of the country (target variable)
- Log GDP per capita: Market values of goods and services in a country
- Social support: How people feel they are supported by those around them
- Healthy life expectancy: Country’s rank based on happiness score
- Freedom to make life choices: Freedom contributing to happiness
- Generosity: How much people donate
- Perceptions of corruption: How people perceive corruption in their country
- Positive affect: How often people experience happiness, laughter, and enjoyment
- Negative affect: How often people experience worry, anger, or sadness

Tech Stack

- Programming Language: Python
- Libraries/Frameworks: 
  - Pandas
  - NumPy
  - Matplotlib/Seaborn (for visualization)
  - Scikit-Learn
  - Statsmodels
  - Jupyter Notebooks (for notebook-based analysis)
- Regression Techniques:
  - Linear Regression
  - Polynomial Regression
  - Ridge, Lasso, Elastic Net Regularization
  - Stochastic Gradient Descent (SGD)

Installation

To run this project locally, make sure you have Python 3.x installed, then follow the steps below:

1. Clone this repository:
   git clone https://github.com/yourusername/Happiness-Regression-Model.git

2. Navigate to the project directory:
   cd Happiness-Regression-Model

3. Install the required dependencies:
   pip install -r requirements.txt

Usage

To run the analysis and model training:

1. Open the Jupyter Notebook in your preferred environment:
   jupyter notebook

2. Open the `Happiness-Regression-Analysis.ipynb` notebook and follow the steps to explore the dataset, preprocess data, train models, and evaluate performance.

Key Analysis Steps:
- Data Exploration: Understand the features and their distributions.
- Data Cleaning: Handle missing values, outliers, and data transformation.
- Feature Engineering: Select relevant features for the regression model.
- Model Training: Use Linear Regression, Polynomial Regression, and regularization techniques.
- Evaluation: Assess model performance using appropriate metrics (e.g., RMSE, R²).

Project Structure

Happiness-Regression-Model/
│
├── data/
│   └── happiness_data.csv      # Dataset used for regression analysis
│
├── notebooks/
│   └── Happiness-Regression-Analysis.ipynb  # Jupyter notebook with the project analysis
│
├── src/
│   ├── data_preprocessing.py   # Data cleaning and preprocessing functions
│   ├── model_training.py       # Functions for training the regression models
│   └── model_evaluation.py     # Functions to evaluate model performance
│
├── requirements.txt            # Python dependencies
├── README.md                   # Project description
└── LICENSE                     # Project license

Results

Key Findings:
- The linear regression model performed well, but adding polynomial features and regularization improved the predictions.
- Ridge, Lasso, and Elastic Net regularization helped prevent overfitting by penalizing large coefficients.
- The test set results showed an R² score of **XX** and RMSE of **YY**.

Hyperparameters Tuning:
- SGD learning rate: Adjusting the learning rate had a significant impact on the model's convergence.
- Regularization strength: Stronger regularization led to better generalization but slightly lower performance on the training set.

Conclusions

The regression model provided a useful approach to predict happiness scores based on socio-economic factors. By experimenting with different techniques such as polynomial regression and regularization, we were able to improve the model’s accuracy and reduce overfitting. Future work could involve more advanced techniques like ensemble methods or deep learning to improve predictive performance further.

License

This project is licensed under the MIT License - see the LICENSE file for details.

