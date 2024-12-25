# Happiness Regression Model

## **Project Overview**

This project focuses on analyzing and predicting happiness scores (Life Ladder) using various attributes related to a country's economic, social, and emotional factors. The model uses linear regression, polynomial regression, and regularization techniques to make predictions based on features such as GDP, social support, freedom of choice, generosity, and perceptions of corruption.


## **Data Description**

The dataset contains the following attributes:

* Country name: Name of the country
* Year: Year data was collected (not used for prediction)
* Life ladder: The happiness score of the country (target variable)
* Log GDP per capita: Market values of goods and services in a country
* Social support: How people feel they are supported by those around them
* Healthy life expectancy: Country’s rank based on happiness score
* Freedom to make life choices: Freedom contributing to happiness
* Generosity: How much people donate
* Perceptions of corruption: How people perceive corruption in their country
* Positive affect: How often people experience happiness, laughter, and enjoyment
* Negative affect: How often people experience worry, anger, or sadness

## **Tech Stack**

* Programming Language: Python
* Libraries/Frameworks:
    * Pandas
    * NumPy
    * Matplotlib/Seaborn (for visualization)
    * Scikit-Learn
    * Statsmodels
    * Jupyter Notebooks (for notebook-based analysis)
* Regression Techniques:
    * Linear Regression
    * Polynomial Regression
    * Ridge, Lasso, Elastic Net Regularization
    * Stochastic Gradient Descent (SGD)

## **Key Analysis Steps**

* Data Exploration: Understand the features and their distributions.
* Data Cleaning: Handle missing values, outliers, and data transformation.
* Feature Engineering: Select relevant features for the regression model.
* Model Training: Use Linear Regression, Polynomial Regression, and regularization techniques.
* Evaluation: Assess model performance using appropriate metrics (e.g., RMSE, R²).


## **Results**

1. **Exploratory Data Analysis (EDA)**  
   - Histograms and scatter plots to visualize distributions and correlations.  
   - A heatmap of correlation coefficients shows relationships between features.  

2. **Key Findings**  
   - Positive correlations of `Log GDP per capita`, `Social support`, and `Healthy life expectancy` with happiness.  
   - Negative correlation between `Perceptions of corruption` and happiness.  

3. **Predictive Models**  
   - **Linear Regression:**  
     Achieved \( R^2 \): 0.8579 on training data, 0.8594 on testing data.  
   - **Regularized Models:**  
     Ridge, Lasso, and ElasticNet were used to address overfitting and improve generalization.

4. **Model Comparisons**  
   - Ridge Regression was less sensitive to regularization.  
   - Lasso Regression showed strong feature selection capabilities.  
   - ElasticNet balanced Ridge and Lasso for combined advantages.

**Conclusions**

The regression model provided a useful approach to predict happiness scores based on socio-economic factors. By experimenting with different techniques such as polynomial regression and regularization, we were able to improve the model's accuracy and reduce overfitting. Future work could involve more advanced techniques like ensemble methods or deep learning to improve predictive performance further.
