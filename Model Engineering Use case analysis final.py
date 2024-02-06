#!/usr/bin/env python
# coding: utf-8

# In[56]:


# Note: Before running this code, ensure that you have installed the required libraries.
# Note: Replace the file path with the correct path to the "PSP_Jan_Feb_2019.xlsx" Excel file.
# This code reads data from the specified Excel file into a DataFrame.

# Import necessary libraries
import pandas as pd  # For data manipulation and analysis
import matplotlib.pyplot as plt  # For data visualization
import seaborn as sns  # For statistical data visualization
import numpy as np  # For numerical operations
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score  # For model selection and evaluation
from sklearn.neighbors import KNeighborsClassifier  # For k-nearest neighbors classification
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, make_scorer  # For model evaluation metrics
from sklearn.preprocessing import StandardScaler, LabelBinarizer, OneHotEncoder, label_binarize  # For data preprocessing
from imblearn.over_sampling import SMOTE  # For handling imbalanced datasets
from sklearn.linear_model import LogisticRegression  # For logistic regression modeling
from sklearn.ensemble import RandomForestClassifier  # For random forest classification modeling
from sklearn.svm import SVC  # For support vector machine classification modeling
import shap  # For SHAP (SHapley Additive exPlanations) values
from sklearn.metrics import precision_recall_curve, auc, roc_curve  # For precision-recall and ROC curve analysis
from sklearn.compose import ColumnTransformer  # For applying transformers to columns in a dataset
from sklearn.model_selection import learning_curve  # For plotting learning curves

# Read data from Excel file into a DataFrame
df = pd.read_excel(r"C:\Users\HP\OneDrive\Desktop\Model Engineering\Dataset_excel\PSP_Jan_Feb_2019.xlsx")

# Print column names of the DataFrame
print(df.columns)


# In[57]:


# Check the number of missing values for each column
missing_values = df.isnull().sum()

# Check the percentage of missing values for each column
percentage_missing = (missing_values / len(df)) * 100

# Create a DataFrame to display the missing values information
missing_data = pd.DataFrame({
    'Missing Values': missing_values,
    'Percentage Missing': percentage_missing
})

# Print the missing values information
print(missing_data)


# In[58]:


# Determine the data types for each column in the DataFrame
data_types = df.dtypes

# Print the data types information
print(data_types)


# In[59]:


# Iterate through each column in the DataFrame
for column in df.columns:
    # Get unique values in the current column
    unique_values = df[column].unique()
    
    # Print the column name and its unique values
    print(f"Column: {column}\nUnique Values: {unique_values}\n")


# In[60]:


# Print the number of rows in the DataFrame
print("Number of rows in ideal_data set:", len(df))

# Print the number of columns in the DataFrame
print("Number of columns in ideal_data set:", len(df.columns))


# In[61]:


# Get the DataFrame's first five rows.
first_five_rows = df.head(5)

# Print the first five rows
print(first_five_rows)


# In[62]:


# Get the last five rows of the DataFrame
last_five_rows = df.tail(5)

# Print the last five rows
print(last_five_rows)


# In[63]:


# Print information about the DataFrame, including data types and memory usage
print(df.info())


# In[64]:


# Drop the 'Unnamed: 0' column from the DataFrame
df = df.drop('Unnamed: 0', axis=1)

# Print the remaining columns after dropping 'Unnamed: 0'
print(df.columns)


# In[65]:


# Print descriptive statistics for the 'amount' column
print(df['amount'].describe().to_frame().T)

# Visualization of transaction amount distribution
plt.figure(figsize=(10, 6))

# Plot a histogram with 30 bins and a kernel density estimate
sns.histplot(df['amount'], bins=30, kde=True)

# Set plot title and axis labels
plt.title('Distribution of Transaction Amount')
plt.xlabel('Transaction Amount')
plt.ylabel('Frequency')

# Show the plot of transaction amount distribution
plt.show()


# In[66]:


# Print summary statistics for the 'success' column
print("Summary statistics for 'success' column:")
print(df['success'].value_counts())

# Visualization of the distribution of 'success' column
plt.figure(figsize=(6, 4))

# Create a countplot for the 'success' column
sns.countplot(x='success', data=df)

# Set plot title and axis labels
plt.title('Distribution of Success')
plt.xlabel('Success')
plt.ylabel('Count')

# Show the plot distribution of success
plt.show()


# In[67]:


# Print summary statistics for the '3D_secured' column
print("\nSummary statistics for '3D_secured' column:")
print(df['3D_secured'].value_counts())

# Visualization of the distribution of '3D_secured' column
plt.figure(figsize=(6, 4))

# Create a countplot for the '3D_secured' column
sns.countplot(x='3D_secured', data=df)

# Set plot title and axis labels
plt.title('Distribution of 3D Secure Transactions')
plt.xlabel('3D Secured')
plt.ylabel('Count')

# Show the plot distribution of 3D_secured
plt.show()


# In[68]:


# Print summary statistics for the 'country' column
print("\nSummary statistics for 'country' column:")
print(df['country'].value_counts())

# Visualization of the count of transactions by country
plt.figure(figsize=(12, 6))

# Create a countplot for the 'country' column
sns.countplot(x='country', data=df)

# Set plot title and axis labels
plt.title('Count of Transactions by Country')
plt.xlabel('Country')
plt.ylabel('Count')

# Show the plot count of transactions by country
plt.show()


# In[69]:


# Print summary statistics for the 'PSP' column
print("\nSummary statistics for 'PSP' column:")
print(df['PSP'].value_counts())

# Visualization of the distribution of transactions by PSP
plt.figure(figsize=(12, 6))

# Create a countplot for the 'PSP' column
sns.countplot(x='PSP', data=df)

# Set plot title, axis labels, and rotate x-axis labels for better readability
plt.title('Distribution of Transactions by PSP')
plt.xlabel('PSP')
plt.ylabel('Count')
plt.xticks(rotation=45)

# Show the plot distribution of transactions by PSP
plt.show()


# In[70]:


# Print summary statistics for the 'card' column
print("\nSummary statistics for 'card' column:")
print(df['card'].value_counts())

# Visualization of the distribution of cards
plt.figure(figsize=(10, 6))

# Create a countplot for the 'card' column
sns.countplot(x='card', data=df)

# Set plot title and axis labels
plt.title('Distribution of Card')
plt.xlabel('Card')
plt.ylabel('Count')

# Show the plot distribution of cards
plt.show()


# In[71]:


# Visualization of the relationship between success and transaction amount using a boxplot
plt.figure(figsize=(8, 6))

# Create a boxplot with 'success' on the x-axis and 'amount' on the y-axis
sns.boxplot(x='success', y='amount', data=df)

# Set plot title and axis labels
plt.title('Relationship between Success and Transaction Amount')
plt.xlabel('Success')
plt.ylabel('Transaction Amount')

# Show the plot relationship between success and transaction amount
plt.show()


# In[72]:


# Create a pair plot for selected variables: 'amount', 'success', and '3D_secured'
sns.pairplot(df[['amount', 'success', '3D_secured']])

# Set the overall title for the pair plot
plt.suptitle('Pair Plot of Variables')

# Show the plot
plt.show()


# In[73]:


# Create a pair plot for selected variables: 'amount', 'success', and '3D_secured', with hue based on 'card' column
sns.pairplot(df, hue='card', vars=['amount', 'success', '3D_secured'])

# Set the overall title for the pair plot
plt.suptitle('Pair Plot of Variables by Card Type')

# Show the plot
plt.show()


# In[74]:


# Select relevant columns for correlation analysis
selected_columns = ['amount', 'success', '3D_secured']

# Calculate the correlation matrix
correlation_matrix = df[selected_columns].corr()

# Create a heatmap to visualize the correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=.5)

# Set plot title
plt.title('Correlation Matrix: Amount, Success, and 3D_secured')

# Show the plot
plt.show()


# In[75]:


# Create a bar plot to visualize success rate by card type
plt.figure(figsize=(8, 6))

# Use a bar plot with 'card' on the x-axis, 'success' on the y-axis, and no confidence interval (ci=None)
sns.barplot(x='card', y='success', data=df, ci=None)

# Set plot title and axis labels
plt.title('Success Rate by Card Type')
plt.xlabel('Card Type')
plt.ylabel('Success Rate')

# Show the plot
plt.show()


# In[76]:


# Convert 'tmsp' column to datetime format
df['tmsp'] = pd.to_datetime(df['tmsp'])

# Set 'tmsp' as the index of the DataFrame
df.set_index('tmsp', inplace=True)

# Resample the data by day and plot the daily transaction volume
df.resample('D').size().plot(title='Daily Transaction Volume', figsize=(12, 6))

# Set x-axis and y-axis labels
plt.xlabel('Date')
plt.ylabel('Transaction Volume')

# Show the plot
plt.show()


# In[77]:


# Create a boxplot to visualize transaction amounts by country
plt.figure(figsize=(14, 8))

# Use a boxplot with 'country' on the x-axis and 'amount' on the y-axis
sns.boxplot(x='country', y='amount', data=df)

# Set plot title, axis labels, and rotate x-axis labels for better readability
plt.title('Boxplot of Transaction Amounts by Country')
plt.xlabel('Country')
plt.ylabel('Transaction Amount')
plt.xticks(rotation=45)

# Show the plot
plt.show()


# In[78]:


# Resample the DataFrame by day and calculate the sum for each day
df_resampled = df.resample('D').sum()

# Create a line plot for success and failure over time
plt.figure(figsize=(12, 6))

# Use lineplot to plot 'success' and 'failure' against the resampled dates
sns.lineplot(x=df_resampled.index, y='success', data=df_resampled, label='Success')
sns.lineplot(x=df_resampled.index, y=1 - df_resampled['success'], label='Failure')

# Set plot title, x-axis label, y-axis label, and add legend
plt.title('Time Series of Success and Failure')
plt.xlabel('Date')
plt.ylabel('Count')
plt.legend()

# Show the plot
plt.show()


# In[79]:


# Create a figure 
plt.figure(figsize=(12, 6))

# Calculate the daily success rate and plot it over time
success_rate_by_date = df.resample('D')['success'].mean()

# Use a line plot with markers to visualize the success rate over time
success_rate_by_date.plot(marker='o', linestyle='-')

# Set plot title, x-axis label, y-axis label
plt.title('Success Rate Over Time')
plt.xlabel('Date')
plt.ylabel('Success Rate')

# Show the plot
plt.show()


# In[80]:


# Reset the index of the DataFrame
df.reset_index(inplace=True)

# Define dictionaries for mapping success and failure transaction fees based on card type
success_fee_mapping = {
    'Moneycard': 5,
    'Goldcard': 10,
    'UK_Card': 3,
    'Simplecard': 1
}
failed_fee_mapping = {
    'Moneycard': 2,
    'Goldcard': 5,
    'UK_Card': 1,
    'Simplecard': 0.5
}

# Create a new column 'transaction_fee' based on success or failure, using np.where
df['transaction_fee'] = np.where(df['success'] == 1, df['PSP'].map(success_fee_mapping), df['PSP'].map(failed_fee_mapping))

# Print the first few rows of the updated DataFrame
print(df.head())


# In[81]:


# Convert 'tmsp' column to datetime format
df['tmsp'] = pd.to_datetime(df['tmsp'])

# Add 'day_of_week' column with numerical mapping (Monday 0, ..., Sunday 6)
df['day_of_week'] = df['tmsp'].dt.dayofweek

# Add 'minute_of_day' column
df['minute_of_day'] = df['tmsp'].dt.hour * 60 + df['tmsp'].dt.minute

# Print the first few rows of the updated DataFrame
print(df.head())


# In[82]:


# Sort the DataFrame by 'country', 'amount', 'day_of_week', 'minute_of_day'
df.sort_values(by=['country', 'amount', 'minute_of_day'], inplace=True)

# Create a new column 'payment_attempts' and initialize it with 1
df['payment_attempts'] = 1

# Identify rows where consecutive attempts have the same 'country', 'amount', 'day_of_week', and 'minute_of_day'
# Increment the 'payment_attempts' for those rows
df['payment_attempts'] = df.groupby(['country', 'amount', 'minute_of_day']).cumcount() + 1

# Reset the DataFrame index
df.reset_index(drop=True, inplace=True)

# Display the updated DataFrame
print(df.head())


# In[83]:


# Drop the 'tmsp' column
df.drop('tmsp', axis=1, inplace=True)
print(df.head())


# In[84]:


# Select numeric columns for summary statistics
numeric_columns = ['transaction_fee']

# Calculate descriptive statistics for numeric columns
numeric_summary = df[numeric_columns].describe()

# Transpose the summary for better readability
transposed_summary = numeric_summary.transpose()

# Print the transposed summary
print(transposed_summary)

# Create a histogram to visualize the distribution of transaction fees
plt.hist(df['transaction_fee'], bins=10, edgecolor='black')

# Set plot title, x-axis label, y-axis label
plt.title('Transaction Fee Histogram')
plt.xlabel('Transaction Fee')
plt.ylabel('Frequency')

# Show the plot
plt.show()


# In[85]:


# Create a histogram to visualize the distribution of 'day_of_week'
plt.hist(df['day_of_week'], bins=7, edgecolor='black')

# Set x-axis label, y-axis label, and plot title
plt.xlabel('Day of Week')
plt.ylabel('Frequency')
plt.title('Distribution of Day of Week')

# Show the plot
plt.show()


# In[86]:


# Create a histogram to visualize the distribution of 'minute_of_day'
plt.hist(df['minute_of_day'], bins=24, edgecolor='black')

# Set x-axis label, y-axis label, and plot title
plt.xlabel('Minute of Day')
plt.ylabel('Frequency')
plt.title('Distribution of Minute of Day')

# Show the plot
plt.show()


# In[87]:


# Count the occurrences of each value in the 'payment_attempts' column and sort by index
attempt_counts = df['payment_attempts'].value_counts().sort_index()

# Print the count of payment attempts for each value
print(attempt_counts)

# Create a histogram to visualize the distribution of 'payment_attempts'
plt.hist(df['payment_attempts'], bins=7, edgecolor='black')

# Set x-axis label, y-axis label, and plot title
plt.xlabel('Payment Attempts')
plt.ylabel('Frequency')
plt.title('Distribution of Payment Attempts')

# Show the plot
plt.show()


# In[88]:


# Examine unique values in 'country' column
unique_countries = df['country'].unique()
print("Unique values in 'country' column:", unique_countries)

# Examine unique values in 'PSP' column
unique_psps = df['PSP'].unique()
print("Unique values in 'PSP' column:", unique_psps)

# Examine unique values in 'card' column
unique_cards = df['card'].unique()
print("Unique values in 'card' column:", unique_cards)


# In[89]:


# Box plot for 'amount'
plt.figure(figsize=(8, 6))
sns.boxplot(x=df['amount'])
plt.title('Box Plot for Amount')
plt.show()

# Box plot for 'transaction_fee'
plt.figure(figsize=(8, 6))
sns.boxplot(x=df['transaction_fee'])
plt.title('Box Plot for Transaction Fee')
plt.show()


# In[90]:


# Set a threshold for identifying rare categories
threshold = 10

# Identify rare categories for 'country', 'PSP', and 'card'
rare_country = df['country'].value_counts()[df['country'].value_counts() < threshold].index
rare_PSP = df['PSP'].value_counts()[df['PSP'].value_counts() < threshold].index
rare_card = df['card'].value_counts()[df['card'].value_counts() < threshold].index

# Print the rare categories
print("Rare countries:", rare_country)
print("Rare PSPs:", rare_PSP)
print("Rare cards:", rare_card)


# In[91]:


# Calculate correlations between variables
correlation_matrix = df.corr()

# Visualize the correlation matrix using a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix')
plt.show()


# In[92]:


# Create a copy of the DataFrame
df_1 = df.copy()

# Drop the 'transaction_fee' column from the copied DataFrame
df_1 = df_1.drop('transaction_fee', axis=1)

# Print the modified DataFrame without the 'transaction_fee' column
print(df_1)


# In[93]:


# Building Baseline Model 1 
def evaluate_model(model, X, y, set_name):
    # Make predictions
    y_pred = model.predict(X)
    
    # Calculate performance metrics
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    roc_auc = roc_auc_score(y, y_pred)
    conf_matrix = confusion_matrix(y, y_pred)
    
    # Print the performance metrics with set name
    print(f'Model Performance on the {set_name}:')
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1-Score: {f1:.4f}')
    print(f'AUC-ROC: {roc_auc:.4f}')
    print('Confusion Matrix:')
    print(conf_matrix)
    print('\n')

# Encode categorical variables using one-hot encoding
df_1_encoded = pd.get_dummies(df_1, columns=['country', 'PSP', 'card'])

# Split the dataset into train, validation, and test sets
X_original = df_1_encoded.drop('success', axis=1)
y_original = df_1_encoded['success']
X_train_original, X_temp_original, y_train_original, y_temp_original = train_test_split(X_original, y_original, test_size=0.2, random_state=42)
X_validation_original, X_test_original, y_validation_original, y_test_original = train_test_split(X_temp_original, y_temp_original, test_size=0.5, random_state=42)

# Create and train a baseline KNN model
baseline_knn_model = KNeighborsClassifier()
baseline_knn_model.fit(X_train_original, y_train_original)

# Evaluate the model on the validation set
evaluate_model(baseline_knn_model, X_validation_original, y_validation_original, "validation set")

# Evaluate the model on the test set
evaluate_model(baseline_knn_model, X_test_original, y_test_original, "test set")


# In[94]:


# Encode categorical variables using one-hot encoding
df_1 = pd.get_dummies(df_1, columns=['country', 'PSP', 'card'])

# Print the DataFrame after one-hot encoding
print(df_1)


# In[95]:



# Drop the target variable 'success' from the features
X = df_1.drop('success', axis=1)

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit and transform the features using StandardScaler
X_scaled = scaler.fit_transform(X)

# Create a DataFrame with scaled features and include the 'success' column
df_1_scaled = pd.DataFrame(X_scaled, columns=X.columns)
df_1_scaled['success'] = df_1['success']

# Print the DataFrame with scaled features
print(df_1_scaled)


# In[96]:


# Extract features and target variable for SMOTE
X_smote = df_1_scaled.drop('success', axis=1)
y_smote = df_1_scaled['success']

# Apply SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_smote, y_smote)

# Create a new DataFrame with the resampled features and the target variable
df_1_resampled = pd.DataFrame(X_resampled, columns=X_smote.columns)
df_1_resampled['success'] = y_resampled

# Display the updated DataFrame with SMOTE
print(df_1_resampled)


# In[97]:


# Development of Model 1

# Split the resampled dataset into features (X_1) and target variable (y_1)
X_1 = df_1_resampled.drop('success', axis=1)
y_1 = df_1_resampled['success']

# Split the dataset into train, validation, and test sets
X_train_1, X_temp, y_train_1, y_temp = train_test_split(X_1, y_1, test_size=0.2, random_state=42)
X_valid_1, X_test_1, y_valid_1, y_test_1 = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Initialize a StandardScaler and scale the features for training, validation, and test sets
scaler = StandardScaler()
X_train_scaled_1 = scaler.fit_transform(X_train_1)
X_valid_scaled_1 = scaler.transform(X_valid_1)
X_test_scaled_1 = scaler.transform(X_test_1)

# Create and train Logistic Regression model
logreg_model_1 = LogisticRegression(random_state=42)
logreg_model_1.fit(X_train_scaled_1, y_train_1)

# Create and train Random Forest model
rf_model_1 = RandomForestClassifier(random_state=42)
rf_model_1.fit(X_train_scaled_1, y_train_1)

# Create and train Support Vector Machine (SVM) model
svm_model_1 = SVC(probability=True, random_state=42)
svm_model_1.fit(X_train_scaled_1, y_train_1)

# Define a function to evaluate the performance of a model
def evaluate_model(model, X_scaled, y, set_name):
    y_pred = model.predict(X_scaled)
    y_pred_proba = model.predict_proba(X_scaled)[:, 1]

    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    roc_auc = roc_auc_score(y, y_pred_proba)
    accuracy = accuracy_score(y, y_pred)
    conf_matrix = confusion_matrix(y, y_pred)

    print(f'Model Performance on {set_name} set - {type(model).__name__}:')
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1-Score: {f1:.4f}')
    print(f'AUC-ROC: {roc_auc:.4f}')
    print('Confusion Matrix:')
    print(conf_matrix)
    print('\n')

# Evaluate the models on the validation set
evaluate_model(logreg_model_1, X_valid_scaled_1, y_valid_1, set_name="Validation")
evaluate_model(rf_model_1, X_valid_scaled_1, y_valid_1, set_name="Validation")
evaluate_model(svm_model_1, X_valid_scaled_1, y_valid_1, set_name="Validation")

# Evaluate the models on the test set
evaluate_model(logreg_model_1, X_test_scaled_1, y_test_1, set_name="Test")
evaluate_model(rf_model_1, X_test_scaled_1, y_test_1, set_name="Test")
evaluate_model(svm_model_1, X_test_scaled_1, y_test_1, set_name="Test")


# In[98]:


# Random Forest Classification with Cross-Validation and Hyperparameter Tuning

# Define the Random Forest model
rf_model = RandomForestClassifier(random_state=42)

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Create GridSearchCV
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)

# Fit the model on the training data
grid_search.fit(X_train_scaled_1, y_train_1)

# Print the best parameters and best score
print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)

# Get the best model from GridSearchCV
best_rf_model = grid_search.best_estimator_

# Evaluate the best model on the validation and test sets
evaluate_model(best_rf_model, X_valid_scaled_1, y_valid_1, set_name="Validation")
evaluate_model(best_rf_model, X_test_scaled_1, y_test_1, set_name="Test")


# In[44]:


#Summary bar plot of the importance of individual features in Model 1

# Create a SHAP TreeExplainer for the best random forest Model 1
explainer = shap.TreeExplainer(best_rf_model)

# Obtain SHAP values for the validation set
shap_values = explainer.shap_values(X_valid_scaled_1)

# Define feature names for summary plot
feature_names = ["amount", "3D_secured", "day_of_week", "minute_of_day", "payment_attempts",
                 "country_Austria", "country_Germany", "country_Switzerland",
                 "card_Diners", "card_Master", "card_Visa",
                 "PSP_Moneycard", "PSP_Simplecard", "PSP_UK_Card", "PSP_Goldcard", "success"]

# Create a summary plot using SHAP values
shap.summary_plot(shap_values[1], X_valid_scaled_1, feature_names=feature_names, plot_type="bar", show=False)

# Show the plot
plt.show()


# In[45]:


# Precision-Recall Curve for validation and test set of Model 1

# Predict probabilities on the validation set
y_valid_pred_proba = best_rf_model.predict_proba(X_valid_scaled_1)[:, 1]

# Predict probabilities on the test set
y_test_pred_proba = best_rf_model.predict_proba(X_test_scaled_1)[:, 1]

# Compute precision-recall curve values for validation set
precision_valid, recall_valid, thresholds_valid = precision_recall_curve(y_valid_1, y_valid_pred_proba)

# Compute area under the curve (AUC) for validation set
pr_auc_valid = auc(recall_valid, precision_valid)

# Plot the precision-recall curve for validation set
plt.figure(figsize=(8, 6))
plt.plot(recall_valid, precision_valid, label=f'Validation Set (AUC = {pr_auc_valid:.2f})', color='b')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve for Validation Set')
plt.legend(loc='lower left')
plt.show()

# Compute precision-recall curve values for test set
precision_test, recall_test, thresholds_test = precision_recall_curve(y_test_1, y_test_pred_proba)

# Compute area under the curve (AUC) for test set
pr_auc_test = auc(recall_test, precision_test)

# Plot the precision-recall curve for test set
plt.figure(figsize=(8, 6))
plt.plot(recall_test, precision_test, label=f'Test Set (AUC = {pr_auc_test:.2f})', color='r')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve for Test Set')
plt.legend(loc='lower left')
plt.show()


# In[46]:


# Impact of threshold adjustment on Precision and Recall metrics of Model 1

def evaluate_model(model, X, y, set_name=""):
    """
    Evaluate the performance of a classification model on a given dataset.

    Parameters:
    - model: The trained classification model.
    - X: The feature matrix of the dataset.
    - y: The true labels of the dataset.
    - set_name: The name of the dataset (e.g., "Train", "Validation", "Test").

    Prints:
    - Accuracy, Precision, Recall, F1 Score, ROC AUC Score, Confusion Matrix.
    """
    # Make predictions
    y_pred = model.predict(X)

    # Convert y to 1D array if it's a DataFrame
    if hasattr(y, 'values'):
        y = y.values.ravel()

    # Calculate and print relevant evaluation metrics
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, average='weighted')
    recall = recall_score(y, y_pred, average='weighted')
    f1 = f1_score(y, y_pred, average='weighted')
    roc_auc = roc_auc_score(y, y_pred)
    confusion_mat = confusion_matrix(y, y_pred)

    print(f"{set_name} Set Evaluation:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print("Confusion Matrix:")
    print(confusion_mat)
    print("\n")

# Predict probabilities on the validation set
y_valid_pred_proba = best_rf_model.predict_proba(X_valid_scaled_1)[:, 1]

# Calculate precision and recall for different thresholds
precision, recall, thresholds = precision_recall_curve(y_valid_1, y_valid_pred_proba)

# Find the threshold for a desired trade-off (e.g., balance between precision and recall)
desired_precision = 0.85
desired_recall = 0.85

# Find the index of the point on the curve closest to the desired trade-off
closest_point_index = np.argmin(np.abs(precision - desired_precision) + np.abs(recall - desired_recall))

# Get the corresponding threshold
desired_threshold = thresholds[closest_point_index]

# Print the desired threshold
print(f"Desired Threshold: {desired_threshold:.4f}")

# Adjust the decision threshold for the validation set
adjusted_predictions_valid = (y_valid_pred_proba >= desired_threshold).astype(int)

# Now, use the adjusted predictions in your evaluation or downstream tasks for the validation set
evaluate_model(best_rf_model, X_valid_scaled_1, y_valid_1, set_name="Validation with Adjusted Threshold")

# Predict probabilities on the test set
y_test_pred_proba = best_rf_model.predict_proba(X_test_scaled_1)[:, 1]

# Adjust the decision threshold for the test set
adjusted_predictions_test = (y_test_pred_proba >= desired_threshold).astype(int)

# Now, use the adjusted predictions in your evaluation or downstream tasks for the test set
evaluate_model(best_rf_model, X_test_scaled_1, y_test_1, set_name="Test with Adjusted Threshold")


# In[49]:


# ROC Curve on validation and test set of Model 1

# Get probabilities on the validation set
y_valid_pred_proba = best_rf_model.predict_proba(X_valid_scaled_1)[:, 1]
# Get probabilities on the test set
y_test_pred_proba = best_rf_model.predict_proba(X_test_scaled_1)[:, 1]

# Adjust the decision threshold for both validation and test sets
adjusted_threshold = 0.445
adjusted_predictions_valid = (y_valid_pred_proba >= adjusted_threshold).astype(int)
adjusted_predictions_test = (y_test_pred_proba >= adjusted_threshold).astype(int)

# Compute ROC Curve for both validation and test sets
fpr_valid, tpr_valid, _ = roc_curve(y_valid_1, y_valid_pred_proba)
roc_auc_valid = auc(fpr_valid, tpr_valid)

fpr_test, tpr_test, _ = roc_curve(y_test_1, y_test_pred_proba)
roc_auc_test = auc(fpr_test, tpr_test)

# Plot ROC Curves separately
import matplotlib.pyplot as plt

# Plot ROC Curve for Validation set
plt.figure(figsize=(8, 8))
plt.plot(fpr_valid, tpr_valid, color='darkorange', lw=2, label=f'Validation ROC Curve (AUC = {roc_auc_valid:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve - Validation Set')
plt.legend()
plt.show()

# Plot ROC Curve for Test set
plt.figure(figsize=(8, 8))
plt.plot(fpr_test, tpr_test, color='green', lw=2, label=f'Test ROC Curve (AUC = {roc_auc_test:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve - Test Set')
plt.legend()
plt.show()


# In[50]:


# Pie chart for analyzing correctly classified and misclassified on validation Set of Model 1

# Confusion matrix of Model 1 on Validation Set
confusion_matrix = np.array([[3260, 785], [768, 3223]])

# Calculate total correctly classified and misclassified instances
correctly_classified = np.trace(confusion_matrix)
misclassified = np.sum(confusion_matrix) - correctly_classified

# Print the results
print("Total Correctly Classified Instances on Validation Set of model 1:", correctly_classified)
print("Total Misclassified Instances on Validation Set of model 1:", misclassified)

# Plot a pie chart
labels = ['Correctly Classified', 'Misclassified of model 1']
sizes = [correctly_classified, misclassified]
colors = ['lightgreen', 'lightcoral']
explode = (0.1, 0)  # explode the first slice

plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

# Display the pie chart
plt.title('Correctly vs Misclassified Instances on Validation Set of model 1')
plt.show()


# In[51]:


# Pie chart for analyzing correctly classified and misclassified  on Test Set of Model 1
# Confusion matrix of Model 1 on Test Set
confusion_matrix = np.array([[3264, 759], [803, 3211]])

# Calculate total correctly classified and misclassified instances
correctly_classified = np.trace(confusion_matrix)
misclassified = np.sum(confusion_matrix) - correctly_classified

# Print the results
print("Total Correctly Classified Instances on Test Set of model 1:", correctly_classified)
print("Total Misclassified Instances on Test Set of model 1:", misclassified)

# Plot a pie chart
labels = ['Correctly Classified', 'Misclassified']
sizes = [correctly_classified, misclassified]
colors = ['lightgreen', 'lightcoral']
explode = (0.1, 0)  # explode the first slice

plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

# Display the pie chart
plt.title('Correctly vs Misclassified Instances on Test Set of model 1')
plt.show()


# In[52]:


#Evaluate Cross-Validation Stability of Model 1
# Define the parameter grid for hyperparameter tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Create GridSearchCV
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)

# Fit the model on the training data
grid_search.fit(X_train_scaled_1, y_train_1)

# Access cross-validation results
cv_results = grid_search.cv_results_

# Print performance metrics for each fold
for i in range(grid_search.n_splits_):
    print(f"\nResults for Fold {i + 1}:")
    print(f"Mean Test Score: {cv_results[f'mean_test_score'][i]}")
    print(f"Standard Deviation Test Score: {cv_results[f'std_test_score'][i]}")
    print(f"Params: {cv_results['params'][i]}")


# In[53]:


# Overfitting Analysis for Model 1
# Function to plot learning curve
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("F1 Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring='f1_macro')
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

# Plot learning curve
title = "Learning Curve (Random Forest)"
cv = 5  # Cross-validation folds
plot_learning_curve(best_rf_model, title, X_train_scaled_1, y_train_1, cv=cv)

plt.show()


# In[99]:


# Drop 'success' and 'transaction_fee' columns from the original DataFrame
X_original = df.drop(['success', 'transaction_fee'], axis=1)

# One-hot encode categorical columns: 'country', 'PSP', 'card'
X_original_encoded = pd.get_dummies(X_original, columns=['country', 'PSP', 'card'])

# Identify additional columns in X_original_encoded not present in X_train_1
additional_columns = set(X_original_encoded.columns) - set(X_train_1.columns)
if additional_columns:
    print(f"Additional columns in X_original_encoded: {additional_columns}")

# Identify missing columns in X_original_encoded compared to X_train_1
missing_columns = set(X_train_1.columns) - set(X_original_encoded.columns)
if missing_columns:
    print(f"Missing columns in X_original_encoded: {missing_columns}")

# Keep only columns present in X_train_1 in X_original_encoded
X_original_encoded = X_original_encoded[X_train_1.columns]

# Scale the features using the previously defined 'scaler'
X_original_scaled = scaler.transform(X_original_encoded)

# Predict success probabilities using the trained random forest Model 1
success_probabilities = best_rf_model.predict_proba(X_original_scaled)[:, 1]

# Add the success probabilities as a new column to the original DataFrame
df['success_probabilities'] = success_probabilities

# Display the updated DataFrame
print(df)


# In[100]:


# Create a copy of the DataFrame
df_2 = df.copy()

# Print the copied DataFrame
print(df_2)


# In[101]:


# Building Baseline Model 2 
# Encode categorical variables using one-hot encoding
df_2_encoded = pd.get_dummies(df_2, columns=['country', 'card'])

# Split the dataset into features (X) and target variable (y)
X = df_2_encoded.drop(['PSP', 'success'], axis=1)
y = df_2_encoded['PSP']

# Split the dataset into train, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Create and train a KNN model
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)

# Define a function to evaluate the performance of the model
def evaluate_model(model, X, y, set_name):
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X) if hasattr(model, 'predict_proba') else None

    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, average='micro')
    recall = recall_score(y, y_pred, average='micro')
    f1 = f1_score(y, y_pred, average='micro')
    roc_auc = roc_auc_score(y, y_proba, multi_class='ovr') if y_proba is not None else None
    conf_matrix = confusion_matrix(y, y_pred)

    print(f'Model Performance on {set_name} set - {type(model).__name__}:')
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1-Score: {f1:.4f}')
    if roc_auc is not None:
        print(f'AUC-ROC: {roc_auc:.4f}')
    print('Confusion Matrix:')
    print(conf_matrix)
    print('\n')

# Evaluate the KNN model on the validation set
evaluate_model(knn_model, X_val, y_val, set_name='Validation')

# Evaluate the KNN model on the test set
evaluate_model(knn_model, X_test, y_test, set_name='Test')


# In[102]:


# Encode categorical variables using one-hot encoding
df_encoded_2 = pd.get_dummies(df_2, columns=['country', 'card'])

# Print the DataFrame after one-hot encoding
print(df_encoded_2)


# In[103]:


# Drop the target variables 'PSP' and 'success' from the features
X_2 = df_encoded_2.drop(['PSP', 'success'], axis=1)

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit and transform the features using StandardScaler
X_scaled_2 = scaler.fit_transform(X_2)

# Create a DataFrame with scaled features and include the 'PSP' column
df_scaled_2 = pd.DataFrame(X_scaled_2, columns=X_2.columns)
df_scaled_2[['PSP']] = df_encoded_2[['PSP']]

# Print the DataFrame with scaled features
print(df_scaled_2)


# In[104]:


# Split the scaled DataFrame into features (X_smote_2) and target variable (y_smote_2)
X_smote_2 = df_scaled_2.drop('PSP', axis=1)
y_smote_2 = df_scaled_2['PSP']

# Initialize the SMOTE with a random state
smote_2 = SMOTE(random_state=42)

# Resample the dataset using SMOTE
X_resampled_2, y_resampled_2 = smote_2.fit_resample(X_smote_2, y_smote_2)

# Create a DataFrame with resampled features and include the 'PSP' column
df_2_resampled = pd.DataFrame(X_resampled_2, columns=X_smote_2.columns)
df_2_resampled['PSP'] = y_resampled_2

# Print the resampled DataFrame
print(df_2_resampled)


# In[105]:


# Split the resampled DataFrame into features (X_2) and target variable (y_2)
X_2 = df_2_resampled.drop('PSP', axis=1)
y_2 = df_2_resampled['PSP']

# Split the dataset into train, validation, and test sets
X_train_2, X_temp_2, y_train_2, y_temp_2 = train_test_split(X_2, y_2, test_size=0.2, random_state=42)
X_valid_2, X_test_2, y_valid_2, y_test_2 = train_test_split(X_temp_2, y_temp_2, test_size=0.5, random_state=42)

# Normalize Data
scaler = StandardScaler()
X_train_scaled_2 = scaler.fit_transform(X_train_2)
X_valid_scaled_2 = scaler.transform(X_valid_2)
X_test_scaled_2 = scaler.transform(X_test_2)


# In[106]:


def evaluate_model(model, X, y, set_name):
    # Make predictions on the data
    y_pred = model.predict(X)

    # Extract the values of the target variable
    y_values = y.values.ravel()  # Convert DataFrame to 1D array

    # Calculate and print relevant evaluation metrics
    precision = precision_score(y_values, y_pred, average='weighted')
    recall = recall_score(y_values, y_pred, average='weighted')
    f1 = f1_score(y_values, y_pred, average='weighted')

    # For binary classification, set multi_class to 'ovr'
    if len(model.classes_) == 2:
        roc_auc = roc_auc_score(y_values, model.predict_proba(X)[:, 1], average='weighted')
    else:
        roc_auc = roc_auc_score(pd.get_dummies(y_values), model.predict_proba(X), average='weighted', multi_class='ovr')
    
    accuracy = accuracy_score(y_values, y_pred)

    print(f"{set_name} Set Evaluation:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_values, y_pred))


# In[107]:


# Development of Model 2

# Create a Logistic Regression model with specified parameters
logreg_model = LogisticRegression(random_state=42, C=1)

# Perform cross-validation on the training set
cv_scores = cross_val_score(logreg_model, X_train_scaled_2, y_train_2, scoring='f1_weighted', cv=StratifiedKFold(n_splits=5))

# Print cross-validation scores
print("Cross-Validation Scores of Logistic Regression:")
print(cv_scores)
print(f"Average F1 Weighted Score: {cv_scores.mean():.4f}\n")

# Fit the Logistic Regression model on the training set
logreg_model.fit(X_train_scaled_2, y_train_2)

# Print model performance on the validation set
print("Model Performance on Validation set - Logistic Regression:")
evaluate_model(logreg_model, X_valid_scaled_2, y_valid_2, set_name="Validation")

# Print model performance on the test set
print("\nModel Performance on Test set - Logistic Regression:")
evaluate_model(logreg_model, X_test_scaled_2, y_test_2, set_name="Test")


# In[108]:


# Create a Random Forest model with specified parameters
rf_model = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=None)

# Perform cross-validation on the training set
cv_scores_rf = cross_val_score(rf_model, X_train_scaled_2, y_train_2, scoring='f1_weighted', cv=StratifiedKFold(n_splits=5))

# Print cross-validation scores
print("Cross-Validation Scores:")
print(cv_scores_rf)
print(f"Average F1 Weighted Score: {cv_scores_rf.mean():.4f}\n")

# Fit the Random Forest model on the training set
rf_model.fit(X_train_scaled_2, y_train_2)

# Print model performance on the validation set
print("Model Performance on Validation set - Random Forest:")
evaluate_model(rf_model, X_valid_scaled_2, y_valid_2, set_name="Validation")

# Print model performance on the test set
print("Model Performance on Test set - Random Forest:")
evaluate_model(rf_model, X_test_scaled_2, y_test_2, set_name="Test")


# In[109]:



# Create a Support Vector Machine (SVM) model with specified parameters
svm_model = SVC(probability=True, random_state=42, C=1, gamma='scale')

# Perform cross-validation on the training set
cv_scores_svm = cross_val_score(svm_model, X_train_scaled_2, y_train_2, scoring='f1_weighted', cv=StratifiedKFold(n_splits=5))

# Print cross-validation scores
print("Cross-Validation Scores of SVC:")
print(cv_scores_svm)
print(f"Average F1 Weighted Score: {cv_scores_svm.mean():.4f}\n")

# Fit the SVM model on the training set
svm_model.fit(X_train_scaled_2, y_train_2)

# Print model performance on the validation set
print("Model Performance on Validation set - SVC:")
evaluate_model(svm_model, X_valid_scaled_2, y_valid_2, set_name="Validation")

# Print model performance on the test set
print("Model Performance on Test set - SVC:")
evaluate_model(svm_model, X_test_scaled_2, y_test_2, set_name="Test")


# In[110]:


# Random Forest Model 2 after hyperparameter tuning and cross-validation

# Create a Random Forest model with a set random state
rf_model = RandomForestClassifier(random_state=42)

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30]
}

# Use GridSearchCV for hyperparameter tuning
rf_grid = GridSearchCV(rf_model, param_grid, scoring='f1_weighted', cv=StratifiedKFold(n_splits=5), verbose=1)
rf_grid.fit(X_train_scaled_2, y_train_2)

# Get the best model from the grid search
rf_best_model = rf_grid.best_estimator_

# Print the best hyperparameters found
print("Best Hyperparameters:", rf_grid.best_params_)

# Perform cross-validation on the training set with the best model
cv_scores_rf = cross_val_score(rf_best_model, X_train_scaled_2, y_train_2, scoring='f1_weighted', cv=StratifiedKFold(n_splits=5))

# Print cross-validation scores
print("Cross-Validation Scores with Best Model of Random Forest Classification:")
print(cv_scores_rf)
print(f"Average F1 Weighted Score: {cv_scores_rf.mean():.4f}\n")

# Fit the best Random Forest model on the training set
rf_best_model.fit(X_train_scaled_2, y_train_2)

# Print model performance on the validation set
print("Model Performance on Validation set - RandomForestClassifier:")
evaluate_model(rf_best_model, X_valid_scaled_2, y_valid_2, set_name="Validation")

# Print model performance on the test set
print("Model Performance on Test set - RandomForestClassifier:")
evaluate_model(rf_best_model, X_test_scaled_2, y_test_2, set_name="Test")


# In[ ]:


#Summary bar plot of the importance of individual features in Model 2

# Create a SHAP TreeExplainer for the best Random Forest Model 2
explainer = shap.TreeExplainer(rf_best_model)

# Calculate SHAP values for the validation set
shap_values = explainer.shap_values(X_valid_scaled_2)

# Define the feature names 
feature_names = ["amount", "3D_secured", "transaction_fee", "day_of_week", "minute_of_day",
                  "payment_attempts", "success_probabilities", "country_Austria", "country_Germany",
                  "country_Switzerland", "card_Diners", "card_Master", "card_Visa", "PSP"]

# Create a summary plot of SHAP values
shap.summary_plot(shap_values[1], X_valid_scaled_2, feature_names=feature_names, plot_type="bar", show=False)

# Show the plot
plt.show()


# In[112]:


from sklearn.preprocessing import label_binarize
# Precision-Recall curve for validation and test set of Model 2

# Binarize the labels for precision-recall curve
y_valid_2_bin = label_binarize(y_valid_2, classes=np.unique(y_valid_2))
y_test_2_bin = label_binarize(y_test_2, classes=np.unique(y_test_2))

# Predict probabilities on the validation set
y_valid_pred_proba_rf = rf_best_model.predict_proba(X_valid_scaled_2)

# Calculate precision and recall for various thresholds
precision_rf_valid, recall_rf_valid, thresholds_rf_valid = precision_recall_curve(y_valid_2_bin.ravel(), y_valid_pred_proba_rf.ravel())

# Calculate AUC for validation set
auc_rf_valid = auc(recall_rf_valid, precision_rf_valid)


# Plot the precision-recall curve for validation set
plt.figure(figsize=(8, 6))
plt.plot(recall_rf_valid, precision_rf_valid, color='blue', label=f'Random Forest (Validation) - AUC: {auc_rf_valid:.4f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve - RandomForestClassifier (Validation)')
plt.legend()
plt.grid(True)
plt.show()

# Predict probabilities on the test set
y_test_pred_proba_rf = rf_best_model.predict_proba(X_test_scaled_2)

# Calculate precision and recall for various thresholds
precision_rf_test, recall_rf_test, thresholds_rf_test = precision_recall_curve(y_test_2_bin.ravel(), y_test_pred_proba_rf.ravel())

# Calculate AUC for test set
auc_rf_test = auc(recall_rf_test, precision_rf_test)


# Plot the precision-recall curve for test set
plt.figure(figsize=(8, 6))
plt.plot(recall_rf_test, precision_rf_test, color='green', label=f'Random Forest (Test) - AUC: {auc_rf_test:.4f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve - RandomForestClassifier (Test)')
plt.legend()
plt.grid(True)
plt.show()


# In[113]:


#ROC Curve on validation and test set of Model 2

def evaluate_model_with_roc(model, X, y, set_name="Set"):
    lb = LabelBinarizer()
    y_bin = lb.fit_transform(y)

    y_pred_prob = model.predict_proba(X)

    # Compute ROC curve and ROC area for each class
    n_classes = y_bin.shape[1]
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_pred_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], lw=2, label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {set_name}')
    plt.legend(loc='lower right')
    plt.show()

# Evaluate on Validation Set with ROC Curve
print("Model Performance on Validation set - RandomForestClassifier:")
evaluate_model_with_roc(rf_best_model, X_valid_scaled_2, y_valid_2, set_name="Validation Set")

# Evaluate on Test Set with ROC Curve
print("Model Performance on Test set - RandomForestClassifier:")
evaluate_model_with_roc(rf_best_model, X_test_scaled_2, y_test_2, set_name="Test Set")


# In[114]:


# Pie chart for analyzing correctly and misclassified classified on the validation Set of Model  2

conf_matrix = np.array([[2604, 4, 0, 0],
                       [15, 2604, 0, 1],
                       [0, 0, 2616, 10],
                       [0, 0, 3, 2727]])

# Calculate total correctly classified instances
correctly_classified_total = np.sum(np.diag(conf_matrix))

# Calculate total misclassified instances
misclassified_total = np.sum(conf_matrix) - correctly_classified_total

# Print the results
print(f"Total Correctly Classified Instances on validation set of model 2: {correctly_classified_total}")
print(f"Total Misclassified Instances on validation set of model 2: {misclassified_total}")

# Data for validation set
labels_validation = ['Correctly Classified', 'Misclassified']
sizes_validation = [correctly_classified_total, misclassified_total]
colors_validation = ['lightgreen', 'lightcoral']  # Match the colors used in code1

# Plot pie chart for validation set
plt.figure(figsize=(8, 8))
plt.pie(sizes_validation, labels=labels_validation, colors=colors_validation, autopct='%1.1f%%', startangle=90, shadow=True)
plt.title('Validation Set - Correctly vs. Misclassified Instances')
plt.show()


# In[115]:


# Pie chart for analyzing correctly and misclassified classified on the Test Set of Model  2
conf_matrix = np.array([[2584, 6, 0, 0],
                       [17, 2604, 0, 0],
                       [0, 0, 2633, 16],
                       [0, 0, 5, 2719]])

# Calculate total correctly classified instances
correctly_classified_total = np.sum(np.diag(conf_matrix))

# Calculate total misclassified instances
misclassified_total = np.sum(conf_matrix) - correctly_classified_total

# Print the results
print(f"Total Correctly Classified Instances on test set of model 2: {correctly_classified_total}")
print(f"Total Misclassified Instances on test set of model 2: {misclassified_total}")

# Data for test set
labels_test = ['Correctly Classified', 'Misclassified']
sizes_test = [correctly_classified_total, misclassified_total]
colors_test = ['lightgreen', 'lightcoral']  # Match the colors used in code1

# Plot pie chart for test set
plt.figure(figsize=(8, 8))
plt.pie(sizes_test, labels=labels_test, colors=colors_test, autopct='%1.1f%%', startangle=90, shadow=True)
plt.title('Test Set - Correctly vs. Misclassified Instances')
plt.show()


# In[116]:


# Cross-Validation stability of Model 2

# Cross-validation scores with the best model
cv_scores_rf = cross_val_score(rf_best_model, X_train_scaled_2, y_train_2, scoring='f1_weighted', cv=StratifiedKFold(n_splits=5))
print("Cross-Validation Scores with  Best Model of Random Forest Classification:")
print(cv_scores_rf)
print(f"Average F1 Weighted Score: {cv_scores_rf.mean():.4f}")
print(f"Standard Deviation of F1 Weighted Scores: {cv_scores_rf.std():.4f}")

# Individual cross-validation scores
for i, score in enumerate(cv_scores_rf):
    print(f"Fold {i + 1}: {score:.4f}")

# Train the model on the entire training set
rf_best_model.fit(X_train_scaled_2, y_train_2)


# In[117]:


# Overfitting Analysis of Model 2

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("F1 Score")
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring='f1_weighted')
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

# Plot learning curves
plot_learning_curve(rf_best_model, "Model 2 Learning Curve - Random Forest", X_train_scaled_2, y_train_2, cv=StratifiedKFold(n_splits=5), n_jobs=-1)
plt.show()


# In[118]:


# Optimal PSP prediction for each exiting transaction

# Extract 'PSP' column for comparison
psp_column = df_2['PSP']

# Drop 'PSP' and 'success' columns from the features
X_predict = df_2.drop(['PSP', 'success'], axis=1)

# Define numeric features and apply StandardScaler
numeric_features = ['amount', '3D_secured', 'transaction_fee', 'day_of_week', 'minute_of_day', 'payment_attempts', 'success_probabilities']
numeric_transformer = StandardScaler()

# Define categorical features and apply OneHotEncoder
categorical_features = ['country', 'card']
categorical_transformer = OneHotEncoder()

# Create a ColumnTransformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Transform the features using the preprocessor
X_predict_transformed = preprocessor.fit_transform(X_predict)

# Predict the 'PSP' values using the best Random Forest model
df_2['Predicted_PSP'] = rf_best_model.predict(X_predict_transformed)

# Restore the original 'PSP' values for comparison
df_2['PSP'] = psp_column

# Print the DataFrame with predicted 'PSP' values
print(df_2)


# In[119]:


# Handle ties and choose the most cost-effective PSP

def choose_best_psp(row):
    if pd.Series(row['Predicted_PSP']).nunique() == 1:
        # No tie, return the predicted PSP
        return row['Predicted_PSP']
    else:
        # There is a tie, choose the most cost-effective PSP based on fee comparison
        min_fee_psp = row.loc[row['transaction_fee'].idxmin()]['Predicted_PSP']
        
        # Filter rows with tied predictions
        tied_rows = row[row['Predicted_PSP'].duplicated(keep=False)]
        
        # Check if the row with the minimum fee is part of the tied rows
        if min_fee_psp in tied_rows['Predicted_PSP'].values:
            return min_fee_psp
        else:
            # If not, choose the PSP with the lowest transaction fee
            return tied_rows.loc[tied_rows['transaction_fee'].idxmin()]['Predicted_PSP']

# Apply the function to each row in df_2
df_2['Best_PSP'] = df_2.apply(choose_best_psp, axis=1)

# Display the updated DataFrame with the 'Best_PSP' column
print(df_2)


# In[120]:


# Count occurrences of Predicted_PSP
predicted_psp_counts = df_2['Predicted_PSP'].value_counts()

# Count occurrences of Best_PSP
best_psp_counts = df_2['Best_PSP'].value_counts()

# Combine the counts into a DataFrame
matching_counts = pd.DataFrame({
    'Predicted_PSP_Count': predicted_psp_counts,
    'Best_PSP_Count': best_psp_counts
})

# Display the counts
print(matching_counts)


# In[ ]:




