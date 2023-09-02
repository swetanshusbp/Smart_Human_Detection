# Smart-human-detection-
A ml based project that can have certian Iot based Implementations 
 The code attempts to classify human activities (e.g., walking, standing, laying) based on sensor readings. Let's break down what the notebook does:

Setting up Google Colab: It appears the data is stored on Google Drive. The user mounts their drive to access the dataset.

Importing necessary libraries: A range of libraries from data manipulation (pandas, numpy) to machine learning (sklearn) are imported.

Data Import: Paths are set up for several files (train and test datasets among others). The dataset seems to be from the UCI Machine Learning repository, particularly the HAR dataset.

Data Cleaning: The dataset is checked for duplicates and NaN/null values.

Data Analysis:

Some meta-information about the data is shown using .info().
Renaming of feature columns, removing any symbols like () or -.
Descriptive statistics for certain activities (like 'STANDING').
Visualization of data distribution among subjects and activities using seaborn.
Exploratory Data Analysis (EDA):

Visualization of data for different activities.
Analysis of static (sitting, standing, laying) vs dynamic (walking, walking upstairs, walking downstairs) activities.
Analysis of the magnitude of acceleration to differentiate activities.
Analysis of gravity's angle components for activity differentiation.
Data Preprocessing:

![image](https://github.com/SAMUDRABAN/Smart-human-detection-/assets/97033991/e9a39456-869a-4603-84b3-dc4e8cd5a250)
![image](https://github.com/SAMUDRABAN/Smart-human-detection-/assets/97033991/61ec0056-dd64-4d41-aab4-48a59161108f)

Drops unnecessary columns.
Splits the data into features (X_train, X_test) and target variables (y_train, y_test).
Utility Function for Visualization:
![image](https://github.com/SAMUDRABAN/Smart-human-detection-/assets/97033991/e9c5d6fb-5cce-477a-82f2-bf0db68082c3)

A function plot_confusion_matrix is defined. This will likely be used later in the notebook (after the provided code) to visualize the performance of machine learning models in terms of correctly predicted activities.
