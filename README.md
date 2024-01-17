# Deep Learning
by YK, UofT

![image](https://github.com/YargKlnc/Deep-Learning/assets/142269763/7da93479-e78b-4f7d-b990-492cebfa6023)


**Background**

The nonprofit foundation Alphabet Soup wants a tool that can help it select the applicants for funding with the best chance of success in their ventures. With your knowledge of machine learning and neural networks, you’ll use the features in the provided dataset to create a binary classifier that can predict whether applicants will be successful if funded by Alphabet Soup.

Received a CSV file from Alphabet Soup’s business team, containing more than 34.000 organizations that have received funding from Alphabet Soup over the years. Within this dataset are a number of columns that capture metadata about each organization, such as:

•	EIN and NAME—Identification columns

•	APPLICATION_TYPE—Alphabet Soup application type

•	AFFILIATION—Affiliated sector of industry

•	CLASSIFICATION—Government organization classification

•	USE_CASE—Use case for funding

•	ORGANIZATION—Organization type

•	STATUS—Active status

•	INCOME_AMT—Income classification

•	SPECIAL_CONSIDERATIONS—Special considerations for application

•	ASK_AMT—Funding amount requested

•	IS_SUCCESSFUL—Was the money used effectively

**Instructions**

**Step 1: Preprocessing the Data**

1. Starter file was uploaded to Google Colab. Using the information provided in the Challenge files, preprocessing steps were completed.

2. Pandas DataFrame was created by reading in the charity_data.csv file. The target variable(s) and feature variable(s) were identified.

3. EIN and NAME columns were dropped.

4. Number of unique values for each column was determined.

5. For columns with more than 10 unique values, the number of data points for each unique value was determined.

6. "Rare" categorical variables were binned together into a new value, Other, using pd.get_dummies() to encode categorical variables.

7. The preprocessed data was split into a features array, X, and a target array, y, using train_test_split.

8. Training and testing features datasets were scaled using StandardScaler.

**Step 2: Compile, Train, and Evaluating the Model**

1. Preprocessing steps were completed in the Google Colab file.

2. A neural network model was created using TensorFlow and Keras, specifying the number of input features and nodes for each layer.

3. The first hidden layer with an appropriate activation function was created.

4. If necessary, a second hidden layer with an appropriate activation function was added.

5. An output layer with an appropriate activation function was created.

6. The structure of the model was checked.

7. The model was compiled and trained.

8. A callback was created to save the model's weights every five epochs.

9. The model was evaluated using the test data to determine the loss and accuracy.

10. Results were saved and exported to an HDF5 file named AlphabetSoupCharity.h5.

**Step 3: Optimizing the Model**

1. A new Google Colab file named AlphabetSoupCharity_Optimization.ipynb was created.

2. Dependencies were imported, and charity_data.csv was read into a Pandas DataFrame.

3. The dataset was preprocessed, considering modifications made during model optimization.

4. A neural network model was designed, adjusting for modifications to achieve a target predictive accuracy higher than 75%.

5. Results were saved and exported to an HDF5 file named AlphabetSoupCharity_Optimization.h5.

Certainly! Here's the content in plain text:

**Step 4: Reporting on the Neural Network Model**

*Overview:*

The objective of this analysis was to develop a deep learning model for Alphabet Soup, predicting the success of funded organizations based on provided features.

*Results:*

Data Preprocessing:

- Target Variable(s): Identified during preprocessing.
- Feature Variable(s): Identified during preprocessing.
- Variables Removed: EIN and NAME columns.

Model Attempts:

Attempt 1:
- Cutoff Value Application: 500
- CutOff Value Classification: 300
- Layer 1 Hidden Nodes: 10
- Layer 1 Activation: RELU
- Layer 2 Hidden Nodes: 5
- Layer 2 Activation: RELU
- Output Layer Activation: SIGMOID
- Compiling: Loss=binary_crossentropy, optimizer=adam, metrics=accuracy
- Training: 50 epochs

**Results:**
**Loss: 0.5524, Accuracy: 0.7265**

Attempt 2:
- Cutoff Value Application: 500
- CutOff Value Classification: 300
- Layer 1 Hidden Nodes: 10
- Layer 1 Activation: RELU
- Layer 2 Hidden Nodes: 10
- Layer 2 Activation: RELU
- Layer 3 Hidden Nodes: 10
- Layer 3 Activation: TANH
- Output Layer Activation: SIGMOID
- Compiling: Loss=binary_crossentropy, optimizer=adam, metrics=accuracy
- Training: 100 epochs

**Results:**
**Loss: 0.5516, Accuracy: 0.7249**

Model Performance:

- Steps to Increase Model Performance: Detailed in the summary report.

**Summary:**

The deep learning model results indicate that further optimization is crucial, as Attempt 1 achieved an accuracy of 72.65%, and Attempt 2 achieved 72.49%, both falling below the minimum acceptable accuracy of 90%. Given the sensitivity of model performance to data quality, exploring enhancements in data preprocessing and feature engineering is recommended. To improve model accuracy, adjustments to the neural network architecture, experimentation with activation functions, and an increase in training epochs are also suggested. These findings underscore the iterative nature of model development, prompting a thorough exploration of alternative models and fine-tuning strategies in future iterations to meet the minimum accuracy threshold and better align with Alphabet Soup's objectives.

  
**References**

Head photo rights: Shutterstock


