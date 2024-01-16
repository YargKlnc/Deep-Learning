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

**Step 4: Reporting on the Neural Network Model**

1. **Overview of the analysis:** The purpose of this analysis was to create a deep learning model for Alphabet Soup to predict the success of funded organizations.

2. **Results:**
   - **Data Preprocessing**
      - Target variable(s): Identified during preprocessing.
      - Feature variable(s): Identified during preprocessing.
      - Variables removed: EIN and NAME columns.

   - **Compiling, Training, and Evaluating the Model**
      - Neurons, layers, and activation functions: Selected based on design considerations.
      - Target model performance: Achieved/Not Achieved.
      - Steps to increase model performance: Detailed in the report.

3. **Summary:** Overall results of the deep learning model were summarized. A recommendation for a different model to solve the classification problem was provided, along with an explanation.

  
**References**

Head photo rights: Shutterstock


