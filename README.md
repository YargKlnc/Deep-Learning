# Neural Networks & Deep-Learning 
by YK, UofT

![image](https://github.com/YargKlnc/Deep-Learning/assets/142269763/7da93479-e78b-4f7d-b990-492cebfa6023)


**Background**

The nonprofit foundation Alphabet Soup wants a tool that can help it select the applicants for funding with the best chance of success in their ventures. With your knowledge of machine learning and neural networks, you’ll use the features in the provided dataset to create a binary classifier that can predict whether applicants will be successful if funded by Alphabet Soup. Received a CSV file from Alphabet Soup’s business team, containing more than 34.000 organizations that have received funding from Alphabet Soup over the years. Within this dataset are a number of columns that capture metadata about each organization, such as:

•	EIN and NAME—Identification columns •	APPLICATION_TYPE—Alphabet Soup application type •	AFFILIATION—Affiliated sector of industry •	CLASSIFICATION—Government organization classification •	USE_CASE—Use case for funding •	ORGANIZATION—Organization type •	STATUS—Active status •	INCOME_AMT—Income classification •	SPECIAL_CONSIDERATIONS—Special considerations for application •	ASK_AMT—Funding amount requested •	IS_SUCCESSFUL—Was the money used effectively


**Instructions**


**Step 1: Preprocessing the Data**

Starter file was uploaded to Google Colab. Using the information provided in the Challenge files, preprocessing steps were completed. Pandas DataFrame was created by reading in the charity_data.csv file. The target variable(s) and feature variable(s) were identified. EIN and NAME columns were dropped. Number of unique values for each column was determined. For columns with more than 10 unique values, the number of data points for each unique value was determined. "Rare" categorical variables were binned together into a new value, Other, using pd.get_dummies() to encode categorical variables. The preprocessed data was split into a features array, X, and a target array, y, using train_test_split. Training and testing features datasets were scaled using StandardScaler.


**Step 2: Compile, Train, and Evaluating the Model**

Preprocessing steps were completed in the Google Colab file. A neural network model was created using TensorFlow and Keras, specifying the number of input features and nodes for each layer. The first hidden layer with an appropriate activation function was created. If necessary, a second hidden layer with an appropriate activation function was added. An output layer with an appropriate activation function was created. The structure of the model was checked. The model was compiled and trained. A callback was created to save the model's weights every five epochs. The model was evaluated using the test data to determine the loss and accuracy. Results were saved and exported to an HDF5 file named AlphabetSoupCharity.h5.


**Step 3: Optimizing the Model**

A new Google Colab file named AlphabetSoupCharity_Optimization.ipynb was created. Dependencies were imported, and charity_data.csv was read into a Pandas DataFrame. The dataset was preprocessed, considering modifications made during model optimization. A neural network model was designed, adjusting for modifications to achieve a target predictive accuracy higher than 75%. Results were saved and exported to an HDF5 file named AlphabetSoupCharity_Optimization.h5.


**Step 4: Reporting on the Neural Network Model**

**Overview:** The objective of this analysis was to develop a deep learning model for Alphabet Soup, predicting the success of funded organizations based on provided features.

**Results:**

*Data Preprocessing:*

*Target Variable(s):* In the provided code snippet, the target variable is denoted by `numeric_application_df.IS_SUCCESSFUL`. The inclusion of this variable suggests that the model aims to predict whether a certain event or condition is successful, and this information is crucial for the training process.

*Feature Variable(s):* It is implied that the feature variables are represented by the columns in the DataFrame `numeric_application_df` excluding the "IS_SUCCESSFUL" column. These features serve as the independent variables used as inputs to the model for predicting the target variable.

*In summary;* the target variable is `IS_SUCCESSFUL`, and the feature variables are the remaining columns in the DataFrame `numeric_application_df`. The code snippet performs the necessary steps to split the data into features (X) and the target (y) arrays, followed by splitting them into training and testing datasets for model training and evaluation.

*Variables Removed in Attempt 1:* In the preprocessing stage of **Attempt 1**, the EIN (Employer Identification Number) and NAME columns were eliminated from the dataset. The exclusion of these variables indicates that they were not considered relevant or contributory to the model's predictive performance for the specific task at hand. The decision to remove variables is often made based on factors such as redundancy, lack of information, or other considerations that might impact the model negatively.

*Variables Removed in Attempt 1:* In the preprocessing stage of **Attempt 2**, the EIN (Employer Identification Number) column was dropped from the dataset. The decision to exclude this variable suggests that it was deemed non-essential or non-contributory to the model's predictive performance for the specific task at hand. Removal of variables is typically based on factors such as redundancy, lack of information, or other considerations that might adversely affect the model. This strategic pruning aims to streamline the dataset and enhance the model's ability to discern meaningful patterns during training.

*Compiling, Training, and Evaluating the Model:*

**Architecture and Training Details (Attempt 1):**

Cutoff Value Application: 500
CutOff Value Classification: 300
Layer 1 Hidden Nodes: 10
Layer 1 Activation: RELU (Rectified Linear Unit)
Layer 2 Hidden Nodes: 5
Layer 2 Activation: RELU
Output Layer Activation: SIGMOID
Compiling: Loss=binary_crossentropy, optimizer=adam, metrics=accuracy
Training: The model was trained for 50 epochs, indicating that it went through 50 cycles of adjusting its weights based on the training data to improve its predictive performance.

![image](https://github.com/YargKlnc/Deep-Learning/assets/142269763/2f78e526-88ff-4304-a223-1d4c67532e33)

![image](https://github.com/YargKlnc/Deep-Learning/assets/142269763/69c305cb-5d87-4dcd-a670-e270b27c76d3)

These details provide insight into the architecture and training parameters used in Attempt 1, offering a glimpse into how the model was structured and trained for the specified binary classification task.

**Results Attempt 1:**
**Loss: 0.5524, Accuracy: 0.7278**

**Architecture and Training Details (Attempt 2):**

Cutoff Value Application: 50
CutOff Value Classification: 120
Layer 1 Hidden Nodes: 8
Layer 1 Activation: RELU (Rectified Linear Unit)
Layer 2 Hidden Nodes: 10
Layer 2 Activation: RELU
Layer 3 Hidden Nodes: 12
Layer 3 Activation: RELU
Output Layer Activation: SIGMOID
Compiling: Loss=binary_crossentropy, optimizer=adam, metrics=accuracy
Training: The model underwent training for 100 epochs, indicating that it iteratively adjusted its weights over 100 cycles based on the training data to enhance its predictive performance.

![image](https://github.com/YargKlnc/Deep-Learning/assets/142269763/71ad8a64-7630-44d6-9a67-601be127a4d4)

![image](https://github.com/YargKlnc/Deep-Learning/assets/142269763/22af7068-2b68-4f93-9dc6-94921b40d41d)

**Results Attempt 2:**
**Loss: 0.4832, Accuracy: 0.7618**


**Summary:**

The deep learning model results highlight the ongoing need for optimization efforts. **Attempt 1** yielded an accuracy of **72.58%**, and **Attempt 2** showed a notable improvement with an accuracy of **76.18%**, surpassing the targeted **75%**. In **Attempt 2**, the neural network architecture was refined with the inclusion of three hidden layers comprising **8**, **10**, and **12** nodes, respectively, each activated by the **RELU** (Rectified Linear Unit) activation function. This architectural adjustment, coupled with the utilization of the **SIGMOID** activation function in the output layer, contributed to the observed improvement in accuracy. The **SIGMOID** activation function, employed in the output layer, is well-suited for binary classification tasks, providing a probability-like output that aids in distinguishing between classes. The model was trained over **30 epochs**, allowing it to iteratively adjust its weights to better capture the underlying patterns in the training data. As the model's performance is sensitive to data quality, there is a recommendation to further explore enhancements in data preprocessing and feature engineering to potentially boost accuracy. Additionally, extending the training epochs and experimenting with alternative activation functions may further contribute to model refinement. These results emphasize the iterative nature of model development, prompting a comprehensive exploration of alternative models and fine-tuning strategies in future iterations to consistently meet or exceed **Alphabet Soup's accuracy objectives**.

 
**References**

Head photo rights: Shutterstock


