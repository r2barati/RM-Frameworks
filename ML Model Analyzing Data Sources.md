# ML Models Analyzing Data Sources

## Sample 1: Analyzing customer churn using machine learning
```python
# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv('customer_churn.csv')

# Prepare the data
X = df.drop('Churn', axis=1)  # Features
y = df['Churn']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
```

In this sample case, we assume that the dataset is stored in a CSV file called 'customer_churn.csv'. The dataset contains features related to customer information and a target variable 'Churn', indicating whether a customer has churned (1) or not (0).

The code performs the following steps:

1. Imports the necessary libraries, including pandas for data manipulation, scikit-learn for machine learning algorithms, and RandomForestClassifier for the classification model.

2. Loads the dataset using the `read_csv()` function from pandas.

3. Prepares the data by separating the features (X) and the target variable (y).

4. Splits the data into training and testing sets using the `train_test_split()` function from scikit-learn.

5. Creates an instance of the Random Forest classifier with 100 decision trees.

6. Trains the classifier on the training data using the `fit()` method.

7. Makes predictions on the test set using the `predict()` method.

8. Evaluates the model's accuracy by comparing the predicted labels with the true labels using the `accuracy_score()` function.

9. Prints the accuracy score as the evaluation result.

## Sample 2: Sentiment analysis using Natural Language Processing (NLP) with the TextBlob library

```python
# Import necessary libraries
from textblob import TextBlob

# Define a sample text
text = "I absolutely loved the movie! The acting was superb and the plot kept me engaged throughout."

# Perform sentiment analysis using TextBlob
blob = TextBlob(text)
sentiment = blob.sentiment

# Print the sentiment polarity and subjectivity
print("Sentiment Polarity:", sentiment.polarity)
print("Sentiment Subjectivity:", sentiment.subjectivity)
```

In this sample case, we are using the TextBlob library to perform sentiment analysis on a sample text. The code performs the following steps:

1. Imports the necessary library, TextBlob, for performing sentiment analysis.

2. Defines a sample text for sentiment analysis. Feel free to modify the text as needed for your case.

3. Creates a TextBlob object by passing the text to the TextBlob() function.

4. Uses the `sentiment` property of the TextBlob object to retrieve the sentiment analysis results, including polarity (ranging from -1 to 1, indicating negative to positive sentiment) and subjectivity (ranging from 0 to 1, indicating objective to subjective sentiment).

5. Prints the sentiment polarity and subjectivity using the `polarity` and `subjectivity` attributes of the sentiment object.

Make sure to install the TextBlob library (`pip install textblob`) and any necessary dependencies before running the code.

## Sample 3: Customer segmentation using K-means clustering with the scikit-learn library

```python
# Import necessary libraries
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv('customer_data.csv')

# Select relevant features for clustering
X = df[['Age', 'Income']]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform K-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)

# Add the cluster labels to the dataset
df['Cluster'] = kmeans.labels_

# Display the results
print(df.head())
```

In this sample case, we assume that the dataset is stored in a CSV file called 'customer_data.csv'. The dataset contains customer information such as age, income, and other relevant attributes.

The code performs the following steps:

1. Imports the necessary libraries, including pandas for data manipulation, scikit-learn for machine learning algorithms, KMeans for clustering, and StandardScaler for feature scaling.

2. Loads the dataset using the `read_csv()` function from pandas.

3. Selects the relevant features for clustering. In this case, we consider 'Age' and 'Income' as the features for segmentation. Modify this step to select the appropriate features for your case.

4. Standardizes the features using the `StandardScaler` to ensure they have a mean of 0 and standard deviation of 1, which helps in clustering.

5. Performs K-means clustering with `n_clusters` set to 3, indicating the desired number of clusters. Adjust the number of clusters as per your requirements.

6. Fits the K-means model to the standardized data using the `fit()` method.

7. Adds the cluster labels to the original dataset by creating a new column 'Cluster' and assigning the cluster labels from the K-means model.

8. Displays the first few rows of the dataset with the added cluster labels using the `head()` function.

# Sample 4: Anomaly detection using the Isolation Forest algorithm from the scikit-learn library

```python
# Import necessary libraries
import pandas as pd
from sklearn.ensemble import IsolationForest

# Load the dataset
df = pd.read_csv('transaction_data.csv')

# Select relevant features for anomaly detection
X = df[['Amount', 'Time']]

# Perform anomaly detection using Isolation Forest
clf = IsolationForest(contamination=0.05, random_state=42)
y_pred = clf.fit_predict(X)

# Add the anomaly labels to the dataset
df['Anomaly'] = y_pred

# Display the results
print(df.head())
```

In this sample case, we assume that the dataset is stored in a CSV file called 'transaction_data.csv'. The dataset contains transaction information, including features such as 'Amount' and 'Time'.

The code performs the following steps:

1. Imports the necessary libraries, including pandas for data manipulation and scikit-learn for machine learning algorithms. Specifically, we use the Isolation Forest algorithm for anomaly detection.

2. Loads the dataset using the `read_csv()` function from pandas.

3. Selects the relevant features for anomaly detection. In this case, we consider 'Amount' and 'Time'. Modify this step to select the appropriate features for your case.

4. Performs anomaly detection using the Isolation Forest algorithm. We set the `contamination` parameter to 0.05, indicating that we expect approximately 5% of the data points to be anomalies. Adjust this value based on your requirements.

5. Generates the anomaly predictions using the `fit_predict()` method, which assigns a label of -1 to anomalies and 1 to normal data points.

6. Adds the anomaly labels to the original dataset by creating a new column 'Anomaly' and assigning the predictions from the Isolation Forest model.

7. Displays the first few rows of the dataset with the added anomaly labels using the `head()` function.
