# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prettytable import PrettyTable

# Extract the fraud data from the CSV file and create a dataframe
data = pd.read_csv('creditcard.csv')

# Print the shape of the data
#data = data.sample(frac=0.1, random_state = 1)
print(data.shape)

# print(data["Amount"].describe())


# # Determine number of fraud cases in dataset
# fraud = data[data['Class'] == 1]
# valid = data[data['Class'] == 0]
# outlier_fraction = len(fraud) / float(len(valid))
# print(outlier_fraction)
# print('Fraud Cases: {}'.format(len(fraud)))
# print('Valid Cases: {}'.format(len(valid)))

# # create plotted table of fraud and valid cases
# table = PrettyTable(["Fraud Cases", "Valid Cases"])
# table.add_row([len(fraud), len(valid)])
# print(table)

# # Check for Null values
# print(data.isnull().sum())

# # Plot histograms of each parameter using seaborn
# sns.pairplot(data=data, hue='Class', vars=['Amount', 'Time'])

# fig = plt.figure(figsize = (12, 9))
# plt.show()


# # Creatre a correlation matrix
# plt.subplots(figsize=(25,15)) # set size of plot
# corrmat = data.corr().round(2) # create correlation matrix (rounded to 2 decimals)
# mask = np.triu(np.ones_like(corrmat, dtype=bool)) # create a mask to hide the upper triangle of the matrix
# sns.heatmap(corrmat, annot= True, square = True, cmap="YlGnBu", mask=mask, annot_kws={"fontsize":8}) # plot heatmap


# Create the test and train splits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Drop the target class
X = data.drop(['Class'], axis=1)
y = data['Class']

# Create the training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Create the scaler
scaler = StandardScaler()

# Fit the scaler to the training data and transform
X_train = scaler.fit_transform(X_train)

# Transform the test data
X_test = scaler.transform(X_test)


# Create the SGD Classifier model
from sklearn.linear_model import SGDClassifier
model = SGDClassifier(loss='log_loss')

# Fit the model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Reshape the prediction values to 0 for valid, 1 for fraud
y_pred[y_pred == 1] = 0
y_pred[y_pred == -1] = 1

# Calculate the number of errors
errors = (y_pred != y_test).sum()   

# Print the accuracy score
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))

# Print the classification report and confusion matrix
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Plot the confusion matrix
LABELS = ['Valid', 'Fraud']
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12, 12))
sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d", annot_kws={"fontsize":20})
plt.title("Confusion matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()

