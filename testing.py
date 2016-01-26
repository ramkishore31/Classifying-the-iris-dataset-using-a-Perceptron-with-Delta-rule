import pandas as pd
import copy
import numpy as np
import pickle

#Reading the dataset into a dataframe
test_data = pd.read_csv('iris_test.data')
#Adding column names to the dataframe
test_data.columns = ['sepal_length','sepal_width','petal_length','petal_width','class_label']
#Initializing the weight vector
weight_vector = [0] * 5
#Initializing the learning rate
learning_rate = 1
prev_error = 1000000000

f = open('weights','r')
weight_vector = pickle.load(f)


#Assigning categorical values for class labels
for i in range(0,len(test_data['class_label'])):
    if test_data['class_label'][i] == 'Iris-setosa':
        test_data['class_label'][i] = 0
    else:
        test_data['class_label'][i] = 1


test_label = test_data['class_label']
del test_data['class_label']

#Iterate over each feature of the dataframe and replace its individual values by its z-score.
for feature in test_data:
    test_data[feature] = (test_data[feature] - test_data[feature].mean()) / test_data[feature].std()


#Adding the bias feature to dataframe
test_data['bias'] = [1] * len(test_data)

predicted_output = []
error = 0
#Finding the training output by taking dot product of input and weight vector
for index,row in test_data.iterrows():
    predicted_output.append(sum([a*b for a,b in zip(row,weight_vector)]))

#Update the values of training output based on whether they are positive or negative
for i in range(0,len(predicted_output)):
    if predicted_output[i] > 0:
        predicted_output[i] = 1
    else:
        predicted_output[i] = 0

error = 0
#Find the number of misclassifications
for i in range(0,len(predicted_output)):
    if predicted_output[i] != test_label[i]:
        error += 1

print error

