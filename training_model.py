import pandas as pd
import copy
import numpy as np
import pickle

#Reading the dataset into a dataframe
train_data = pd.read_csv('iris_train.data')
#Adding column names to the dataframe
train_data.columns = ['sepal_length','sepal_width','petal_length','petal_width','class_label']
#Initializing the weight vector
weight_vector = [0] * 5
#Initializing the learning rate
learning_rate = 1
prev_error = 1000000000


#Assigning categorical values for class labels
for i in range(0,len(train_data['class_label'])):
    if train_data['class_label'][i] == 'Iris-setosa':
        train_data['class_label'][i] = 0
    else:
        train_data['class_label'][i] = 1


train_label = train_data['class_label']
del train_data['class_label']

#Iterate over each feature of the dataframe and replace its individual values by its z-score.
for feature in train_data:
    train_data[feature] = (train_data[feature] - train_data[feature].mean()) / train_data[feature].std()


#Adding the bias feature to dataframe
train_data['bias'] = [1] * len(train_data)



predicted_output = []
error = 0
while(True):
    del predicted_output[:]
    #Finding the training output by taking dot product of input and weight vector
    for index,row in train_data.iterrows():
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
        if predicted_output[i] != train_label[i]:
            error += 1

    print error
    if prev_error <= error:
        break
    else:
        prev_error = error

    #Applying the delta rule to update the weight vector
    for i in range(0,len(predicted_output)):
        weight_vector = np.array(weight_vector) + np.array((train_label[i] - predicted_output[i]) * learning_rate * np.array(train_data.loc[i,:]))

f = open('weights','wb')
pickle.dump(weight_vector, f)
f.close()

print weight_vector