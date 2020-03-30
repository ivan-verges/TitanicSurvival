import codecademylib3_seaborn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#Load the passenger data into a dataframe
passengers = pd.read_csv("passengers.csv")

#Update the sex column to hold numerical values
passengers["Sex"] = passengers["Sex"].map({"female" : 1, "male" : 0})

#Fill the nan values in the age column with the mean of all ages
avg = 0
for i in range(len(passengers["Age"])):
  if(np.isnan(passengers["Age"].values[i]) == False):
    avg += passengers["Age"][i]
avg /= len(passengers["Age"])
avg = round(avg, 0)
passengers["Age"].fillna(value = avg, inplace = True)

#Create a class column for First Class Passengers
passengers["FirstClass"] = passengers["Pclass"].apply(lambda x : 1 if x == 1 else 0)

#Create a class column for Second Class Passengers
passengers["SecondClass"] = passengers["Pclass"].apply(lambda x : 1 if x == 2 else 0)

#Create a class column for Third Class Passengers
passengers["ThirdClass"] = passengers["Pclass"].apply(lambda x : 1 if x == 3 else 0)

#Select the desired features to traing the model
features = passengers[["Sex", "Age", "FirstClass", "SecondClass", "ThirdClass"]]
survival = passengers["Survived"]

#Perform train and test split by 80% Train and 20% Test
x_train, x_test, y_train, y_test = train_test_split(features, survival, test_size = 0.2)

#Scale the feature data so it has mean = 0 and standard deviation = 1
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#Create and train the model
model = LogisticRegression()
model.fit(x_train, y_train)

#Prints the Score of the model on the train data
print(model.score(x_train, y_train))

#Score the Score of the model on the test data
print(model.score(x_test, y_test))

#Print the coefficients to Analyze
print(list(zip(["Sex", "Age", "FirstClass", "SecondClass", "ThirdClass"], model.coef_[0])))

#Creates some Sample passenger features
Jack = np.array([0.0, 20.0, 0.0, 0.0, 1.0])
Rose = np.array([1.0, 17.0, 1.0, 0.0, 0.0])
Jose = np.array([0.0, 35.0, 0.0, 0.0, 1.0])
Mary = np.array([1.0, 40.0, 0.0, 1.0, 0.0])

#Combine created passenger arrays
sample_passengers = np.array([Jack, Rose, Jose, Mary])

#Scale the sample passenger features
sample_passengers = scaler.transform(sample_passengers)

#Print the sample passengers
print(sample_passengers)

#Makes survival predictions and their probabilities
print(model.predict(sample_passengers))
print(model.predict_proba(sample_passengers))