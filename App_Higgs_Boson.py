
import numpy as np
from keras.utils import np_utils
import pandas as pd
import streamlit as st
import warnings
warnings.filterwarnings("ignore") 

from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest
from collections import Counter

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM,Dropout,SimpleRNN

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import matplotlib.pyplot as plt
import seaborn as sb


# In[12]:


# if the user chooses to upload the data
file = st.file_uploader('Dataset')
class DL_models:

    def __init__(self,X_train,y_train,X_test,y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test


    def simple_ANN(self):
        st.subheader("SIMPLE ARTIFICIAL NEURAL NETWORK")
        model = Sequential()
        model.add(Dense(units = 128, activation = 'relu'))
        model.add(Dropout(0.2))
        model.add(Dense(units = 64))
        model.add(Dropout(0.2))
        model.add(Dense(units = 1))
        model.compile(optimizer = 'adam', loss = 'mean_squared_error',metrics = ["acc"])
        model.fit(self.X_train,self.y_train, batch_size = 32, epochs = 10)
        y_pred = model.predict(self.X_test)
        y_pred = [0 if y_pred[i] <=0.5 else 1 for i in range(len(y_pred))]
        st.write("ACCURACY : ",accuracy_score(self.y_test, y_pred))
        fig=plt.plot(self.y_test, y_pred)
        st.write(fig)
        
        st.write("CONFUSION MATRIX : ") 
        st.write(confusion_matrix(self.y_test, y_pred))
        st.write("CLASSIFICATION REPORT : ")
        st.write(classification_report(self.y_test, y_pred, target_names = ["b","s"], output_dict=True))



    def RNN(self):
        st.subheader("RECURRENT NEURAL NETWORK")
        Xtrain = np.reshape(self.X_train, (self.X_train.shape[0], self.X_train.shape[1], 1))
        # Reshape the data
        Xtest = np.reshape(self.X_test, (self.X_test.shape[0], self.X_test.shape[1], 1 ))
        model = Sequential()
        model.add(SimpleRNN(units=4, input_shape=(Xtrain.shape[1], Xtrain.shape[2])))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam',metrics = ["acc"])
        model.fit(Xtrain, self.y_train, epochs=10, batch_size=32)
        # predicting the opening prices
        prediction = model.predict(Xtest)
        predictions = [0 if prediction[i] <=0.5 else 1 for i in range(len(prediction))]
        st.write("ACCURACY : ",accuracy_score(self.y_test, predictions))
        fig=plt.plot(self.y_test, y_pred)
        st.write(fig)
        st.write("CONFUSION MATRIX : ")
        st.write(confusion_matrix(self.y_test, predictions))
        st.write("CLASSIFICATION REPORT : ")
        st.write(classification_report(self.y_test, predictions, target_names = ["b","s"], output_dict=True))
        
    def dl_LSTM(self):
        st.subheader("LSTM(Long Short Term Memory)")
        X_train = self.X_train.reshape(self.X_train.shape+(1,))
        X_test = self.X_test.reshape(self.X_test.shape+(1,))
        
        model = Sequential()
        model.add(LSTM(units = 50,dropout = 0.2, return_sequences = True, input_shape = (self.X_train.shape[1], 1), activation = 'tanh'))
        model.add(LSTM(units = 50, activation = 'tanh'))
        model.add(Dense(units = 2,activation='softmax'))
        model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
        model.fit(X_train,self.y_train, batch_size = 32, epochs = 10)
        
        y_pred = model.predict(X_test)
        y_pred = [0 if y_pred[i] <=0.5 else 1 for i in range(len(y_pred))]
        
        st.write("ACCURACY : ",accuracy_score(self.y_test, y_pred))
        fig=plt.plot(self.y_test, y_pred)
        st.write(fig)
        
        st.write("CONFUSION MATRIX : ") 
        st.write(confusion_matrix(self.y_test, y_pred))
        
        st.write("CLASSIFICATION REPORT : ")
        st.write(classification_report(self.y_test, y_pred, target_names = ["b","s"], output_dict=True))
        

if file is not None:

    dataset = pd.read_csv(file)
    # flag is set to true as data has been successfully read
    flag = "True"
    st.header('**HIGGS BOSON DATA**')
    st.write(dataset.head())
    
    dataset.index=dataset.EventId
    dataset=dataset.drop(columns=['EventId'])

    st.write("DATA PRE-PROCESSING")
    st.subheader("Correlation plot of the features")
    corr = dataset.corr()#to find the pairwise correlation of all columns in the dataframe
    fig, ax = plt.subplots()
    sb.heatmap(corr,cmap="Blues", ax=ax) #Plot rectangular data as a color-encoded matrix.
    st.write(fig)
    
    
    st.subheader("Labels distribution")
    st.bar_chart(dataset["Label"].value_counts())
    
    st.subheader("Finding no. of null values per column in the dataset")
    st.write(dataset.isna().sum())
    
    st.subheader("Statistical information about the datset")
    st.write(dataset.describe())
    
    data = dataset.iloc[:,:-1] # Extracting features
    imp_mean = SimpleImputer(missing_values = -999.0, strategy='mean') 
    # the placeholder for the missing values. All occurrences of missing_values will be imputed.
    #If “mean”, then replace missing values using the mean along each column. Can only be used with numeric data.
    
    X = imp_mean.fit_transform(data)
    y = dataset.iloc[:,-1].values # extracting the labels
    train_label = y.tolist()
    class_names = list(set(train_label))
    y=np_utils.to_categorical(y)
    
    st.success("Data cleaned!")
    st.subheader('Test size split of users choice:')
    st.text('Default is set to 20%')
    
    k = st.number_input('',step = 5,min_value=10, value = 20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = k * 0.01, random_state = 0)
    st.write("Data is being split into testing and training data!")
    # Splitting the data into 20% test and 80% training data
    # Outlier detection and removal using Isolation Forest
    iso = IsolationForest(contamination='auto')
    train_hat = iso.fit_predict(X_train)
    test_hat = iso.fit_predict(X_test)
    st.success("Data split successfuly")
    st.write("No of Training data outliers :",Counter(train_hat)[-1],"out of ",X_train.shape[0],"data points")
    st.write("No of Testing data outliers :",Counter(test_hat)[-1],"out of ",X_test.shape[0],"data points")
    # select all rows that are not outliers
    mask_train = train_hat != -1
    mask_test = test_hat != -1
    X_train, y_train = X_train[mask_train, :], y_train[mask_train]
    X_test, y_test = X_test[mask_test, :], y_test[mask_test]
    st.success("Outliers removed successfully!")
    std_sc = StandardScaler()
    X_train = std_sc.fit_transform(X_train) # Scaling the training data
    X_test = std_sc.transform(X_test) # Scaling the testing data

    dl = DL_models(X_train,y_train,X_test,y_test)
    st.subheader('Choose the Deep Learning model :')
    mopt = st.multiselect("Select :",["Simple ANN","RNN","LSTM","All"])
    # "Click to select",
    if(st.button("START TRAINING AND TESTING THE MODEL(S) SELECTED")):

        if "Basic ANN" in mopt:
            dl.simple_ANN()
            
        if "RNN" in mopt:
            dl.RNN()
            
        if "LSTM" in mopt:
            dl.dl_LSTM()

        if "All" in mopt:

            dl.basic_ANN()
            dl.RNN()
            dl.dl_LSTM()
            

    if(st.button("FINISH")):
        st.info("THANK YOU FOR YOUR PATIENCE. WE ARE DONE. HOPE YOU ARE SATISFIED")
        st.balloons()

else:
    st.warning("No file has been chosen yet")






