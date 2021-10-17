
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
from tensorflow.keras.layers import Conv1D,MaxPooling1D,BatchNormalization

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import matplotlib.pyplot as plt
import seaborn as sns

#Upload Data
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
        st.write("ACCURACY : ",accuracy_score(self.y_test, y_pred)*100,"%")
        plt.plot(self.y_test, y_pred)
        st.write(plt.show())
        
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
        y_pred = model.predict(Xtest)
        y_pred = [0 if y_pred[i] <=0.5 else 1 for i in range(len(y_pred))]
        st.write("ACCURACY : ",accuracy_score(self.y_test, predictions)*100,"%")
        plt.plot(self.y_test, y_pred)
        st.write(plt.show())
        st.write("CONFUSION MATRIX : ")
        st.write(confusion_matrix(self.y_test, y_pred))
        st.write("CLASSIFICATION REPORT : ")
        st.write(classification_report(self.y_test, y_pred, target_names = ["b","s"], output_dict=True))
        
    def dl_LSTM(self):
        st.subheader("LSTM(Long Short Term Memory)")
        Xtrain = np.reshape(self.X_train, (self.X_train.shape[0], self.X_train.shape[1], 1))
        # Reshape the data
        Xtest = np.reshape(self.X_test, (self.X_test.shape[0], self.X_test.shape[1], 1 ))
        
        model = Sequential()
        model.add(LSTM(units = 50,dropout = 0.2, return_sequences = True, input_shape = (Xtrain.shape[1], 1), activation = 'tanh'))
        model.add(LSTM(units = 50, activation = 'tanh'))
        model.add(Dense(units = 2,activation='softmax'))
        model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
        model.fit(Xtrain,self.y_train, batch_size = 32, epochs = 2)
        
        y_pred = model.predict(Xtest)
        y_pred = [0 if y_pred[i] <=0.5 else 1 for i in range(len(y_pred))]
        
        st.write("ACCURACY : ",accuracy_score(self.y_test, y_pred)*100,"%")
        plt.plot(self.y_test, y_pred)
        st.write(plt.show())
        
        st.write("CONFUSION MATRIX : ") 
        st.write(confusion_matrix(self.y_test, y_pred))
        
        st.write("CLASSIFICATION REPORT : ")
        st.write(classification_report(self.y_test, y_pred, target_names = ["b","s"], output_dict=True))
        
        
    def gru_lstm(self):
        st.subheader("Hybrid Model")
        trainx = np.reshape(self.X_train, (self.X_train.shape[0], self.X_train.shape[1], 1))
        testx =  np.reshape(self.X_test, (self.X_test.shape[0], self.X_test.shape[1], 1 ))
       
        model = Sequential()
        model.add(Conv1D(filters=32, kernel_size=9, padding='same', activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Conv1D(filters=16, kernel_size=9, padding='same', activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(LSTM(16, dropout=0.2, recurrent_dropout=0.2,return_sequences=True))
        model.add(LSTM(8, dropout=0.2, recurrent_dropout=0.2))
        model.add(Dense(2, activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        model.fit(trainx,self.y_train,epochs=2,validation_data=(testx, self.y_test),verbose=1)
        
        y_pred = model.predict(testx)
        y_pred = [0 if y_pred[i] <=0.5 else 1 for i in range(len(y_pred))]
        
        st.write("ACCURACY : ",accuracy_score(self.y_test, y_pred)*100,"%")
        plt.plot(self.y_test, y_pred)
        st.write(plt.show())
        
        st.write("CONFUSION MATRIX : ") 
        st.write(confusion_matrix(self.y_test, y_pred))
        
        st.write("CLASSIFICATION REPORT : ")
        st.write(classification_report(self.y_test, y_pred, target_names = ["b","s"], output_dict=True))
        
        

if file is not None:

    dataset = pd.read_csv(file,nrows=30000)
    # flag is set to true as data has been successfully read
    flag = "True"
    st.header('**HIGGS BOSON DATA**')
    st.write(dataset.head())
    
    dataset.index=dataset.EventId
    dataset=dataset.drop(columns=['EventId'])

    st.subheader("DATA PRE-PROCESSING")
    st.write("Number of features",dataset.shape[1])
    st.subheader("Features List")
    st.write(dataset.columns)
    st.write("Label is the column to be predicted")
    #st.subheader("Correlation plot of the features")
    #corr = dataset.corr()#to find the pairwise correlation of all columns in the dataframe
    #fig, ax = plt.subplots()
    #sns.heatmap(corr,cmap="Greens", ax=ax) #Plot rectangular data as a color-encoded matrix.
    #st.write(fig)
    
    
    st.subheader("Labels distribution")
    st.bar_chart(dataset["Label"].value_counts())
    
    #st.subheader("Finding no. of null values per column in the dataset")
    #st.write(Counter(dataset.isna()))
    
    st.subheader("Statistical information about the datset")
    st.write(dataset.describe())
    
    data = dataset.iloc[:,:-1] # Extracting features
    imp_mean = SimpleImputer(missing_values = -999.0, strategy='mean') 
    # the placeholder for the missing values. All occurrences of missing_values will be imputed.
    #If “mean”, then replace missing values using the mean along each column. Can only be used with numeric data.
    
    X = imp_mean.fit_transform(data)
    y = dataset.iloc[:,-1] # extracting the labels
    train_label = y.tolist()
    class_names = list(set(train_label))
    le = LabelEncoder()  
    y = le.fit_transform(y) # Encoding categorical data to numeric data
    
    st.success("Data cleaned!")
    st.subheader('Test size split of users choice:')
    st.text('Default is set to 20%')
    
    k = st.number_input('',step = 5,min_value=10, value = 20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = k * 0.01, random_state = 0)
    st.write("Data is being split into testing and training data!")
    # Splitting the data into 20% test and 80% training data
    # Outlier detection and removal using Isolation Forest
    iso = IsolationForest(contamination='auto')
    train_iso = iso.fit_predict(X_train)
    test_iso = iso.fit_predict(X_test)
    st.success("Dataset is split into Training and Testing data ")
    st.write(Counter(train_hat)[-1],"outliers out of ",X_train.shape[0],"data points are removed which makes it easier for prediction")
    st.write(Counter(test_hat)[-1],"outliers out of ",X_test.shape[0],"data points are removed which makes it easier for prediction")
    # select all rows that are not outliers
    mask_train = train_iso != -1 #-1 refers to outliers while 1 refers to Inliers
    mask_test = test_iso != -1
    X_train, y_train = X_train[mask_train, :], y_train[mask_train]
    X_test, y_test = X_test[mask_test, :], y_test[mask_test]
    st.success("Outliers removed successfully!")
    std_sc = StandardScaler()
    X_train = std_sc.fit_transform(X_train) # Scaling the training data
    X_test = std_sc.transform(X_test) # Scaling the testing data

    dl = DL_models(X_train,y_train,X_test,y_test)
    st.subheader('Choose the Deep Learning model :')
    mopt = st.multiselect("Select :",["Simple ANN","RNN","LSTM","GRU_LSTM","All"])
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
            dl.gru_lstm()
            

    if(st.button("FINISH")):
        st.info("THANK YOU FOR YOUR PATIENCE. WE ARE DONE. HOPE YOU ARE SATISFIED")
        st.balloons()

else:
    st.warning("No file has been chosen yet")






