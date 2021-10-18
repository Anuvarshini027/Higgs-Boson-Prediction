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


from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# if the user chooses to upload the data
file = st.file_uploader('Dataset')
class Models:

    def __init__(self,trainx,trainy,testx,testy):
        self.trainx = trainx
        self.trainy = trainy
        self.testx = testx
        self.testy = testy


    def simple_ANN(self):
        st.subheader("SIMPLE ARTIFICIAL NEURAL NETWORK")
            
        with st.spinner('Loading...'):
            model = Sequential()
            model.add(Dense(30, input_dim=self.trainx.shape[1], activation='relu'))
            model.add(Dense(2, activation='sigmoid'))
            opt = tensorflow.keras.optimizers.Adam(learning_rate=0.01)
            model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        
            st.subheader('Number of epochs as per user(s) choice:)')
            st.text('Default is set to 50')
            f = st.number_input('',step = 10,min_value=50, value = 50)
            model.fit(self.trainx,self.trainy,validation_data =(self.testx,self.testy), epochs=f)
            st.success("Done!")
        st.subheader("Summary of the Model")
        st.write(model.summary())
        # evaluate the model
        scores = model.evaluate(self.trainx, self.trainy, verbose=0)
        st.write("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
        

    def RNN(self):
        st.subheader("RECURRENT NEURAL NETWORK")
         
        trainx_dl=self.trainx.reshape(self.trainx.shape+(1,))
        testx_dl=self.testx.reshape(self.testx.shape+(1,))
        
        model = Sequential()
        model.add(SimpleRNN(units=100, input_shape= (trainx_dl.shape[1], 1)))
        model.add(Dense(2))
        model.compile(loss='mean_squared_error', optimizer='adam',metrics = ["accuracy"])
        
        st.subheader('Number of epochs as per user(s) choice:)')
        st.text('Default is set to 10')
        f = st.number_input('',step = 1,min_value=10, value = 10)
        model.fit(trainx_dl, self.trainy, epochs=f,validation_data=(testx_dl,self.testy),verbose=1)
        
        st.subheader("Summary of the Model")
        st.write(model.summary())
        # evaluate the model
        scores = model.evaluate(trainx_dl, self.trainy, verbose=0)
        st.write("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
        
    def LSTM(self):
        st.subheader("LSTM(Long Short Term Memory)")
         
        trainx_dl=self.trainx.reshape(self.trainx.shape+(1,))
        testx_dl=self.testx.reshape(self.testx.shape+(1,))
        
        model = Sequential()
        model.add(LSTM(units = 50,dropout = 0.2, return_sequences = True, input_shape = (trainx_dl.shape[1], 1), activation = 'tanh'))
        model.add(LSTM(units = 50, activation = 'tanh'))
        model.add(Dense(units = 2,activation='softmax'))
        model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
        
        st.subheader('Number of epochs as per user(s) choice:)')
        st.text('Default is set to 10')
        f = st.number_input('',step = 1,min_value=10, value = 10)
        model.fit(trainx_dl,self.trainy,epochs=f,validation_data=(testx_dl,self.testy),verbose=1)
        st.write(model.summary())
        
        st.subheader("Summary of the Model")
        st.write(model.summary())
        # evaluate the model
        scores = model.evaluate(trainx_dl, self.trainy, verbose=0)
        st.write("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
        
    def gru_lstm(self):
        st.subheader("Hybrid Model")
         
        trainx_dl=self.trainx.reshape(self.trainx.shape+(1,))
        testx_dl=self.testx.reshape(self.testx.shape+(1,))
        
        model = Sequential()
        model.add(Conv1D(filters=32, kernel_size=9, padding='same', activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Conv1D(filters=16, kernel_size=9, padding='same', activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(LSTM(16, dropout=0.2, recurrent_dropout=0.2,return_sequences=True))
        model.add(LSTM(8, dropout=0.2, recurrent_dropout=0.2))
        model.add(Dense(2, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        
        st.subheader('Number of epochs as per user(s) choice:)')
        st.text('Default is set to 10')
        f = st.number_input('',step = 1,min_value=10, value = 10)
        model.fit(trainx_dl,self.trainy,epochs=f,validation_data=(testx_dl, self.testy),verbose=1)
        st.write(model.summary())
        
        st.subheader("Summary of the Model")
        st.write(model.summary())
        # evaluate the model
        scores = model.evaluate(trainx_dl, self.trainy, verbose=0)
        st.write("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
       
        

if file is not None:

    dataset = pd.read_csv(file,nrows=30000)
    # flag is set to true as data has been successfully read
    flag = "True"
    st.header('**HIGGS BOSON DATA**')
    st.write(dataset.head())
    
    dataset.index=dataset.EventId
    dataset=dataset.drop(columns=['EventId'])

    st.write("DATA PRE-PROCESSING")   
    st.subheader("Labels distribution")
    st.bar_chart(dataset["Label"].value_counts())
    
    st.subheader("Features")
    st.write(dataset.columns)
    
    st.subheader("Statistical information about the datset")
    st.write(dataset.describe())
    
    st.write("We are dropping weight column as it contains almost similar properties as the label column. So if we train the model using weight column it will give us 100 percent score.")
    st.subheader("Excluding the feature weight")
    
    Xt1 = dataset.iloc[:,:-2] 
    st.write(Xt1.head())
    yt = dataset.iloc[:,-1] # extracting the label 
    Xt=np.asarray(Xt1)
    Y=pd.get_dummies(yt) #one hot encoding
    Y=np.asarray(Y)
    
    #Feature selection
    st.subheader("Feature selection")
  
    st.subheader('Number of features to be selected as per user(s) choice:)')
    st.text('Default is set to 15')
    f = st.number_input('',step = 5,min_value=10, value = 15)
    
    m = LogisticRegression()
    rfe = RFE(m, f) #extracts 15 best features from the dataset
    fit = rfe.fit(Xt, yt)
    ans=fit.support_
    index=[]
    for i in range(len(ans)):
        if ans[i] == True:
            index.append(i)
    
    a=[Xt1.iloc[:,i] for i in index]
    a=pd.DataFrame(a)
    a=a.T
    st.subheader("After extracting the features")
    st.write(a.head())
    st.write(a.columns)
    st.write(a.shape)
    
    st.subheader("Correlation plot of the features")
    corr = a.corr()#to find the pairwise correlation of all columns in the dataframe
    fig, ax = plt.subplots()
    sns.heatmap(corr,cmap="Greens", ax=ax) #Plot rectangular data as a color-encoded matrix.
    st.write(fig)
    
    st.success("Data cleaned!")
    st.subheader('Test size split of users choice:)')
    st.text('Default is set to 20%')
    
    k = st.number_input('',step = 5,min_value=10, value = 20)
    trainx,testx,trainy,testy = train_test_split(X, y, test_size = k * 0.01, random_state = 0)
    st.write("Data is being split into testing and training data!")
    # Splitting the data into 20% test and 80% training data   

    algo = Models(trainx,trainy,testx,testy)
    st.subheader('Choose the Deep Learning model :')
    options = st.multiselect("Select :",["Simple ANN","RNN","LSTM","GRU_LSTM(HYBRID)","All"])
    # "Click to select",
    if(st.button("START")):

        if "Simple ANN" in options:
            algo.simple_ANN()
            
        if "RNN" in options:
            algo.RNN()
            
        if "LSTM" in options:
            algo.LSTM()
        
        if "GRU_LSTM(HYBRID)" in options:
            algo.gru_lstm()
            
        if "All" in options:

            algo.basic_ANN()
            algo.RNN()
            algo.LSTM()
            algo.gru_lstm()
            

    if(st.button("FINISH")):
        st.info("Thank You for your Patience!")
        st.balloons()

else:
    st.warning("No file has been chosen yet")
