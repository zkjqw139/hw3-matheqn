# -*- coding: utf-8 -*-
"""
Created on Mon May  7 17:27:56 2018

@author: hasee
"""

import itertools
import numpy as np
import random

from keras.models import Sequential
from keras.layers import LSTM,RepeatVector,Dense,Activation
from keras.layers.wrappers import TimeDistributed,Bidirectional
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.models import load_model

CHARS = [str(n) for n in range(10)] + ['*','+','-', ' ', '\0']
CHAR_TO_INDEX = {i: c for c, i in enumerate(CHARS)}
INDEX_TO_CHAR = {c: i for c, i in enumerate(CHARS)}
          
MIN_NUMBER = 100
MAX_NUMBER = 999

MAX_N_EXAMPLES = (MAX_NUMBER - MIN_NUMBER) ** 2
N_EXAMPLES = 100000
N_FEATURES = len(CHARS)
MAX_NUMBER_LENGTH_LEFT_SIDE = len(str(MAX_NUMBER))
MAX_NUMBER_LENGTH_RIGHT_SIDE = MAX_NUMBER_LENGTH_LEFT_SIDE *2
MAX_EQUATION_LENGTH = (MAX_NUMBER_LENGTH_LEFT_SIDE * 2) + 4
MAX_RESULT_LENGTH = MAX_NUMBER_LENGTH_RIGHT_SIDE + 1

SPLIT = .1
EPOCHS = 500
LEARNING_RATE = 0.001
BATCH_SIZE = 256
HIDDEN_SIZE = 128
ENCODER_DEPTH = 1
DECODER_DEPTH = 1
DROPOUT = 0
BATCH_NORM = True

class colors:
    ok = '\033[92m'
    fail = '\033[91m'
    close = '\033[0m'

def generate_equations(max_count=None):
        number_permutations=itertools.permutations(range(100,1000),2)
        if max_count is not None:
            number_permutations=itertools.islice(number_permutations,max_count)
        number_permutations=list(number_permutations)
        random.shuffle(number_permutations)
        for x,y in number_permutations:
            p=random.randint(1,2)
            if p==1:
                if x>y:
                    yield'{}-{}'.format(x,y)
                else:
                    yield'{}-{}'.format(y,x)
            if p==2:
                    yield'{}+{}'.format(x,y)
            if p==3 or p==4:
                    yield'{}*{}'.format(x,y)
 

                    
                    
                    
                    
                    
                    
def one_hot_to_index(vector):
    if not np.any(vector):
        return -1
    return np.argmax(vector)

def one_hot_to_char(vector):
    index=one_hot_to_index(vector)
    if index==-1:
        return''
    return INDEX_TO_CHAR[index]

def one_hot_to_string(matrix):
    return ''.join(one_hot_to_char(vector) for vector in matrix)              
                
def equations_to_x_y(equations,n):
     x=np.zeros((n,MAX_EQUATION_LENGTH,N_FEATURES),dtype=np.bool)
     y=np.zeros((n,MAX_RESULT_LENGTH,N_FEATURES),dtype=np.bool)             
     for i,equation in enumerate(itertools.islice(equations,n)):
         result=str(eval(equation))
         result=' ' *(MAX_RESULT_LENGTH-1-len(result))+result
         
         equation +='\0'
         result   +='\0'
         for t,char in enumerate(equation):
            x[i,t,CHAR_TO_INDEX[char]]=1
         for t,char in enumerate(result):
            y[i,t,CHAR_TO_INDEX[char]]=1
     return x,y

def gen_data(equations,n):
    dataset=[]
    for i,equation in enumerate(itertools.islice(equations,n)):     
        dataset.append(equation)
    dataset=np.array(dataset)
    train_name="C:/Users/hasee/Desktop/lstm-sub/y_train.txt"
    np.savetxt(train_name,np.array(dataset),fmt="%s", delimiter=",")
     
def build_dataset():
    generator=generate_equations(max_count=1000000)
    n_test=round(SPLIT*N_EXAMPLES)
    n_train=N_EXAMPLES-n_test
    
    x_test,y_test=equations_to_x_y(generator,n_test)
    x_train,y_train=equations_to_x_y(generator,n_train)
  
    
    return x_test,y_test,x_train,y_train
    
def print_example_prediction(count,model,x_test,y_test):
    print('Examples:')
    prediction_indices=np.random.choice(x_test.shape[0],size=count,replace=False)
    print(np.array(x_test[prediction_indices,:]).shape)
    predictions=model.predict(x_test[prediction_indices,:])
    for i in range(count): 
        correct=one_hot_to_string(y_test[prediction_indices[i]])
        guess=one_hot_to_string(predictions[i])
        
        print('Q {} - {} '.format(one_hot_to_string(x_test[prediction_indices[i]]),one_hot_to_string(predictions[i])),end=' ')
        print('T', correct, end=' ')  
        if correct == guess:
            print( 'ok',end=' ')
        else:
            print( 'not ok',end=' ')
        print('\n')
      
        
def build_model():
    """
    Builds and returns the model based on the global config.
    """
    input_shape = (MAX_EQUATION_LENGTH, N_FEATURES)

    model = Sequential()

    # Encoder:
    model.add(Bidirectional(LSTM(20), input_shape=input_shape))
    model.add(BatchNormalization())

    # The RepeatVector-layer repeats the input n times
    model.add(RepeatVector(MAX_RESULT_LENGTH))

    # Decoder:
    model.add(Bidirectional(LSTM(20, return_sequences=True)))
    model.add(BatchNormalization())

    model.add(TimeDistributed(Dense(N_FEATURES)))
    model.add(Activation('softmax'))

    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(lr=0.01),
        metrics=['accuracy'],
    )

    return model


    
    
def main():
     
    
    
    model=build_model()
    model.summary()
    print()    
    
    x_test,y_test,x_train,y_train=build_dataset()
    
    print()
    print_example_prediction(5,model,x_test,y_test)
    print()
    
    for iteration in range(100):
        print()
        print('-'*50)
        print('Iteration',iteration)
        model.fit(x_train, y_train,batch_size=BATCH_SIZE,epochs=1,validation_data=(x_test, y_test))
        print_example_prediction(10,model,x_test,y_test)
        model.save('my_model.h5')
        
 
def test():
      
     model=load_model('my_model.h5')
     model.summary()
     print()    
     
     x_train_name="x_train.txt"
     y_train_name="y_train.txt"
  
     x=np.loadtxt(x_train_name,dtype='str',delimiter=" ")
     y=np.loadtxt(y_train_name,dtype='str',delimiter=" ")
 
     x_train=np.zeros((90000,MAX_EQUATION_LENGTH,N_FEATURES),dtype=np.bool)
     y_train=np.zeros((90000,MAX_RESULT_LENGTH,N_FEATURES),dtype=np.bool)             
     for i,equation in enumerate(x):
         result=str(eval(equation))
         result=' ' *(MAX_RESULT_LENGTH-1-len(result))+result
         
         equation +='\0'
         result   +='\0'
         for t,char in enumerate(equation):
            x_train[i,t,CHAR_TO_INDEX[char]]=1
         for t,char in enumerate(result):
            y_train[i,t,CHAR_TO_INDEX[char]]=1
     
     x_test=np.zeros((10000,MAX_EQUATION_LENGTH,N_FEATURES),dtype=np.bool)
     y_test=np.zeros((10000,MAX_RESULT_LENGTH,N_FEATURES),dtype=np.bool)             
     for i,equation in enumerate(y):
         result=str(eval(equation))
         result=' ' *(MAX_RESULT_LENGTH-1-len(result))+result
         
         equation +='\0'
         result   +='\0'
         for t,char in enumerate(equation):
            x_test[i,t,CHAR_TO_INDEX[char]]=1
         for t,char in enumerate(result):
            y_test[i,t,CHAR_TO_INDEX[char]]=1
     

     print(x_train.shape,y_train.shape)
     print(x_test.shape,y_test.shape)
     
     prediction=model.predict(x_train)
     print(prediction.shape)
     
     count=0
     for i in range(prediction.shape[0]):
         correct=one_hot_to_string(y_train[i,:,:])
         guess=one_hot_to_string(prediction[i,:,:])
         if correct==guess:
            count=count+1
         if i%100==0:
             print(count)
     acc=count/90000
     print(acc)
     
     test_prediction=model.predict(x_test)
     print(test_prediction.shape)
     
     count=0
     for i in range(test_prediction.shape[0]):
         correct=one_hot_to_string(y_test[i,:,:])
         guess=one_hot_to_string(test_prediction[i,:,:])
         if correct==guess:
            count=count+1
         if i%100==0:
             print(count)
     acc=count/10000
     print(acc)
     
    
if __name__=='__main__':
    ##train model use
    ##main()
    
    ##test model use
    ##test()
    test() 
    
    
    
    
    
    
    

    