import numpy as np
from keras.models import Sequential
from keras.layers import Dense,LSTM,Embedding
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
data = """ Jack and Jill went up the hill\n
        To fetch a pail of water\n
        Jack fell down and broke his crown\n
        And Jill came tumbling after\n """
#object of tokenizer class created 
tokenizer=Tokenizer()
#data fitted by tokenizer
tokenizer.fit_on_texts([data])
#encoded_data stores the value of repeated score of the word 1= highly repeated 21= least repeated 
encoded_data=tokenizer.texts_to_sequences([data])[0]
print(encoded_data)
''' Data Formation

calculate the vocab size

tokenizer.word_size gives dictionary of words corresponidng to the there label no.

eg: "and" is repeated most in the data and thus it has label 1

now the list will be created containing current word and next word. this list will be fed into the neural network 

and training will be done on the list created

list format ----> [current word :next word]'''
vocab_size=len(tokenizer.word_index)+1
print(vocab_size)
word_sequence=[] #list_word_sequence created 
for i in range(1,len(encoded_data)):
    sequence=encoded_data[i-1:i+1]        #saving encoder current word and next word in sequence variable
    word_sequence.append(sequence)
word_sequence=np.array(word_sequence)   #converting list into array
print(word_sequence)
'''# Splitting of Data

now we have generated data in right format and now we can use the data to in our ML model
'''
x_train,y_train = word_sequence[:,0],word_sequence[:,1]  #slicing coloumn of the array into x_train and y_train
print(f"x_train:{x_train} \ny_train:{y_train}")  #print statement
#converting categorical data to numerical data using one hot encoding 
#The binary variables are often called “dummy variables"
y_train=to_categorical(y_train,num_classes=vocab_size)
model=Sequential()
model.add(Embedding(vocab_size,10,input_length=1))
model.add(LSTM(70))
model.add(Dense(vocab_size,activation="softmax"))
model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])
print(model.summary())
model.fit(x_train,y_train,epochs=500,verbose=1)
x_test=input("enter word whose next word you wanna predict:")
no_next_word=int(input("enter no. of predictions you want:"))
#in_test variable created to prevent chnages form testing element. can be ignored and changes can be done directly
#on the x_test
in_test,next_word=x_test,x_test
print(f"entered word:{x_test}")
for i in range(no_next_word):
    print(i)
    out_word=""
    encoded_test=tokenizer.texts_to_sequences([in_test])[0]
    encoded_test=np.array(encoded_test)
    #prediction:
    y_predict=model.predict_classes(encoded_test,verbose=0)
    for word, index in tokenizer.word_index.items():
        if y_predict==index:
            out_word=word
            break
    in_test=out_word
    print(f"predicted next words:{out_word}")
