import numpy as np
from Bio import SeqIO
import gzip
import os
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv1D, MaxPooling1D, Flatten

def load_data(path):
    data = gzip.open(os.path.join(path,"sequences.fa.gz"),"rt")
    return data


def run():
    protein = "1_PARCLIP_AGO1234_hg19"
    training_data = load_data("data/clip/%s/5000/training_sample_0"% protein)
    x_train = []
    y_train = []
    
    for record in SeqIO.parse(training_data,"fasta"):
        sequence = list(record.seq)
        #print(sequence)                
        nucleotide = {'A' : 0, 'T' : 1, 'G' : 2, 'C' : 3, 'N' : 4} 
        num_seq = list() #sekvenca v številskem formatu
        

        for i in range(0,len(sequence)):
            num_seq.append(nucleotide[sequence[i]])


        X = np.zeros((len(num_seq),4))
        for i in range (len(num_seq)):
            if num_seq[i] <= 3:
                X[i,num_seq[i]] = 1
        #print (X)
        #print (num_seq)
        #print (len(record.seq))

        
        
        y_train.append([int((record.description).split(":")[1])])
    	#class dobim tako da opis fasta locim po : in dobim vrednost


        
        x_train.append(X)
    x_train = np.array(x_train)
    #print(x_train, y_train)
        
  

    model = Sequential()
    model.add(Conv1D(32,kernel_size = 26, input_shape=(101,4), strides = 1, padding='valid', activation='relu'))
    
    model.add(MaxPooling1D(pool_size=13, strides=13, padding='valid'))
    
    model.add(Flatten())

    model.add(Dense(input_dim=640, units=100))
    
    model.add(Activation('relu'))
    
    model.add(Dense(input_dim=100, units=1))
    
    model.add(Activation('sigmoid'))
        

    model.summary()
    

    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    model.fit(np.array(x_train), np.array(y_train), epochs=10, batch_size=32)#so podatki v pravilni obliki?


    score = model.evaluate(np.array(x_train), np.array(y_train), batch_size=32)
    print (score)

    print(model.predict(x_train)[13])
    print(model.predict(x_train)[14])
    print(model.predict(x_train)[15])


    
