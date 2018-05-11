import numpy as np
from Bio import SeqIO
import gzip
import os
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv1D, MaxPooling1D, Flatten

def load_data(path):
    data = gzip.open(os.path.join(path,"sequences.fa.gz"),"rt")
    return data


def get_seq(protein, t_data):
    training_data = load_data("data/clip/%s/5000/training_sample_0"% protein)
    x_train = []
    
    
    for record in SeqIO.parse(training_data,"fasta"):
        sequence = list(record.seq)                
        nucleotide = {'A' : 0, 'T' : 1, 'G' : 2, 'C' : 3, 'N' : 4} 
        num_seq = list() #sekvenca v številskem formatu
        

        for i in range(0,len(sequence)):
            num_seq.append(nucleotide[sequence[i]])


        X = np.zeros((len(num_seq),4))

        for i in range (len(num_seq)):
            if num_seq[i] <= 3:
                X[i,num_seq[i]] = 1               
               
        x_train.append(X)


    x_train = np.array(x_train)
    return (x_train)
        
  
def get_class(protein, t_data):
    y_train = []
    

    if t_data == 'train':
        data = load_data("data/clip/%s/5000/training_sample_0"% protein)

    elif t_data == 'test':
        data = load_data("data/clip/%s/5000/test_sample_0"% protein)


    for record in SeqIO.parse(data,"fasta"):
        y_train.append([int((record.description).split(":")[1])])


    return np.array(y_train)
    
def run (protein):
    model = Sequential()
    model.add(Conv1D(100,kernel_size = 26, input_shape=(101,4), strides = 1, padding='valid', activation='relu'))
    
    model.add(MaxPooling1D(pool_size=13, strides=13, padding='valid'))
    
    model.add(Flatten())

    model.add(Dense(input_dim=640, units=100))
    
    model.add(Activation('relu'))
    
    model.add(Dense(input_dim=100, units=1))
    
    model.add(Activation('sigmoid'))
        

    model.summary()
    

    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    model.fit(get_seq(protein,"train"), get_class(protein,"train"), epochs=10, batch_size=100)


    score = model.evaluate(get_seq(protein,"train"), get_class(protein,"train"), batch_size=100) #test data je drugacne velikosti samo 1000 primerov
    
    print (score)

