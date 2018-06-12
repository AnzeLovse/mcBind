import numpy as np
from Bio import SeqIO
import gzip
import os
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv1D, MaxPooling1D, Flatten, GlobalAveragePooling1D

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
               
        x_train.append(X.flatten())


    x_train = np.array(x_train)
    x_train = np.expand_dims(x_train, axis=2)
    print(x_train.shape)
    print(x_train)
    return x_train
        
  
def get_class(protein, t_data):
    y_train = []
    

    if t_data == 'train':
        data = load_data("data/clip/%s/5000/training_sample_0"% protein)

    elif t_data == 'test':
        data = load_data("data/clip/%s/5000/test_sample_0"% protein)


    for record in SeqIO.parse(data,"fasta"):
        v = int((record.description).split(":")[1])
        y_train.append([int(v == 0), int(v != 0)])

    y_train = np.array(y_train)
#    y_train = np.expand_dims(y_train, axis=2)
    print(y_train.shape)
    print(y_train)
    return y_train
    
def run (protein):
    model = Sequential()
    model.add(Conv1D(20, 6, input_shape=(404, 1), strides = 4, padding='valid')) #, activation='relu'))
    #model.add(MaxPooling1D(pool_size=13, strides=13, padding='valid'))
    model.add(MaxPooling1D(4)) #, padding='valid'))
    model.add(Conv1D(20, 6, activation='relu'))
    model.add(MaxPooling1D(4))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(2, activation='sigmoid'))

    print(model.summary())
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(get_seq(protein,"train"), get_class(protein,"train"), epochs=10, batch_size=16, verbose=1)


    score = model.evaluate(get_seq(protein,"test"), get_class(protein,"test"), batch_size=20) #test data je drugacne velikosti samo 1000 primerov
    
    print (score)

