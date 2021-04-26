#Training the model on 30k datasets.

import random
import numpy as np
from Bio import SeqIO
import gzip
import os
import pickle
from tensorflow import set_random_seed
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dense, Activation, Conv1D, MaxPooling1D, Flatten, GlobalAveragePooling1D, Dropout
from sklearn.metrics import roc_auc_score
import pandas as pd
from keras.models import model_from_json

#Set random seeds for reproducibility.
np.random.seed(4)
random.seed(5)
set_random_seed(4) 



def load_data(path):   
    data = gzip.open(os.path.join(path,"sequences.fa.gz"),"rt")
    return data


def get_seq(protein, t_data, training_set): 
    if t_data == "train":     
        training_data = load_data("datasets/clip/%s/30000/training_sample_%s"% (protein, training_set))
        x_train = np.zeros((30000,101,4))          
    
    elif t_data == "test":          
        training_data = load_data("datasets/clip/%s/30000/test_sample_%s"% (protein, training_set))
        x_train = np.zeros((10000,101,4))      
    r = 0 
    for record in SeqIO.parse(training_data,"fasta"):
        sequence = list(record.seq)                
        nucleotide = {'A' : 0, 'T' : 1, 'G' : 2, 'C' : 3, 'N' : 4} 
        num_seq = list() #Encoded sequence
        for i in range(0,len(sequence)):
                num_seq.append(nucleotide[sequence[i]])

        X = np.zeros((1,len(num_seq),4))
        for i in range (len(num_seq)):
                if num_seq[i] <= 3:
                    X[:,i,num_seq[i]] = 1               
        x_train[r,:,:] = X
        r = r + 1
    
    return x_train


def get_class(protein, t_data,training_set):
    y_train = []
    if t_data == 'train':
        data = load_data("datasets/clip/%s/30000/training_sample_%s"% (protein, training_set))
    elif t_data == 'test':
        data = load_data("datasets/clip/%s/30000/test_sample_%s"% (protein, training_set))
    for record in SeqIO.parse(data,"fasta"):
        v = int((record.description).split(":")[1])
        # [1,0] if there was no observed binding and [0,1] for sequences where binding was observed.
        y_train.append([int(v == 0), int(v != 0)])

    y_train = np.array(y_train)
    return y_train


def get_cobinding(protein, t_data, training_set):
    if t_data == "train":
        with gzip.open(("datasets/clip/%s/30000/training_sample_%s/matrix_Cobinding.tab.gz"% (protein, training_set)), "rt") as f:
            cobinding_data = np.loadtxt(f, skiprows=1)       
        cobinding = np.zeros((30000,101,int(cobinding_data.shape[1]/101)),dtype=np.int)    
    elif t_data == "test":
        with gzip.open(("datasets/clip/%s/30000/test_sample_%s/matrix_Cobinding.tab.gz"% (protein, training_set)), "rt") as f:
            cobinding_data = np.loadtxt(f, skiprows=1) 
        cobinding = np.zeros((10000,101,int(cobinding_data.shape[1]/101)),dtype=np.int)
    for n in range(0,cobinding_data.shape[1],101):
        a = cobinding_data[:,n:(n+101)]
        cobinding[:,:,int(n/101)] = a
    
    return cobinding
    

def get_region (protein, t_data, training_set):
    if t_data == "train":
        with gzip.open(("datasets/clip/%s/30000/training_sample_%s/matrix_RegionType.tab.gz"% (protein, training_set)), "rt") as f:
            region_data = np.loadtxt(f, skiprows=1)
        region = np.zeros((30000,101,int(region_data.shape[1]/101)),dtype=np.int)
                             
    elif t_data == "test":
        with gzip.open(("datasets/clip/%s/30000/test_sample_%s/matrix_RegionType.tab.gz"% (protein, training_set)), "rt") as f:
            region_data = np.loadtxt(f, skiprows=1) 
        region = np.zeros((10000,101,int(region_data.shape[1]/101)),dtype=np.int)
    for n in range(0,region_data.shape[1],101):
        a = region_data[:,n:(n+101)]
        region[:,:,int(n/101)] = a
        
    return region


def get_fold (protein, t_data, training_set):
    if t_data == "train":
        with gzip.open(("datasets/clip/%s/30000/training_sample_%s/matrix_RNAfold.tab.gz"% (protein, training_set)), "rt") as f:
            fold_data = np.loadtxt(f, skiprows=1) 
        fold = np.zeros((30000,101,int(fold_data.shape[1]/101)), dtype=np.int)
                             
    elif t_data == "test":
        with gzip.open(("datasets/clip/%s/30000/test_sample_%s/matrix_RNAfold.tab.gz"% (protein, training_set)), "rt") as f:
            fold_data = np.loadtxt(f, skiprows=1) 
        fold = np.zeros((10000,101,int(fold_data.shape[1]/101)),dtype=np.int)




    for n in range(0,fold_data.shape[1],101):
        a = fold_data[:,n:(n+101)]
        fold[:,:,int(n/101)] = a
    
    
    return fold

def load_data_sources(protein, t_data, training_set, *args):
    X = np.array([])
    data_sources = []
    for arg in args:
        
        if arg == 'KMER':
            if X.size == 0:
                X = get_seq(protein, t_data, training_set)
            else: 
                X = np.dstack((X, get_seq(protein, t_data, training_set)))
        if arg == 'RNA': 
            if X.size == 0:
                X = get_fold(protein, t_data, training_set)
            else: 
                X = np.dstack((X, get_fold(protein, t_data, training_set)))
        if arg == 'RG':   
            if X.size == 0:
                X = get_region(protein, t_data, training_set)
            else: 
                X = np.dstack((X, get_region(protein, t_data, training_set)))
        if arg == 'CLIP': 
            if X.size == 0:
                X = get_cobinding(protein, t_data, training_set)
            else: 
                X = np.dstack((X, get_cobinding(protein, t_data, training_set)))
        data_sources.append(arg)
        
    data_sources = ','.join(data_sources)
    return data_sources, X


def train_model(protein, training_set, sources):
        print (f"Training model for {protein} from set {training_set}")
        _, X = load_data_sources(protein, 'train', training_set, *sources)
        y = get_class(protein,"train",training_set)
        size = X.shape[2]
        # ideep receptive = 9
        model = Sequential()
        model.add(Conv1D(10,6, data_format='channels_last', input_shape=(101, size) , strides = 1, padding='valid'))
        model.add(MaxPooling1D(pool_size=2, strides=1, padding='valid'))
        model.add(Conv1D(10, 3, activation='relu',strides=1, padding='valid'))
        # Removed layers
        #-------------------------------------------------------------------
        # model.add(Dropout(0.1))
        # model.add(MaxPooling1D(pool_size=40, strides=1, padding='valid'))
        # model.add(Conv1D(15, 4, activation='relu'))
        # model.add(MaxPooling1D(pool_size=30, strides=1, padding='valid'))
        # model.add(Conv1D(15, 3, activation='relu'))
        #-------------------------------------------------------------------
        model.add(GlobalAveragePooling1D())
        model.add(Dense(200, activation='relu'))
        model.add(Dropout(0.1))
        model.add(Dense(2, activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        checkpointer = ModelCheckpoint(filepath="models/" + protein + "_weights.hdf5", verbose=1, save_best_only=True)
        earlystopper = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
        metrics = model.fit(X, y, validation_split = 0.2, epochs=12, batch_size=200, verbose=1, callbacks=[earlystopper])
        
        #Save model and weights to .json file.
        json_model = model.to_json()
        with open("shallow_model/set_%s/%s/model.json" % (training_set, protein), "w") as json_file:
            json_file.write(json_model)
        with open ("shallow_model/set_%s/%s/weights.h5" % (training_set, protein), "w") as weights_file:
            model.save_weights("shallow_model/set_%s/%s/weights.h5" % (training_set, protein))
        
        #Save the model metrics generated with model fit.
        with open("shallow_model/set_%s/%s/metrics" % (training_set, protein), "wb") as pickle_file:
            pickle.dump(metrics, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)


def run_predictions(protein, training_set, sources):
    with open("shallow_model/set_%s/%s/model.json" % (training_set, protein), "r") as json_file:
        json = json_file.read()
        loaded_model = model_from_json(json)
        loaded_model.load_weights("shallow_model/set_%s/%s/weights.h5" % (training_set, protein))

    #Load data for testing purposes.
    data_sources, X_test = load_data_sources(protein, 'test', training_set, *sources)
    y_test = get_class(protein,"test",training_set)


    #Run predictions on test dataset and save them.
    predictions = loaded_model.predict(X_test)
    y_scores = predictions[:,0:1]
    y_test = y_test[:,0:1]

    score = roc_auc_score(y_test, y_scores)
                                              
    with open ("shallow_model/set_%s/%s/predictions" % (training_set, protein), "wb") as predictions_file:
        np.save(predictions_file, predictions)
    
    print(f"Model for {protein}({training_set}) achieved AUC {score}.")
    
    return score


if __name__ == "__main__":
    protein_list = ["1_PARCLIP_AGO1234_hg19", "2_PARCLIP_AGO2MNASE_hg19","3_HITSCLIP_Ago2_binding_clusters","4_HITSCLIP_Ago2_binding_clusters_2","5_CLIPSEQ_AGO2_hg19", "6_CLIP-seq-eIF4AIII_1","7_CLIP-seq-eIF4AIII_2","8_PARCLIP_ELAVL1_hg19","9_PARCLIP_ELAVL1MNASE_hg19", "10_PARCLIP_ELAVL1A_hg19", "11_CLIPSEQ_ELAVL1_hg19", "12_PARCLIP_EWSR1_hg19", "13_PARCLIP_FUS_hg19", "14_PARCLIP_FUS_mut_hg19", "15_PARCLIP_IGF2BP123_hg19", "16_ICLIP_hnRNPC_Hela_iCLIP_all_clusters", "17_ICLIP_HNRNPC_hg19", "18_ICLIP_hnRNPL_Hela_group_3975_all-hnRNPL-Hela-hg19_sum_G_hg19--ensembl59_from_2337-2339-741_bedGraph-cDNA-hits-in-genome", "19_ICLIP_hnRNPL_U266_group_3986_all-hnRNPL-U266-hg19_sum_G_hg19--ensembl59_from_2485_bedGraph-cDNA-hits-in-genome", "20_ICLIP_hnRNPlike_U266_group_4000_all-hnRNPLlike-U266-hg19_sum_G_hg19--ensembl59_from_2342-2486_bedGraph-cDNA-hits-in-genome", "21_PARCLIP_MOV10_Sievers_hg19", "22_ICLIP_NSUN2_293_group_4007_all-NSUN2-293-hg19_sum_G_hg19--ensembl59_from_3137-3202_bedGraph-cDNA-hits-in-genome", "23_PARCLIP_PUM2_hg19", "24_PARCLIP_QKI_hg19", "25_CLIPSEQ_SFRS1_hg19","26_PARCLIP_TAF15_hg19", "27_ICLIP_TDP43_hg19", "28_ICLIP_TIA1_hg19", "29_ICLIP_TIAL1_hg19", "30_ICLIP_U2AF65_Hela_iCLIP_ctrl_all_clusters", "31_ICLIP_U2AF65_Hela_iCLIP_ctrl+kd_all_clusters"]
    sources = ['KMER' , 'RNA', 'RG', 'CLIP']
    scores = {}
    for training_set in range (3):
        if not os.path.exists("shallow_model/set_%s" % training_set):
            os.makedirs("shallow_model/set_%s" % training_set)
        for protein in protein_list:
            if not os.path.exists("shallow_model/set_%s/%s" % (training_set, protein)):
                os.makedirs("shallow_model/set_%s/%s" % (training_set, protein))
                print(f"Created new directory results/set_{training_set}/{protein}.")

            train_model(protein, training_set, sources)
            score = run_predictions(protein, training_set, sources)

            if protein in scores:
                scores[protein].append(score)
            else:
                scores[protein] = [score]
    pd.DataFrame.from_dict(scores, orient='index').to_csv("shallow_model/scores.tsv", sep="\t")


