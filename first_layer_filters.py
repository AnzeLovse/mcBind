"""Visualise first layer filters.

This creates weblogos of filter motifs. Steps are as follows:
- get filter activation for samples with binding
- calculate threshold based on maximum activation in the dataset
- write sequence to file for locations that pass the threshold
- visualise sequences using weblogo
"""
import gzip
import os
import random

import numpy as np
from Bio import SeqIO
from keras.models import model_from_json, Model
from tensorflow import set_random_seed
from Bio.Seq import Seq

#Set random seeds for reproducibility.
np.random.seed(4)
random.seed(5)
set_random_seed(4) 


def load_data(path):   
    data = gzip.open(os.path.join(path,"sequences.fa.gz"),"rt")
    return data


def get_seq(protein, t_data, training_set_number): 
    if t_data == "train":
        training_data = load_data("datasets/clip/%s/30000/training_sample_%s"% (protein, training_set_number))
        x_train = np.zeros((30000,101,4))          
    
    elif t_data == "test":    
        training_data = load_data("datasets/clip/%s/30000/test_sample_%s"% (protein, training_set_number))
        x_train = np.zeros((10000,101,4))      
      
    r = 0    
    
    for record in SeqIO.parse(training_data,"fasta"):
        sequence = list(record.seq)                
        nucleotide = {'A' : 0, 'T' : 1, 'G' : 2, 'C' : 3, 'N' : 4} 
        num_seq = list()

        for i in range(0,len(sequence)):
                num_seq.append(nucleotide[sequence[i]])

        X = np.zeros((1,len(num_seq),4))

        for i in range (len(num_seq)):
                if num_seq[i] <= 3:
                    X[:,i,num_seq[i]] = 1               

        x_train[r,:,:] = X
        r = r + 1
    
    return x_train


def get_class(protein, t_data,training_set_number):
    y_train = []
    if t_data == 'train':
        data = load_data("datasets/clip/%s/30000/training_sample_%s"% (protein, training_set_number))

    elif t_data == 'test':
        data = load_data("datasets/clip/%s/30000/test_sample_%s"% (protein, training_set_number))

    for record in SeqIO.parse(data,"fasta"):
        v = int((record.description).split(":")[1])
        # [1,0] if there was no observed binding and [0,1] for sequences where binding was observed.
        y_train.append([int(v == 0), int(v != 0)])

    y_train = np.array(y_train)
    return y_train


def get_cobinding(protein, t_data, training_set_number):
    if t_data == "train":
        with gzip.open(("datasets/clip/%s/30000/training_sample_%s/matrix_Cobinding.tab.gz"% (protein, training_set_number)), "rt") as f:
            cobinding_data = np.loadtxt(f, skiprows=1) 
            
        cobinding = np.zeros((30000,101,int(cobinding_data.shape[1]/101)),dtype=np.int)    
    elif t_data == "test":
        with gzip.open(("datasets/clip/%s/30000/test_sample_%s/matrix_Cobinding.tab.gz"% (protein, training_set_number)), "rt") as f:
            cobinding_data = np.loadtxt(f, skiprows=1) 
        cobinding = np.zeros((10000,101,int(cobinding_data.shape[1]/101)),dtype=np.int)
   
    for n in range(0,cobinding_data.shape[1],101):
        a = cobinding_data[:,n:(n+101)]
        cobinding[:,:,int(n/101)] = a
    
    return cobinding
    

def get_region (protein, t_data, training_set_number):
    if t_data == "train":
        with gzip.open(("datasets/clip/%s/30000/training_sample_%s/matrix_RegionType.tab.gz"% (protein, training_set_number)), "rt") as f:
            region_data = np.loadtxt(f, skiprows=1)
        region = np.zeros((30000,101,int(region_data.shape[1]/101)),dtype=np.int)
                             
    elif t_data == "test":
        with gzip.open(("datasets/clip/%s/30000/test_sample_%s/matrix_RegionType.tab.gz"% (protein, training_set_number)), "rt") as f:
            region_data = np.loadtxt(f, skiprows=1) 
        region = np.zeros((10000,101,int(region_data.shape[1]/101)),dtype=np.int)
    
    for n in range(0,region_data.shape[1],101):
        a = region_data[:,n:int(n+101)]
        region[:,:,int(n/101)] = a
        
    return region


def get_fold (protein, t_data, training_set_number):
    if t_data == "train":
        with gzip.open(("datasets/clip/%s/30000/training_sample_%s/matrix_RNAfold.tab.gz"% (protein, training_set_number)), "rt") as f:
            fold_data = np.loadtxt(f, skiprows=1) 
        fold = np.zeros((30000,101,int(fold_data.shape[1]/101)), dtype=np.int)
                             
    elif t_data == "test":
        with gzip.open(("datasets/clip/%s/30000/test_sample_%s/matrix_RNAfold.tab.gz"% (protein, training_set_number)), "rt") as f:
            fold_data = np.loadtxt(f, skiprows=1)
        fold = np.zeros((10000,101,int(fold_data.shape[1]/101)), dtype=np.int)

    for n in range(0,fold_data.shape[1],101):
        a = fold_data[:,n:(n+101)]
        fold[:,:,int(n/101)] = a
    
    return fold

def load_data_sources(protein, t_data, training_set_number, *args):
    X = np.array([])
    data_sources = []
    for arg in args:
        if arg == 'KMER':
            if X.size == 0:
                X = get_seq(protein, t_data, training_set_number)
            else: 
                X = np.dstack((X, get_seq(protein, t_data, training_set_number)))
        if arg == 'RNA': 
            if X.size == 0:
                X = get_fold(protein, t_data, training_set_number)
            else: 
                X = np.dstack((X, get_fold(protein, t_data, training_set_number)))
        if arg == 'RG':   
            if X.size == 0:
                X = get_region(protein, t_data, training_set_number)
            else: 
                X = np.dstack((X, get_region(protein, t_data, training_set_number)))
        if arg == 'CLIP': 
            if X.size == 0:
                X = get_cobinding(protein, t_data, training_set_number)
            else: 
                X = np.dstack((X, get_cobinding(protein, t_data, training_set_number)))
        data_sources.append(arg)
        
    data_sources = ','.join(data_sources)
    return data_sources, X


def get_positive_samples(protein, t_data, training_set_number, *args ):
    data_sources, X_test = load_data_sources(protein, t_data, training_set_number, *args)
    y_test = get_class(protein, t_data, training_set_number)
    y_test = y_test [:,1]
    positive_samples = np.zeros((2000, X_test.shape[1], X_test.shape[2]))
    n = 0

    for i, value in enumerate(y_test):
        if value == 1:
            positive_samples[n] = X_test[i]
            n += 1 
 
    return positive_samples


def create_filter_motifs(protein, set_number, experiment_set, max_pct=0.5):
    """Creates weblogos of filter motifs.

    If the argument `max_pct` isn't passed in, the default 0.5 is used.

    Parameters
    ----------
    protein : str
        Name of protein
    set_number : int
        Dataset number
    experiment_set : list
        List of strings with names of datasources to use.
    max_pct : float, optional
        Threshold factor for filter actvation (default is 0.5)
    """
    X_samples = get_positive_samples(protein, 'test', set_number, *experiment_set)

    with open(f"results/set_{set_number}/{protein}/model.json", "r") as json_file:
        json = json_file.read()
        model = model_from_json(json)
        model.load_weights(f"results/set_{set_number}/{protein}/weights.h5")
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        
    first_layer_outputs = model.layers[0].output
    activation_model = Model(inputs = model.input, outputs = first_layer_outputs)
    activations = activation_model.predict(X_samples)

    test_sequences = load_data(f"datasets/clip/{protein}/30000/test_sample_{set_number}")
    y_test = get_class(protein, "test", set_number)
    y_test = y_test [:,1]
    positive_seq = [seq for seq, label in zip(SeqIO.parse(test_sequences,"fasta"), y_test) if label == 1]

    filters_sequences = {}
    for filter_number in range(10):
        filter_activations = activations[:,:,filter_number]

        all_activations = np.ravel(filter_activations)
        activations_mean = all_activations.mean()
        activations_norm = all_activations - activations_mean
        threshold = max_pct * activations_norm.max() + activations_mean

        for case_num, act in enumerate(filter_activations):
            locations = np.argwhere(act > threshold)
            if locations.size != 0:
                for loc in locations.flatten():
                    loc = int(loc)
                    nucleotides = list(positive_seq[case_num].seq)[loc: loc + 6]
                    motif = Seq("".join(nucleotides).replace("T", "U"))

                    if filter_number in filters_sequences:
                        filters_sequences[filter_number].append(motif)
                    else:
                        filters_sequences[filter_number] = [motif]
                    
    for filter_number, fil_seq in filters_sequences.items():
        if not os.path.exists('test_motifs/{}'.format(protein)):
            os.mkdir('test_motifs/{}'.format(protein))
        with open(f"test_motifs/{protein}/filter{filter_number}_motif.txt", "w") as motif_file:
            for motif in fil_seq:
                motif_file.write(str(motif) + "\n")

    for filter_number, fil_seq in filters_sequences.items():
        cmd = (f"weblogo -f test_motifs/{protein}/filter{filter_number}_motif.txt -F png_print -o test_motifs/{protein}/motif{filter_number}.png -P '' "
        f"--title 'Filter motif {filter_number} ({len(fil_seq)})' --size large --errorbars NO --show-xaxis YES --show-yaxis YES -A rna "
        f"--composition none --color '#00CC00' 'A' 'A' --color '#0000CC' 'C' 'C' --color '#FFB300' 'G' 'G' --color '#CC0000' 'U' 'U'")
        os.system(cmd)


if __name__ == "__main__":
    set_number = 0
    experiment_set = ['KMER' , 'RNA', 'RG', 'CLIP']
    protein_list = [
        "1_PARCLIP_AGO1234_hg19",
        "2_PARCLIP_AGO2MNASE_hg19",
        "3_HITSCLIP_Ago2_binding_clusters",
        "4_HITSCLIP_Ago2_binding_clusters_2",
        "5_CLIPSEQ_AGO2_hg19", 
        "6_CLIP-seq-eIF4AIII_1",
        "7_CLIP-seq-eIF4AIII_2",
        "8_PARCLIP_ELAVL1_hg19",
        "9_PARCLIP_ELAVL1MNASE_hg19",
        "10_PARCLIP_ELAVL1A_hg19",
        "12_PARCLIP_EWSR1_hg19",
        "13_PARCLIP_FUS_hg19",
        "14_PARCLIP_FUS_mut_hg19", 
        "15_PARCLIP_IGF2BP123_hg19",
        "16_ICLIP_hnRNPC_Hela_iCLIP_all_clusters", 
        "17_ICLIP_HNRNPC_hg19",
        "18_ICLIP_hnRNPL_Hela_group_3975_all-hnRNPL-Hela-hg19_sum_G_hg19--ensembl59_from_2337-2339-741_bedGraph-cDNA-hits-in-genome",
        "19_ICLIP_hnRNPL_U266_group_3986_all-hnRNPL-U266-hg19_sum_G_hg19--ensembl59_from_2485_bedGraph-cDNA-hits-in-genome", 
        "20_ICLIP_hnRNPlike_U266_group_4000_all-hnRNPLlike-U266-hg19_sum_G_hg19--ensembl59_from_2342-2486_bedGraph-cDNA-hits-in-genome", 
        "21_PARCLIP_MOV10_Sievers_hg19", "22_ICLIP_NSUN2_293_group_4007_all-NSUN2-293-hg19_sum_G_hg19--ensembl59_from_3137-3202_bedGraph-cDNA-hits-in-genome", 
        "23_PARCLIP_PUM2_hg19",
        "24_PARCLIP_QKI_hg19",
        "25_CLIPSEQ_SFRS1_hg19",
        "26_PARCLIP_TAF15_hg19",
        "27_ICLIP_TDP43_hg19",
        "28_ICLIP_TIA1_hg19",
        "29_ICLIP_TIAL1_hg19", 
        "30_ICLIP_U2AF65_Hela_iCLIP_ctrl_all_clusters",
        "31_ICLIP_U2AF65_Hela_iCLIP_ctrl+kd_all_clusters"
    ]
    for protein in protein_list:
        create_filter_motifs(protein, set_number, experiment_set, max_pct=0.5)