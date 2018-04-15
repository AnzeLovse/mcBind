import numpy as np
from Bio import SeqIO
import gzip
import os


def load_data(path):
    data = gzip.open(os.path.join(path,"sequences.fa.gz"),"rt")
    return data


def run():
    protein = "1_PARCLIP_AGO1234_hg19"
    training_data = load_data("data/clip/%s/5000/training_sample_0"% protein)
        
    for record in SeqIO.parse(training_data,"fasta"):
        
        print(record.id)
        sequence = list(record.seq)
        print(sequence)
                
        nucleotide = {'A' : 0, 'T' : 1, 'G' : 2, 'C' : 3, 'N' : 4} #kaj nerediti v primeru ko ni nukleotida?
        num_seq = list() #sekvenca v številskem formatu
        	
        for i in range(0,len(sequence)):
            num_seq.append(nucleotide[sequence[i]])


        X = np.zeros((len(num_seq),4))
        for i in range (len(num_seq)):
            if num_seq[i] <= 3:
                X[i,num_seq[i]] = 1
        print (X)
        #print (num_seq)
        #print (len(record.seq))

       
        #class dobim tako da opis fasta locim po : in dobim vrednost
        print((record.description).split(":")[1])

'''
X = np.zeros((101,4))

sequence = [0,1,2,3]

print (sequence)
for i in range(4):
	if sequence[i] <= 3:
		X[i,sequence[i]] = 1

print (X)
'''
