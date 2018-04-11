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
        print(record.seq)
        #class dobim tako da opis fasta locim po : in dobim vrednost
        print((record.description).split(":")[1])


	



