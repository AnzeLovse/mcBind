# Generates commands that were run for model training.
import os

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
    "11_CLIPSEQ_ELAVL1_hg19",
    "12_PARCLIP_EWSR1_hg19",
    "13_PARCLIP_FUS_hg19",
    "14_PARCLIP_FUS_mut_hg19",
    "15_PARCLIP_IGF2BP123_hg19",
    "16_ICLIP_hnRNPC_Hela_iCLIP_all_clusters",
    "17_ICLIP_HNRNPC_hg19",
    "18_ICLIP_hnRNPL_Hela_group_3975_all-hnRNPL-Hela-hg19_sum_G_hg19--ensembl59_from_2337-2339-741_bedGraph-cDNA-hits-in-genome",
    "19_ICLIP_hnRNPL_U266_group_3986_all-hnRNPL-U266-hg19_sum_G_hg19--ensembl59_from_2485_bedGraph-cDNA-hits-in-genome",
    "20_ICLIP_hnRNPlike_U266_group_4000_all-hnRNPLlike-U266-hg19_sum_G_hg19--ensembl59_from_2342-2486_bedGraph-cDNA-hits-in-genome",
    "21_PARCLIP_MOV10_Sievers_hg19",
    "22_ICLIP_NSUN2_293_group_4007_all-NSUN2-293-hg19_sum_G_hg19--ensembl59_from_3137-3202_bedGraph-cDNA-hits-in-genome",
    "23_PARCLIP_PUM2_hg19",
    "24_PARCLIP_QKI_hg19",
    "25_CLIPSEQ_SFRS1_hg19",
    "26_PARCLIP_TAF15_hg19",
    "27_ICLIP_TDP43_hg19",
    "28_ICLIP_TIA1_hg19",
    "29_ICLIP_TIAL1_hg19",
    "30_ICLIP_U2AF65_Hela_iCLIP_ctrl_all_clusters",
    "31_ICLIP_U2AF65_Hela_iCLIP_ctrl+kd_all_clusters",
]


# Used to iterate through all three datasets.
for i in range (3):
    #Create directories for saving models and predictions.
    os.system('mkdir models_%s; mkdir predictions_%s' % (i, i))
    
    # Commands for model training.
    for protein in protein_list:
        os.system('KERAS_BACKEND="theano" python ideep.py --train=True --data_dir=datasets/clip/%s/30000/training_sample_%s/ --model_dir=models_%s/%s;' % (protein, i, i, protein))
       
    # Commands for running predictions.
    for protein in protein_list:
        os.system ('KERAS_BACKEND="theano" python ideep.py --predict=True --data_dir=datasets/clip/%s/30000/test_sample_%s/ --model_dir=models%s/%s --out_file=predictions_%s/%s; ' % (protein, i, i, protein, i, protein))
