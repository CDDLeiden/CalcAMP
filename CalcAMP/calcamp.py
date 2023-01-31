# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
from modlamp.descriptors import PeptideDescriptor, GlobalDescriptor
import pickle
from CalcAMP.PyBioMed.PyBioMed import Pyprotein

def LoadModel(model):
    """Load prepared sklearn model to be able to make predictions.

    Args:
        model (__path__): path to the pickled model

    Returns:
        loaded_model: sklearn model ready to be used as such
    """    
    loaded_model = pickle.load(open(model, 'rb'))
    return loaded_model


def OpenFasta(fasta_file, seq_length=True):
    """Open a Fasta file and returns a DataFrame containing the 
    IDs, the sequences and optionnaly the length of the peptides.

    Args:
        fasta_file (_path_): path to the Fasta file
        seq_length (bool, optional): If True return in a column the length 
        of the sequences. Default to True.
    
    Returns:
        df: dataframe containing IDs/Sequences/Lengths of the 
        peptides
    """
    L_ids = []
    L_seqs = []
    L_length = []
    with open(fasta_file, 'r') as fp:
        for line in fp:
            line = line.rstrip('\n')
            if not line:
                continue
            if line.startswith(">"):
                L_ids.append(line.rstrip('\n')[1:])
            else:
                L_seqs.append(line.rstrip('\n'))
                if seq_length:
                    L_length.append(len(line))
    df = pd.DataFrame()
    df["ID"] = L_ids
    df["Sequence"] = L_seqs
    if seq_length:
        df["Length"] = L_length
    
    return df

def OpenCSV(csv_file, seq_length=True, sep=";"):
    """Open a csv file and returns a DataFrame containing the 
    IDs, the sequences and optionnaly the length of the peptides.

    Args:
        csv_file (_path_): path to the CSV file
        seq_length (bool, optional): If True return in a column the length 
        of the sequences. Default to True.
        sep (str): motif separating the values. Default to ",".
    Returns:
        df: dataframe containing IDs/Sequences/Lengths of the 
        peptides
    """
    df = pd.read_csv(csv_file, sep=";")
    if seq_length:
        df["Length"] = [len(seq) for seq in df.Sequence]
    
    return df


def CalculateAllDescriptors(df):
    """Function to calculate the different descriptors used for the 
    predictions of antimicrobial activity.

    Args:
        df (_pd.DataFrame_): dataframe containing the sequences of the 
        peptides

    Returns:
        df_desc_all: new dataframe containing all the values of the 
        descriptors calculated for each peptides
    """    
    aacomp=[]
    ctdcomp=[]
    dpcomp=[]
    paacomp=[]

    for seq in df.Sequence:
        protein_class = Pyprotein.PyProtein(seq)
        aacomp.append(protein_class.GetAAComp()) #aa composition
        ctdcomp.append(protein_class.GetCTD()) #CTD descriptors (147 features)
        dpcomp.append(protein_class.GetDPComp()) #Di peptide descriptors (400 features)
        paacomp.append(protein_class.GetPAAC(lamda=4,weight=0.05)) #Pseudoamino acid composition (24)

    #Create a specific df for each and then concatenate them 

    df_aa = pd.DataFrame(aacomp)
    df_ctd = pd.DataFrame(ctdcomp)
    df_dpc = pd.DataFrame(dpcomp)
    df_dpc.rename(columns={'MW': 'MW_dipep'}, inplace=True) #to avoid two columns with the same name
    df_paa = pd.DataFrame(paacomp)

    #ModelAMP physicochemical descriptors (10)
    B = GlobalDescriptor(list(df.Sequence))
    B.calculate_all()
    df_desc = pd.DataFrame(B.descriptor, columns=B.featurenames)
    df_desc_all = pd.concat([df_aa, df_ctd, df_dpc, df_desc, df_paa], axis=1)
    return df_desc_all

def GetAllPred(L_pep, model):
    """Return the prediction (probability or as binary) for 
    a list of peptides

    Args:
        L_pep (_list_): list of peptide sequences
        model (_sklearnmodel_): model to use for the predictions

    Returns:
        df_pred: DataFrame containing the predictions
    """     
    df_pred = pd.DataFrame()
    df_pred['Sequence'] = L_pep
    df_desc = CalculateAllDescriptors(df_pred)
    if not model.n_features_in_ == len(df_desc.columns):
    #keep only features used by the model
        df_desc = df_desc[np.intersect1d(df_desc.columns, model.feature_names)]
    
    pred_prod = model.predict_proba(df_desc)

    df_pred['Proba_pred_non_amp'] = [np.round(i[0], 3) for i in pred_prod]
    df_pred['Proba_pred_amp'] = [np.round(i[1], 3) for i in pred_prod]
    df_pred['AMP'] = model.predict(df_desc)
    return df_pred

def Pred_proba(seq, model):
    """Return the prediction (probability) for a simple peptide sequence
    or from a list of peptides

    Args:
        seq (_str_ or _list_): string or list of peptide sequences
        model (_sklearnmodel_): model to use for the predictions

    Returns:
        df_pred: DataFrame containing the prediction probabilities
    """     
    df_pred = pd.DataFrame()
    
    if type(seq) == str:
        df_pred['Sequence'] = [seq]
    elif type(seq) == list:
        df_pred['Sequence'] = seq
    else:
        return "Check Input"

    df_desc = CalculateAllDescriptors(df_pred)
    if not model.n_features_in_ == len(df_desc.columns):
    #keep only features used by the model
        df_desc = df_desc[np.intersect1d(df_desc.columns, model.feature_names)]
    
    pred_prod = model.predict_proba(df_desc)

    df_pred['Proba_pred_non_amp'] = [np.round(i[0], 3) for i in pred_prod]
    df_pred['Proba_pred_amp'] = [np.round(i[1], 3) for i in pred_prod]
    return df_pred

def Pred(seq, model):
    """Return the binary predictions for a simple peptide sequence
    or from a list of peptides

    Args:
        seq (_str_ or _list_): string or list of peptide sequences
        model (_sklearnmodel_): model to use for the predictions

    Returns:
        df_pred: DataFrame containing the prediction probabilities
    """     
    df_pred = pd.DataFrame()
    
    if type(seq) == str:
        df_pred['Sequence'] = [seq]
    elif type(seq) == list:
        df_pred['Sequence'] = seq
    else:
        return "Check Input"

    df_desc = CalculateAllDescriptors(df_pred)
    if not model.n_features_in_ == len(df_desc.columns):
    #keep only features used by the model
        df_desc = df_desc[np.intersect1d(df_desc.columns, model.feature_names)]
    
    pred_prod = model.predict(df_desc)
    df_pred['AMP'] = pred_prod
    
    return df_pred