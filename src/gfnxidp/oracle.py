import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Import DeePhase dependencies
import pickle
import math
import tempfile
import subprocess
from sklearn.ensemble import RandomForestClassifier
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio.SeqUtils.IsoelectricPoint import IsoelectricPoint as IP
from gensim.models import word2vec
from gfnxidp.utils import X_from_seq, Model, AttrSetter

import warnings
from Bio import BiopythonDeprecationWarning
warnings.filterwarnings("ignore", category=BiopythonDeprecationWarning)
warnings.filterwarnings("ignore", message="Trying to unpickle estimator")

    
class IDPOracle:
    def __init__(self, args, tokenizer):
        """
        Unified Oracle class for both DG and CSAT predictions.
        
        Args:
            args: Configuration arguments
            tokenizer: Tokenizer for sequence processing
            model_type: Type of model to use ('dg' or 'csat')
        """
        self.model_type = args.oracle_mode
        self.tokenizer = tokenizer
        self.charge_termini = args.charge_termini
        self.temperature = args.temperature
        self.ionic_strength = args.ionic_strength
        self.nu_model = args.nu_model
        self.residues = pd.read_csv(args.residues_file).set_index('one')
        self.feature = ['mean_lambda', 'faro', 'shd', 'ncpr', 'fcr', 'scd', 'ah_ij', 'nu_svr']
        
        # Set model file and target based on type
        if self.model_type == 'dg':
            self.model_file = args.dg_model
            self.target = 'dg'
        elif self.model_type == 'csat':
            self.model_file = args.csat_model
            self.target = 'logcdil_mgml'
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}. Use 'dg' or 'csat'.")
        
        self.model = self.load_model()
        
    def load_model(self):
        """Load the appropriate model based on model_type."""
        return joblib.load(self.model_file)
    
    def predict(self, tokens):
        """Predict for a single sequence."""
        seq = self.tokenizer.detokenize(tokens)
        X = X_from_seq(seq, self.feature, residues=self.residues, 
                       charge_termini=self.charge_termini, nu_file=self.nu_model)
        pred = self.model.predict(X)
        mean_pred = np.mean(pred)
        return np.float32(mean_pred)
    
    def batch_predict(self, batch_tokens):
        """Predict for a batch of sequences."""
        seqs = [self.tokenizer.detokenize(tokens) for tokens in batch_tokens]
        output = []
        for seq in seqs:
            X = X_from_seq(seq, self.feature, residues=self.residues,
                          charge_termini=self.charge_termini, nu_file=self.nu_model)
            pred = self.model.predict(X)
            mean_pred = np.mean(pred)
            output.append(mean_pred)
        return np.float32(output)
    

def get_oracle(args, tokenizer):
    return IDPOracle(args, tokenizer)
