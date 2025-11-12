import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

from lib.utils import X_from_seq

class IDPOracle:
    def __init__(self, args, tokenizer):
        self.dg_file = args.dg_file
        self.nu_file = args.nu_file
        self.tokenizer = tokenizer
        self.charge_termini = args.charge_termini
        self.temperature = args.temperature
        self.ionic_strength = args.ionic_strength
        self.residues = pd.read_csv(args.residues_file).set_index('one')
        self.feature = ['mean_lambda', 'faro', 'shd', 'ncpr', 'fcr', 'scd', 'ah_ij','nu_svr']
        self.models, self.targets = self.load_model()
        
    def load_model(self):
        models = {}
        models['dg'] = joblib.load(self.dg_file)
        targets = ['dg']
        return models, targets
    
    def predict(self, tokens):
        seq = self.tokenizer.detokenize(tokens)
        X = X_from_seq(seq, self.feature, residues=self.residues, 
                       charge_termini=self.charge_termini, nu_file=self.nu_file)
        for target in self.targets:
            pred = self.models[target].predict(X)
            mean_pred = np.mean(pred)
            if target == 'dg':
                output = mean_pred
        return np.float32(output)
    
    def batch_predict(self, batch_tokens):
        seqs = [self.tokenizer.detokenize(tokens) for tokens in batch_tokens]
        for target in self.targets:
            output = []
            for seq in seqs:
                X = X_from_seq(seq, self.feature, residues=self.residues,
                              charge_termini=self.charge_termini, nu_file=self.nu_file)
                pred = self.models[target].predict(X)
                mean_pred = np.mean(pred)
                if target == 'dg':
                    output.append(mean_pred)
        return np.float32(output)
    
def get_oracle(args, tokenizer):
    return IDPOracle(args, tokenizer)
