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

import warnings
from Bio import BiopythonDeprecationWarning
warnings.filterwarnings("ignore", category=BiopythonDeprecationWarning)
warnings.filterwarnings("ignore", message="Trying to unpickle estimator")

# DeePhase constants and setup
SEED = 42
np.random.seed(SEED)

AA_array = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
       'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

kd = {"A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5,
      "C": 2.5,  "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2,
      "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9,
      "F": 2.8, "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3,
      "V": 4.2}

Hydrophobic_AAs = ['A', 'I', 'L', 'M', 'F', 'V']
Polar_AAs = ['S', 'Q', 'N', 'G', 'C', 'T', 'P']
Cation_AAs = ['K', 'R', 'H']
Anion_AAs = ['D', 'E']
Arom_AAs = ['W', 'Y', 'F']

# DeePhase helper functions
def hydrophobicity(seq):
    sequence = ProteinAnalysis(seq)
    HB = 0
    for k in range(0, len(AA_array)):
        HB = HB + sequence.count_amino_acids()[AA_array[k]] * kd[AA_array[k]]        
    return HB

def Shannon_entropy(seq):
    sequence =  ProteinAnalysis(seq)
    entropy = 0
    for k in range(0, len(AA_array)):
        if sequence.get_amino_acids_percent()[AA_array[k]] == 0:
            entropy = entropy + 0
        else:
            entropy = entropy - math.log2(sequence.get_amino_acids_percent()[AA_array[k]]) * sequence.get_amino_acids_percent()[AA_array[k]]        
    return entropy


def extract_LCR(seq):
    tmp_LCR = tempfile.NamedTemporaryFile(delete=False)  
    with open(tmp_LCR.name, 'w') as f_LCR:
         f_LCR.write('>1\n' + str(seq))
    tmp_LCR.seek(0)
    
    out = subprocess.Popen(['segmasker', '-in', str(tmp_LCR.name)], 
           stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    stdout_LCR, stderr_LCR = out.communicate() 
    stdout_LCR = stdout_LCR.split()[1:]
    
    LCR_start_values = []; LCR_end_values = []    
    for i in range(0, int(len(stdout_LCR)/3)):
        LCR_start_values.append(int(str(stdout_LCR[3*i],'utf-8')))
        LCR_end_values.append(int(str(stdout_LCR[3*i + 2],'utf-8')))
    LCR_residues = []
    for i in range(0, len(LCR_start_values)):
        LCR_residues.extend(list(np.linspace(LCR_start_values[i], LCR_end_values[i], (LCR_end_values[i] - LCR_start_values[i] + 1) )))
    LCR_residues = sorted(list(set(LCR_residues)))
    LCR_sequence = ''
    for i in range(0, len(LCR_residues)):
        LCR_sequence = LCR_sequence + seq[int(LCR_residues[i]-1)]
    
    os.unlink(tmp_LCR.name)
    return len(LCR_residues), LCR_sequence

def extract_IDR(seq):
    tmp_IDR = tempfile.NamedTemporaryFile(delete=False)  
    with open(tmp_IDR.name, 'w') as f_IDR:
         f_IDR.write('>1\n' + str(seq))
    tmp_IDR.seek(0)
    
    out = subprocess.Popen(['python', 'lib/tools/iupred2a.py', str(tmp_IDR.name), 'long'], 
           stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    stdout_IDR, stderr_IDR = out.communicate()
    stdout_IDR = stdout_IDR.split()[40:]
    
    IDR_prob = []
    for i in range(0, int(len(stdout_IDR)/3)):
        IDR_prob.append(float(str(stdout_IDR[3*i + 2], 'utf-8')))
       
    TH1 = 0.5
    TH2 = 20
    IDR_residues = []
    current = 0
    for t in range(0, len(IDR_prob)):
        if IDR_prob[t] > TH1:
            current = current + 1
            if t == len(IDR_prob) - 1:
                if current > TH2:
                    IDR_residues.extend(range(t - current , t + 1))
        else:
            if current > TH2:
                IDR_residues.extend(range(t - current , t + 1))
                current = 0
            else:
                current = 0
    
    os.unlink(tmp_IDR.name)
    return len(IDR_residues)

def get_AA_count(seq, AA):
    if type(seq) == float:
        count = 0
    else:
        sequence = ProteinAnalysis(seq)
        count = sequence.count_amino_acids()[str(AA)]
    return count

def split_ngrams(seq, n):
    a, b, c = zip(*[iter(seq)]*n), zip(*[iter(seq[1:])]*n), zip(*[iter(seq[2:])]*n)
    str_ngrams = []
    for ngrams in [a,b,c]:
        x = []
        for ngram in ngrams:
            x.append("".join(ngram))
        str_ngrams.append(x)
    return str_ngrams

# ProtVec class definition (needed for unpickling the model)
class ProtVec(word2vec.Word2Vec):
    def __init__(self, fasta_fname=None, corpus=None, n=3, size=100, corpus_fname="corpus.txt",  
                 sg=1, window=25, min_count=1, workers=20):
        self.n = n
        self.size = size
        self.fasta_fname = fasta_fname

        if corpus is None and fasta_fname is None:
            raise Exception("Either fasta_fname or corpus is needed!")

        if fasta_fname is not None:
            print('Generate Corpus file from fasta file...')
            generate_corpusfile(fasta_fname, n, corpus_fname)
            corpus = word2vec.Text8Corpus(corpus_fname)

        word2vec.Word2Vec.__init__(self, corpus, size=size, sg=sg, window=window, 
                                   min_count=min_count, workers=workers)

    def to_vecs(self, seq):
        ngram_patterns = split_ngrams(seq, self.n)
        protvecs = []
        for ngrams in ngram_patterns:
            ngram_vecs = []
            for ngram in ngrams:
                try:
                    ngram_vecs.append(self.wv[ngram])
                except:
                    raise Exception("Model has never trained this n-gram: " + ngram)
            protvecs.append(sum(ngram_vecs))
        return protvecs
    
    def get_vector(self, seq):
        return sum(self.to_vecs(seq))

def load_protvec(model_fname):
    return word2vec.Word2Vec.load(model_fname)

def create_features_dict(seq, pv):
    """Create features as a dictionary instead of DataFrame"""
    features = {}
    
    # Basic properties
    seq_length = len(seq)
    features['Sequence_length'] = seq_length
    
    # LCR features
    lcr_length, lcr_seq = extract_LCR(seq)
    features['LCR_frac'] = lcr_length / seq_length if seq_length > 0 else 0
    features['LCR_length'] = len(lcr_seq)
    
    # Physical properties
    features['Hydrophobicity'] = hydrophobicity(seq)
    features['Shannon_entropy'] = Shannon_entropy(seq)
    features['IDR_frac'] = extract_IDR(seq) / seq_length if seq_length > 0 else 0
    features['pI'] = IP(seq).pi()
    
    # Amino acid counts in full sequence
    for aa in AA_array:
        features[f'AA_{aa}'] = get_AA_count(seq, aa)
    
    # AA type fractions in full sequence
    features['HB'] = sum(features[f'AA_{aa}'] for aa in Hydrophobic_AAs)
    features['HB_frac'] = features['HB'] / seq_length if seq_length > 0 else 0
    
    features['Polar'] = sum(features[f'AA_{aa}'] for aa in Polar_AAs)
    features['Polar_frac'] = features['Polar'] / seq_length if seq_length > 0 else 0
    
    features['Arom'] = sum(features[f'AA_{aa}'] for aa in Arom_AAs)
    features['Arom_frac'] = features['Arom'] / seq_length if seq_length > 0 else 0
    
    features['Cation'] = sum(features[f'AA_{aa}'] for aa in Cation_AAs)
    features['Cation_frac'] = features['Cation'] / seq_length if seq_length > 0 else 0
    
    features['Anion'] = sum(features[f'AA_{aa}'] for aa in Anion_AAs)
    features['Anion_frac'] = features['Anion'] / seq_length if seq_length > 0 else 0
    
    # LCR amino acid counts
    for aa in AA_array:
        features[f'AA_LCR_{aa}'] = get_AA_count(lcr_seq, aa)
    
    # AA type fractions in LCR
    features['HB_LCR'] = sum(features[f'AA_LCR_{aa}'] for aa in Hydrophobic_AAs)
    features['HB_LCR_frac'] = features['HB_LCR'] / features['LCR_length'] if features['LCR_length'] > 0 else 0
    
    features['Polar_LCR'] = sum(features[f'AA_LCR_{aa}'] for aa in Polar_AAs)
    features['Polar_LCR_frac'] = features['Polar_LCR'] / features['LCR_length'] if features['LCR_length'] > 0 else 0
    
    features['Arom_LCR'] = sum(features[f'AA_LCR_{aa}'] for aa in Arom_AAs)
    features['Arom_LCR_frac'] = features['Arom_LCR'] / features['LCR_length'] if features['LCR_length'] > 0 else 0
    
    features['Cation_LCR'] = sum(features[f'AA_LCR_{aa}'] for aa in Cation_AAs)
    features['Cation_LCR_frac'] = features['Cation_LCR'] / features['LCR_length'] if features['LCR_length'] > 0 else 0
    
    features['Anion_LCR'] = sum(features[f'AA_LCR_{aa}'] for aa in Anion_AAs)
    features['Anion_LCR_frac'] = features['Anion_LCR'] / features['LCR_length'] if features['LCR_length'] > 0 else 0
    
    # Word2Vec embeddings
    w2v_vector = pv.get_vector(seq)
    for i in range(200):
        features[i] = w2v_vector[i]
    
    return features

def predict_multiclass_dict(model, features_dict, feature_subset):
    """Predict using dictionary features"""
    X = np.array([[features_dict[key] for key in feature_subset]])
    proba = model.predict_proba(X)
    return proba[0, 0] + 0.5 * proba[0, 1]

def compute_deephase_score(seq, pv, phys_model, w2v_model):
    """Compute DeePhase score from a single sequence"""

    features = create_features_dict(seq, pv)
    phys_features = sorted(['Hydrophobicity', 'Shannon_entropy', 'LCR_frac', 
                           'IDR_frac', 'Arom_frac', 'Cation_frac'])

    w2v_features = sorted([str(i) for i in range(200)])
    
    phys_X = np.array([[features[key] for key in phys_features]])
    phys_proba = phys_model.predict_proba(phys_X)
    phys_score = phys_proba[0, 0] + 0.5 * phys_proba[0, 1]
    
    w2v_X = np.array([[features[int(key)] for key in w2v_features]])
    w2v_proba = w2v_model.predict_proba(w2v_X)
    w2v_score = w2v_proba[0, 0] + 0.5 * w2v_proba[0, 1]
    
    deephase_score = 0.5 * (phys_score + w2v_score)
    
    return deephase_score

class IDPOracle:
    def __init__(self, args, tokenizer):
        self.tokenizer = tokenizer
        
        # Load DeePhase models
        self.phys_model = pickle.load(open('lib/tools/Models/phys_multi.sav', 'rb'))
        self.w2v_model = pickle.load(open('lib/tools/Models/w2v_multi.sav', 'rb'))
        
        # Load ProtVec model
        self.pv = load_protvec('lib/tools/Embeddings/swissprot_size200_window25.model')
        
    def predict(self, tokens):
        """Predict DeePhase score for a single sequence"""
        seq = self.tokenizer.detokenize(tokens)
        score = compute_deephase_score(seq, self.pv, self.phys_model, self.w2v_model)
        return np.float32(score)

    def batch_predict(self, batch_tokens):
        """Predict DeePhase scores for a batch of sequences"""
        seqs = [self.tokenizer.detokenize(tokens) for tokens in batch_tokens]
        scores = []
        
        for i, seq in enumerate(seqs):
            score = compute_deephase_score(seq, self.pv, self.phys_model, self.w2v_model)
            scores.append(score)
        
        return np.array(scores, dtype=np.float32)
    
def get_oracle(args, tokenizer):
    return IDPOracle(args, tokenizer)


# Test code
if __name__ == "__main__":
    import argparse
    from tokenizer import get_tokenizer
    from args import get_default_args
    
    print("="*60)
    print("Testing DeePhase Oracle")
    print("="*60)
    
    # Create mock args
    args = get_default_args()
    
    # Get tokenizer
    tokenizer = get_tokenizer(args)
    
    # Create oracle
    oracle = IDPOracle(args, tokenizer)
    
    # Test sequences
    test_seqs = [
        'MEVEQEQRRRKVEAGRTKLAHFRQRKTKGDSSHSEKKTAKRKGSAVDASVQEESPVTKEDSALCGGGDICKSTSCDDTPDGAGGAFAAQPEDCDGEKREDLEQLQQKQVNDHPPEQCGMFTVSDHPPEQHGMFTVGDHPPEQRGMFTVSDHPPEQHGMFTVSDHPPEQRGMFTISDHQPEQRGMFTVSDHTPEQRGIFTISDHPAEQRGMFTKECEQECELAITDLESGREDEAGLHQSQAVHGLELEALRLSLSNMHTAQLELTQANLQKEKETALTELREMLNSRRAQELALLQSRQQHELELLREQHAREKEEVVLRCGQEAAELKEKLQSEMEKNAQIVKTLKEDWESEKDLCLENLRKELSAKHQSEMEDLQNQFQKELAEQRAELEKIFQDKNQAERALRNLESHHQAAIEKLREDLQSEHGRCLEDLEFKFKESEKEKQLELENLQASYEDLKAQSQEEIRRLWSQLDSARTSRQELSELHEQLLARTSRVEDLEQLKQREKTQHESELEQLRIYFEKKLRDAEKTYQEDLTLLQQRLQGAREDALLDSVEVGLSCVGLEEKPEKGRKDHVDELEPERHKESLPRFQAELEESHRHQLEALESPLCIQHEGHVSDRCCVETSALGHEWRLEPSEGHSQELPWVHLQGVQDGDLEADTERAARVLGLETEHKVQLSLLQTELKEEIELLKIENRNLYGKLQHETRLKDDLEKVKHNLIEDHQKELNNAKQKTELMKQEFQRKETDWKVMKEELQREAEEKLTLMLLELREKAESEKQTIINKFELREAEMRQLQDQQAAQILDLERSLTEQQGRLQQLEQDLTSDDALHCSQCGREPPTAQDGELAALHVKEDCALQLMLARSRFLEERKEITEKFSAEQDAFLQEAQEQHARELQLLQERHQQQLLSVTAELEARHQAALGELTASLESKQGALLAARVAELQTKHAADLGALETRHLSSLDSLESCYLSEFQTIREEHRQALELLRADFEEQLWKKDSLHQTILTQELEKLKRKHEGELQSVRDHLRTEVSTELAGTVAHELQGVHQGEFGSEKKTALHEKEETLRLQSAQAQPFHQEEKESLSLQLQKKNHQVQQLKDQVLSLSHEIEECRSELEVLQQRRERENREGANLLSMLKADVNLSHSERGALQDALRRLLGLFGETLRAAVTLRSRIGERVGLCLDDAGAGLALSTAPALEETWSDVALPELDRTLSECAEMSSVAEISSHMRESFLMSPESVRECEQPIRRVFQSLSLAVDGLMEMALDSSRQLEEARQIHSRFEKEFSFKNEETAQVVRKHQELLECLKEESAAKAELALELHKTQGTLEGFKVETADLKEVLAGKEDSEHRLVLELESLRRQLQQAAQEQAALREECTRLWSRGEATATDAEAREAALRKEVEDLTKEQSETRKQAEKDRSALLSQMKILESELEEQLSQHRGCAKQAEAVTALEQQVASLDKHLRNQRQFMDEQAAEREHEREEFQQEIQRLEGQLRQAAKPQPWGPRDSQQAPLDGEVELLQQKLREKLDEFNELAIQKESADRQVLMQEEEIKRLEEMNINIRKKVAQLQEEVEKQKNIVKGLEQDKEVLKKQQMSSLLLASTLQSTLDAGRCPEPPSGSPPEGPEIQLEVTQRALLRRESEVLDLKEQLEKMKGDLESKNEEILHLNLKLDMQNSQTAVSLRELEEENTSLKVIYTRSSEIEELKATIENLQENQKRLQKEKAEEIEQLHEVIEKLQHELSLMGPVVHEVSDSQAGSLQSELLCSQAGGPRGQALQGELEAALEAKEALSRLLADQERRHSQALEALQQRLQGAEEAAELQLAELERNVALREAEVEDMASRIQEFEAALKAKEATIAERNLEIDALNQRKAAHSAELEAVLLALARIRRALEQQPLAAGAAPPELQWLRAQCARLSRQLQVLHQRFLRCQVELDRRQARRATAHTRVPGAHPQPRMDGGAKAQVTGDVEASHDAALEPVVPDPQGDLQPVLVTLKDAPLCKQEGVMSVLTVCQRQLQSELLLVKNEMRLSLEDGGKGKEKVLEDCQLPKVDLVAQVKQLQEKLNRLLYSMTFQNVDAADTKSLWPMASAHLLESSWSDDSCDGEEPDISPHIDTCDANTATGGVTDVIKNQAIDACDANTTPGGVTDVIKNWDSLIPDEMPDSPIQEKSECQDMSLSSPTSVLGGSRHQSHTAEAGPRKSPVGMLDLSSWSSPEVLRKDWTLEPWPSLPVTPHSGALSLCSADTSLGDRADTSLPQTQGPGLLCSPGVSAAALALQWAESPPADDHHVQRTAVEKDVEDFITTSFDSQETLSSPPPGLEGKADRSEKSDGSGFGARLSPGSGGPEAQTAGPVTPASISGRFQPLPEAMKEKEVRPKHVKALLQMVRDESHQILALSEGLAPPSGEPHPPRKEDEIQDISLHGGKTQEVPTACPDWRGDLLQVVQEAFEKEQEMQGVELQPRLSGSDLGGHSSLLERLEKIIREQGDLQEKSLEHLRLPDRSSLLSEIQALRAQLRMTHLQNQEKLQHLRTALTSAEARGSQQEHQLRRQVELLAYKVEQEKCIAGDLQKTLSEEQEKANSVQKLLAAEQTVVRDLKSDLCESRQKSEQLSRSLCEVQQEVLQLRSMLSSKENELKAALQELESEQGKGRALQSQLEEEQLRHLQRESQSAKALEELRASLETQRAQSSRLCVALKHEQTAKDNLQKELRIEHSRCEALLAQERSQLSELQKDLAAEKSRTLELSEALRHERLLTEQLSQRTQEACVHQDTQAHHALLQKLKEEKSRVVDLQAMLEKVQQQALHSQQQLEAEAQKHCEALRREKEVSATLKSTVEALHTQKRELRCSLEREREKPAWLQAELEQSHPRLKEQEGRKAARRSAEARQSPAAAEQWRKWQRDKEKLRELELQRQRDLHKIKQLQQTVRDLESKDEVPGSRLHLGSARRAAGSDADHLREQQRELEAMRQRLLSAARLLTSFTSQAVDRTVNDWTSSNEKAVMSLLHTLEELKSDLSRPTSSQKKMAAELQFQFVDVLLKDNVSLTKALSTVTQEKLELSRAVSKLEKLLKHHLQKGCSPSRSERSAWKPDETAPQSSLRRPDPGRLPPAASEEAHTSNVKMEKLYLHYLRAESFRKALIYQKKYLLLLIGGFQDSEQETLSMIAHLGVFPSKAERKITSRPFTRFRTAVRVVIAILRLRFLVKKWQEVDRKGALAQGKAPRPGPRARQPQSPPRTRESPPTRDVPSGHTRDPARGRRLAAAASPHSGGRATPSPNSRLERSLTASQDPEHSLTEYIHHLEVIQQRLGGVLPDSTSKKSCHPMIKQ',  # FUS fragment
    ]
    
    print("\n--- Single Predictions ---\n")
    for i, seq in enumerate(test_seqs):
        tokens = tokenizer.tokenize(seq)
        score = oracle.predict(tokens)
        print(f"Seq {i+1} (len={len(seq)}): DeePhase = {score:.4f}")
    
    print("\n--- Batch Prediction ---\n")
    batch_tokens = [tokenizer.tokenize(seq) for seq in test_seqs]
    batch_scores = oracle.batch_predict(batch_tokens)
    print(f"Batch scores: {batch_scores}")
    print(f"Mean: {batch_scores.mean():.4f}, Std: {batch_scores.std():.4f}")
    
    print("\n" + "="*60)
    print("Test completed!")