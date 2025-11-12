import os
import numba as nb
import numpy as np
import random

from re import findall
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

from scipy.integrate import quad
from MDAnalysis import Universe
from Bio import SeqIO, SeqUtils
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from polyleven import levenshtein

from scipy.stats import linregress, spearmanr 
from joblib import load
from tqdm import tqdm


def read_fasta(ffasta):
    records = SeqIO.to_dict(SeqIO.parse(ffasta, "fasta"))
    return records

def write_fasta(new_records,fout):
    """ seqs: list of sequences """
    if os.path.isfile(fout):
        records = []
        ids = []
        for record in SeqIO.parse(fout, "fasta"):
            records.append(record)
            ids.append(record.id)
        for record in new_records:
            if record.id not in ids: # avoid duplicates
                records.append(record)
        SeqIO.write(records, fout, "fasta")
    else:
        SeqIO.write(new_records, fout, "fasta")

def record_from_seq(seq,name):
    record = SeqRecord(
        Seq(seq),
        id=name,
        name='',
        description=''
    )
    return(record)

def get_qs(seq,flexhis=False,pH=7,residues=[]):
    """ charges and absolute charges vs. residues """
    qs = []
    # scaled charges
    qcoeff = 1.
    if len(residues) > 0: # residue charges provided
        for s in seq:
            if flexhis and s == 'H':
                q = qcoeff / ( 1 + 10**(pH-6) )
            else:
                q = residues.loc[s].q
            qs.append(q)
    else: # guess residue charges
        # histidines
        if flexhis:
            qhis = qcoeff / ( 1 + 10**(pH-6) )
        else:
            qhis = 0.
        # loop through sequence
        for s in seq:
            if s in ['R','K']:
                qs.append(qcoeff)
            elif s in ['E','D']:
                qs.append(-1.*qcoeff)
            elif s == 'H':
                qs.append(qhis)
            else:
                qs.append(0.)
    qs = np.array(qs)
    qs_abs = np.abs(qs)
    return qs, qs_abs

def frac_charges(qs):
    N = len(qs)
    fpos = np.sum(np.where(qs>0, 1, 0)) / N
    fneg = np.sum(np.where(qs<0, 1, 0)) / N
    return fpos, fneg

def patch_terminal_qs(qs,loc='both'):
    qsnew = qs.copy()
    qcoeff = 1.

    if loc in ['N','both']:
        qsnew[0] += qcoeff
    if loc in ['C','both']:
        qsnew[-1] -= qcoeff
    return qsnew

@nb.jit(nopython=True)
def calc_SCD(seq,charge_termini=False):
    """ Sequence charge decoration, eq. 14 in Sawle & Ghosh, JCP 2015 """
    qs, _ = get_qs_fast(seq)
    if charge_termini:
        qs[0] = qs[0] + 1.
        qs[-1] = qs[-1] - 1.
    N = len(seq)
    scd = 0.
    for idx in range(1,N):
        for jdx in range(0,idx):
            s = qs[idx] * qs[jdx] * (idx - jdx)**0.5
            scd = scd + s
    scd = scd / N
    return scd

def calc_SHD(seq,lambda_map,beta=-1.):
    """ Sequence hydropathy decoration, eq. 4 in Zheng et al., JPC Letters 2020"""
    N = len(seq)
    shd = 0.

    for idx in range(0, N-1):
        seqi = seq[idx]
        for jdx in range(idx+1, N):
            seqj = seq[jdx]
            s = lambda_map[(seqi,seqj)] * (jdx - idx)**beta
            shd = shd + s
    shd = shd / N
    return shd

def calc_mean_lambda(seq,residues):
    """ Mean hydropathy """
    lambdas_sum = 0.
    for idx, x in enumerate(seq):
        lambdas_sum += residues.lambdas[x]
    lambdas_mean = lambdas_sum / len(seq)
    return  lambdas_mean

def calc_aromatics(seq):
    """ Fraction of aromatics """
    seq = str(seq)
    N = len(seq)    
    rY = len(findall('Y',seq)) / N
    rF = len(findall('F',seq)) / N
    rW = len(findall('W',seq)) / N
    return rY, rF, rW

def calc_mw(fasta,residues=[]):
    seq = "".join(fasta)
    if len(residues) > 0:
        mw = 0.
        for s in seq:
            m = residues.loc[s,'MW']
            mw += m
    else:
        mw = SeqUtils.molecular_weight(seq,seq_type='protein')
    return mw

### SEQUENCE MANIPULATION
def shuffle_str(seq):
    l = list(seq)
    random.shuffle(l)
    seq_shuffled = "".join(l)
    return(seq_shuffled)

def split_seq(seq):
    """ split sequence in positive, negative, neutral residues """
    seqpos = []
    seqneg = []
    seqneu = []
    for s in seq:
        if s in ['K','R']:
            seqpos.append(s)
        elif s in ['D','E']:
            seqneg.append(s)
        else:
            seqneu.append(s)
    seqpos = shuffle_str(seqpos)
    seqneg = shuffle_str(seqneg)
    seqneu = shuffle_str(seqneu)
    return seqpos, seqneg, seqneu

@nb.jit(nopython=True)
def lj_potential(r,sig,eps):
    ulj = 4.*eps*((sig/r)**12 - (sig/r)**6)
    return ulj

@nb.jit(nopython=True)
def ah_potential(r,sig,eps,l,rc):
    if r <= 2**(1./6.)*sig:
        ah = lj_potential(r,sig,eps) - l * lj_potential(rc,sig,eps) + eps * (1 - l)
    elif r <= rc:
        ah = l * (lj_potential(r,sig,eps) - lj_potential(rc,sig,eps))
    else:
        ah = 0.
    return ah

def ah_scaled(r,sig,eps,l,rc):
    ah = ah_potential(r,sig,eps,l,rc)
    ahs = ah*4*np.pi*r**2
    return ahs

def make_ah_intgrl_map(residues,rc=2.0,eps = 0.2 * 4.184):
    ah_intgrl_map = {}
    for key0, val0 in residues.iterrows():
        sig0, l0 = val0['sigmas'], val0['lambdas']
        for key1, val1 in residues.iterrows():
            sig1, l1 = val1['sigmas'], val1['lambdas']
            sig, l = 0.5*(sig0+sig1), 0.5*(l0+l1)
            res = quad(lambda r: ah_scaled(r,sig,eps,l,rc), 2**(1./6.)*sig, rc)
            ah_intgrl_map[(key0,key1)] = res[0]
            ah_intgrl_map[(key1,key0)] = res[0]
    return ah_intgrl_map

def make_lambda_map(residues):
    lambda_map = {}
    for key0, val0 in residues.iterrows():
        l0 = val0['lambdas']
        for key1, val1 in residues.iterrows():
            l1 = val1['lambdas']
            l = l0+l1
            lambda_map[(key0,key1)] = l
            lambda_map[(key1,key0)] = l
    return lambda_map

def calc_ah_ij(seq,ah_intgrl_map):
    U = 0.
    seq = list(seq)
    N = len(seq)
    for idx in range(N):
        seqi = seq[idx]
        for jdx in range(idx,N):
            seqj = seq[jdx]
            ahi = ah_intgrl_map[(seqi,seqj)]
            U += ahi
    U /= (N * (N-1) / 2. + N)
    return U


@nb.jit(nopython=True)
def check_dmax(seq,dmax,seqmax):
    qs, _ = get_qs_fast(seq)
    d = calc_delta(qs)
    if d > dmax:
        return seq, d
    else:
        return seqmax, dmax

@nb.jit(nopython=True)
def calc_case0(seqpos,seqneg,seqneu):
    seqmax = ''
    dmax = 0.
    N = len(seqpos) + len(seqneg) + len(seqneu)
    if len(seqpos) == 0:
        seqcharge = seqneg[:]
    elif len(seqneg) == 0:
        seqcharge = seqpos[:]
    if len(seqneu) > len(seqcharge):
        for pos in range(0, N - len(seqcharge) + 1):
            seqout = ''
            seqout += seqneu[:pos]
            seqout += seqcharge
            seqout += seqneu[pos:]
            seqmax, dmax = check_dmax(seqout,dmax,seqmax)
    else:
        for pos in range(0, N - len(seqneu) + 1):
            seqout = ''
            seqout += seqcharge[:pos]
            seqout += seqneu
            seqout += seqcharge[pos:]
            seqmax, dmax = check_dmax(seqout,dmax,seqmax)
    return seqmax

@nb.jit(nopython=True)
def calc_case1(seqpos,seqneg,seqneu):
    seqmax = ''
    dmax = 0.
    N = len(seqpos) + len(seqneg) + len(seqneu)
    if len(seqpos) > len(seqneg):
        for pos in range(0, N - len(seqneg) + 1):
            seqout = ''
            seqout += seqpos[:pos]
            seqout += seqneg
            seqout += seqpos[pos:]
            seqmax, dmax = check_dmax(seqout,dmax,seqmax)
    else:
        for neg in range(0, N - len(seqpos) + 1):
            seqout = ''
            seqout += seqneg[:neg]
            seqout += seqpos
            seqout += seqneg[neg:]
            seqmax, dmax = check_dmax(seqout,dmax,seqmax)
    return seqmax

@nb.jit(nopython=True)
def calc_case2(seqpos,seqneg,seqneu):
    seqmax = ''
    dmax = 0.
    for startNeuts in range(0, 7):
        for endNeuts in range(0, 7):
            startBlock = seqneu[:startNeuts]
            endBlock = seqneu[startNeuts:startNeuts+endNeuts]
            midBlock = seqneu[startNeuts+endNeuts:]

            seqout = ''
            seqout += startBlock
            seqout += seqpos
            seqout += midBlock
            seqout += seqneg
            seqout += endBlock
            seqmax, dmax = check_dmax(seqout,dmax,seqmax)
    return seqmax

@nb.jit(nopython=True)
def calc_case3(seqpos,seqneg,seqneu):
    seqmax = ''
    dmax = 0.
    for midNeuts in range(0, len(seqneu)+1):
        midBlock = seqneu[:midNeuts]
        for startNeuts in range(0, len(seqneu) - midNeuts + 1):
            startBlock = seqneu[midNeuts:midNeuts+startNeuts]
            seqout = ''
            seqout += startBlock
            seqout += seqpos
            seqout += midBlock
            seqout += seqneg
            seqout += seqneu[midNeuts+startNeuts:]
            seqmax, dmax = check_dmax(seqout,dmax,seqmax)
    return seqmax

def construct_deltamax(seq):
    seqpos, seqneg, seqneu = split_seq(seq)

    if (len(seqpos) == 0) or (len(seqneg) == 0):
        seqmax = calc_case0(seqpos,seqneg,seqneu)
    elif len(seqneu) == 0:
        seqmax = calc_case1(seqpos,seqneg,seqneu)
    elif len(seqneu) >= 18:
        seqmax = calc_case2(seqpos,seqneg,seqneu)
    else:
        seqmax = calc_case3(seqpos,seqneg,seqneu)
    return seqmax

def calc_kappa_manual(seq):
    qs, qs_abs = get_qs_fast(seq)
    if np.sum(qs_abs) == 0:
        return -1
    else:
        seqpos, seqneg, seqneu = split_seq(seq)
        if (len(seqneu) == 0): 
            if (len(seqneg) == 0) or (len(seqpos) == 0):
                return -1

    delta = calc_delta(qs)
    
    seq_max = construct_deltamax(seq)
    qs_max, _ = get_qs_fast(seq_max)
    delta_max = calc_delta(qs_max)
    
    kappa = delta / delta_max
    return kappa

@nb.jit(nopython=True)
def calc_delta(qs):
    d5 = calc_delta_form(qs,window=5)
    d6 = calc_delta_form(qs,window=6)
    return (d5 + d6) / 2.

@nb.jit(nopython=True)
def calc_delta_form(qs,window=5):
    sig_m = calc_sigma(qs)

    nw = len(qs)-window + 1
    sigs = np.zeros((nw))

    for idx in range(0,nw):
        q_window = qs[idx:idx+window]
        sigs[idx] = calc_sigma(q_window)
    delta = np.sum((sigs-sig_m)**2) / nw
    return delta

@nb.jit(nopython=True)
def frac_charges(qs):
    N = len(qs)
    fpos = 0.
    fneg = 0.
    for idx in range(N):
        if qs[idx] > 0:
            fpos = fpos + 1.
        elif qs[idx] < 0:
            fneg = fneg + 1.
    fpos = fpos / N
    fneg = fneg / N
    return fpos, fneg

@nb.jit(nopython=True)
def calc_sigma(qs):
    fpos, fneg = frac_charges(qs)
    ncpr = fpos-fneg
    fcr = fpos+fneg
    if fcr == 0:
        return 0.
    else:
        return ncpr**2 / fcr

@nb.jit(nopython=True)
def get_qs_fast(seq):
    """ charges and absolute charges vs. residues """
    qs = np.zeros(len(seq))
    qs_abs = np.zeros(len(seq))

    # loop through sequence
    for idx in range(len(seq)):
        if seq[idx] in ['R','K']:
            qs[idx] = 1.
            qs_abs[idx] = 1.
        elif seq[idx] in ['E','D']:
            qs[idx] = -1.
            qs_abs[idx] = 1.
        else:
            qs[idx] = 0.
            qs_abs[idx] = 0.
    return qs, qs_abs

class SeqFeatures:
    def __init__(self,seq,residues=None,charge_termini=False,calc_dip=False,
    nu_file=None,ah_intgrl_map=None,lambda_map=None):
        self.seq = seq
        self.N = len(seq)
        self.qs, self.qs_abs = get_qs_fast(seq)
        if charge_termini:
            self.qs[0] += 1.
            self.qs[-1] -= 1.
        self.charge = np.sum(self.qs)
        self.fpos, self.fneg = frac_charges(self.qs)
        self.ncpr = self.charge / self.N
        self.fcr = self.fpos+self.fneg
        self.scd = calc_SCD(seq,charge_termini=charge_termini)
        self.rY, self.rF, self.rW = calc_aromatics(seq)
        self.faro = self.rY + self.rF + self.rW

        if residues is not None:
            self.mean_lambda = calc_mean_lambda(seq,residues)
            if lambda_map is None:
                lambda_map = make_lambda_map(residues)
            self.shd = calc_SHD(seq,lambda_map,beta=-1.)
            self.mw = calc_mw(seq,residues=residues)
            if ah_intgrl_map is None:
                ah_intgrl_map = make_ah_intgrl_map(residues)
            self.ah_ij = calc_ah_ij(seq,ah_intgrl_map)

        if nu_file is not None:
            self.kappa = calc_kappa_manual(seq)
            if self.kappa == -1: # no charges
                self.kappa = 0.
            feats_for_nu = [self.scd, self.shd, self.kappa, self.fcr, self.mean_lambda]
            model_nu = load(nu_file)
            X_nu = np.reshape(np.array(feats_for_nu),(1,-1))
            self.nu_svr = model_nu.predict(X_nu)[0]

def calc_box(N):
    if N > 350:
        box = [25., 25., 300.] # nm
    else:
        box = [20., 20., 200.] # nm
    return box

def calc_nprot_slab(N,box,pbeads=90):
    ''' pbeads: beads per nm^2 in xy directions '''
    beads = pbeads * box[0] * box[1]
    nprot = int(beads / N)
    return nprot

def calc_mw(fasta,residues=[]):
    seq = "".join(fasta)
    if len(residues) > 0:
        mw = 0.
        for s in seq:
            m = residues.loc[s,'MW']
            mw += m
    else:
        mw = SeqUtils.molecular_weight(seq,seq_type='protein')
    return mw

def bin_data(xs,ys,nbins,drange=None):
    """ bin data ys in xs bins, based on numpy.histogram_bin_edges """
    if drange == None:
        xmin, xmax = np.min(xs), np.max(xs)
    else:
        xmin, xmax = drange[0], drange[1]
    bins = np.linspace(xmin,xmax,nbins+1)
    y_binned = [[] for _ in range(nbins)]
    for x, y in zip(xs,ys):
        if x <= bins[0]:
            y_binned[0].append(y)
        elif x >= bins[-1]:
            y_binned[nbins-1].append(y)
        else:
            for idx in range(nbins):
                if x >= bins[idx] and x < bins[idx+1]:
                    y_binned[idx].append(y)
    return bins, y_binned

def name_to_index(df,name):
    return df.index[df['seq_name'] == name]

def predict_single(X,model):
    y = model.predict(X)
    return y

def predict_multimodels(X,models):
    ys = np.zeros(len(models))
    for idx, model in enumerate(models):
        ys[idx] = predict_single(X,model)
    return ys

def X_from_seq(seq,feats,residues=[],charge_termini=True,nu_file=None,ah_intgrl_map=None,lambda_map=None,
              seq_feats=None):
    X = []
    if seq_feats == None:
        seq_feats = SeqFeatures(seq,residues=residues,charge_termini=charge_termini,nu_file=nu_file,
                                       ah_intgrl_map=ah_intgrl_map,lambda_map=lambda_map)
    for feat in feats:
        X.append(getattr(seq_feats,feat))
    X = np.array(X)
    X = np.reshape(X,(1,-1))
    return X

def makeXy(df,feats,target=None):
    """ Make feature (X) -- target (y) pairs from dataframe """
    X, y, X_keys = [], [], []
    
    for key, val in df.iterrows():
        features = []

        for feat in feats: # feats is a list of string
            features.append(val[feat]) # features is a list of values

        X.append(features)
        X_keys.append(key)

        if target is not None:
            target_sim = val[target]
            y.append(target_sim)

    X = np.array(X)
    if target is not None:
        y = np.array(y)
        return X, y, X_keys
    else:
        return X, X_keys

class AttrSetter:
    def __init__(self,**kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)

class Model:
    def __init__(self,**kwargs):
        self.mltype = kwargs.get('mltype','svr')
        self.layers = kwargs.get('layers',(5,5))
        self.alpha = kwargs.get('alpha',10)
        self.C = kwargs.get('C',10)
        self.epsilon = kwargs.get('epsilon',1e-2)
        self.ptrain = kwargs.get('ptrain',0.8)
        self.ncrossval = kwargs.get('ncrossval',50)

    @staticmethod
    def split_data(X,y,X_keys,ptrain):
        """ Split data into train and test set and return corresponding indices """
        nsamp = len(X)
        if nsamp != len(y):
            raise ValueError("X and y size is not equal!")
    
        random_idx = np.random.choice(nsamp, size=nsamp, replace=False)
        ntrain = int(nsamp * ptrain)
        train_idx = random_idx[:ntrain]
        test_idx = random_idx[ntrain:]
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        X_train_keys = [X_keys[idx] for idx in train_idx]
        X_test_keys = [X_keys[idx] for idx in test_idx]
    
        return X_train, X_test, y_train, y_test, X_train_keys, X_test_keys
    
    @staticmethod
    def calc_statistics(y, ypred, verbose=True):
        # Pearson
        fit = linregress(y, ypred)
        rp = fit.rvalue

        # Spearmanx
        rs = spearmanr(y, ypred).statistic

        # Root mean squared deviation
        rmsd = np.sqrt(np.mean((y - ypred)**2))

        if verbose:
            print(f'Pearson: {rp:.3f}, Spearman: {rs:.3f}, RMSD: {rmsd:.3f}')
        return rp, rs, rmsd

    @staticmethod
    def calc_statistics_multimodel(y, ypred, verbose=True):
        nmodels = len(ypred)
        
        rp = np.zeros((nmodels))
        rs = np.zeros((nmodels))
        rmsd = np.zeros((nmodels))
        
        for idx, yp in enumerate(ypred):
            rp[idx], rs[idx], rmsd[idx] = Model.calc_statistics(y, yp, verbose=verbose)
        return rp, rs, rmsd

    def predict(self,X):
        ypred = np.zeros((self.ncrossval, len(X)))
        for idx, crossval in enumerate(self.crossvals):
            ypred[idx] = crossval.mlmodel.predict(X)
        return ypred

    def train(self,X,y,X_keys,**kwargs):
        self.models = []
        self.crossvals = []
        verbose = kwargs.get('verbose',True)

        for idx in range(self.ncrossval):
            X_train, X_test, y_train, y_test, X_train_keys, X_test_keys = self.split_data(X,y,X_keys,self.ptrain)

            if self.mltype == 'svr':
                mlmodel = make_pipeline(StandardScaler(), SVR(C=self.C, epsilon=self.epsilon))
            elif self.mltype == 'mlp':
                mlmodel = make_pipeline(
                    StandardScaler(),
                    MLPRegressor(
                        hidden_layer_sizes=self.layers,activation='tanh',
                        solver='lbfgs',max_iter=10000,alpha=self.alpha),
                )
            mlmodel.fit(X_train, y_train)

            ypred_train = mlmodel.predict(X_train)
            ypred_test = mlmodel.predict(X_test)

            rp, rs, rmsd = self.calc_statistics(y_test, ypred_test, verbose=verbose)

            self.crossvals.append(AttrSetter(
                X_train = X_train,
                X_test = X_test,
                y_train = y_train,
                y_test = y_test,
                X_train_keys = X_train_keys,
                X_test_keys = X_test_keys,
                mlmodel = mlmodel,
                ypred_train = ypred_train,
                ypred_test = ypred_test,
                rp = rp,
                rs = rs,
                rmsd = rmsd
            ))
        self.rp_mean = np.mean([cval.rp for cval in self.crossvals])
        self.rs_mean = np.mean([cval.rs for cval in self.crossvals])
        self.rmsd_mean = np.mean([cval.rmsd for cval in self.crossvals])

def add_seq(df,records):
    for key, val in df.iterrows():
        add = False
        if 'fasta' not in df.keys():
            add = True
        elif not isinstance(val['fasta'], str):
            add = True
        if add:
            if key in records:
                df.loc[key,'fasta'] = str(records[key].seq)
            else:
                print(f'Could not find {key} in records')
    return df

def add_seq_from_df_full(df,df_full):
    for name, val in df.iterrows():
        if not isinstance(val['fasta'], str):
            key = name_to_index(df_full,name)[0]
            print(key)
            df.loc[name,'fasta'] = df_full.loc[key,'fasta']
    return df

def calc_cdil_mgml(N,mw,val):
    box = calc_box(N)
    nprot = calc_nprot_slab(N,box)

    if val['dG'] == 0.:
        cdil = nprot / (box[0] * box[1] * box[2]) / 6.022e23 * 1e24 * 1e3 # mM, bulk
    else:
        cdil = val['cdil']

    if cdil < np.exp(-7.5):
        cdil = np.exp(-7.5)
        logcdil = -7.5
    else:
        logcdil = np.log(cdil)

    cdil_mgml = cdil * mw  / 1e3

    if cdil_mgml < np.exp(-4):
        cdil_mgml = np.exp(-4)
        logcdil_mgml = -4.
        log10cdil_mgml = -1.737 # np.log10(np.exp(-4))
    else:
        logcdil_mgml = np.log(cdil_mgml)
        log10cdil_mgml = np.log10(cdil_mgml)
    
    return cdil, logcdil, cdil_mgml, logcdil_mgml, log10cdil_mgml

def convert_cdil(df):
    for key, val in df.iterrows():
        seq = val['fasta']
        N = len(seq)
        # calc mw
        mw = calc_mw(seq)
        df.loc[key,'mw'] = mw
        # convert cdil
        cdil, logcdil, cdil_mgml, logcdil_mgml, log10cdil_mgml = calc_cdil_mgml(N,mw,val)
        df.loc[key,'cdil'] = cdil
        df.loc[key,'logcdil'] = logcdil
        df.loc[key,'cdil_mgml'] = cdil_mgml
        df.loc[key,'logcdil_mgml'] = logcdil_mgml
        df.loc[key,'log10cdil_mgml'] = log10cdil_mgml
    return df

def add_features(df,feats,charge_termini=True,residues=None,nu_file=None,verbose=False,
ah_intgrl_map=None, lambda_map=None):
    for key, val in tqdm(df.iterrows(),total=len(df)):
        seq = val['fasta']
        if verbose:
            print(key, seq)
        N = len(seq)
        seq_feats = SeqFeatures(seq,charge_termini=charge_termini,
                                            residues=residues,nu_file=nu_file,
                                            ah_intgrl_map=ah_intgrl_map,lambda_map=lambda_map)
        for feat in feats:
            df.loc[key,feat] = getattr(seq_feats,feat)
    return df


def is_similar(seq_a, seq_b, dist_type="edit", threshold=0):
    if dist_type == "edit":
        return edit_dist(seq_a, seq_b) < threshold
    return False


def edit_dist(seq1, seq2):
    return levenshtein(seq1, seq2) / 1