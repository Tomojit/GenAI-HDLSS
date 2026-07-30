import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import rbf_kernel
from scipy.spatial.distance import pdist, cdist
from scipy.spatial.distance import jensenshannon
#from utilityDBN import *

# Install POT first if needed:
# pip install POT
import ot


def standardizeData(data,mu=[],std=[]):
    #data: a m x n matrix where m is the no of observations and n is no of features
    #if any(mu) == None and any(std) == None:
    if not(len(mu) and len(std)):
        #pdb.set_trace()
        std = np.std(data,axis=0)
        mu = np.mean(data,axis=0)
        std[np.where(std==0)[0]] = 1.0 #This is for the constant features.
        standardizeData = (data - mu)/std
        return mu,std,standardizeData
    else:
        standardizeData = (data - mu)/std
        return standardizeData
		
def unStandardizeData(data,mu,std):
    return std * data + mu

def load_matrix(path):
    """
    Loads a CSV file as a numeric matrix.
    Assumes rows = samples and columns = genes/features.
    """
    #df = pd.read_csv(path, index_col=0)
    #pdb.set_trace()
    df = pd.read_csv(path)
    return df.values.astype(float)
    
def subspaceSimilarity(X,Y):
    # Given two datasets X,Y it'll return the angles between the orthogonal subspaces computed from X and Y
    # Samples are arranged in rows, i.e., each row is a sample and each column is a feature/gene.
    
    # Mean substruct each column for both the data set
    X = X - np.mean(X,axis=0)
    Y = Y - np.mean(Y,axis=0)
    
    # transpose the data matrix to make each column a sample
    X = X.T
    Y = Y.T
    
    # Find the orthogonal basis for each datasets. I can use either QR or SVD.
    U_X,_,_ = np.linalg.svd(X,full_matrices=False)
    U_Y,_,_ = np.linalg.svd(Y,full_matrices=False)
    
    # Now run SVD to to calculate the angles between the optimal basis of X and Y
    _,sigmas,_ = np.linalg.svd(np.dot(U_X.T,U_Y))

    # Sanity check of singular values
    if np.any(sigmas < -1e-12) or np.any(sigmas > 1 + 1e-12):
        print("Warning: Singular values outside [0,1]")
    
    # The singular values(the vector EV) are sorted in descending order. These values are the cosine of the angle between the orthogonal subspaces of X and Y
    # The bigger the value of EV the smaller the angle
    # Return the average angle
    
    return np.mean(sigmas)

def kl_divergence(p, q):
    """Calculate KL divergence between two probability distributions"""
    epsilon = 1e-10
    p = p + epsilon
    q = q + epsilon
    return np.sum(p * np.log(p / q))
    
def js_divergence(p, q, epsilon=1e-3):
    p = np.asarray(p, dtype=float) + epsilon
    q = np.asarray(q, dtype=float) + epsilon
    p = p / p.sum()
    q = q / q.sum()
    return jensenshannon(p, q, base=np.e) ** 2

def symmetric_kl_divergence(Origdata, Synthdata, nBins=50):

    # Calculate pairwise distances within each dataset
    orig_distances = pdist(Origdata, metric='euclidean')
    synth_distances = pdist(Synthdata, metric='euclidean')
    
    # Create histograms of distances to get probabilities
    min_dist = min(orig_distances.min(), synth_distances.min())
    max_dist = max(orig_distances.max(), synth_distances.max())
    
    # Create bin edges which requires nBins+1 no of bins to create nBins
    bin_edges = np.linspace(min_dist,max_dist,nBins+1)
    
    # Calculate probability distributions
    p_orig_counts, _ = np.histogram(orig_distances, bins=bin_edges)
    p_synth_counts, _ = np.histogram(synth_distances, bins=bin_edges)
    p_orig = p_orig_counts / np.sum(p_orig_counts)
    p_synth = p_synth_counts / np.sum(p_synth_counts)
    
    # Compute KL Divergence
    kl_orig_to_synth = kl_divergence(p_orig, p_synth)
    kl_synth_to_orig = kl_divergence(p_synth, p_orig)

    #return js_divergence(p_orig, p_synth)
    #pdb.set_trace()
    return 0.5* (kl_orig_to_synth + kl_synth_to_orig)

def median_heuristic_sigma(X, Y):
    """
    Computes RBF kernel bandwidth using the median pairwise distance heuristic.
    """
    Z = np.vstack([X, Y])
    D = pairwise_distances(Z, metric="euclidean")
    D = D[np.triu_indices_from(D, k=1)]
    sigma = np.median(D[D > 0])
    if sigma <= 0 or np.isnan(sigma):
        sigma = 1.0
    #print("Min. Distance :",np.min(D))
    #print("Max. Distance :",np.max(D))
    #print("Mean Distance :",np.mean(D))
    #print("Median Distance :",np.median(D))
    #print("Standard deviations of Distance :",np.std(D))
    return sigma


def mmd_rbf(X, Y, sigma=None):
    """
    Biased MMD^2 with RBF kernel.
    Lower value means empirical and synthetic distributions are closer.
    """
    if sigma is None:
        sigma = median_heuristic_sigma(X, Y)
        #print('Sigma:',sigma)

    gamma = 1.0 / (2.0 * sigma ** 2)

    Kxx = rbf_kernel(X, X, gamma=gamma)
    Kyy = rbf_kernel(Y, Y, gamma=gamma)
    Kxy = rbf_kernel(X, Y, gamma=gamma)

    mmd2 = Kxx.mean() + Kyy.mean() - 2.0 * Kxy.mean()
    return max(mmd2, 0.0)


def wasserstein_pot(X, Y):
    """
    Multivariate Wasserstein / Earth Mover's Distance using POT.
    Lower value means empirical and synthetic distributions are closer.
    """
    n = X.shape[0]
    m = Y.shape[0]

    a = np.ones(n) / n
    b = np.ones(m) / m

    # Cost matrix between empirical and synthetic samples
    M = ot.dist(X, Y, metric="euclidean")

    return ot.emd2(a, b, M)


def load_IBD_Data():
    dataFile = '../Dataset/GDS1615Dataset.csv'
    df = pd.read_csv(dataFile,delimiter=',',header=0,low_memory=False)
    data = df.values[:-1,2:].astype('float64')
    
    return data.T

def evaluate_one_dataset(dataSetName, itrs, models):
    """
    dataSetName: a string either IBD, BreastCancer, or GEO_RNA
    itrs: no. of runs a generative model is run to create synthetic CSV files with different random seeds
    models: a lits of string
    """
    
    X = load_IBD_Data()

    if dataSetName in ['BreastCancer','IBD']:
        nFea = 22282
    elif dataSetName in ['RNA_GEO','RNA']:
        nFea = 6658
    X = X[:,:nFea]
    
    # Apply Z-transformation on the original data. All the generative models were trained on the Z-transformed training set.
    # So we are applying the Z-transformation on the empirical data before computing the metrics.
    # We don't need to apply Z-transformation on the synthetic data. 
    mu,sd,X = standardizeData(X)

    KLD_VAE = np.array([])
    KLD_scDiffusion = np.array([])
    KLD_LSH_GAN = np.array([])

    MMD_VAE = np.array([])
    MMD_scDiffusion = np.array([])
    MMD_LSH_GAN = np.array([])
    
    SSA_VAE = np.array([])
    SSA_scDiffusion = np.array([])
    SSA_LSH_GAN = np.array([])    
    
    Wass_Dist_VAE = np.array([])
    Wass_Dist_scDiffusion = np.array([])
    Wass_Dist_LSH_GAN = np.array([])

    #print("Dataset:",dataSetName,'Min. value: ',np.min(X),' Max. value: ',np.max(X),' after Z-transformation')
    print("Dataset:",dataSetName)
    
    for model in models:

        for i in range (itrs):
            #load the synthetic data
            #print('File no: ',i+1)
            fileName = './'+model+'/'+dataSetName+'_Synthetic_Data_Run'+str(i+1)+'.csv'	
            Y = load_matrix(fileName)
            Y = Y[:,:nFea]
            #print("Synthetic data. Run ",i+1,'Min. value: ',np.min(Y),' Max. value: ',np.max(Y))
            #pdb.set_trace()
            if X.shape[1] != Y.shape[1]:
                raise ValueError(f"Feature mismatch: real has {X.shape[1]} features, "f"synthetic has {Y.shape[1]} features in {syn_path}")
            if model == 'VAE':
                KLD_VAE = np.append(KLD_VAE, symmetric_kl_divergence(X, Y))
                MMD_VAE = np.append(MMD_VAE, mmd_rbf(X, Y))
                SSA_VAE = np.append(SSA_VAE, subspaceSimilarity(X, Y))
                Wass_Dist_VAE = np.append(Wass_Dist_VAE, wasserstein_pot(X, Y))
            elif model == 'scDiffusion':
                KLD_scDiffusion = np.append(KLD_scDiffusion, symmetric_kl_divergence(X, Y))
                MMD_scDiffusion = np.append(MMD_scDiffusion, mmd_rbf(X, Y))
                SSA_scDiffusion = np.append(SSA_scDiffusion, subspaceSimilarity(X, Y))
                Wass_Dist_scDiffusion = np.append(Wass_Dist_scDiffusion, wasserstein_pot(X, Y))
            else:
                KLD_LSH_GAN = np.append(KLD_LSH_GAN, symmetric_kl_divergence(X, Y))
                MMD_LSH_GAN = np.append(MMD_LSH_GAN, mmd_rbf(X, Y))
                SSA_LSH_GAN = np.append(SSA_LSH_GAN, subspaceSimilarity(X, Y))
                Wass_Dist_LSH_GAN = np.append(Wass_Dist_LSH_GAN, wasserstein_pot(X, Y)) 
            #results.append({"seed": i+1,"MMD_RBF": mmd_value,"Wasserstein": wass_value})
        
        print("\t Model:", model)
        if model == 'VAE':
            print("\t \t Average MMD: {:.5f}%".format(np.mean(MMD_VAE)),'+/-','{:.5f}'.format(np.std(MMD_VAE)))
            print("\t \t Average Wass Dist: {:.5f}%".format(np.mean(Wass_Dist_VAE)),'+/-','{:.5f}'.format(np.std(Wass_Dist_VAE)))
            print("\t \t Average SCS: {:.5f}%".format(np.mean(SSA_VAE)),'+/-','{:.5f}'.format(np.std(SSA_VAE)))
            print("\t \t Average KLD: {:.5f}%".format(np.mean(KLD_VAE)),'+/-','{:.5f}'.format(np.std(KLD_VAE)))
        elif model == 'scDiffusion':
            print("\t \t Average MMD: {:.5f}%".format(np.mean(MMD_scDiffusion)),'+/-','{:.5f}'.format(np.std(MMD_scDiffusion)))
            print("\t \t Average Wass Dist: {:.5f}%".format(np.mean(Wass_Dist_scDiffusion)),'+/-','{:.5f}'.format(np.std(Wass_Dist_scDiffusion)))
            print("\t \t Average SCS: {:.5f}%".format(np.mean(SSA_scDiffusion)),'+/-','{:.5f}'.format(np.std(SSA_scDiffusion)))
            print("\t \t Average KLD: {:.5f}%".format(np.mean(KLD_scDiffusion)),'+/-','{:.5f}'.format(np.std(KLD_scDiffusion)))
        else:
            print("\t \t Average MMD: {:.5f}%".format(np.mean(MMD_LSH_GAN)),'+/-','{:.5f}'.format(np.std(MMD_LSH_GAN)))
            print("\t \t Average Wass Dist: {:.5f}%".format(np.mean(Wass_Dist_LSH_GAN)),'+/-','{:.5f}'.format(np.std(Wass_Dist_LSH_GAN)))
            print("\t \t Average SCS: {:.5f}%".format(np.mean(SSA_LSH_GAN)),'+/-','{:.5f}'.format(np.std(SSA_LSH_GAN)))
            print("\t \t Average KLD: {:.5f}%".format(np.mean(KLD_LSH_GAN)),'+/-','{:.5f}'.format(np.std(KLD_LSH_GAN)))

if __name__ == "__main__":

    # -------------------------
    # Example usage
    # -------------------------
    
    dataSetName = "IBD"
    models = ['VAE','scDiffusion','LSH_GAN']
    itrs = 5
    evaluate_one_dataset(dataSetName, itrs, models)

    
    
