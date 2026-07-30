from __future__ import print_function
import pdb
import argparse
import torch
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.nn import functional as F
#from torchvision import datasets, transforms
#from torchvision.utils import save_image
import torch.utils.data
import torch.utils.data as Data
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=1024, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

#torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}


def standardizeData(data,mu=None,std=None):
	#data: a m x n matrix where m is the no of observations and n is no of features
	#if any(mu) == None and any(std) == None:
	if mu is None or std is None:
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

class VAE(nn.Module):
    def __init__(self,kld_weight=0.1):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(dataDim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc31 = nn.Linear(256, 50)
        self.fc32 = nn.Linear(256, 50)
        self.fc4 = nn.Linear(50, 256)
        self.fc5 = nn.Linear(256, 512)
        self.fc6 = nn.Linear(512, dataDim)
        self.kld_Weight = kld_weight

    def encode(self, x):
        h1 = F.leaky_relu(self.fc1(x))
        h2 = F.leaky_relu(self.fc2(h1))
        
        return self.fc31(h2), self.fc32(h2)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h4 = F.leaky_relu(self.fc4(z))
        h5 = F.leaky_relu(self.fc5(h4))
        
        return self.fc6(h5)

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, dataDim))
        # Clamp logvar to prevent numerical instability
        logvar = torch.clamp(logvar, min=-20.0, max=20.0)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def loadIBDData():
	
	dataFile = '../Dataset/GDS1615Dataset.csv'
	#pdb.set_trace()
	df = pd.read_csv(dataFile,delimiter=',',header=0,low_memory=False)
	probe_ids = df['ID_REF'].iloc[:-1].values
	gene_ids = df['IDENTIFIER'].iloc[:-1].values
	sample_ids = df.columns[2:]
	#D = np.array(D)
	trData = df.values[:-1,2:].astype('float64')
	trData = trData.T
	L = df.values[-1,2:]
	#pdb.set_trace()
	trLabels = np.zeros([len(L)])
	trLabels[np.where(L=='Ulcerative Colitis')[0]] = 1
	trLabels[np.where(L=='Crohn\'s Disease')[0]] = 2
	return trData

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    MSE = 0.5*((x - recon_x)**2).sum()
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KLD = model.kld_Weight*KLD
    
    return MSE, KLD, MSE+ KLD


def train(epoch):
    model.train()
    train_loss = 0
    mse_loss,kld_loss, total_loss = [],[],[]
    for batch_idx, data in enumerate(train_loader):
        data = data[0].to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        l1,l2,l3 = loss_function(recon_batch, data, mu, logvar)
        l3.backward()
        optimizer.step()
        mse_loss.append(l1.item())
        kld_loss.append(l2.item())
        total_loss.append(l3.item())

    mse_loss,kld_loss, total_loss = np.array(mse_loss),np.array(kld_loss), np.array(total_loss)
    losses_total.append(np.mean(total_loss))
    losses_mse.append(np.mean(mse_loss))
    losses_kld.append(np.mean(kld_loss))
    if ((epoch+1) % (args.epochs*0.1)) == 0:
        print ('Epoch [{}/{}], MSE loss: {:.6f}, KL Divergence loss: {:.6f}, Total loss: {:.6f}'.format(epoch+1, args.epochs, losses_mse[-1],losses_kld[-1], losses_total[-1]))


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                      recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                         'VAE/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

def runPCA(data):
	#data = [m x n] array where m: no of samples and n: no of features.
	#p: no. of principle components for dim. reduction
	#pdb.set_trace()
	data = data.astype('float64')
	data = data - np.mean(data,axis=0) # substruct the mean.
	cov = np.dot(data.T,data)
	eVals,eVecs = np.linalg.eigh(cov)

	#the eigen values are returned in ascending order. Need to flip the eigen values and eigrn vectors
	eVecs_flip = np.flip(eVecs,axis=1)
	eVals_flip = np.flip(eVals)
	return eVals_flip,eVecs_flip

def Fast_PCA(X,m):
	#X = [n x d] array where n: no of samples and d: no of features.
	X = X.astype('float64')
	X = X - np.mean(X,axis=0) # substruct the mean.
	X = X.T
	M = np.dot(X.T,X)
	
	eVals,eVecs = np.linalg.eigh(M)
	#the eigen values are returned in ascending order. Need to flip the eigen values and eigen vectors
	eVecs_flip = np.flip(eVecs,axis=1)
	eVals_flip = np.abs(np.flip(eVals))
	sigma = np.diag(np.power(eVals_flip,-0.5))
	U = np.dot(X,np.dot(eVecs_flip,sigma))
	#Now project the data on the first m eigen vector
	projectedData = np.dot(U[:,:m].T,X)
	return projectedData.T,eVals_flip,U

if __name__ == "__main__":
	    
    # Load IBD/Crohn Data
    X_ = loadIBDData()

    # Apply Z-transformation on the data
    mu,std,X = standardizeData(X_)

    sampleCnt = len(X)
    dataDim = np.shape(X)[1]

    #Prepare data for torch
    trDataTorch = Data.TensorDataset(torch.from_numpy(X).float())
    train_loader = Data.DataLoader(dataset=trDataTorch,batch_size=args.batch_size,shuffle=True)
    train_loader1 = Data.DataLoader(dataset=trDataTorch,batch_size=100000,shuffle=False)#load the full data
    
    kld_weight = 0.5
    model = VAE(kld_weight).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    losses_total,losses_mse,losses_kld = [],[],[]

    for epoch in range(1, args.epochs + 1):
        train(epoch)
    #pass the training data thru the network
    with torch.no_grad():
        for i, data in enumerate(train_loader1):
            data = data[0].to(device)
            recon_X, _, _ = model(data)
            
    recon_X = recon_X.detach().cpu().numpy()

    #merge reconstructed samples with the original samples and run PCA
    merged_X = np.vstack((X,recon_X))
    _,eVals,eVecs = Fast_PCA(merged_X,2)
    X_2D_pca = np.dot(eVecs[:,:2].T,merged_X.T).T
    
    #plot the merged data
    plt.scatter(X_2D_pca[:sampleCnt,0],X_2D_pca[:sampleCnt,1],color='r',marker='o',label='Original Data')
    plt.scatter(X_2D_pca[sampleCnt:2*sampleCnt,0],X_2D_pca[sampleCnt:2*sampleCnt,1],color='b',marker='*',label='Synthetic Data')
    plt.legend()
    plt.show()
    
    # Ask the user whether to save the reconstructed data
    save_file = input("Do you want to save the reconstructed data to a CSV file? (y/n): ").strip().lower()

    if save_file in ("y", "yes"):
        pd.DataFrame(recon_X).to_csv(f"VAE_synthetic_data_epoch_{epoch}.csv", index=False)
        
        print("Reconstructed data saved to 'reconstructed_breast_cancer.csv'.")
    else:
        print("File was not saved.")

