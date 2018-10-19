
import numpy as np
X_train = np.load("DATA/3.npy")
np.random.seed(42)


# initialisation of the weights
def weights(in_dim,hidden_nodes):

    W_mean = np.random.normal(0,1e-5,(in_dim,hidden_nodes)) # Mean
    W_covar = np.random.normal(0,1e-5,(in_dim,hidden_nodes)) # log covariance
    W_latent = np.random.normal(0,1e-5,(hidden_nodes,in_dim)) # corresponding to latent part
    b_mean=np.random.normal(0,1e-5,(hidden_nodes,1))
    b_covar=np.random.normal(0,1e-5,(hidden_nodes,1))
    b_latent=np.random.normal(0,1e-5,(in_dim,1))
    return W_mean,W_covar,W_latent,b_mean,b_covar,b_latent

def layer(w,x,b):
    out = np.dot(x,w)+b.T
    return out
# the sigmoid function
def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

# derivative of sigmoid function
def derivative_sigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))


def backprop(delta,z):
    bias=np.sum(delta,axis=0)
    bias=bias.reshape(len(bias),1)
    sigma = np.matmul(z.T,delta)
    return sigma,bias

def sigmaDelta(delta,W,z):
    delta_l1 = np.matmul(delta,W.T)
    sm = derivative_sigmoid(z)*delta_l1
    return sm


print(X_train[0].shape)
flatArr = X_train[0].shape
print(flatArr)
epochs = 10
in_dim = flatArr[0]*flatArr[1]
X=X_train.reshape(len(X_train),flatArr[0]*flatArr[1])
print(X.shape)

# Normalise
X=X/255.0

#parameters
hidden_nodes=3
learning_rate = 5e-4
Lambda=1
nsamples = len(X)
epochs = 60
batchsize=600
num_iter = (int)(nsamples/batchsize)

# weights

W_mean,W_covar,W_latent,b_mean,b_covar,b_latent=weights(in_dim,hidden_nodes)


for i in range(epochs):
    loss = 0       
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X = X[indices] 
    for j in range(num_iter):
        sigmaMean = np.zeros(W_mean.shape)
        sigmaVar  = np.zeros(W_covar.shape)
        sigma     = np.zeros(W_latent.shape)
        biasMean  = np.zeros(b_mean.shape)
        biasVar   = np.zeros(b_covar.shape)
        bias      = np.zeros(b_latent.shape)
    
    # forward pass
        X_batch = X[j*batchsize:(j+1)*batchsize]
        mean = np.zeros(len(b_mean))
        var = np.eye(len(b_mean))
        e = np.random.multivariate_normal(mean,var,batchsize)
        e = e.reshape(len(e),len(e[0]))
        out_1 = layer(W_mean,X_batch,b_mean)
        out_2 = layer(W_covar,X_batch,b_covar)
        zmean = sigmoid(out_1)
        logVar = sigmoid(out_2)    
        z = zmean + np.exp(0.5*logVar)*e
        out_3 = layer(W_latent,z,b_latent)
        y_pred = sigmoid(out_3)
    
    # the loss
        loss1 = np.sum((y_pred-X_batch)**2)
        loss2 = Lambda*0.5*np.sum(np.exp(logVar)+(zmean)**2-1-logVar)
        loss = loss+loss1+loss2
    
    # backpropagation

        delta = 2*(y_pred-X_batch)*derivative_sigmoid(out_3)
        tot,bia = backprop(delta,z)
        sigma = sigma + tot
        bias = bias + bia
    
    # mean
        sm = sigmaDelta(delta,W_latent,out_1)
        sm_mean = sm + Lambda*zmean*derivative_sigmoid(out_1)
        tot,bia = backprop(sm_mean,X_batch)
        sigmaMean = sigmaMean + tot
        biasMean = biasMean +bia
    
    # covar  
        sm1 = sigmaDelta(delta,W_latent,out_2)
        sm_covar = 0.5*sm1*e*np.exp(0.5*logVar) + Lambda*0.5*(np.exp(logVar)-1)*derivative_sigmoid(out_2)
        tot,bia = backprop(sm_covar,X_batch)
        sigmaVar = sigmaVar + tot
        biasVar = biasVar + bia
    
    # update

        b_latent = b_latent - learning_rate*bias
        b_mean = b_mean - learning_rate*biasMean
        b_covar = b_covar - learning_rate*biasVar
        W_latent = W_latent - learning_rate*sigma
        W_mean = W_mean - learning_rate*sigmaMean
        W_covar = W_covar - learning_rate*sigmaVar

    print("Epoch: "+str(i) +"    "+"loss: "+str(loss))


np.save('b_latent3.npy',b_latent)
np.save('b_covar3.npy',b_covar)
np.save('b_mean3.npy',b_mean)
np.save('W_latent3.npy',W_latent)
np.save('W_covar3.npy',W_covar)
np.save('W_mean3.npy',W_mean)


