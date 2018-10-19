import numpy as np
import matplotlib.pyplot as plt


#layer
def layer(w,x,b):
    out = np.dot(x,w)+b.T
    return out

# the sigmoid function
def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

# derivative of sigmoid function
def derivative_sigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))

#forward path
def forward_path(W_mu,W_Var,W,b_mu,b_si,b,X,hidden_nodes):
    
    out_mu  = layer(W_mu,X,b_mu)
    out_logVar = layer(W_Var,X,b_si)
    mu = sigmoid(out_mu)
    logVar = sigmoid(out_logVar)
    e = np.random.normal(0,1,(len(X),hidden_nodes))
    z = e*np.exp(logVar/2.0)+mu
    out=layer(W,z,b)
    y_pred=sigmoid(out)
    
    return out_mu,mu,out_logVar,z,logVar,out,y_pred,e

    



hidden_nodes =3
# np.random.seed(42)

bias=np.load('b_latent3.npy')
W=np.load('W_latent3.npy')

#testing
print('\nTest:\n')

fig=plt.figure(figsize=(8, 8))

for i in range(1, 21):
    z = np.random.normal(0, 1, 3)
    out = layer(W,z,bias)
    img = sigmoid(out)*255
    img = img.reshape(28,28)
    fig.add_subplot(4, 5, i)
    plt.imshow(img, cmap='gray')

plt.show()