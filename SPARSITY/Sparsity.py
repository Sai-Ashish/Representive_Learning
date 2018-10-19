import numpy as np
import matplotlib.pyplot as plt


X_train = np.load("IMAGES.npy")
np.random.seed(42)

# initialisation of the weights
def weights(inputDim,hidden_nodes):

	W1=np.random.normal(1e-4,1,(inputDim,hidden_nodes))
	W2=np.random.normal(1e-4,1,(hidden_nodes,inputDim))
	b1=np.random.normal(1e-4,1,(hidden_nodes,1))
	b2=np.random.normal(1e-4,1,(inputDim,1))
	return W1,W2,b1,b2

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
def forward_path(W1,W2,b1,b2,X):
	out_1 = layer(W1,X,b1)
	z = sigmoid(out_1)
	out_2 = layer(W2,z,b2)
	y_pred= sigmoid(out_2)
	return out_1,z,out_2,y_pred

# back propagation
def back_propagation(W1,W2,b1,b2,out_1,z,out_2,y_pred,s1,Lambda,X):
    sigmaAlpha = np.zeros(W1.shape)
    sigmaBeta = np.zeros(W2.shape)
    bias_1 =  np.zeros(b1.shape)
    bias_2 =  np.zeros(b2.shape)

    zm = np.mean(z,axis=0)
    zm=zm.reshape(1,len(zm))
    Der_Kl = -s1/zm + (1-s1)/(1-zm)
    
    delta = 2*(y_pred-X)*derivative_sigmoid(out_2)
    sigma1=np.sum(delta,axis=0)
    sigma1=sigma1.reshape(len(sigma1),1)
    bias_2=sigma1
    sigmaBeta = np.matmul(z.T,delta)
    
    
    derLayer1 = derivative_sigmoid(out_1)
    Derterm_KL = Lambda*Der_Kl*derLayer1
    
    
    delta_l1 = np.matmul(delta,W2.T)
    sm = derLayer1*delta_l1
    sm = sm + Derterm_KL
    
    
    sm1 = np.sum(sm,axis=0)
    sm1 = sm1.reshape(len(sm1),1)
    
    bias_1 = bias_1+sm1
    sigmaAlpha=sigmaAlpha+np.matmul(X.T,sm)

    return sigmaAlpha,sigmaBeta,bias_1,bias_2,zm

def test(X_train,W1,W2,b1,b2):
	
	flag=109
	#forward pass
	out_1,z,out_2,y_pred=forward_path(W1,W2,b1,b2,X_train[flag].reshape(1,196)/255.0)
	y_pred = y_pred.reshape(14,14)
	y_pred=y_pred*255

	print(np.min(z))
	print(np.max(y_pred))
	print(np.mean(z))

	
	fig = plt.figure()
	ax1 = fig.add_subplot(221)
	ax2 = fig.add_subplot(222)
	ax1.imshow(X_train[flag],cmap='gray')
	ax2.imshow(y_pred,cmap='gray')
	ax1.title.set_text('ORIGINAL IMAGE')
	ax2.title.set_text('IMAGE FROM AE')
	plt.show()

#####################################################

flat = X_train[0].shape
print('X:')
print(X_train.shape)

inputDim = flat[0]*flat[1]
X=X_train.reshape(len(X_train),inputDim)

#normalising 
X=X/255.0

hidden_nodes=250
learning_rate = 1e-4
Lambda = 1
s1 = 0.05
W1,W2,b1,b2=weights(inputDim,hidden_nodes)

nsamples = len(X)

epochs=200

sigmaAlpha = np.zeros(W1.shape)
sigmaBeta = np.zeros(W2.shape)
bias_1 = np.zeros(b1.shape)
bias_2 = np.zeros(b2.shape)

for k in range(0,epochs):
	loss = 0
	temp = 0
	# forward pass
	out_1,z,out_2,y_pred=forward_path(W1,W2,b1,b2,X)

	#backpropagation
	sigmaAlpha,sigmaBeta,bias_1,bias_2,zm =back_propagation(W1,W2,b1,b2,out_1,z,out_2,y_pred,s1,Lambda,X)

	temp=np.sum((s1*np.log(s1/zm))+((1-s1)*np.log((1-s1)/(1-zm))),axis=1)

	loss = np.sum((np.linalg.norm(y_pred-X))**2)-Lambda*temp
	#print the loss in each epoch
	print('Epoch:'+str(k+1)+'         Loss:'+str(loss))

	b2=b2-learning_rate*bias_2
	b1=b1-learning_rate*bias_1
	W2=W2-learning_rate*sigmaBeta
	W1=W1-learning_rate*sigmaAlpha

np.save('b1.npy',b1)
np.save('b2.npy',b2)
np.save('w1.npy',W1)
np.save('w2.npy',W2)

#testing
print('\nTest:\n')
test(X_train,W1,W2,b1,b2)