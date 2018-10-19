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
def forward_path(W1,W2,b1,b2,X):
	out_1 = layer(W1,X,b1)
	print out_1.shape
	z = sigmoid(out_1)
	print W1.shape
	out_2 = layer(W2,z,b2)
	y_pred= sigmoid(out_2)
	return out_1,z,out_2,y_pred

def test(X_train,W1,W2,b1,b2):
	
	flag=1000#change this see an other image X_train[flag]
	#forward pass
	out_1,z,out_2,y_pred=forward_path(W1,W2,b1,b2,X_train[flag].reshape(1,196)/255.0)
	y_pred = y_pred.reshape(14,14)
	y_pred=y_pred*255

	print((z))
	print("We can see that many z's are closer to ZERO")

	fig = plt.figure()
	ax1 = fig.add_subplot(221)
	ax2 = fig.add_subplot(222)
	ax1.imshow(X_train[flag],cmap='gray')
	ax2.imshow(y_pred,cmap='gray')
	ax1.title.set_text('ORIGINAL IMAGE')
	ax2.title.set_text('IMAGE FROM AE')
	plt.show()

X_train = np.load("IMAGES.npy")
np.random.seed(42)

b1=np.load('b1.npy')
b2=np.load('b2.npy')
W1=np.load('w1.npy')
W2=np.load('w2.npy')

#testing
print('\nTest:\n')
test(X_train,W1,W2,b1,b2)