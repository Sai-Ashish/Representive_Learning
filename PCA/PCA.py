#RGB format

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt 

def center(X):
	Y=X-(np.mean(X,axis=1).reshape((X.shape[0],1)))
	return Y

img=Image.open('test2.jpg')
img=np.array(img)
plt.imshow(img)
plt.show()
img=img.astype('float64')


flatten=img.reshape(-1,3)
flatten=flatten.T


mean_flat=center(flatten)
cov=np.matmul(mean_flat,mean_flat.T)/(1.0*len(mean_flat[0]))

print('\nCovariance matrix of initial data:')
print(cov) 

eigen_values,eigen_vectors=np.linalg.eig(cov)


y=np.matmul(eigen_vectors.T,mean_flat)

print('\nCovariance matrix of transformed data:')
print(np.cov(y)) 

output=y.T
output=output.reshape(img.shape)
output=output.astype('int')



plt.imshow(output)
plt.show()



