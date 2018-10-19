
#BGR format

import numpy as np
import cv2
import matplotlib.pyplot as plt
from random import randint

def verify_clusters(Centre,img):
    Cent=[]
    flat=img.reshape([img.shape[0]*img.shape[1],img.shape[2]])
    c=np.zeros((len(flat),len(Centre)))    
    
    for k in range(len(Centre)):
        c[:,k]=np.linalg.norm((Centre[k]-flat),axis=1)

    label=np.argmin(c.T,axis=1)
    
    for i in range(0,len(label)):
        Cent.append(flat[label[i]])
    
    return np.array(Cent).astype('uint8')


def Cluster_init(img,K):
	Cent=[]
	print('Initial cluster centres:')

	flat=img.reshape([img.shape[0]*img.shape[1],img.shape[2]])
	mean=np.mean(flat,axis=0)
	print(mean.shape)
	var=np.var(flat,axis=0)*10
	print (var.shape)
	for i in range(0,K):
		Cent.append(np.random.randn(1, img.shape[2])*np.sqrt(var)+mean)
	
	return np.array(Cent).astype('uint8')


# Clustering 
def Clustering(Centre,img):
    
    label = []
    
    c=np.zeros((len(img),len(Centre)))
    
    for k in range(len(Centre)):
        c[:,k]=np.linalg.norm((Centre[k]-img),axis=1)
    
    label=np.argmin(c,axis=1)
    
    return np.array(label)


def Centroid(img,label,K):

	Centre=[]
	for i in range(K):
		Centre.append(np.mean(img[np.argwhere(label==i)],axis=0))
	return np.array(Centre)

def Kmeans(img,K,error):

	Centre=[]
	Centre=np.array(Cluster_init(img,K))
	Centre=verify_clusters(Centre,img)
	print (Centre)
	print('\n')

	flatten=img.reshape([img.shape[0]*img.shape[1],img.shape[2]])

	label=Clustering(Centre,flatten)

	new_Centre = []

	diff=2*error

	while(diff>error):
		label=Clustering(Centre,flatten)
		new_Centre=Centroid(flatten,label,K)
		diff=np.linalg.norm(new_Centre-Centre)
		print('Error ='+str(diff))
		Centre = new_Centre

	return Centre


def image_show(Centre,k,name):

	img=cv2.imread(name)

	flatten=img.reshape([img.shape[0]*img.shape[1],3])

	label=Clustering(Centre,flatten)

	flatten[np.argwhere(label!=k)]=0

	image=flatten.reshape([img.shape[0],img.shape[1],3])

	plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

	plt.show()





#############################################################

name='K_means.jpg'
			
img=cv2.imread(name)
print (img.shape)


### Asking inputs
print("Number of clusters :")
K = int(input())
print("Threshold E :")
error = float(input())

Centre=Kmeans(img,K,error)

print('\nThe cluster centers are:\n'+str(Centre)) 


for i in range(0,K):
	image_show(Centre,i,name)



