import numpy as np
import GMM_lib as GMM

np.random.seed(425)

# function which  generates data
def datagen(d,n,k):
    Mean = []
    Cov = []
    data = []
    for i in range(k):
        Mean.append(np.random.uniform(1,2,d))
        temp1 = np.random.uniform(1,2,(d,d))
        Cov.append(np.matmul(temp1,temp1.T))
        for j in range(n):
            data.append(np.random.multivariate_normal(Mean[i],Cov[i]))

    print Mean
    print Cov
    return data


## function which initialises the parameters
def initialisation(k,d):
    mean=np.random.uniform(1,5,(k,d))
    covar=[]
    for i in range(k):
        covar.append(((i+1)*1.0/k)*np.eye(d))
    covar=np.array(covar)
    w=np.random.uniform(2,3,k)
    w=w/np.sum(w)
    return mean,covar,w

#"Please rerun the code If singular matrix or nans appear"
#this happens if the initialisation doesnot happen preoperly

print("Number of data samples per mode:")
n = int(input())
print("Dimension of input(d):")
d = int(input())
print("Modes of gaussians(k):")
k = int(input())
print("Enter threshold: ")
err=float(input())


##### data generation ####
X=datagen(d,n,k)
X=np.array(X)
print(X.shape)
#### initialisation of parameters ####
mean,covar,w=initialisation(k,d)

#call the fit function from the library written by me ie. GMM_lib
GMM.fit(X,k,d,n*k,mean,covar,w,err)