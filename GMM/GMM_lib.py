import numpy as np


np.random.seed(425)

def normal(x,mean,covar):
    inv=[]
    mean=mean.reshape(1,len(mean))
    det=np.linalg.det(covar)
    inv=np.linalg.inv(covar)
    nr=np.exp(-0.5*(np.matmul((np.matmul(x-mean,inv)),(x-mean).T)))
    dr=np.sqrt(det*((2*np.pi)**(len(x))))
    return np.asscalar(nr/dr)


def posterior(x,w,mean,covar):

    tot=[]
    post=[]
    x=x.reshape(1,len(x))
    for j in range(len(w)):
    	tot.append(w[j]*normal(x,mean[j],covar[j]))
    tot=np.array(tot)
    post=tot/np.sum(tot)
    return post,tot

def likelihood(X,n,mean,covar,w,k):

    normals=[]
    for i in range(n):
        _,tmp=posterior(X[i],w,mean,covar)
        normals.append(tmp)
    normals=np.array(normals)
    normals=normals.reshape(n,k)

    return np.sum(np.log(np.sum(normals,axis=1).reshape(n,1)))


def new_mean(post,X):

    Nk=np.sum(post,axis=0)
    Nk=Nk.reshape(len(Nk),1)
    mean=np.matmul(post.T,X)
    mean=mean.reshape(post.shape[1],X.shape[1])
    mean=mean/Nk
    return mean

def new_covar(post,X,mean,covar):

    for i in range(post.shape[1]):
        temp=(post.T)[i]
        temp=temp.reshape(len(temp),1)
        mean_temp=mean[i]
        mean_temp=mean_temp.reshape(1,len(mean_temp))
        Nk=np.sum(temp)
        covar[i]=(np.matmul((temp*(X-mean_temp)).T,X-mean_temp))/Nk

    return covar

def new_weigths(post):

    Nk=np.sum(post,axis=0)
    Nk=Nk.reshape(len(Nk),1)	

    return Nk/(len(post)*1.0)

def fit(X,k,d,n,mean,covar,w,err):

    error=100
    print err
    Likelihood=0
    iter=0
    Likelihood=likelihood(X,n,mean,covar,w,k)
    while(error>err):
    	post=[]
    	for i in range(n):
    	    tmp,_=posterior(X[i],w,mean,covar)
    	    post.append(tmp)

    	post=np.array(post)
    	post=post.reshape(n,k)

    	mean=new_mean(post,X)
    	
    	covar=new_covar(post,X,mean,covar)

    	w=new_weigths(post)

    	error=likelihood(X,n,mean,covar,w,k)-Likelihood

    	Likelihood=likelihood(X,n,mean,covar,w,k)

    	print('error:'+str(error)+'				Log_Likelihood:'+str(Likelihood))

    	iter=iter+1

    print('mean\n'+str(mean))
    print('________________________________________________________')
    print('covar\n'+str(covar))
    print('________________________________________________________')
    print('w\n'+str(w))
    print('________________________________________________________')
	
############################################################



