import numpy as np
import matplotlib.pyplot as plt


def hist_plot(Ground_truth,generated,N_bins):

	fig, axis = plt.subplots(1, 3, sharey=True, tight_layout=True)
	axis[0].hist(Ground_truth, bins=N_bins,alpha=0.6,label="Ground Truth")
	axis[0].legend()
	axis[1].hist(generated, bins=N_bins,alpha=0.6,label="generated")
	axis[1].legend()
	axis[2].hist(generated,bins=N_bins,alpha=0.6,label="generated")
	axis[2].hist(Ground_truth,bins=N_bins,alpha=0.6,label="Ground Truth")
	axis[2].legend()
	plt.show()

def Gaussian_MLE(Ground_truth,N_points):

	generated_mean=np.mean(Ground_truth)
	generated_std=np.std(Ground_truth)
	generated=np.random.normal(generated_mean,generated_std,N_points)
	return generated

def Laplacian_MLE(Ground_truth,N_points):

	generated_mu=np.median(Ground_truth)
	generated_b=np.mean(np.abs(Ground_truth-generated_mu))
	generated=np.random.laplace(generated_mu,generated_b,N_points)
	return generated

def exponential_MLE(Ground_truth,N_points):

	generated_beta=np.mean(Ground_truth)
	generated=np.random.exponential(generated_beta,N_points)
	return generated

def poisson_MLE(Ground_truth,N_points):

	generated_lambda=np.mean(Ground_truth)
	generated=np.random.poisson(generated_lambda,N_points)
	return generated

def binomial_MLE(Ground_truth,N_points):

	generated_prob=np.mean(Ground_truth)/num_trials*1.0
	generated=np.random.binomial(num_trials,generated_prob,N_points)
	return generated


N_points=10000
N_bins=100
print('1)Gaussian\n2)Laplacian\n3)Exponential\n4)Poisson\n5)Binomial\n')
while(1):
	choice=int(raw_input("Enter choice of distribution: "))
	if(choice==1):
		mean=float(raw_input("Ground truth mean: "))
		std=float(raw_input("Ground truth standard deviation: "))
		Ground_truth=np.random.normal(mean,std,N_points)
		generated=Gaussian_MLE(Ground_truth,N_points)
		hist_plot(Ground_truth,generated,N_bins)

	if(choice==2):		
		mu=float(raw_input("Ground truth loc: "))
		b=float(raw_input("Ground truth scale: "))
		Ground_truth=np.random.laplace(mu,b,N_points) 
		generated=Laplacian_MLE(Ground_truth,N_points)
		hist_plot(Ground_truth,generated,N_bins)
	
	if(choice==3):
		beta=float(raw_input("Ground truth beta: "))
		Ground_truth=np.random.exponential(beta,N_points)
		generated=exponential_MLE(Ground_truth,N_points)
		hist_plot(Ground_truth,generated,N_bins)
	
	if(choice==4):
		Lambda=float(raw_input("Ground truth lambda: "))
		Ground_truth=np.random.poisson(Lambda,N_points)
		generated=poisson_MLE(Ground_truth,N_points)
		hist_plot(Ground_truth,generated,N_bins)
	
	if(choice==5):
		prob=float(raw_input("The probability of success: "))
		num_trials=int(raw_input("Number of trials: "))
		Ground_truth=np.random.binomial(num_trials,prob,N_points)
		generated=binomial_MLE(Ground_truth,N_points)
		hist_plot(Ground_truth,generated,N_bins)
