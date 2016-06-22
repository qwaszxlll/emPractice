import random as ra
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pat
from numpy import linalg as LA
from numpy import ma as ma
from scipy.misc import logsumexp

#---------------------------------------------------------------------------------
# Utility Functions - There is no need to edit code in this section.
#---------------------------------------------------------------------------------

# Reads a data matrix from file.
# Output: X: data matrix.
def readData(file):
    X = []
    with open(file,"r") as f:
        for line in f:
            X.append(map(float,line.split(" ")))
    return np.array(X)
    

# plot 2D toy data
# input: X: n*d data matrix;
#        K: number of mixtures;
#        Mu: K*d matrix, each row corresponds to a mixture mean;
#        P: K*1 matrix, each entry corresponds to the weight for a mixture;
#        Var: K*1 matrix, each entry corresponds to the variance for a mixture;
#        Label: n*K matrix, each row corresponds to the soft counts for all mixtures for an example
#        title: a string represents the title for the plot
def plot2D(X,K,Mu,P,Var,Label,title):
    r=0.25
    color=["r","b","k","y","m","c"]
    n,d = np.shape(X)
    per= Label/(1.0*np.tile(np.reshape(np.sum(Label,axis=1),(n,1)),(1,K)))
    fig=plt.figure()
    plt.title(title)
    ax=plt.gcf().gca()
    ax.set_xlim((-20,20))
    ax.set_ylim((-20,20))
    for i in xrange(len(X)):
        angle=0
        for j in xrange(K):
            cir=pat.Arc((X[i,0],X[i,1]),r,r,0,angle,angle+per[i,j]*360,edgecolor=color[j])
            ax.add_patch(cir)
            angle+=per[i,j]*360
    for j in xrange(K):
        sigma = np.sqrt(Var[j])
        circle=plt.Circle((Mu[j,0],Mu[j,1]),sigma,color=color[j],fill=False)
        ax.add_artist(circle)
        text=plt.text(Mu[j,0],Mu[j,1],"mu=("+str("%.2f" %Mu[j,0])+","+str("%.2f" %Mu[j,1])+"),stdv="+str("%.2f" % np.sqrt(Var[j])))
        ax.add_artist(text)
    plt.axis('equal')
    plt.show()

#---------------------------------------------------------------------------------



#---------------------------------------------------------------------------------
# K-means methods - There is no need to edit code in this section.
#---------------------------------------------------------------------------------

# initialization for k means model for toy data
# input: X: n*d data matrix;
#        K: number of mixtures;
#        fixedmeans: is an optional variable which is
#        used to control whether Mu is generated from a deterministic way
#        or randomized way
# output: Mu: K*d matrix, each row corresponds to a mixture mean;
#         P: K*1 matrix, each entry corresponds to the weight for a mixture;
#         Var: K*1 matrix, each entry corresponds to the variance for a mixture;    
def init(X,K,fixedmeans=False):
    n, d = np.shape(X)
    P=np.ones((K,1))/float(K)

    if (fixedmeans):
        assert(d==2 and K==3)
        Mu = np.array([[4.33,-2.106],[3.75,2.651],[-1.765,2.648]])
    else:
        # select K random points as initial means
        rnd = np.random.rand(n,1)
        ind = sorted(range(n),key = lambda i: rnd[i])
        Mu = np.zeros((K,d))
        for i in range(K):
            Mu[i,:] = np.copy(X[ind[i],:])

    Var=np.mean( (X-np.tile(np.mean(X,axis=0),(n,1)))**2 )*np.ones((K,1))
    return (Mu,P,Var)


# K Means method
# input: X: n*d data matrix;
#        K: number of mixtures;
#        Mu: K*d matrix, each row corresponds to a mixture mean;
#        P: K*1 matrix, each entry corresponds to the weight for a mixture;
#        Var: K*1 matrix, each entry corresponds to the variance for a mixture;
# output: Mu: K*d matrix, each row corresponds to a mixture mean;
#         P: K*1 matrix, each entry corresponds to the weight for a mixture;
#         Var: K*1 matrix, each entry corresponds to the variance for a mixture;
#         post: n*K matrix, each row corresponds to the soft counts for all mixtures for an example
def kMeans(X, K, Mu, P, Var):
    prevCost=-1.0; curCost=0.0
    n=len(X)
    d=len(X[0])
    while abs(prevCost-curCost)>1e-4:
        post=np.zeros((n,K))
        prevCost=curCost
        #E step
        for i in xrange(n):
            post[i,np.argmin(np.sum(np.square(np.tile(X[i,],(K,1))-Mu),axis=1))]=1
        #M step
        n_hat=np.sum(post,axis=0)
        P=n_hat/float(n)
        curCost = 0
        for i in xrange(K):
            Mu[i,:]= np.dot(post[:,i],X)/float(n_hat[i])
            # summed squared distance of points in the cluster from the mean
            sse = np.dot(post[:,i],np.sum((X-np.tile(Mu[i,:],(n,1)))**2,axis=1))
            curCost += sse
            Var[i]=sse/float(d*n_hat[i])
        print curCost
    # return a mixture model retrofitted from the K-means solution
    return (Mu,P,Var,post) 
#---------------------------------------------------------------------------------



#---------------------------------------------------------------------------------
# PART 1 - EM algorithm for a Gaussian mixture model
#---------------------------------------------------------------------------------

def variance(X_t, Mu_j):
    return LA.norm(X_t - Mu_j) ** 2

# Same as N(x;mu,varI)    
def spherical_Gaussian(X_t, Mu_j, Var_j):
    d = len(X_t)
    exp = - variance(X_t, Mu_j) / (2.0 * Var_j)
    bot = (2 * np.pi * Var_j) ** (d/2.0)
    return np.exp( exp ) / bot

def sum_p_N(X_t, Mu, Var, P, k):
    bot_sum = 0
    for l in range(k):
        bot_sum = bot_sum + P[l][0] * spherical_Gaussian(X_t, Mu[l], Var[l])
        # print spherical_Gaussian(X_t, Mu[l], Var[l], d)
    return bot_sum

def log_likelihood(X, Mu, Var, P, n, k):
    log_sum = 0
    for t in range(n):
        log_sum = log_sum + math.log(sum_p_N(X[t], Mu, Var, P, k))
    return log_sum


# E step of EM algorithm
# input: X: n*d data matrix;
#        K: number of mixtures;
#        Mu: K*d matrix, each row corresponds to a mixture mean;
#        P: K*1 matrix, each entry corresponds to the weight for a mixture;
#        Var: K*1 matrix, each entry corresponds to the variance for a mixture;
# output:post: n*K matrix, each row corresponds to the soft counts for all mixtures for an example
#        LL: a Loglikelihood value
def Estep(X, K, Mu, P, Var):
    n,d = np.shape(X) # n data points of dimension d
    post = np.zeros((n,K)) # posterior probabilities to compute
    LL = log_likelihood(X, Mu, Var, P, n, K)

    for t in range(n):
        bot_sum = sum_p_N(X[t], Mu, Var, P, K)
        for j in range(K):
            N = spherical_Gaussian(X[t], Mu[j], Var[j])
            post[t][j] = P[j] * N / bot_sum

    return (post,LL)


# M step of EM algorithm
# input: X: n*d data matrix;
#        K: number of mixtures;
#        Mu: K*d matrix, each row corresponds to a mixture mean;
#        P: K*1 matrix, each entry corresponds to the weight for a mixture;
#        Var: K*1 matrix, each entry corresponds to the variance for a mixture;
#        post: n*K matrix, each row corresponds to the soft counts for all mixtures for an example
# output:Mu: updated Mu, K*d matrix, each row corresponds to a mixture mean;
#        P: updated P, K*1 matrix, each entry corresponds to the weight for a mixture;
#        Var: updated Var, K*1 matrix, each entry corresponds to the variance for a mixture;
def Mstep(X, K, Mu, P, Var, post):
    n,d = np.shape(X) # n data points of dimension d
    N = np.zeros(K)
    post_T = np.transpose(post);
    
    for j in range(K):
        #Update parameters
        N[j] = math.fsum(post_T[j])
        P[j] = N[j]/n
        Mu[j] = np.dot(post_T[j], X) / N[j]

        variances = np.array([variance(X[t], Mu[j]) for t in range(n)])
        Var[j] = np.dot(post_T[j], variances) / (d * N[j]) 

    return (Mu,P,Var)


# Mixture of Guassians
# input: X: n*d data matrix;
#        K: number of mixtures;
#        Mu: K*d matrix, each row corresponds to a mixture mean;
#        P: K*1 matrix, each entry corresponds to the weight for a mixture;
#        Var: K*1 matrix, each entry corresponds to the variance for a mixture;
# output: Mu: updated Mu, K*d matrix, each row corresponds to a mixture mean;
#         P: updated P, K*1 matrix, each entry corresponds to the weight for a mixture;
#         Var: updated Var, K*1 matrix, each entry corresponds to the variance for a mixture;
#         post: updated post, n*K matrix, each row corresponds to the soft counts for all mixtures for an example
#         LL: Numpy array for Loglikelihood values
def mixGauss(X, K, Mu, P, Var):
    n,d = np.shape(X) # n data points of dimension d
    post = np.zeros((n,K)) # posterior probabilities\
    ll_array = []

    while True:

        (post, curr_LL) = Estep(X, K, Mu, P, Var)
        (Mu, P, Var) = Mstep(X, K, Mu, P, Var, post)

        if len(ll_array) > 1 and (abs(curr_LL - ll_array[-1]) <= 10 ** -6 * abs(curr_LL)):
            return (Mu,P,Var,post,np.array(ll_array)) 

        ll_array.append(curr_LL)

        # if len(ll_array) > 2: 
        #     print abs(ll_array[-1] - ll_array[-2])
        # title = "Iteration " + str(len(ll_array))
        # plot2D(X, K, Mu, P, Var, post, title)

    
    return (Mu,P,Var,post,np.array(ll_array))


# Bayesian Information Criterion (BIC) for selecting the number of mixture components
# input:  n*d data matrix X, a list of K's to try 
# output: the highest scoring choice of K
def BICmix(X, Kset):
    n,d = np.shape(X)
    l = log_likelihood
    max_bic = -100000000
    max_k = 0
    for K in Kset:
        (Mu, P, Var) = init(X, K)
        (Mu, P, Var, post, LL) = mixGauss(X, K, Mu, P, Var)
        p = K * (d+2)-1
        bic = LL[-1] - p/2 * math.log(n)
        print bic
        if bic > max_bic:
            max_bic = bic
            max_k = K
    print max_k, max_bic
    return max_k
#---------------------------------------------------------------------------------



#---------------------------------------------------------------------------------
# PART 2 - Mixture models for matrix completion
#---------------------------------------------------------------------------------

def get_partial_X(X):
    n,d = np.shape(X)
    X_Cu = []
    for x in range(n):
        X_Cu.append([i for i in X[x] if i > 0])
    return np.array(X_Cu)

def get_indicator_arr(X_u):
    return np.array([(X_u[i] > 0) for i in range(len(X_u)) ])

def get_Mu_Cu(X_u, Mu_j):
    return np.array([Mu_j[i] for i in range(len(X_u)) if X_u[i] > 0])

def f_u_i(u, i, P, X, X_Cu, Mu, Var):
    Mu_Cu_i = get_Mu_Cu(X[u], Mu[i])

    d = len(X_Cu[u])
    exp = - variance(X_Cu[u], Mu_Cu_i) / (2 * Var[i])
    bot = (2 * np.pi * Var[i])
    return math.log(P[i]) + exp - (d/2.0) * math.log(bot)

def movie_log_likelihood(X, X_Cu, Mu, Var, P, n, K):
    return math.fsum([logsumexp([f_u_i(t, j, P, X, X_Cu, Mu, Var) for j in range(K) ]) for t in range(n)])

# RMSE criteria
# input: X: n*d data matrix;
#        Y: n*d data matrix;
# output: RMSE
def rmse(X,Y):
    return np.sqrt(np.mean((X-Y)**2))


# E step of EM algorithm with missing data
# input: X: n*d data matrix;
#        K: number of mixtures;
#        Mu: K*d matrix, each row corresponds to a mixture mean;
#        P: K*1 matrix, each entry corresponds to the weight for a mixture;
#        Var: K*1 matrix, each entry corresponds to the variance for a mixture;
# output:post: n*K matrix, each row corresponds to the soft counts for all mixtures for an example
#        LL: a Loglikelihood value
def Estep_part2(X, K, Mu, P, Var):
    n = len(X) # n data points of dimension d
    post = np.zeros((n,K)) # posterior probabilities to compute
    X_Cu = get_partial_X(X)
    LL = movie_log_likelihood(X, X_Cu, Mu, Var, P, n, K)

    # print "---------------------STARTING E STEP---------------------"
    for t in range(n):
        logSum = logsumexp([f_u_i(t, i, P, X, X_Cu, Mu, Var) for i in range(K)])
        for j in range(K):
            ftj = f_u_i(t, j, P, X, X_Cu, Mu, Var)
            post[t][j] = ftj - logSum
    
    return (np.exp(post),LL)

	
# M step of EM algorithm
# input: X: n*d data matrix;
#        K: number of mixtures;
#        Mu: K*d matrix, each row corresponds to a mixture mean;
#        P: K*1 matrix, each entry corresponds to the weight for a mixture;
#        Var: K*1 matrix, each entry corresponds to the variance for a mixture;
#        post: n*K matrix, each row corresponds to the soft counts for all mixtures for an example
# output:Mu: updated Mu, K*d matrix, each row corresponds to a mixture mean;
#        P: updated P, K*1 matrix, each entry corresponds to the weight for a mixture;
#        Var: updated Var, K*1 matrix, each entry corresponds to the variance for a mixture;
def Mstep_part2(X,K,Mu,P,Var,post, minVariance=0.25):
    n,d = np.shape(X) # n data points of dimension d
    N = np.zeros(K)
    mask = (X[:]==0)
    post_T = np.transpose(post)
    X_T = np.transpose(X)
    X_Cu = ma.array(X,mask=mask)
    # X_Cu = get_partial_X(X)

    # print "---------------------STARTING M STEP---------------------"
    N = np.sum(post,axis=0)
    P = np.divide(N,n)

    # update means
    Mu_new = np.dot(post_T,X_Cu)
    Mu_bot = np.dot(post_T,~mask)
    mask_2 = (Mu_bot[:]<1)
    Mu_bot = ma.array(Mu_bot,mask=mask_2) #masks out values <1
    #print Mu_bot
    Mu_new = np.divide(Mu_new, Mu_bot)
    Mu = (Mu * mask_2) + Mu_new.filled(0) #keep original masked out values

    # update variances
    nonzeros = np.apply_along_axis(np.count_nonzero,1,~mask)
    sig_denoms= np.sum(post * np.transpose([nonzeros]),axis=0)
    #print post

    for j in xrange(K):
        norm = lambda x: LA.norm(x - Mu[j])**2
        Var[j] = max(minVariance, np.sum(np.multiply(post_T[j],ma.apply_along_axis(norm,1,X_Cu)))/(sig_denoms[j]))

    # for j in range(K):
    #     # Update parameters
    #     N[j] = math.fsum(post_T[j])
    #     P[j] = N[j]/n
    #     for l in range(d):
    #         Mu_bot = math.fsum([post[t][j] for t in range(n) if X[t][l] > 0])
    #         Mu_top = math.fsum([post[t][j] * X[t][l] for t in range(n) if X[t][l] > 0])
    #         if Mu_bot >= 1:
    #             Mu[j][l] = Mu_top / Mu_bot

        # variances = np.array([variance(X_Cu[t], Mu[j]) for t in range(n)])
        # var_top = np.dot(post_T[j], variances)
        # var_bot = math.fsum( [post[t][j] * len(X_Cu[t]) for t in range(n)] )
        # Var[j] = max(var_top / var_bot, minVariance)
    return (Mu,P,Var)

	
# mixture of Guassians
# input: X: n*d data matrix;
#        K: number of mixtures;
#        Mu: K*d matrix, each row corresponds to a mixture mean;
#        P: K*1 matrix, each entry corresponds to the weight for a mixture;
#        Var: K*1 matrix, each entry corresponds to the variance for a mixture;
# output: Mu: updated Mu, K*d matrix, each row corresponds to a mixture mean;
#         P: updated P, K*1 matrix, each entry corresponds to the weight for a mixture;
#         Var: updated Var, K*1 matrix, each entry corresponds to the variance for a mixture;
#         post: updated post, n*K matrix, each row corresponds to the soft counts for all mixtures for an example
#         LL: Numpy array for Loglikelihood values
def mixGauss_part2(X,K,Mu,P,Var):
    n,d = np.shape(X) # n data points of dimension d
    post = np.zeros((n,K)) # posterior probs tbd
    ll_array = []

    while True:
        # print "Var: ", Var
        (post, curr_LL) = Estep_part2(X, K, Mu, P, Var)
        (Mu, P, Var) = Mstep_part2(X, K, Mu, P, Var, post)

        if len(ll_array) > 1 and (abs(curr_LL - ll_array[-1]) <= (10 ** -6) * abs(curr_LL)):
            return (Mu,P,Var,post,np.array(ll_array)) 

        ll_array.append(curr_LL)
	
    
    return (Mu,P,Var,post,np.array(ll_array))


# fill incomplete Matrix
# input: X: n*d incomplete data matrix;
#        K: number of mixtures;
#        Mu: K*d matrix, each row corresponds to a mixture mean;
#        P: K*1 matrix, each entry corresponds to the weight for a mixture;
#        Var: K*1 matrix, each entry corresponds to the variance for a mixture;
# output: Xnew: n*d data matrix with unrevealed entries filled
def fillMatrix(X,K,Mu,P,Var):
    n,d = np.shape(X)
    Xnew = np.copy(X)
    mask = (Xnew[:]!=0)

    (post, curr_LL) = Estep_part2(X, K, Mu, P, Var)
    X_pred = np.dot(post, Mu)
    X_pred_masked = ma.array(X_pred,mask=mask)
    Xnew = Xnew + X_pred_masked.filled(0)

    return Xnew
#---------------------------------------------------------------------------------