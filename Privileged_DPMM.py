""""
Collapsed variational inference (CVI) as in Kurihara et al. 2007.
"""

import numpy as np
from scipy.special import digamma, gammaln
from sklearn.cluster import KMeans
from Auxiliary_functions import E_log_p_of_z_given_other_z, ELBO_term_for_pz


def Privileged_CV_DPMM(X, info, T=10, max_iter=100, alpha=1., thresh=1e-3):

    """
    Prior distributions:
    ___________________

    z ~ p_TSB as in Eq. (10) of Kurihara et al. (2007)
    mu_t ~ Normal(prior_mean, prior_cov)
    tau_t ~ Gamma(a_0, b_0)
    X ~ N(mu_t, (tau_t^-1)*I_d)

    Variational distributions:
    _________________________

    q(zn) ~ Discrete(zn | phi)
    q(mu_t) ~ Normal(mean_mu, cov_mu)
    q(tau_t) ~ Gamma(a_tao, b_tao)
    """
    #Get dimensions:
        
    # B is the nº of batches,
    # N is the nº of items in each batch
    # M is the dimension of the data points

    B, N, M = X.shape

    # Prior initializations
    # Parameters for mu_t
    prior_mean = np.zeros((T, M))
    prior_cov = np.empty((T, M, M))
    for t in range(T):
        prior_cov[t] = np.identity(M)

    # Parameters for tau_t
    a_0 = np.ones(T)
    b_0 = np.ones(T)
    # Variational initializations
    # Var. parameters for mu_t
    # kmeans = KMeans(n_clusters=T, init='k-means++', max_iter=1).fit(X[0])
    mean_mu = np.zeros((T, M)) #np.array(kmeans.cluster_centers_)
    cov_mu = np.empty((T, M, M))
    for t in range(T):
        cov_mu[t] = np.identity(M)
    # Var. parameters for tau_t
    a_tao = np.ones(T)
    b_tao = np.ones(T)
    # Cluster assignments parameter
    np.random.seed(999)
    phi = np.random.rand(T, N)
    phi = phi/np.sum(phi, axis=0)    
    N_t = np.sum(phi, axis=1)    

    # Initialize other auxiliary variables
    log_lik = [0]*B
    clusters = np.empty((B, N), dtype=int)
    cluster_centers = [0]*B
    cluster_covariances = [0]*B    
    params = [0]*B
    
    for batch in range(B):

        # Update priors with the variational posteriors of the previous batch
        if batch > 0:
            if info[batch] == info[batch-1]:
                a_0 = a_tao
                b_0 = b_tao
                prior_mean = mean_mu
                prior_cov = cov_mu
            else:
                a_0 = np.ones(T)
                b_0 = np.ones(T)
                prior_mean = np.zeros((T, M))
                prior_cov = np.empty((T, M, M))
                for t in range(T):
                    prior_cov[t] = np.identity(M)     
            # a_0 = np.ones(T)
            # b_0 = np.ones(T)
            # prior_mean = np.zeros((T, M))
            # prior_cov = np.empty((T, M, M))
            # for t in range(T):
            #     prior_cov[t] = np.identity(M)    
        # Variational initializations
        # Var. parameters for mu_t
        # kmeans = KMeans(n_clusters=T, init='k-means++', max_iter=1).fit(X[batch])
        mean_mu = np.zeros((T, M))
        cov_mu = np.empty((T, M, M))
        for t in range(T):
            cov_mu[t] = np.identity(M)
        # Var. parameters for tau_t
        a_tao = np.ones(T)
        b_tao = np.ones(T)
        # Cluster assignments parameters
        phi = np.random.rand(T, N)
        phi = phi/np.sum(phi, axis=0)
        N_t = np.sum(phi, axis=1)

        log_l = []

        for iteration in range(max_iter):

            # Update parameters
             # Update mean_mu, cov_mu, variational parameters of mu_t
            for t in range(T):
                tao_t = a_tao[t]/b_tao[t]
                cov_mu[t] = np.linalg.inv((tao_t*N_t[t] + 1./prior_cov[t, 0, 0])*np.eye(M))
                mean_mu[t] = np.matmul(np.linalg.inv(prior_cov[t]).dot(prior_mean[t]) +\
                             tao_t*X[batch].T.dot(phi[t]), cov_mu[t])

            # Compute auxiliary \eta_{x_i, t} function
            suff = np.zeros((T, N))
            for t in range(T):
                suff[t] = np.sum((X[batch] - mean_mu[t])**2, axis=1) + np.trace(cov_mu[t])

           # Update tao
            for t in range(T):
                a_tao[t] = a_0[t] + .5*M*N_t[t]
                b_tao[t] = b_0[t] + .5*np.sum(np.multiply(phi[t], suff[t]))

            # Update latent parameter
            # Compute Eq. (18) as in [1]
            likx = np.zeros(suff.shape)
            for t in range(T):
                likx[t, :] = .5*M*(digamma(a_tao[t]) - np.log(b_tao[t]) - np.log(2*np.pi))
                tao_t = a_tao[t]/b_tao[t]
                likx[t, :] -= .5*tao_t*suff[t]     
            # Compute Eq. (15) as in [1]
            likz = np.zeros(suff.shape)
            likz = E_log_p_of_z_given_other_z(alpha, phi)
            s = likz + likx                     
            # Update phi, the variational parameter of cluster assignments
            phi = log_normalize(s, axis=0)
            N_t = np.sum(phi, axis=1)

            # # Reorder clusters
            # inds = N_t.argsort()[::-1]
            # a_tao = a_tao[inds]
            # b_tao = b_tao[inds]
            # mean_mu = mean_mu[inds]
            # cov_mu = cov_mu[inds]
            # phi = phi[inds]
            # if batch > 0:
            #     a_0 = a_0[inds]
            #     b_0 = b_0[inds]
            #     prior_mean = prior_mean[inds]
            #     prior_cov = prior_cov[inds] 
            # # Compute and store the ELBO
            # log_l.append(lowerbound(X, T, N, M, alpha, prior_mean, prior_cov, a_0, b_0,\
            #                         mean_mu, cov_mu, a_tao, b_tao, phi, likx, N_t, batch))


            # if len(log_l) > 1 and 100*np.abs(log_l[-1] - log_l[-2])/np.abs(log_l[-2]) < thresh:
            #     break

        # Assign clusters & cluster centers
        log_lik[batch] = log_l
        clusters[batch] = np.argmax(phi, axis=0)
        n_clust = np.unique(clusters[batch])
        cluster_cent = np.zeros((len(n_clust), M))
        if M == 2:
            cluster_cov = np.zeros((len(n_clust), M, M))
        #Compute covariances for further plotting (not vital)
        for i in range(len(n_clust)):
            cluster_cent[i] = mean_mu[n_clust[i]]
            if M == 2:
                cluster_cov[i] = np.diag([b_tao[n_clust[i]]/a_tao[n_clust[i]]]*2)

        cluster_centers[batch] = cluster_cent
        if M == 2:
            cluster_covariances[batch] = cluster_cov
        # Print info.
        print('ELBO of batch', batch)
        print("----------------")
        print(np.round(log_l, 2))
        print()
        
        params[batch] = [np.copy(mean_mu), np.copy(cov_mu), np.copy(a_tao), np.copy(b_tao), np.copy(phi)]

    return params, clusters, N_t, iteration, log_lik, cluster_centers, cluster_covariances



def lowerbound(X, T, N, M, alpha, prior_mean, prior_cov, a_0, b_0, mean_mu, cov_mu, a_tao, b_tao, phi, likx, N_t, batch):
    """
    Computes the ELBO of the log-likelihood function 
    """
    elbo = 0

    #mu
    lpmu = 0
    lqmu = 0
    for t in range(T):
        aux_inv = np.linalg.pinv(prior_cov[t])
        aux_mean = prior_mean[t] - mean_mu[t]
        # Eq[log p(mu)]
        lpmu += -.5*(aux_mean.T.dot(aux_inv.dot(aux_mean)) + np.trace(np.matmul(aux_inv, cov_mu[t])) - M)
        sign_q, logdet_q = np.linalg.slogdet(cov_mu[t])
        sign_p, logdet_p = np.linalg.slogdet(prior_cov[t])
        # Eq[log q(mu | mean_mu, cov_mu)]
        lqmu += -.5*(sign_p*logdet_p - sign_q*logdet_q)
    elbo += lpmu + lqmu

    #tao
    # Eq[log p(tau)] - Eq[log q(tau | a_tao, b_tao)]
    ltao = np.sum(gammaln(a_tao) - (a_tao-a_0)*digamma(a_tao) \
        - a_0*(np.log(b_tao) - np.log(b_0)) - gammaln(a_0) + a_tao - b_0*a_tao/b_tao)
    elbo += ltao 

    #z
    elbo += ELBO_term_for_pz(alpha, phi)

    #x
    # Eq[log p(X)]
    lpx = np.sum(phi*likx)
    elbo += lpx
    
    return elbo



def logsumexp(arr, axis=0):  
    """
    Computes the sum of arr assuming arr is in the log domain.
    Returns log(sum(exp(arr))) while minimizing the possibility of
    over/underflow
    """
    arr = np.rollaxis(arr, axis)
    # Use the max to normalize, as with the log this is what accumulates
    # the less errors
    vmax = arr.max(axis=0)
    out = np.log(np.sum(np.exp(arr - vmax), axis=0))
    out += vmax
    return out

def log_normalize(v, axis=0):
    """
    Normalized probabilities from unnormalized log-probabilites
    """
    v = np.rollaxis(v, axis)
    v = v.copy()
    v -= v.max(axis=0)
    out = logsumexp(v)
    v = np.exp(v - out)
    v += np.finfo(np.float32).eps
    v /= np.sum(v, axis=0)
    return np.swapaxes(v, 0, axis)