import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.lines as mlines
import seaborn as sns
import numpy as np
from Privileged_DPMM import Privileged_CV_DPMM
from PP_CVI_DPMM import PP_DPMM
from MHPP_CVI_DPMM import MHPP_DPMM
from HPP_DPM import HPP_DPMM
from CVI_for_DPMM import CV_DPMM
from Auxiliary_functions import compute_mixture_test_likelihood, mixture_lik
from Synth_drift_generator import drift_generator_2D
import bnpy
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn import metrics

def purity_score(y_true, y_pred):
    # compute contingency matrix
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency_matrix, axis=0))/np.sum(contingency_matrix) 


# Use numbers 1 to n for the nonempty clusters so that there is no confusion
# with the colors
def regularize_for_print(array):
    for t in range(len(np.unique(array))):
        array[array == np.unique(array)[t]] = t
    return array

# Aux. function to print ellipses
def eigsorted(cov):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    return vals[order], vecs[:, order]

colors = sns.color_palette("Paired", n_colors=100)
n_batches=20
params_svi = [0]*n_batches

# Truncation
T=10
# Number of times the experiment is repeated 
K=10
for n_samples in [1000]:
    # Arrays to store metrics
    lik = np.zeros((6, n_batches))
    score = np.zeros((n_batches, 6))
    nmi_score = np.zeros((n_batches, 6))
    ari_score = np.zeros((n_batches, 6))
    pur_score = np.zeros((n_batches, 6))
    for k in range(K):
        # Execute algorithms
        X, test_data, info, centers, stds, y = drift_generator_2D(n_samples=n_samples, n_test_samples=int(n_samples/2), n_batches=n_batches,\
                                                               n_stationary_batches=1, random_seed=999)

        params, clusters, N_t, iteration, log_lik, cluster_centers, cluster_covs = CV_DPMM(X,
                                                                              alpha=2.,
                                                                              thresh=1.e-3,
                                                                              max_iter=100,
                                                                              T=T)

        params_1, clusters_1, N_t, iteration, log_lik_1, cluster_centers_1, cluster_covs_1 = Privileged_CV_DPMM(X,
                                                                                              info,
                                                                                              alpha=2.,
                                                                                              thresh=1.e-3,
                                                                                              max_iter=100,
                                                                                              T=T)

        params_2, clusters_2, N_t, iteration, log_lik_2, cluster_centers_2, cluster_covs_2 = MHPP_DPMM(X,
                                                                              alpha=2.,
                                                                              thresh=1.e-3,
                                                                              max_iter=100,
                                                                              T=T)

        params_3, clusters_3, N_t, iteration, log_lik_3, cluster_centers_3, cluster_covs_3 = PP_DPMM(X,
                                                                              alpha=2.,
                                                                              thresh=1.e-3,
                                                                              max_iter=100,
                                                                              T=T,
                                                                              pp=.9)
        
        params_4, clusters_4, N_t, iteration, log_lik_4, cluster_centers_4, cluster_covs_4 = HPP_DPMM(X,
                                                                              alpha=2.,
                                                                              thresh=1.e-3,
                                                                              max_iter=100,
                                                                              T=T)
        # Execute SVI
        for i in range(n_batches):
            data = bnpy.data.XData(X[i])
            svi_mean_mu = np.zeros((T, 2))
            svi_cov = np.zeros((T, 2, 2))

            if i == 0:
                prev_start_model, prev_info_dict = bnpy.run(
                    data, 'DPMixtureModel', 'DiagGauss', 'soVB',
                    output_path='/home/icasado/Desktop/soVB/{}'.format(i),
                    nLap=1, nTask=1, nBatch=5,
                    rhoexp=0.55, rhodelay=1,
                    sF=1, ECovMat='eye', K=T, gamma0=2,
                    initname='randexamples',
                    )

            else:
                prev_start_model, prev_info_dict = bnpy.run(
                    data, 'DPMixtureModel', 'DiagGauss', 'soVB',
                    output_path='/home/icasado/Desktop/soVB/{}'.format(i),
                    nLap=1, nTask=1, nBatch=5,
                    rhoexp=0.55, rhodelay=1,
                    sF=1, ECovMat='eye', K=T, gamma0=2,
                    initname=prev_info_dict['task_output_path'],
                    )

            LP = prev_start_model.calc_local_params(data)
            resp=LP['resp']
            E_proba_K = prev_start_model.allocModel.get_active_comp_probs()
            inds = E_proba_K.argsort()[::-1]
            aux = resp.argmax(axis=1).astype(int)
            # Compute metrics for SVI
            nmi_score[i, 4] += normalized_mutual_info_score(y[i], aux)
            score[i, 4] += silhouette_score(X[i], aux)
            ari_score[i, 4] += adjusted_rand_score(y[i], aux)
            pur_score[i, 4] += purity_score(y[i], aux)
            lik[4][i] += mixture_lik(test_data[i], resp, prev_start_model.obsModel.get_mean_for_comp,\
                                      prev_start_model.obsModel.get_covar_mat_for_comp)  
            # Store SVI parameters for tracking
            for t in range(T):
                svi_mean_mu[t] = prev_start_model.obsModel.get_mean_for_comp(t)
                svi_cov[t] = prev_start_model.obsModel.get_covar_mat_for_comp(t)
            params_svi[i] = [np.copy(svi_mean_mu[inds]), np.copy(svi_cov[inds])]
        # Compute and plot log-likelihood
        for i in range(n_batches):
            lik[0][i] += compute_mixture_test_likelihood(test_data[i], params[i][4], params[i][0], params[i][1], params[i][2], params[i][3], 2)
            lik[1][i] += compute_mixture_test_likelihood(test_data[i], params_1[i][4], params_1[i][0], params_1[i][1], params_1[i][2], params_1[i][3], 2)
            lik[2][i] += compute_mixture_test_likelihood(test_data[i], params_2[i][4], params_2[i][0], params_2[i][1], params_2[i][2], params_2[i][3], 2)
            lik[3][i] += compute_mixture_test_likelihood(test_data[i], params_3[i][4], params_3[i][0], params_3[i][1], params_3[i][2], params_3[i][3], 2)
            lik[5][i] += compute_mixture_test_likelihood(test_data[i], params_4[i][4], params_4[i][0], params_4[i][1], params_4[i][2], params_4[i][3], 2)

            score[i, 0] += silhouette_score(X[i], clusters[i])
            score[i, 1] += silhouette_score(X[i], clusters_1[i])
            score[i, 2] += silhouette_score(X[i], clusters_2[i])
            score[i, 3] += silhouette_score(X[i], clusters_3[i])
            score[i, 5] += silhouette_score(X[i], clusters_4[i])

            nmi_score[i, 0] += normalized_mutual_info_score(y[i], clusters[i])
            nmi_score[i, 1] += normalized_mutual_info_score(y[i], clusters_1[i])
            nmi_score[i, 2] += normalized_mutual_info_score(y[i], clusters_2[i])
            nmi_score[i, 3] += normalized_mutual_info_score(y[i], clusters_3[i])
            nmi_score[i, 5] += normalized_mutual_info_score(y[i], clusters_4[i])

            ari_score[i, 0] += adjusted_rand_score(y[i], clusters[i])
            ari_score[i, 1] += adjusted_rand_score(y[i], clusters_1[i])
            ari_score[i, 2] += adjusted_rand_score(y[i], clusters_2[i])
            ari_score[i, 3] += adjusted_rand_score(y[i], clusters_3[i])
            ari_score[i, 5] += adjusted_rand_score(y[i], clusters_4[i])

            pur_score[i, 0] += purity_score(y[i], clusters[i])
            pur_score[i, 1] += purity_score(y[i], clusters_1[i])
            pur_score[i, 2] += purity_score(y[i], clusters_2[i])
            pur_score[i, 3] += purity_score(y[i], clusters_3[i])
            pur_score[i, 5] += purity_score(y[i], clusters_4[i])

        print(k)
    # Plot log-likelihood
    fig, ax1 = plt.subplots(1, 1, figsize=(12, 6))
    ax1.plot(lik[0]/K, 'r-.', linewidth=4, label='SVB')
    ax1.plot(lik[1]/K, 'b--', linewidth=4, label='Privileged')
    ax1.plot(lik[2]/K, 'g-', linewidth=5, label='MHPP')
    ax1.plot(lik[3]/K, 'm', marker='s', markersize=10, linewidth=3, label='0.9-PP')
    ax1.plot(lik[4]/K, marker='x', markersize=10, linewidth=4, label='SVI')
    ax1.plot(lik[5]/K, marker='o', markersize=10, linewidth=4, label='HPP')

    plt.locator_params(axis="x", integer=True, tight=True)
    ax1.legend(frameon=False, fontsize=15)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=15)
    plt.xlabel('Batch nº', fontsize=20)
    plt.ylabel('test-likelihood / N', fontsize=20)
    plt.show()

    fig.savefig('synth_lik.pdf', format='pdf', bbox_inches='tight')
    
    # Plot cluster metric table
    print('SVB, PRIV, MHPP, PP, SVI, HPP')
    print('NMI', np.sum(nmi_score, axis=0)/(n_batches*K))
    print('Silh', np.sum(score, axis=0)/(n_batches*K))
    print('ARI', np.sum(ari_score, axis=0)/(n_batches*K))
    print('Pur', np.sum(pur_score, axis=0)/(n_batches*K))

    
    print(np.std(nmi_score, axis=0))    
    print(np.std(score, axis=0))    
    print(np.std(ari_score, axis=0))
    print(np.std(pur_score, axis=0))



# Scatterplot of the clusters found by MHPP

for i in range(n_batches):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    clusters_2[i] = regularize_for_print(clusters_2[i])
    n_clust = len(np.unique(clusters_2[i]))
    for n in range(n_clust):
        data = X[i, clusters_2[i] == n]
        ax1.scatter(data[:, 0], data[:, 1], s=10, color=colors[n])
        ax1.scatter(cluster_centers_2[i][n, 0], cluster_centers_2[i][n, 1], color='black')
        # Print cov. ellipses with 2 std.
        cov = cluster_covs_2[i][n]
        vals, vecs = eigsorted(cov)
        theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
        w, h = 4 * np.sqrt(vals)
        ell = Ellipse(xy=(cluster_centers_2[i][n, 0], cluster_centers_2[i][n, 1]),
                  width=w, height=h,
                  angle=theta, color='black')
        ell.set_facecolor('none') 
        ax1.add_artist(ell)
    ax1.set_xlim([-3,3])
    ax1.set_ylim([-3,3])
    fig.suptitle("Streaming Variational DPM Clustering")
    ax2.plot(log_lik[i])
    ax2.set(xlabel="nº of iterations", ylabel="log likelihood")
    plt.show()



# Plot mean track
cents = np.zeros((n_batches, 4, 2))
cents_2 = np.zeros((n_batches, 4, 2))
cents_svi = np.zeros((n_batches, 4, 2))

for i in range(n_batches):
    cents[i] = params[i][0][:4]
    cents_2[i] = params_2[i][0][:4]
    cents_svi[i] = params_svi[i][0][:4]

centers = np.moveaxis(centers, 0, 1)
cents = np.moveaxis(cents, 0, 1)
cents_2 = np.moveaxis(cents_2, 0, 1)
cents_svi = np.moveaxis(cents_svi, 0, 1)

fig, ax1 = plt.subplots(1, 1, figsize=(12, 6))

for i in range(4):
    ax1.plot(centers[i, :, 0], centers[i, :, 1], color='black')
    for j in range(n_batches):
        ax1.annotate(j//4, (centers[i, j, 0], centers[i, j, 1]+0.08), fontsize=30)
    ax1.scatter(cents_svi[i, :, 0], cents_svi[i, :, 1], zorder=2, facecolor='orange', edgecolors='black', marker='s',s=300)
    ax1.scatter(cents_2[i, :, 0], cents_2[i, :, 1], zorder=3, facecolor='green', edgecolors='black', s=200)
    ax1.scatter(cents[i, :, 0], cents[i, :, 1], zorder=1, facecolor='yellow', edgecolors='black', marker='v',s=500)

blue = mlines.Line2D([], [], markerfacecolor='orange', markeredgecolor='black', marker='s', linestyle='None',
                          markersize=10, label='SVI')
green = mlines.Line2D([], [], markerfacecolor='green', marker='.', markeredgecolor='black', linestyle='None',
                          markersize=20, label='MHPP')
yellow = mlines.Line2D([], [], markerfacecolor='yellow', markeredgecolor='black', marker='v', linestyle='None',
                          markersize=10, label='SVB')

ax1.legend(handles=[green,yellow,blue], frameon=False, fontsize=17,loc='lower right')
plt.title('Position of the cluster centers', fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=15)
plt.xlabel('X axis', fontsize=20)
plt.ylabel('Y axis', fontsize=20)

fig.savefig('mean_track.pdf', format='pdf', bbox_inches='tight')
plt.show()


# Plot std track
var = np.zeros((n_batches, 4))
var_2 = np.zeros((n_batches, 4))
var_svi = np.zeros((n_batches, 4))
for i in range(n_batches):
    var[i] = params[i][3][:4]/params[i][2][:4]
    var_2[i] = params_2[i][3][:4]/params_2[i][2][:4]
    for j in range(4):
        var_svi[i][j] = params_svi[i][1][:4][j,0,0]
stds = np.moveaxis(stds, 0, 1)
var = np.moveaxis(var, 0, 1)
var_2 = np.moveaxis(var_2, 0, 1)
var_svi = np.moveaxis(var_svi, 0, 1)

fig, ax1 = plt.subplots(1, 1, figsize=(12, 6))


for i in range(4):
    ax1.plot(stds[i], color='black')
    for j in range(n_batches):
        ax1.scatter(j, np.sqrt(var_2[i][j]), zorder=3, facecolor='green',edgecolors='black', s=180)
        ax1.scatter(j, np.sqrt(var[i][j]), zorder=1, facecolor='yellow', edgecolors='black', marker='v', s=300)
        ax1.scatter(j, np.sqrt(var_svi[i][j]), zorder=2, facecolor='orange', edgecolors='black', marker='s', s=150)

blue = mlines.Line2D([], [], markerfacecolor='orange', markeredgecolor='black', marker='s', linestyle='None',
                          markersize=10, label='SVI')
green = mlines.Line2D([], [], color='green', marker='.', linestyle='None',
                          markersize=20, label='MHPP')
yellow = mlines.Line2D([], [], markerfacecolor='yellow', markeredgecolor='black', marker='v', linestyle='None',
                          markersize=10, label='SVB')
ax1.legend(handles=[green,yellow, blue], frameon=False, fontsize=17,loc='upper left')
plt.locator_params(axis="x", integer=True, tight=True)
plt.title('Cluster standard deviations', fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=15)
plt.xlabel('Batch nº', fontsize=20)
plt.ylabel('std', fontsize=20)
fig.savefig('std_track.pdf', format='pdf', bbox_inches='tight')

plt.show()