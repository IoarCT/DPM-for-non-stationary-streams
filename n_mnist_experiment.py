import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import scipy.io
import matplotlib.pyplot as plt
from PP_CVI_DPMM import PP_DPMM
from Privileged_DPMM import Privileged_CV_DPMM
from MHPP_CVI_DPMM import MHPP_DPMM
from CVI_for_DPMM import CV_DPMM
from HPP_DPM import HPP_DPMM
from Auxiliary_functions import compute_mixture_test_likelihood, mixture_lik
import struct
import bnpy
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn import metrics

def purity_score(y_true, y_pred):
    # compute contingency matrix
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency_matrix, axis=0))/np.sum(contingency_matrix) 


# Load datasets
# Original MNIST data
# Train data
with open('/n_mnist/train-images.idx3-ubyte','rb') as f:
    magic, size = struct.unpack(">II", f.read(8))
    nrows, ncols = struct.unpack(">II", f.read(8))
    train_mnist = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
    train_mnist = train_mnist.reshape((size, nrows*ncols))
# Test data
with open('/n_mnist/t10k-images.idx3-ubyte','rb') as f:
    magic, size = struct.unpack(">II", f.read(8))
    nrows, ncols = struct.unpack(">II", f.read(8))
    test_mnist = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
    test_mnist = test_mnist.reshape((size, nrows*ncols))
# Train labels
with open('/n_mnist/train-labels.idx1-ubyte','rb') as f:
    magic, size = struct.unpack('>II', f.read(8))
    train_mnist_labels = np.fromfile(f, dtype=np.dtype(np.uint8)).newbyteorder(">")   
# Test labels    
with open('/n_mnist/t10k-labels.idx1-ubyte','rb') as f:
    magic, size = struct.unpack('>II', f.read(8))
    test_mnist_labels = np.fromfile(f, dtype=np.dtype(np.uint8)).newbyteorder(">")   


# n_MNIST variants
mnist_awgn = scipy.io.loadmat('/n_mnist/mnist-with-awgn.mat')
mnist_blur = scipy.io.loadmat('/n_mnist/mnist-with-motion-blur.mat')
mnist_contrast_awgn = scipy.io.loadmat('/n_mnist/mnist-with-reduced-contrast-and-awgn.mat')


# Gaussian noise train/test
train_awgn = mnist_awgn['train_x']
train_awgn_labels = np.zeros(train_awgn.shape[0], dtype=int)
for i in range(train_awgn.shape[0]):
    train_awgn_labels[i] = list(mnist_awgn['train_y'][i]).index(1)

test_awgn = mnist_awgn['test_x']
test_awgn_labels = np.zeros(test_awgn.shape[0], dtype=int)
for i in range(test_awgn.shape[0]):
    test_awgn_labels[i] = list(mnist_awgn['test_y'][i]).index(1)
 
# Blur train/test
train_blur = mnist_blur['train_x']
train_blur_labels = np.zeros(train_blur.shape[0], dtype=int)
for i in range(train_blur.shape[0]):
    train_blur_labels[i] = list(mnist_blur['train_y'][i]).index(1)

test_blur = mnist_blur['test_x']
test_blur_labels = np.zeros(test_blur.shape[0], dtype=int)
for i in range(test_blur.shape[0]):
    test_blur_labels[i] = list(mnist_blur['test_y'][i]).index(1)

# Contrast + gaussian noise train/test
train_con_awgn = mnist_contrast_awgn['train_x']
train_con_awgn_labels = np.zeros(train_con_awgn.shape[0], dtype=int)
for i in range(train_con_awgn.shape[0]):
    train_con_awgn_labels[i] = list(mnist_contrast_awgn['train_y'][i]).index(1)

test_con_awgn = mnist_contrast_awgn['test_x']
test_con_awgn_labels = np.zeros(test_con_awgn.shape[0], dtype=int)
for i in range(test_con_awgn.shape[0]):
    test_con_awgn_labels[i] = list(mnist_contrast_awgn['test_y'][i]).index(1)


# Print sample digit
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 8))
ax1.imshow(train_blur[11].reshape((28,28)), cmap='Greys_r')
ax2.imshow(train_awgn[11].reshape((28,28)), cmap='Greys_r')
ax3.imshow(train_con_awgn[11].reshape((28,28)), cmap='Greys_r')
ax1.axis('off')
ax2.axis('off')
ax3.axis('off')
plt.show()
fig.savefig('n_mnist.pdf', format='pdf', bbox_inches='tight')

# Set experiment
n_batches = 15
n_samples = 1000
n_test_samples = 500
# nº of components in PCA
n_comp = 50

# Preprocess
train_total = np.vstack((train_mnist, train_blur, train_awgn))
test_total = np.vstack((test_mnist, test_blur, test_awgn))
pca = PCA(n_components=n_comp)
minmax = MinMaxScaler()
train_total = minmax.fit_transform(train_total)
train_total = pca.fit(train_total)


raw_train = np.zeros((4, train_mnist.shape[0], n_comp))
raw_test = np.zeros((4, test_mnist.shape[0], n_comp))
raw_train_labels = np.zeros((4, train_mnist.shape[0]))
raw_test_labels = np.zeros((4, test_mnist.shape[0]))

raw_train[0] = pca.transform(minmax.fit_transform(train_mnist))
raw_test[0] = pca.transform(minmax.fit_transform(test_mnist))
raw_train_labels[0] = train_mnist_labels
raw_test_labels[0] = test_mnist_labels

raw_train[1] = pca.transform(minmax.fit_transform(train_blur))
raw_test[1] = pca.transform(minmax.fit_transform(test_blur))
raw_train_labels[1] = train_blur_labels
raw_test_labels[1] = test_blur_labels

raw_train[2] = pca.transform(minmax.fit_transform(train_awgn))
raw_test[2] = pca.transform(minmax.fit_transform(test_awgn))
raw_train_labels[2] = train_awgn_labels
raw_test_labels[2] = test_awgn_labels

raw_train[3] = pca.transform(minmax.fit_transform(train_con_awgn))
raw_test[3] = pca.transform(minmax.fit_transform(test_con_awgn))
raw_train_labels[3] = train_con_awgn_labels
raw_test_labels[3] = test_con_awgn_labels



batches = np.zeros((n_batches, n_samples, n_comp))
labels = np.zeros((n_batches, n_samples))
test_data = np.zeros((n_batches, n_test_samples, n_comp))

# Arrays to store metrics
lik = np.zeros((6, n_batches))
score = np.zeros((n_batches, 6))
nmi_score = np.zeros((n_batches, 6))
ari_score = np.zeros((n_batches, 6))
pur_score = np.zeros((n_batches, 6))

# Info controls the forgetting of Privileged
info = np.zeros(n_batches)
for i in range(n_batches):
    info[i] = i//2

# Create data batches with drift
for i in range(n_batches):
    n_digits = np.random.randint(6, 10, size=1)
    digits = np.random.choice(10, n_digits, replace=False)

    source = np.random.randint(0, 4, size=n_digits)
    data_aux = np.zeros(n_digits, dtype=object)
    label_aux = np.zeros(n_digits, dtype=object)
    test_aux = np.zeros(n_digits, dtype=object)
    
    for k in range(n_digits[0]):
        data_aux[k] = raw_train[source[k]][[j for j, num in enumerate(raw_train_labels[source[k]].astype(int)) if num == digits[k]]]
        label_aux[k] = raw_train_labels[source[k]][[j for j, num in enumerate(raw_train_labels[source[k]].astype(int)) if num == digits[k]]]      
        test_aux[k] = raw_test[source[k]][[j for j, num in enumerate(raw_test_labels[source[k]].astype(int)) if num == digits[k]]]
    
    data = np.concatenate(data_aux, axis=0)
    lab = np.concatenate(label_aux, axis=0)
    test_d = np.concatenate(test_aux, axis=0)
    
    idx = np.random.choice(data.shape[0], n_samples, replace=False)
    batches[i] = data[idx, :]
    labels[i] = lab[idx]
    test_data[i] = test_d[np.random.choice(test_d.shape[0], n_test_samples, replace=False), :]


# Run algorithms
params, clusters, N_t, iteration, log_lik, cluster_centers, cluster_covs = CV_DPMM(batches,
                                                                              alpha=3.,
                                                                              thresh=1.e-4,
                                                                              max_iter=100,
                                                                              T=30)

params_1, clusters_1, N_t, iteration, log_lik_1, cluster_centers_1, cluster_covs_1 = Privileged_CV_DPMM(batches,
                                                                                              info,
                                                                                              alpha=3.,
                                                                                              thresh=1.e-4,
                                                                                              max_iter=100,
                                                                                              T=30)

params_2, clusters_2, N_t, iteration, log_lik_2, cluster_centers_2, cluster_covs_2 = MHPP_DPMM(batches,
                                                                              alpha=3.,
                                                                              thresh=1.e-4,
                                                                              max_iter=100,
                                                                              T=30)

params_3, clusters_3, N_t, iteration, log_lik_3, cluster_centers_3, cluster_covs_3 = PP_DPMM(batches,
                                                                              alpha=3.,
                                                                              thresh=1.e-4,
                                                                              max_iter=100,
                                                                              T=30,
                                                                              pp=.9)

params_4, clusters_4, N_t, iteration, log_lik_4, cluster_centers_4, cluster_covs_4 = HPP_DPMM(batches,
                                                                              	alpha=3.,
                                                                              	thresh=1.e-4,
                                                                        	max_iter=100,
										T=30)
for i in range(n_batches):
    data = bnpy.data.XData(batches[i])
    svi_mean_mu = np.zeros((30, n_comp))
    svi_cov = np.zeros((30, n_comp, n_comp))

    if i == 0:
        prev_start_model, prev_info_dict = bnpy.run(
            data, 'DPMixtureModel', 'DiagGauss', 'soVB',
            output_path='/home/icasado/Desktop/soVB/{}'.format(i),
            nLap=1, nTask=1, nBatch=5, # We set minbatches of size 200
            rhoexp=0.55, rhodelay=1,
            sF=1, ECovMat='eye', K=30, gamma0=3,
            initname='randexamples',
            )

    else:
        prev_start_model, prev_info_dict = bnpy.run(
            data, 'DPMixtureModel', 'DiagGauss', 'soVB',
            output_path='/home/icasado/Desktop/soVB/{}'.format(i),
            nLap=1, nTask=1, nBatch=5,
            rhoexp=0.55, rhodelay=1,
            sF=1, ECovMat='eye', K=30, gamma0=3,
            initname=prev_info_dict['task_output_path'],
            )

    LP = prev_start_model.calc_local_params(data)
    resp=LP['resp']
    E_proba_K = prev_start_model.allocModel.get_active_comp_probs()
    inds = E_proba_K.argsort()[::-1]
    aux = resp.argmax(axis=1).astype(int)
    # Compute metrics for SVI
    nmi_score[i, 4] = normalized_mutual_info_score(labels[i], aux)
    score[i, 4] = silhouette_score(batches[i], aux)
    ari_score[i, 4] = adjusted_rand_score(labels[i], aux)
    pur_score[i, 4] = purity_score(labels[i], aux)
    lik[4][i] = mixture_lik(test_data[i], resp, prev_start_model.obsModel.get_mean_for_comp,\
                                       prev_start_model.obsModel.get_covar_mat_for_comp)  


# Compute metrics for the rest of methods
for i in range(n_batches):
    lik[0][i] = compute_mixture_test_likelihood(test_data[i], params[i][4], params[i][0], params[i][1], params[i][2], params[i][3], 2)
    lik[1][i] = compute_mixture_test_likelihood(test_data[i], params_1[i][4], params_1[i][0], params_1[i][1], params_1[i][2], params_1[i][3], 2)
    lik[2][i] = compute_mixture_test_likelihood(test_data[i], params_2[i][4], params_2[i][0], params_2[i][1], params_2[i][2], params_2[i][3], 2)
    lik[3][i] = compute_mixture_test_likelihood(test_data[i], params_3[i][4], params_3[i][0], params_3[i][1], params_3[i][2], params_3[i][3], 2)
    lik[5][i] = compute_mixture_test_likelihood(test_data[i], params_4[i][4], params_4[i][0], params_4[i][1], params_4[i][2], params_4[i][3], 2)

    score[i, 0] = silhouette_score(batches[i], clusters[i])
    score[i, 1] = silhouette_score(batches[i], clusters_1[i])
    score[i, 2] = silhouette_score(batches[i], clusters_2[i])
    score[i, 3] = silhouette_score(batches[i], clusters_3[i])
    score[i, 5] = silhouette_score(batches[i], clusters_4[i])

    nmi_score[i, 0] = normalized_mutual_info_score(labels[i], clusters[i])
    nmi_score[i, 1] = normalized_mutual_info_score(labels[i], clusters_1[i])
    nmi_score[i, 2] = normalized_mutual_info_score(labels[i], clusters_2[i])
    nmi_score[i, 3] = normalized_mutual_info_score(labels[i], clusters_3[i])
    nmi_score[i, 5] = normalized_mutual_info_score(labels[i], clusters_4[i])

    ari_score[i, 0] = adjusted_rand_score(labels[i], clusters[i])
    ari_score[i, 1] = adjusted_rand_score(labels[i], clusters_1[i])
    ari_score[i, 2] = adjusted_rand_score(labels[i], clusters_2[i])
    ari_score[i, 3] = adjusted_rand_score(labels[i], clusters_3[i])
    ari_score[i, 5] = adjusted_rand_score(labels[i], clusters_4[i])

    pur_score[i, 0] = purity_score(labels[i], clusters[i])
    pur_score[i, 1] = purity_score(labels[i], clusters_1[i])
    pur_score[i, 2] = purity_score(labels[i], clusters_2[i])
    pur_score[i, 3] = purity_score(labels[i], clusters_3[i])
    pur_score[i, 5] = purity_score(labels[i], clusters_4[i])

#Plot log-likelihoods
fig, ax1 = plt.subplots(1, 1, figsize=(12, 6))
ax1.plot(lik[0], 'r-.', linewidth=4, label='SVB')
ax1.plot(lik[1], 'b--', linewidth=4, label='Privileged')
ax1.plot(lik[2], 'g', linewidth=5, label='MHPP')
ax1.plot(lik[3], 'm', marker='s', markersize=10, linewidth=3, label='0.99-PP')
ax1.plot(lik[4], marker='x', markersize=10, linewidth=4, label='SVI')
ax1.plot(lik[5], marker='o', markersize=10, linewidth=4, label='HPP')

plt.locator_params(axis="x", integer=True, tight=True)
ax1.legend(frameon=False, fontsize=15)
plt.xticks(fontsize=16) 
plt.yticks(fontsize=15)
plt.xlabel('Batch nº', fontsize=20)
plt.ylabel('test-likelihood / N', fontsize=20)
plt.show()

fig.savefig('real_lik.pdf', format='pdf', bbox_inches='tight')

# Print table with clustering metrics
print('SVB, PRIV, MHPP, PP, SVI, HPP')
print('NMI', np.sum(nmi_score, axis=0)/(n_batches))
print('Silh', np.sum(score, axis=0)/(n_batches))
print('ARI', np.sum(ari_score, axis=0)/(n_batches))
print('Pur', np.sum(pur_score, axis=0)/(n_batches))

print(np.std(nmi_score, axis=0))    
print(np.std(score, axis=0))    
print(np.std(ari_score, axis=0))
print(np.std(pur_score, axis=0))

# for i in range(n_batches):
#     fig, ax = plt.subplots(6, 5, figsize=(20, 20))
#     cluster_centers_2[i] = pca.inverse_transform(cluster_centers_2[i])
#     n_clust = len(np.unique(clusters_2[i]))
#     for j in range(n_clust):
#         ax[j//6, j%5].matshow(cluster_centers_2[i][j].reshape((28, 28)), cmap='Greys')
#     plt.show()


