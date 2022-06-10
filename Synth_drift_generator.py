from sklearn.datasets import make_blobs
import numpy as np


# Create batches with user controlled concept drift
def drift_generator_2D(n_samples, n_test_samples, n_batches, n_stationary_batches, random_seed=999):

    n_clust = 4
    
    X = np.zeros((n_batches, n_samples, 2))
    y = np.zeros((n_batches, n_samples))
    test_data = np.zeros((n_batches, n_test_samples, 2))

    np.random.seed(random_seed)

    center_var = np.random.uniform(-.7, .7, (n_batches//n_stationary_batches, n_clust, 2))
    std_var = np.random.uniform(-.07, .07, (n_batches//n_stationary_batches, n_clust))

    center_drift = np.zeros((n_batches, n_clust, 2))
    std_drift = np.zeros((n_batches, n_clust))
    info = np.zeros(n_batches)
    centers = np.zeros((n_batches, n_clust, 2))
    stds = np.zeros((n_batches, n_clust))

    for i in range(n_batches):
        center_drift[i] = center_var[i//n_stationary_batches]
        std_drift[i] = std_var[i//n_stationary_batches]
        # Stores info about the existence of drift
        info[i] = i//n_stationary_batches
        # Original mixture parameters are hardcoded, can be modified to add more components or change params
        centers[i]=[(1.5, 1.5),
                    (-1, 1),
                    (-.5, -1),
                    (1, -.5)] + center_drift[i]    
        stds[i] = [.15, .15, .15, .15] + std_drift[i]

    np.random.seed()
    for i in range(n_batches):
        X[i], y[i] = make_blobs(n_samples=n_samples,
                          cluster_std=stds[i],
                          n_features=2,
                          centers=centers[i])

        test_data[i]= make_blobs(n_samples=n_test_samples,
                      cluster_std=stds[i],
                      n_features=2,
                      centers=centers[i])[0]
    
    return X, test_data, info, centers, stds, y
        
        

# # Create batches with user controlled concept drift
# def half_drift_generator_2D(n_samples, n_test_samples, n_batches, n_stationary_batches, random_seed=999):

#     X = np.zeros((n_batches, n_samples, 2))
#     test_data = np.zeros((n_batches, n_test_samples, 2))

#     np.random.seed(random_seed)

#     center_aux = np.random.uniform(-.7, .7, (n_batches//n_stationary_batches, 2, 2))
#     std_aux = np.random.uniform(-.1, .1, (n_batches//n_stationary_batches, 2))

#     center_var = np.concatenate((center_aux, np.zeros((n_batches//n_stationary_batches, 2, 2))), axis=1)
#     std_var = np.concatenate((std_aux, np.zeros((n_batches//n_stationary_batches, 2))), axis=1)

#     center_drift = np.zeros((n_batches, 4, 2))
#     std_drift = np.zeros((n_batches, 4))
#     info = np.zeros(n_batches)
#     centers = np.zeros((n_batches, 4, 2))
#     stds = np.zeros((n_batches, 4))

#     for i in range(n_batches):
#         center_drift[i] = center_var[i//n_stationary_batches]
#         std_drift[i] = std_var[i//n_stationary_batches]
#         # Stores info about the existence of drift
#         info[i] = i//n_stationary_batches
    
#         centers[i]=[(1.5, 1.5),
#                     (-1, 1),
#                     (-.5, -1),
#                     (1, -.5)] + center_drift[i]    
#         stds[i] = [.15, .15, .15, .15] + std_drift[i]

#     np.random.seed()
#     for i in range(n_batches):
#         X[i] = make_blobs(n_samples=n_samples,
#                           cluster_std=stds[i],
#                           n_features=2,
#                           centers=centers[i])[0]

#         test_data[i] = make_blobs(n_samples=n_test_samples,
#                       cluster_std=stds[i],
#                       n_features=2,
#                       centers=centers[i])[0]
    
#     return X, test_data, info, centers, stds        
        
