import numpy as np 
import sklearn.metrics

all = ['compute_prdc']

def compute_pairwise_distance(data_x, data_y=None): 
    if data_y is None: 
        data_y = data_x 
    dists = sklearn.metrics.pairwise_distances( data_x, data_y, metric='euclidean', n_jobs=8) 
    return dists

def get_kth_value(unsorted, k, axis=-1): 
    indices = np.argpartition(unsorted, k, axis=axis)[..., :k] 
    k_smallests = np.take_along_axis(unsorted, indices, axis=axis) 
    kth_values = k_smallests.max(axis=axis) 
    return kth_values

def compute_nearest_neighbour_distances(input_features, nearest_k): 
    distances = compute_pairwise_distance(input_features) 
    radii = get_kth_value(distances, k=nearest_k + 1, axis=-1) 
    return radii

def compute_prdc(real_features, fake_features, nearest_k): 
    print('Num real: {} Num fake: {}' .format(real_features.shape[0], fake_features.shape[0]))

    real_nearest_neighbour_distances = compute_nearest_neighbour_distances(real_features, nearest_k)
    fake_nearest_neighbour_distances = compute_nearest_neighbour_distances(fake_features, nearest_k)
    distance_real_fake = compute_pairwise_distance(real_features, fake_features)

    precision = (
            distance_real_fake <
            np.expand_dims(real_nearest_neighbour_distances, axis=1)
    ).any(axis=0).mean()

    recall = (
            distance_real_fake <
            np.expand_dims(fake_nearest_neighbour_distances, axis=0)
    ).any(axis=1).mean()

    density = (1. / float(nearest_k)) * (
            distance_real_fake <
            np.expand_dims(real_nearest_neighbour_distances, axis=1)
    ).sum(axis=0).mean()

    coverage = (
            distance_real_fake.min(axis=1) <
            real_nearest_neighbour_distances
    ).mean()

    return dict(precision=precision, recall=recall, density=density, coverage=coverage)