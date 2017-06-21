# -*- coding: utf-8 -*-

"""Common functions for computing various streamline distances, and
doing efficient nearest neighbors computations, by means of the
dissimilarity representationa and k-d tree.

Copyright 2017 Emanuele Olivetti, Giulia Berto e Pietro Gori

MIT License.
"""

import numpy as np
from numpy.linalg import norm
from dissimilarity import compute_dissimilarity
from dipy.tracking.distances import (bundles_distances_mam,
                                     bundles_distances_mdf)
from sklearn.neighbors import KDTree
from dipy.tracking.streamline import set_number_of_points
from dipy.tracking.utils import affine_for_trackvis
from dipy.tracking.vox2track import streamline_mapping


def pairwise_distances(s1, s2, metric=""):
    """Faster implementation of sklearn.pairwise_distances(), for squared
    Euclidean distance only.
    """
    return (s1 * s1).sum(1)[:, None] - 2.0 * np.dot(s1, s2.T) + \
        (s2 * s2).sum(1)


def k_pdm(sa, sb, sigma2):
    """Gaussian kernel for PDM.
    """
    return np.exp(-pairwise_distances(sa, sb, metric='sqeuclidean') /
                  sigma2).mean()


def pdm_2streamlines(sa, sb, sigma2):
    """PDM streamline distance function for 2 streamlines.
    """
    return np.sqrt(k_pdm(sa, sa, sigma2=sigma2) +
                   k_pdm(sb, sb, sigma2=sigma2) -
                   2.0 * k_pdm(sa, sb, sigma2=sigma2))


def pdm(streamlines_A, streamlines_B, sigma2=1130.0):
    """Point Density Metric distance function between set of streamlines.
    Note: sigma = 42mm is suggested in (1), in (HCP) voxel units
          sigma = 42mm/1.25 = 33.6voxels, so sigma^2 = ~33.6^2 = 1130

    See: (1) Siless et al. PRNI 2013 and (2) Auzias et al. 2009
    """
    result = np.zeros((len(streamlines_A), len(streamlines_B)))
    for i, sa in enumerate(streamlines_A):
        for j, sb in enumerate(streamlines_B):
            result[i, j] = pdm_2streamlines(sa, sb, sigma2=sigma2)

    return result


def voxel_measure(estimated_tract, true_tract,
                  vol_size=(256, 300, 256),
                  voxel_size=(1.25, 1.25, 1.25)):
    """Ratio of overlap between two tracts (in voxels).
    """
    affine = affine_for_trackvis(voxel_size)
    voxel_list_estimated_tract = streamline_mapping(estimated_tract,
                                                    affine=affine).keys()
    voxel_list_true_tract = streamline_mapping(true_tract,
                                               affine=affine).keys()
    TP = len(set(voxel_list_estimated_tract).intersection(set(voxel_list_true_tract)))
    vol_A = len(set(voxel_list_estimated_tract))
    vol_B = len(set(voxel_list_true_tract))
    DSC = 2.0 * float(TP) / float(vol_A + vol_B)

    return DSC, TP, vol_A, vol_B


def streamline_measure(estimated_target_tract_idx, target_tract_idx):
    """Ratio of overlap between two tracts (in number of streamlines).
    """
    a = set(estimated_target_tract_idx.tolist())
    b = set(target_tract_idx.tolist())
    dsc = 2.0 * len(a.intersection(b)) / float(len(a) + len(b))
    return dsc


def streamlines_idx(target_tract, kdt, prototypes, distance_func,
                    warning_threshold=1.0e-4):
    """Retrieve IDs of the streamlines of the target tract.
    """
    dm_target_tract = distance_func(target_tract, prototypes)
    D, I = kdt.query(dm_target_tract, k=1)
    # print("D: %s" % D)
    # print("I: %s" % I)
    # assert((D < 1.0e-4).all())
    if (D > warning_threshold).any():
        print("WARNING (streamlines_idx()): for %s streamlines D > 1.0e-4 !!" % (D > warning_threshold).sum())

    target_tract_idx = I.squeeze()
    return target_tract_idx


def bundles_distances_mam_avg(tracksA, tracksB):
    return bundles_distances_mam(tracksA, tracksB, metric='avg')


def bundles_distances_mam_min(tracksA, tracksB):
    return bundles_distances_mam(tracksA, tracksB, metric='min')


def bundles_distances_mam_max(tracksA, tracksB):
    return bundles_distances_mam(tracksA, tracksB, metric='max')


def compute_kdtree(tractogram, target_tract, distance,
                   num_prototypes=None, nb_points=None,
                   mam_metric='avg'):
    """Compute the dissimilarity representation of the tractogram and
        build the kd-tree.
    """
    if distance == 'MAM':
        print("Using MAM distance with %s" % mam_metric)
        if mam_metric == 'avg':
            distance_func = bundles_distances_mam_avg
        elif mam_metric == 'min':
            distance_func = bundles_distances_mam_min
        elif mam_metric == 'max':
            distance_func = bundles_distances_mam_max
        else:
            raise Exception
    elif distance == 'MDF':
        if nb_points is None:
            nb_points = 32
            print('Using MDF distance and resampling with %s points as in Yoo et al. 2015.' % nb_points)

        print("Resampling the streamline with %d points" % nb_points)
        tractogram = np.array([set_number_of_points(s, nb_points=nb_points)
                               for s in tractogram], dtype=np.object)
        target_tract = np.array([set_number_of_points(s, nb_points=nb_points)
                                 for s in target_tract], dtype=np.object)
        distance_func = bundles_distances_mdf
    elif distance == 'PDM':
        print('Using PDM distance, as in Siless et al. PRNI 2013.')
        distance_func = pdm
    elif distance == 'varifolds':
        print('Using varifolds distance, as in P. Gori et al. MICCAI 2013.')
        distance_func = varifolds
    else:
        print("Distance %s not supported." % distance)
        Exception

    tractogram = np.array(tractogram, dtype=np.object)

    print("Computing dissimilarity matrices")
    if num_prototypes is None:
        num_prototypes = 40
        print("Using %s prototypes as in Olivetti et al. 2012"
              % num_prototypes)

    print("Using %s prototypes" % num_prototypes)
    dm_tractogram, prototype_idx = compute_dissimilarity(tractogram,
                                                         num_prototypes=num_prototypes,
                                                         distance=distance_func,
                                                         prototype_policy='sff',
                                                         n_jobs=-1,
                                                         verbose=False)
    prototypes = tractogram[prototype_idx]

    print("Building the KD-tree of tractogram")
    kdt = KDTree(dm_tractogram)
    target_tract_idx = streamlines_idx(target_tract, kdt, prototypes, distance_func)

    return kdt, prototype_idx, tractogram, target_tract_idx, distance_func


def compute_kdtree_and_dr(source_tract, tractogram, target_tract,
                          distance, num_prototypes=None, nb_points=None,
                          mam_metric='avg'):
    """Compute the dissimilarity representation of the tractogram and of
    the source tract (using the prototypes of the tractogram). Then
    build the kd-tree.
    """
    if distance == 'MAM':
        print("Using MAM distance with %s" % mam_metric)
        if mam_metric == 'avg':
            distance_func = bundles_distances_mam_avg
        elif mam_metric == 'min':
            distance_func = bundles_distances_mam_min
        elif mam_metric == 'max':
            distance_func = bundles_distances_mam_max
        else:
            raise Exception

    elif distance == 'MDF':
        if nb_points is None:
            nb_points = 32
            print('Using MDF distance and resampling with %s points as in Yoo et al. 2015.' % nb_points)

        print("Resampling the streamline with %d points" % nb_points)
        tractogram = np.array([set_number_of_points(s, nb_points=nb_points)
                               for s in tractogram], dtype=np.object)
        source_tract = np.array([set_number_of_points(s, nb_points=nb_points)
                               for s in source_tract], dtype=np.object)
        target_tract = np.array([set_number_of_points(s, nb_points=nb_points)
                               for s in target_tract], dtype=np.object)
        distance_func = bundles_distances_mdf
    elif distance == 'PDM':
        print('Using PDM distance, as in Siless et al. PRNI 2013.')
        distance_func = pdm
    elif distance == 'currents':
        print('Using currents distance, as S. Durrleman et al. NeuroImage 2014.')
        distance_func = currents
    elif distance == 'varifolds':
        print('Using varifolds distance, as in P. Gori et al. MICCAI 2013.')
        distance_func = varifolds
    else:
        print("Distance %s not supported." % distance)
        Exception

    tractogram = np.array(tractogram, dtype=np.object)
    source_tract = np.array(source_tract, dtype=np.object)

    print("Computing dissimilarity matrices")
    if num_prototypes is None:
        num_prototypes = 40
        print("Using %s prototypes as in Olivetti et al. 2012"
              % num_prototypes)

    print("Using %s prototypes" % num_prototypes)
    dm_tractogram, prototype_idx = compute_dissimilarity(tractogram,
                                                         num_prototypes=num_prototypes,
                                                         distance=distance_func,
                                                         prototype_policy='sff',
                                                         n_jobs=-1,
                                                         verbose=False)
    prototypes = tractogram[prototype_idx]
    dm_source_tract = distance_func(source_tract, prototypes)
    print("Building the KD-tree of tractogram")
    kdt = KDTree(dm_tractogram)
    target_tract_idx = streamlines_idx(target_tract, kdt, prototypes, distance_func)
    return kdt, dm_source_tract, prototype_idx, source_tract, tractogram, target_tract_idx, distance_func


def NN(kdt, dm_source_tract):
    """Code for efficient approximate nearest neighbors computation.
    """
    D, I = kdt.query(dm_source_tract, k=1)
    return I.squeeze()


def cent_tang(s):
    """Compute centers and tangents of all segments in a streamline s.
    Compact and faster version.
    """
    tang = np.diff(s, axis=0)  # compute tangent vector of each segment
    cent = (s[:-1, :] + s[1:, :]) / 2.0  # compute center of each segment
    return cent, tang


def inner_varifolds(centa, tanga, centb, tangb, sigma2):
    exp_dist = np.exp(-pairwise_distances(centa, centb,
                                          metric='sqeuclidean') / sigma2)
    norm_tanga = np.sqrt(norm(tanga, axis=1)) + 1.0e-9  # to avoid norm=0
    tanga_normalised = tanga / norm_tanga[:, None]
    norm_tangb = np.sqrt(norm(tangb, axis=1)) + 1.0e-9  # to avoid norm=0
    tangb_normalised = tangb / norm_tangb[:, None]
    gram_matrix = np.power(tanga_normalised.dot(tangb_normalised.T), 2.0)
    return (exp_dist * gram_matrix).sum()


def varifolds_2streamlines(centa, tanga, centb, tangb, sigma2):
    return np.sqrt(inner_varifolds(centa, tanga, centa, tanga, sigma2=sigma2) +
                   inner_varifolds(centb, tangb, centb, tangb, sigma2=sigma2) -
                   2.0 * inner_varifolds(centa, tanga, centb,
                                         tangb, sigma2=sigma2))


def varifolds(streamlines_A, streamlines_B, sigma2=1130.0):
    """Varifolds distance matrix computation between two sets of
    streamlines.

    See: P. Gori et al., Bayesian atlas estimation for the variability
    analysis of shape complexes, in Medical Image Computing and
    Computer-Assisted Intervention—MICCAI, vol. 8149, K. Mori,
    I. Sakuma, Y. Sato, C. Barillot, N. Navab, Eds.,New YorkSpringer,
    2013, pp. 267–274.
    """
    result = np.zeros((len(streamlines_A), len(streamlines_B)))
    for i, sa in enumerate(streamlines_A):
        centa, tanga = cent_tang(sa)
        for j, sb in enumerate(streamlines_B):
            centb, tangb = cent_tang(sb)
            result[i, j] = varifolds_2streamlines(centa, tanga, centb,
                                                  tangb, sigma2=sigma2)

    return result
