# -*- coding: utf-8 -*-

"""Common functions for computing various streamline distances, and
doing efficient nearest neighbors computations, by means of the
dissimilarity representationa and k-d tree.

Copyright 2017 Giulia Berto

MIT License.
"""

from __future__ import print_function, division
from common_functions import (NN, voxel_measure, streamline_measure,
                              compute_kdtree)
import numpy as np
import nibabel as nib
from dipy.tracking.streamline import set_number_of_points
import pickle
from dipy.segment.clustering import QuickBundles
from dipy.align.streamlinear import StreamlineLinearRegistration


def compute_DSC(distance_func, kdt, dm_source_tract, source_tract_tmp,
                tractogram_tmp):
    """Compute segmentation as Nearest Neighbour with the given distance.
    Extract the estimated target tract. Compute the Dice Similarity
    Coefficient (DSC).
    """
    print("Computing segmentation as Nearest Neighbour with %s distance\n" % distance)
    estimated_target_tract_idx = NN(kdt, dm_source_tract)

    print("Extracting the estimated target tract.")
    estimated_target_tract = tractogram[estimated_target_tract_idx]

    DSC, TP, vol_A, vol_B = voxel_measure(estimated_target_tract, target_tract)
    print("Dice Sim. Coeff. (estimated target tract, target tract) is %f\n" % DSC)

    DSC_streamlines = streamline_measure(estimated_target_tract_idx, target_tract_idx)
    print("Dice Sim. Coeff. for Streamlines (estimated target tract, target tract) is %f\n" % DSC_streamlines)

    NVETT = np.sum(vol_A)
    print("Number of voxels estimated target tract: %i" % NVETT)

    NVTT = np.sum(vol_B)
    print("Number of voxels target tract: %i" % NVTT)

    NOV = np.sum(TP)
    print("Number of overlapping voxels: %i" % NOV)

    sorted_idx = sorted(estimated_target_tract_idx)
    count = 1
    NSETT = 1
    while count < len(sorted_idx):
        if sorted_idx[count] == sorted_idx[count-1]:
            NSETT = NSETT
        else:
            NSETT = NSETT + 1
        count = count + 1
    print("Number of streamlines estimated target tract: %i" % NSETT)

    NSTT = len(target_tract)
    print("Number of streamlines target tract: %i\n" % NSTT)

    return DSC, NVETT, NVTT, NOV, NSETT, NSTT,\
        estimated_target_tract_idx, DSC_streamlines


def slr_tractograms_registration(target_tractogram, source_subject_id, tract_name):
    """Quick Bundles + Resampling + Streamlines Linear Registration.
    """
    # Loading the source tractogram
    source_tractogram_filename = 'data/%s/Tractogram/tractogram_b1k_1.25mm_csd_wm_mask_eudx1M.trk' % source_subject_id
    print("Loading source tractogram: %s" % source_tractogram_filename)
    source_tractogram, header = nib.trackvis.read(source_tractogram_filename)
    source_tractogram = [s[0] for s in source_tractogram]

    # Loading the source tract
    source_tract_filename = 'data/%s/wmql_tracts/%s_%s.trk' % (source_subject_id, source_subject_id, tract_name)
    print("Loading source tract: %s" % source_tract_filename)
    source_tract, header = nib.trackvis.read(source_tract_filename)
    source_tract = np.array([streamline[0] for streamline in
                             source_tract], dtype=np.object)

    # Parameters as in [Garyfallidis et al. 2015]
    threshold_length = 40.0  # 50mm / 1.25
    qb_threshold = 16.0  # 20mm / 1.25
    nb_res_points = 20

    # Target tractogram
    tt = np.array([s for s in target_tractogram if len(s) >
                   threshold_length], dtype=np.object)
    qb = QuickBundles(threshold=qb_threshold)
    tt_clusters = [cluster.centroid for cluster in qb.cluster(tt)]
    tt_clusters = set_number_of_points(tt_clusters, nb_res_points)

    # Source tractogram
    st = np.array([s for s in source_tractogram if len(s) >
                   threshold_length], dtype=np.object)
    qb = QuickBundles(threshold=qb_threshold)
    st_clusters = [cluster.centroid for cluster in qb.cluster(st)]
    st_clusters = set_number_of_points(st_clusters, nb_res_points)

    # Linear Registration
    srr = StreamlineLinearRegistration()
    srm = srr.optimize(static=tt_clusters, moving=st_clusters)

    # Transforming the source tract
    source_tract_aligned = srm.transform(source_tract)

    return source_tract_aligned


if __name__ == '__main__':
    from os.path import isfile
    # Fixed parameters
    num_prototypes = 40
    seed = 0
    np.random.seed(seed)

    # Variable parameters
    tract_name_list = ['cg.left', 'cg.right', 'ifof.left',
                       'ifof.right', 'uf.left', 'uf.right',
                       'cc_7', 'cc_2', 'af.left']
    target_subject_id_list = ['100307', '124422', '161731', '199655',
                              '201111', '239944', '245333', '366446',
                              '528446', '856766']
    distance_list = ['MAM', 'MDF', 'PDM', 'varifolds']
    mam_metric_list = ['max', 'avg', 'min']
    nb_points_list = [12, 20, 32]
    source_subject_id_list = ['100307', '124422', '161731', '199655',
                              '201111', '239944', '245333', '366446',
                              '528446', '856766']

    # Initializing the results table
    table_filename = 'table_slr.pickle'
    if isfile(table_filename):
        print("Retrieving past results from %s" % table_filename)
        table = pickle.load(open(table_filename))
    else:
        print("Creating a new table which will be saved in %s" % table_filename)
        table = {}

    for t in range(len(tract_name_list)):
        tract_name = tract_name_list[t]
        for ts in range(len(target_subject_id_list)):
            target_subject_id = target_subject_id_list[ts]
            # Data
            target_tract_filename = 'data/%s/wmql_tracts/%s_%s.trk' % (target_subject_id, target_subject_id, tract_name)
            target_tractogram_filename = 'data/%s/Tractogram/tractogram_b1k_1.25mm_csd_wm_mask_eudx1M.trk' % target_subject_id
            print("Loading true target tract: %s" % target_tract_filename)
            target_tract, header = nib.trackvis.read(target_tract_filename)
            target_tract = np.array([streamline[0] for streamline in target_tract], dtype=np.object)
            print("Loading target tractogram: %s" % target_tractogram_filename)
            target_tractogram, header = nib.trackvis.read(target_tractogram_filename)
            tractogram = np.array([streamline[0] for streamline in target_tractogram], dtype=np.object)
            target_tractogram = [s[0] for s in target_tractogram]
            for d in range(len(distance_list)):
                distance = distance_list[d]
                if distance == 'MAM':
                    nb_points = 'nd'  # not defined
                    for m in range(len(mam_metric_list)):
                        mam_metric = mam_metric_list[m]

                        print("Computing the KD-Tree")
                        kdt, prototype_idx, tractogram_tmp, target_tract_idx, distance_func = compute_kdtree(tractogram, target_tract, distance, num_prototypes, nb_points, mam_metric)
                        for ss in range(len(source_subject_id_list)):
                            if source_subject_id_list[ss] != target_subject_id_list[ts]:
                                source_subject_id = source_subject_id_list[ss]
                                # Alignment of tractograms and computation of aligned source tract
                                print("Alignment of tractograms with the streamline linear registration method")
                                source_tract_aligned = slr_tractograms_registration(target_tractogram, source_subject_id, tract_name)
                                source_tract_aligned = np.array(source_tract_aligned, dtype=np.object)
                                source_tract = source_tract_aligned
                                print("Computing the dissimilarity for the source tract")
                                prototypes = tractogram[prototype_idx]
                                dm_source_tract = distance_func(source_tract, prototypes)
                                source_tract_tmp = source_tract
                                DSC, NVETT, NVTT, NOV, NSETT, NSTT, estimated_target_tract_idx, DSC_streamlines = compute_DSC(distance_func, kdt, dm_source_tract, source_tract_tmp, tractogram_tmp)
                                # Fill dictionary
                                table[source_subject_id, target_subject_id, tract_name, distance, mam_metric, nb_points] = {'estimated_target_tract_idx': estimated_target_tract_idx, 'target_tract_idx': target_tract_idx}
                                pickle.dump(table, open(table_filename, 'w'), protocol=pickle.HIGHEST_PROTOCOL)

                elif distance == 'MDF':
                    mam_metric = 'nd'  # not defined
                    for pt in range(len(nb_points_list)):
                        nb_points = nb_points_list[pt]
                        print("Computing the KD-Tree")
                        kdt, prototype_idx, tractogram_tmp, target_tract_idx, distance_func = compute_kdtree(tractogram, target_tract, distance, num_prototypes, nb_points, mam_metric)
                        for ss in range(len(source_subject_id_list)):
                            if source_subject_id_list[ss] != target_subject_id_list[ts]:
                                source_subject_id = source_subject_id_list[ss]
                                # Alignment of tractograms and
                                # computation of aligned source tract
                                print("Alignment of tractograms with the streamline linear registration method")
                                source_tract_aligned = slr_tractograms_registration(target_tractogram, source_subject_id, tract_name)
                                source_tract_aligned = np.array(source_tract_aligned, dtype=np.object)
                                source_tract = source_tract_aligned
                                print ("Resampling the streamline with %d points" % nb_points)
                                source_tract_tmp = np.array([set_number_of_points(s, nb_points=nb_points) for s in source_tract], dtype=np.object)
                                print("Computing the dissimilarity for the source tract")
                                prototypes = tractogram_tmp[prototype_idx]
                                dm_source_tract = distance_func(source_tract_tmp, prototypes)
                                DSC, NVETT, NVTT, NOV, NSETT, NSTT, estimated_target_tract_idx, DSC_streamlines = compute_DSC(distance_func, kdt, dm_source_tract, source_tract_tmp, tractogram_tmp)
                                # Fill dictionary
                                table[source_subject_id, target_subject_id, tract_name, distance, mam_metric, nb_points] = {'estimated_target_tract_idx': estimated_target_tract_idx, 'target_tract_idx': target_tract_idx}
                                pickle.dump(table, open(table_filename, 'w'), protocol=pickle.HIGHEST_PROTOCOL)

                elif distance == 'PDM':
                    mam_metric = 'nd'  # not defined
                    nb_points = 'nd'  # not defined
                    print("Computing the KD-Tree")   
                    kdt, prototype_idx, tractogram_tmp, target_tract_idx, distance_func = compute_kdtree(tractogram, target_tract, distance, num_prototypes, nb_points, mam_metric) 

                    for ss in range(len(source_subject_id_list)):
                        if source_subject_id_list[ss] != target_subject_id_list[ts]:
                       
                            source_subject_id = source_subject_id_list[ss]
                            # Alignment of tractograms and computation
                            # of aligned source tract
                            print("Alignment of tractograms with the streamline linear registration method")
                            source_tract_aligned = slr_tractograms_registration(target_tractogram, source_subject_id, tract_name)
                            source_tract_aligned = np.array(source_tract_aligned, dtype=np.object)
                            source_tract = source_tract_aligned
                            print("Computing the dissimilarity for the source tract")
                            prototypes = tractogram[prototype_idx]
                            dm_source_tract = distance_func(source_tract, prototypes)
                            source_tract_tmp = source_tract
                            DSC, NVETT, NVTT, NOV, NSETT, NSTT, estimated_target_tract_idx, DSC_streamlines = compute_DSC(distance_func, kdt, dm_source_tract, source_tract_tmp, tractogram_tmp)
                            # Fill dictionary
                            table[source_subject_id, target_subject_id, tract_name, distance, mam_metric, nb_points] = {'estimated_target_tract_idx': estimated_target_tract_idx, 'target_tract_idx': target_tract_idx}
                            pickle.dump(table, open(table_filename, 'w'), protocol=pickle.HIGHEST_PROTOCOL)

                elif distance == 'varifolds':
                    mam_metric = 'nd'  # not defined
                    nb_points = 'nd'  # not defined
                    print("Computing the KD-Tree")   
                    kdt, prototype_idx, tractogram_tmp, target_tract_idx, distance_func = compute_kdtree(tractogram, target_tract, distance, num_prototypes, nb_points, mam_metric)
                    for ss in range(len(source_subject_id_list)):
                        if source_subject_id_list[ss] != target_subject_id_list[ts]:
                            source_subject_id = source_subject_id_list[ss]
                            # Alignment of tractograms and computation of aligned source tract
                            print("Alignment of tractograms with the streamline linear registration method")
                            source_tract_aligned = slr_tractograms_registration(target_tractogram, source_subject_id, tract_name)
                            source_tract_aligned = np.array(source_tract_aligned, dtype=np.object)
                            source_tract = source_tract_aligned
                            print("Computing the dissimilarity for the source tract")
                            prototypes = tractogram[prototype_idx]
                            dm_source_tract = distance_func(source_tract, prototypes)
                            source_tract_tmp = source_tract
                            DSC, NVETT, NVTT, NOV, NSETT, NSTT, estimated_target_tract_idx, DSC_streamlines = compute_DSC(distance_func, kdt, dm_source_tract, source_tract_tmp, tractogram_tmp)
                            # Fill dictionary
                            table[source_subject_id, target_subject_id, tract_name, distance, mam_metric, nb_points] = {'estimated_target_tract_idx': estimated_target_tract_idx, 'target_tract_idx': target_tract_idx}
                            pickle.dump(table, open(table_filename, 'w'), protocol=pickle.HIGHEST_PROTOCOL)

                else:
                    raise Exception
