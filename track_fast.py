import os
import numba
import numpy as np
import pandas as pd
import networkx as nx

from networkx.algorithms import similarity


@numba.njit(parallel=True)
def compute_cell_center(seg_img: np.ndarray, labels: np.ndarray, results: np.ndarray) \
                        -> np.ndarray:
    for label in labels:
        if label != 0:
            all_points_z, all_points_x, all_points_y = np.where(seg_img == label)
            avg_z = np.round(np.mean(all_points_z))
            avg_x = np.round(np.mean(all_points_x))
            avg_y = np.round(np.mean(all_points_y))
            results[label] = [avg_z, avg_x, avg_y]

    return results

def cell_center_fast(seg_img: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    optimized version of cell_enter()

    compute_cell_center(): jitted helper-func
    """
    array_max_idx = max(labels)
    results = np.zeros((array_max_idx + 1, 3))
    results = compute_cell_center(seg_img, labels, results)

    return results

# was opt3
def compute_cell_location_fast(seg_img: np.ndarray, all_labels: np.ndarray) \ 
                               -> (nx.graph, np.ndarray):
    """
    optimized version of compute_cell_location()

    speed gained by passing in all_labels from previous calc,
    only evaluating all_labels once and 
    removing unused draw_board
    """
    g = nx.Graph()
    centers = cell_center_fast(seg_img, all_labels) # was 6

    # Compute vertices
    for i in all_labels:
        if i != 0:
            g.add_node(i)

    # Compute edges
    for i in all_labels:
        if i != 0:
            for j in all_labels:
                if j != 0:
                    if i != j:
                        pos1 = centers[i]
                        pos2 = centers[j]
                        distance = np.sqrt((pos1[0]-pos2[0])**2 +
                                           (pos1[1]-pos2[1])**2 + (pos1[2]-pos2[2])**2)

                        g.add_edge(i, j, weight=distance)
    return g, centers


def tracklet_fast(g1, g2, seg_img1, seg_img2, maxtrackid, time, linelist, tracksavedir, cellcenter1, cellcenter2):
    """
    optimized version of tracklet()

    speed gained by reusing previously calulated cellcenters
    """
    f1 = {}
    f2 = {}
    new_seg_img2 = np.zeros(seg_img2.shape)
    dict_associate = {}
    loc1 = g1.degree(weight='weight')
    loc2 = g2.degree(weight='weight')

    for ele1 in loc1:
        cell = ele1[0]
        f1[cell] = [cellcenter1[cell], ele1[1]]

    for ele2 in loc2:
        cell = ele2[0]
        f2[cell] = [cellcenter2[cell], ele2[1]]

    # TODO: try to write external and jitted version for those loops
    for cell in f2.keys():
        tmp_center = f2[cell][0]
        min_distance = seg_img2.shape[0]**2 + seg_img2.shape[1]**2 + seg_img2.shape[2]**2

        for ref_cell in f1.keys():
            ref_tmp_center = f1[ref_cell][0]
            distance = (tmp_center[0]-ref_tmp_center[0])**2 + (tmp_center[1] -
                                                               ref_tmp_center[1])**2 + (tmp_center[2]-ref_tmp_center[2])**2
            if distance < min_distance:
                dict_associate[cell] = ref_cell
                min_distance = distance

    inverse_dict_ass = {}

    for cell in dict_associate:
        if dict_associate[cell] in inverse_dict_ass:
            inverse_dict_ass[dict_associate[cell]].append(cell)
        else:
            inverse_dict_ass[dict_associate[cell]] = [cell]

    maxtrackid = max(maxtrackid, max(inverse_dict_ass.keys()))

    # TODO: try to write external and jitted version for those loops
    for cell in inverse_dict_ass.keys():
        if len(inverse_dict_ass[cell]) > 1:
            for cellin2 in inverse_dict_ass[cell]:
                maxtrackid = maxtrackid + 1
                new_seg_img2[seg_img2 == cellin2] = maxtrackid
                string = '{} {} {} {}'.format(maxtrackid, time+1, time+1, cell)
                linelist.append(string)
        else:
            cellin2 = inverse_dict_ass[cell][0]
            new_seg_img2[seg_img2 == cellin2] = cell
            i = 0

            for line in linelist:
                i = i+1
                if i == cell:
                    list_tmp = line.split()
                    new_string = '{} {} {} {}'.format(list_tmp[0], list_tmp[1], time+1, list_tmp[3])
                    linelist[i-1] = new_string

    # TODO: each get and write gets one thread to do it, so it's parallel not sequential
    img1 = sitk.GetImageFromArray(seg_img1.astype('uint16'))
    img2 = sitk.GetImageFromArray(new_seg_img2.astype('uint16'))

    filename1 = 'mask'+'%0*d' % (3, time)+'.tif'
    filename2 = 'mask'+'%0*d' % (3, time+1)+'.tif'

    sitk.WriteImage(img1, os.path.join(tracksavedir, filename1))
    sitk.WriteImage(img2, os.path.join(tracksavedir, filename2))

    return maxtrackid, linelist


def track_main_fast(seg_fold: str, track_fold: str):
    """
    optimized version of track_main()

    speed gained by reducing redundancy,
    using accelerated helper-funcs and
    parallelized IO -> TODO
    """
    folder1 = track_fold
    folder2 = seg_fold
    times = len(os.listdir(folder2))
    maxtrackid = 0
    linelist = []
    total_start_time = timing.time()

    for time in range(times-1):
        print('linking frame {} to previous tracked frames'.format(time+1))
        start_time = timing.time()
        threshold = 100

        if time == 0:
            file1 = 'mask000.tif'
            img1 = sitk.ReadImage(os.path.join(folder2, file1))
            img1 = sitk.GetArrayFromImage(img1)
            img1_label, img1_counts = np.unique(img1, return_counts=True)

            for l in range(len(img1_label)):
                if img1_counts[l] < threshold:
                    img1[img1 == img1_label[l]] = 0

            labels = pd.unique(img1.flatten('K'))
            start_label = 0

            # seems odd...
            for label in labels:
                img1[img1 == label] = start_label
                start_label = start_label + 1

            img1 = sitk.GetImageFromArray(img1)
            sitk.WriteImage(img1, os.path.join(folder1, file1))

        file1 = 'mask'+'%0*d' % (3, time)+'.tif'
        file2 = 'mask'+'%0*d' % (3, time+1)+'.tif'
        img1 = sitk.ReadImage(os.path.join(folder1, file1))
        img2 = sitk.ReadImage(os.path.join(folder2, file2))
        img1 = sitk.GetArrayFromImage(img1)
        img2 = sitk.GetArrayFromImage(img2)

        img2_label_counts = np.array(np.unique(img2, return_counts=True)).T
        labels2 = img2_label_counts[..., 0]

        if len(labels2) < 2:
            img2 = img1
            img2_img = sitk.GetImageFromArray(img2)
            sitk.WriteImage(img2_img, os.path.join(folder2, file2))
    
        for i, label in enumerate(labels2):
            if img2_label_counts[i, 1] < threshold:
                img2[img2 == label] = 0

        labels = pd.unique(img1.flatten('K'))
        
        # parallelize
        g1, cellcenter1 = compute_cell_location_fast(img1, labels)
        g2, cellcenter2 = compute_cell_location_fast(img2, labels2)

        if time == 0:
            for cell in labels:
                if cell != 0:
                    string = '{} {} {} {}'.format(cell, time, time, 0)
                    linelist.append(string)
                maxtrackid = max(cell, maxtrackid)

        maxtrackid, linelist = tracklet_fast(g1, g2, img1, img2, maxtrackid, time, linelist, folder1, cellcenter1, cellcenter2)

        print('--------%s seconds-----------' % (timing.time()-start_time))
    
    filetxt = open(os.path.join(folder1, 'res_track.txt'), 'w')
    
    for line in linelist:
        filetxt.write(line)
        filetxt.write("\n")
    
    filetxt.close()
    print('whole time sequence running time %s' % (timing.time() - total_start_time))
