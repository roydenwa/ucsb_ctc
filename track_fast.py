import os
import numba
import numpy as np
import pandas as pd
import networkx as nx
import time as timing
import SimpleITK as sitk
import concurrent.futures

from networkx.algorithms import similarity


def save_img_as_tiff(seg_img: np.ndarray, time: int, track_save_dir: str):
    """
    helper-func to parallelize saving imgs as tiff-files 
    """
    img = sitk.GetImageFromArray(seg_img.astype("uint16"))
    filename = 'mask'+'%0*d' % (3, time) + '.tif'
    sitk.WriteImage(img, os.path.join(track_save_dir, filename))


@numba.njit(parallel=True)
def compute_cell_center(seg_img: np.ndarray, labels: np.ndarray, results: np.ndarray) \
                        -> np.ndarray:
    """
    jitted helper-func to compute cell-centers
    """
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


def tracklet_fast(g1: nx.graph, g2: nx.graph, seg_img1: np.ndarray, seg_img2: np.ndarray, maxtrackid: int, time: int, 
                  linelist: list, tracksavedir: str, cellcenter1: np.ndarray, cellcenter2: np.ndarray) \
                  -> (int, list):
    """
    optimized version of tracklet()

    speed gained by reusing previously calulated cellcenters and
    parallelized IO
    """
    f1 = {}
    f2 = {}
    dict_associate = {}
    loc1 = g1.degree(weight="weight")
    loc2 = g2.degree(weight="weight")
    new_seg_img2 = np.zeros(seg_img2.shape)

    for ele1 in loc1:
        cell = ele1[0]
        f1[cell] = [cellcenter1[cell], ele1[1]]

    for ele2 in loc2:
        cell = ele2[0]
        f2[cell] = [cellcenter2[cell], ele2[1]]

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

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        thread1 = executor.submit(save_img_as_tiff, seg_img1, time, tracksavedir)
        thread2 = executor.submit(save_img_as_tiff, new_seg_img2, time+1, tracksavedir)

    return maxtrackid, linelist


def compute_graph_and_cellcenters_img1(img1: np.ndarray, time: int, linelist: list, maxtrackid: int) \
                                       -> (nx.graph, np.ndarray, list, int):
    """
    helper-func to parallelize the computation of the graph and the cellcenters
    for img1
    """
    labels = compute_unique_vals(img1, return_counts=False)
    g1, cellcenter1 = compute_cell_location_fast(img1, labels)

    if time == 0:
        for cell in labels:
            if cell != 0:
                string = '{} {} {} {}'.format(cell, time, time, 0)
                linelist.append(string)
            maxtrackid = max(cell, maxtrackid)
    
    return g1, cellcenter1, linelist, maxtrackid


def compute_graph_and_cellcenters_img2(img1: np.ndarray, folder2: str, time: int, threshold: int) \
                                       -> (np.ndarray, nx.graph, np.ndarray):
    """
    helper-func to parallelize the computation of the graph and the cellcenters
    for img2
    """
    file2 = 'mask'+'%0*d' % (3, time+1) +'.tif'
    img2_sitk = sitk.ReadImage(os.path.join(folder2, file2))
    img2 = sitk.GetArrayFromImage(img2_sitk)

    img2_label_counts = np.array(compute_unique_vals(img2, return_counts=True)).T
    labels2 = img2_label_counts[..., 0]

    if len(labels2) < 2:
        img2 = img1
        img2_img = sitk.GetImageFromArray(img2)
        sitk.WriteImage(img2_img, os.path.join(folder2, file2))

    for i, label in enumerate(labels2):
        if img2_label_counts[i, 1] < threshold:
            img2[img2 == label] = 0

    g2, cellcenter2 = compute_cell_location_fast(img2, labels2)

    return img2, g2, cellcenter2


def track_main_fast(seg_fold: str, track_fold: str):
    """
    optimized version of track_main()

    speed gained by reducing redundancy,
    using accelerated and parallel helper-funcs and
    parallelizing IO in general
    """
    linelist = []
    maxtrackid = 0
    folder1 = track_fold
    folder2 = seg_fold
    times = len(os.listdir(folder2))
    total_start_time = timing.time()

    for time in range(times-1):
        print('linking frame {} to previous tracked frames'.format(time+1))
        start_time = timing.time()
        threshold = 100

        if time == 0:
            file1 = 'mask000.tif'
            img1 = sitk.ReadImage(os.path.join(folder2, file1))
            img1 = sitk.GetArrayFromImage(img1)
            img1_label, img1_counts = compute_unique_vals(img1, return_counts=True)

            for l in range(len(img1_label)):
                if img1_counts[l] < threshold:
                    img1[img1 == img1_label[l]] = 0

            labels = compute_unique_vals(img1, return_counts=False)
            start_label = 0

            for label in labels:
                img1[img1 == label] = start_label
                start_label = start_label + 1

            img1 = sitk.GetImageFromArray(img1)
            sitk.WriteImage(img1, os.path.join(folder1, file1))

        file1 = 'mask'+'%0*d' % (3, time)+'.tif'
        img1_sitk = sitk.ReadImage(os.path.join(folder1, file1))
        img1 = sitk.GetArrayFromImage(img1_sitk)

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            thread1 = executor.submit(compute_graph_and_cellcenters_img1, img1, time, linelist, maxtrackid)
            thread2 = executor.submit(compute_graph_and_cellcenters_img2, img1, folder2, time, threshold)
            
            g1, cellcenter1, linelist, maxtrackid = thread1.result()
            img2, g2, cellcenter2 = thread2.result()

        # this line takes ~ 0.07s to calc
        maxtrackid, linelist = tracklet_fast(g1, g2, img1, img2, maxtrackid, time, linelist, folder1, cellcenter1, cellcenter2)

        print('--------%s seconds-----------' % (timing.time()-start_time))
    
    filetxt = open(os.path.join(folder1, 'res_track.txt'), 'w')
    
    for line in linelist:
        filetxt.write(line)
        filetxt.write("\n")
    
    filetxt.close()

    print('whole time sequence running time %s' % (timing.time() - total_start_time))

