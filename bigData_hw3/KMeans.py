from pyspark import SparkConf, SparkContext
import numpy as np
import csv
import math
import time
import matplotlib.pyplot as plt
import os


def get_init_centroids(file_path):
    centroids = []
    with open(file_path, "r") as file:
        for line in file.readlines():
            centroid = list(map(float, line.split(' ')))
            centroids.append(centroid)
    return centroids


def classify_and_get_loss(point, centroids, method):
    min_loss = float("inf")
    in_cluster = None
    point = list(map(float, point.split(' ')))
    for centroid_idx, centroid in list(enumerate(centroids)):
        point_loss = 0
        for dim in range(len(point)):
            if method == 'Euclidean':
                loss = math.pow(point[dim] - centroid[dim], 2)
            elif method == 'Manhattan':
                loss = abs(point[dim] - centroid[dim])
            else:
                raise NameError
            point_loss += loss
        if point_loss < min_loss:
            in_cluster = centroid_idx
            min_loss = point_loss
    return [(in_cluster, (min_loss, point))]


def list_add_reducer(x, y):
    list_add = [0] * len(x)
    for i in range(len(x)):
        list_add[i] = x[i] + y[i]
    return list_add


def iterative_k_means(sc,  max_iter, init_centroids_file_path, data_file_path, method):
    centroids = get_init_centroids(init_centroids_file_path)
    losses = []
    for iter in range(1, max_iter + 2):
        print('iteration: {}'.format(iter))
        clusters_with_loss_and_points = sc.textFile(data_file_path).flatMap(
            lambda x: classify_and_get_loss(x, centroids, method)
        )
        clusters_with_loss = clusters_with_loss_and_points.map(lambda x: (x[0], x[1][0]))
        loss = sum(clusters_with_loss.reduceByKey(lambda x, y: x + y).values().collect())
        print('\tloss: {}'.format(loss))
        losses.append(loss)
        if iter == max_iter + 1:
            break
        clusters_with_points = clusters_with_loss_and_points.map(lambda x: (x[0], x[1][1]))
        clusters_points_sum = clusters_with_points.reduceByKey(list_add_reducer).collect()
        clusters_points_num_dict = dict(clusters_with_points.countByKey())
        for cluster_idx, cluster_points_sum in clusters_points_sum:
            cluster_points_num = clusters_points_num_dict[cluster_idx]
            centroids[cluster_idx] = [x / cluster_points_num for x in cluster_points_sum]
    improvement = (losses[0] - losses[-1]) / losses[0] * 100.
    losses = losses[1:]
    return losses, centroids, improvement


def plot_losses(png_save_path, c1_losses, c2_losses, max_iter, method):
    x = ['Round {}'.format(i) for i in range(1, max_iter + 1)]
    _ = plt.figure(0)
    plt.title(method)
    plt.plot(x, c1_losses, label='c1')
    plt.plot(x, c2_losses, label='c2')
    plt.legend(loc='upper right')
    plt.savefig(png_save_path)
    plt.close(0)


def record_centroid_pair_distance_in_csv(csv_file_path, centroids, method):
    with open(csv_file_path, "w", newline='') as csvfile:
        centroid_num = len(centroids)
        writer = csv.writer(csvfile)
        writer.writerow([method] + list(range(1, centroid_num + 1)))
        for i in range(centroid_num):
            row = [i + 1]
            for j in range(centroid_num):
                if i > j:
                    row += ['']
                    continue
                elif i == j:
                    value = 0.00
                else:
                    if method == 'Euclidean':
                        value = np.sqrt(np.sum(np.power(np.array(centroids[i]) - np.array(centroids[j]), 2)))
                    elif method == 'Manhattan':
                        value = np.sum(np.abs(np.array(centroids[i]) - np.array(centroids[j])))
                    else:
                        raise NameError
                row += [np.round(value, 2)]
            writer.writerow(row)


if __name__ == '__main__':

    MAX_ITER = 20

    EUCLIDEAN_METHOD = 'Euclidean'
    MANHATTAN_METHOD = 'Manhattan'
    METHODS = [EUCLIDEAN_METHOD, MANHATTAN_METHOD]

    C1_FILE_PATH = 'hw3-q2-kmeans/c1.txt'
    C2_FILE_PATH = 'hw3-q2-kmeans/c2.txt'
    DATA_FILE_PATH = 'hw3-q2-kmeans/data.txt'

    if not os.path.isdir('results'):
        os.mkdir('results')

    conf = SparkConf().setMaster("local").setAppName("kMeans")
    sc = SparkContext(conf=conf)

    start_time = time.time()

    pc_improvements = []

    for METHOD in METHODS:
        
        print('------------------------------------------------')
        print(METHOD)

        print('\nc1\n')
        c1_losses, c1_centroids, c1_improvement = iterative_k_means(sc, MAX_ITER, C1_FILE_PATH, DATA_FILE_PATH, METHOD)

        print('\nc2\n')
        c2_losses, c2_centroids, c2_improvement = iterative_k_means(sc, MAX_ITER, C2_FILE_PATH, DATA_FILE_PATH, METHOD)

        plot_losses('results/{}.png'.format(METHOD), c1_losses, c2_losses, MAX_ITER, METHOD)

        record_centroid_pair_distance_in_csv('results/{}_c1_{}.csv'.format(METHOD, EUCLIDEAN_METHOD), c1_centroids, EUCLIDEAN_METHOD)
        record_centroid_pair_distance_in_csv('results/{}_c1_{}.csv'.format(METHOD, MANHATTAN_METHOD), c1_centroids, MANHATTAN_METHOD)
        record_centroid_pair_distance_in_csv('results/{}_c2_{}.csv'.format(METHOD, EUCLIDEAN_METHOD), c2_centroids, EUCLIDEAN_METHOD)
        record_centroid_pair_distance_in_csv('results/{}_c2_{}.csv'.format(METHOD, MANHATTAN_METHOD), c2_centroids, MANHATTAN_METHOD)

        pc_improvements += [c1_improvement, c2_improvement]

    print(pc_improvements)

    end_time = time.time()
    print('running time: {:.2f} min'.format((end_time - start_time) / 60))







