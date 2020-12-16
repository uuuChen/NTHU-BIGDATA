from pyspark import SparkConf, SparkContext
import numpy as np
import time


def line_mapper(line):
    out_node, in_node = line.split('\t')
    return [(out_node, in_node)]


def count_per_contribution(x, r, arr_idx_dict, Beta):
    (out_node, (in_node, d)) = x
    part_new_r = Beta * r[arr_idx_dict[int(out_node)]] / d
    return in_node, part_new_r


def manhattan_distance(x, y):
    return np.sum(np.abs(x - y))


def zero_padding(new_vertex_pair, arr_idx_dict):
    zero_padding_list = [0] * len(arr_idx_dict.keys())
    for new_vertex_idx, new_vertex_pageRank in new_vertex_pair:
        zero_padding_list[arr_idx_dict[new_vertex_idx]] = new_vertex_pageRank
    return zero_padding_list


def count_pageRank(file_path, iteration, epsilon):
    conf = SparkConf().setMaster("local").setAppName("count_pageRank")
    sc = SparkContext(conf=conf)
    out_in_nodes = sc.textFile(file_path).flatMap(line_mapper)
    sorted_vertex_idxs = sorted(list(map(int, set((out_in_nodes.keys() + out_in_nodes.values()).collect()))))
    N = len(sorted_vertex_idxs)
    vertex_to_r_idxs_dict = dict(zip(sorted_vertex_idxs, list(range(N))))
    new_r = old_r = np.array([1 / N] * N).T
    dead_end_exist = False
    nodes_conns_counts = out_in_nodes.map(lambda x: (x[0], 1)).reduceByKey(lambda x, y: x + y).collect()
    if len(nodes_conns_counts) < N:
        dead_end_exist = True
    out_in_nodes_with_d = out_in_nodes.join(sc.parallelize(out_in_nodes.countByKey().items()))
    for i in range(iteration):
        print('iteration: {}'.format(i + 1))
        in_nodes_with_value = out_in_nodes_with_d.map(lambda x: count_per_contribution(x, old_r, vertex_to_r_idxs_dict, BETA))
        one_minus_beta_NN = sum(old_r) * (1 - BETA) / N
        new_r = in_nodes_with_value.reduceByKey(lambda x, y: x + y).sortByKey().map(
            lambda x: (int(x[0]), x[1] + one_minus_beta_NN)).sortByKey()
        new_r = np.array(zero_padding(new_r.collect(), vertex_to_r_idxs_dict)).T
        if dead_end_exist:
            new_r = new_r + (1 - sum(new_r)) / N
        m_dist = manhattan_distance(new_r, old_r)
        if m_dist < epsilon:
            break
        else:
            print('\tmanhattan_distance: {}'.format(m_dist))
        old_r = new_r
    sort_r_dict = sorted(dict(zip(sorted_vertex_idxs, new_r)).items(), key=lambda d: d[1], reverse=True)
    return sort_r_dict


def write_results_to_file(results, list_num):
    with open("results.txt", "w") as f:
        for i, result in list(enumerate(results, start=1)):
            vertex, pageRank = result
            write_str = '{}\t{:.13f}\n'.format(vertex, pageRank)
            f.write(write_str)
            if i == list_num:
                break


if __name__ == '__main__':
    FILE_PATH = 'p2p-Gnutella04.txt'
    BETA = 0.8
    ITERATION = 1000
    EPSILON = 1e-15
    LIST_NUM = 10
    start_time = time.time()
    results = count_pageRank(FILE_PATH, ITERATION, EPSILON)
    end_time = time.time()
    print('running time: {:.2f} min'.format((end_time - start_time) / 60))
    write_results_to_file(results, LIST_NUM)
