from pyspark import SparkConf, SparkContext
import random
import numpy as np
import time


def mapper1(line):
    matrix_name, row, col, num = line.split(",")
    row, col, num = int(row), int(col), int(num)
    mapList = []
    for idx in range(MATRIX_SIZE):
        if matrix_name == 'M':
            key = (row, idx, col)
            value = num
        else:
            key = (idx, col, row)
            value = num
        mapList.append([key, value])
    return mapList


def reducer1(x, y):
    return x * y


def mapper2(x):  # x => ((0, 0, 0), 10)
    row, col, idx = x[0]
    key = (row, col)
    value = x[1]
    return [(key, value)]


def reducer2(x, y):
    return x + y


def map_and_reduce(file_path):
    conf = SparkConf().setMaster("local").setAppName("matrix_multiplication")
    sc = SparkContext(conf=conf)
    lines = sc.textFile(file_path).flatMap(mapper1)
    lines = lines.reduceByKey(reducer1)
    lines = lines.flatMap(mapper2)
    lines = lines.reduceByKey(reducer2)
    return lines.collect()


def gen_test_case(matrix_size):
    M = []
    N = []
    with open("{}input.txt".format(matrix_size), 'w') as f:
        for i in range(matrix_size):
            M_row = []
            for j in range(matrix_size):
                item = random.randint(1, 10)
                M_row.append(item)
                write_str = '{},{},{},{}\n'.format('M', i, j, item)
                f.write(write_str)
            M.append(M_row)

        for i in range(matrix_size):
            N_row = []
            for j in range(matrix_size):
                item = random.randint(1, 10)
                N_row.append(item)
                write_str = '{},{},{},{}\n'.format('N', i, j, item)
                f.write(write_str)
            N.append(N_row)

    answer = np.matmul(np.array(M), np.array(N))
    with open("{}input_answer.txt".format(matrix_size), 'w') as f:
        for i in range(answer.shape[0]):
            for j in range(answer.shape[1]):
                item = answer[i, j]
                write_str = '{},{},{}\n'.format(i, j, item)
                f.write(write_str)


def write_results_to_file(results, matrix_size):
    sort_key = lambda x: (x[0], x[1])
    results.sort(key=sort_key)
    with open("{}output.txt".format(matrix_size), 'w') as f:
        for result in results:
            row, col = result[0]
            value = result[1]
            write_str = '{},{},{}\n'.format(row, col, value)
            f.write(write_str)


if __name__ == '__main__':
    MATRIX_SIZE = 500
    # gen_test_case(MATRIX_SIZE)
    FILE_PATH = '{}input.txt'.format(MATRIX_SIZE)
    start_time = time.time()
    results = map_and_reduce(file_path=FILE_PATH)
    end_time = time.time()
    print('running time: {:.2f} min'.format((end_time - start_time) / 60))
    write_results_to_file(results, MATRIX_SIZE)
