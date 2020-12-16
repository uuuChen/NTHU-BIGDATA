from pyspark import SparkConf, SparkContext
import math
import random
import os
import networkx as nx
from numpy.random import choice
import numpy as np

RATINGS_PATH = "data/ft_ratings.txt"
TRUST_PATH = "data/ft_trust.txt"
DATA_DIR = 'data/cv_cold_start/'
DATA_FILE_NUM = 5


def get_trust_network(inputPath):
    global trust_network_graph
    trust_network_graph = nx.DiGraph()
    with open(inputPath, 'r') as file:
        line = file.readline()
        while line is not None and line != "":
            nodes = line.split(" ")
            trust_network_graph.add_edge(int(nodes[0]), int(nodes[1]))
            line = file.readline()


def get_split_rating_paths(test_model, data_dir, file_num):
    if test_model:
        file_idxs = list(range(0, file_num))
        # random.shuffle(file_idxs)
        used_rating_paths = [os.path.join(data_dir, 'ft-{}.csv'.format(rand_idx)) for rand_idx in file_idxs[:-1]]
        test_rating_path = [os.path.join(data_dir, 'ft-{}.csv'.format(file_idxs[-1]))]
    else:
        used_rating_paths = [RATINGS_PATH]
        test_rating_path = None
    return used_rating_paths, test_rating_path


def row_followees_map(row):
    follower, followee, trust_weight = row.split(" ")
    return int(follower), [int(followee)]


def row_followers_map(row):
    follower, followee, trust_weight = row.split(" ")
    return int(followee), [int(follower)]


def get_relations_dict(sc, path):
    followees_dict = dict(sc.textFile(path).map(row_followees_map).reduceByKey(lambda x, y: x + y).collect())
    followers_dict = dict(sc.textFile(path).map(row_followers_map).reduceByKey(lambda x, y: x + y).collect())
    return followers_dict, followees_dict


def row_userItemsRatings_map(row):
    userID, itemID, rating = row.split(" ")
    return int(float(userID)), [(int(float(itemID)), float(rating))]


def row_itemUsersRatings_map(row):
    userID, itemID, rating = row.split(" ")
    return int(float(itemID)), [(int(float(userID)), float(rating))]


def get_user_item_rating_dict(sc, paths):
    user_item_rating_rdd = None
    for path in paths:
        file_user_item_rating_rdd = sc.textFile(path).map(row_userItemsRatings_map).reduceByKey(lambda x, y: x + y)
        if not user_item_rating_rdd:
            user_item_rating_rdd = file_user_item_rating_rdd
        else:
            user_item_rating_rdd = user_item_rating_rdd.union(file_user_item_rating_rdd)
    user_item_rating_dict = dict(user_item_rating_rdd.reduceByKey(lambda x, y: x + y).collect())
    for key in user_item_rating_dict.keys():
        user_item_rating_dict[key] = dict(user_item_rating_dict[key])
    return user_item_rating_dict


def get_item_user_rating_dict(sc, paths):
    item_user_rating_rdd = None
    for path in paths:
        file_item_user_rating_rdd = sc.textFile(path).map(row_itemUsersRatings_map).reduceByKey(lambda x, y: x + y)
        if not item_user_rating_rdd:
            item_user_rating_rdd = file_item_user_rating_rdd
        else:
            item_user_rating_rdd = item_user_rating_rdd.union(file_item_user_rating_rdd)
    item_user_rating_dict = dict(item_user_rating_rdd.reduceByKey(lambda x, y: x + y).collect())
    for key in item_user_rating_dict.keys():
        item_user_rating_dict[key] = dict(item_user_rating_dict[key])

    item_rating_mean_list = sc.parallelize(item_user_rating_dict.items()).map(lambda x: get_rating_mean(x)).sortBy(
        lambda x: -x[1]).collect()
    item_user_rating_rdd = sc.parallelize(item_user_rating_dict.items()).map(lambda x: rating_normalization(x))
    item_norm_user_rating_dict = dict(item_user_rating_rdd.collect())
    return item_user_rating_dict, item_norm_user_rating_dict, item_rating_mean_list


def get_rating_mean(row):
    rating_mean = sum(row[1].values()) / len(row[1])
    return row[0], rating_mean


def rating_normalization(row):
    rating_mean = sum(row[1].values()) / len(row[1])
    for key in row[1].keys():
        row[1][key] = (row[1][key] - rating_mean)
    return row


def get_ratings_mean(sc, double_dict):
    ratings_rdd = sc.parallelize(list(double_dict.items())).flatMap(lambda x: list(x[1].values()))
    ratings_mean = ratings_rdd.sum() / ratings_rdd.count()
    return ratings_mean


def get_user_bias_dict(sc, user_item_rating_dict, all_ratings_mean):
    user_item_rating_rdd = sc.parallelize(list(user_item_rating_dict.items()))
    user_bias_rdd = user_item_rating_rdd.map(lambda x: (x[0], sum(x[1].values())/len(x[1]) - all_ratings_mean))
    user_bias_dict = dict(user_bias_rdd.collect())
    return user_bias_dict


def get_item_bias_dict(sc, item_user_rating_dict, all_ratings_mean):
    item_user_rating_rdd = sc.parallelize(list(item_user_rating_dict.items()))
    item_bias_rdd = item_user_rating_rdd.map(lambda x: (x[0], sum(x[1].values())/len(x[1]) - all_ratings_mean))
    item_bias_dict = dict(item_bias_rdd.collect())
    return item_bias_dict


def get_most_similar_itemID(sc, u_itemIDs, target_itemID, item_user_rating_dict):
    max_similarity = -100
    sum_simialrity = 0
    most_similar_itemID = 0
    target_item_usersRatings = item_user_rating_dict[target_itemID]
    for u_itemID in u_itemIDs:
        u_item_usersRatings = item_user_rating_dict[u_itemID]
        similarity = item_item_similarity(sc, target_item_usersRatings, u_item_usersRatings)
        sum_simialrity += similarity
        if similarity > max_similarity:
            max_similarity = similarity
            most_similar_itemID = u_itemID
    return max_similarity, most_similar_itemID


def join_two_items_map(item1_user_rating, item2_user_rating_dict):
    item1_userID, item1_user_rating = item1_user_rating
    item2_userIDs = item2_user_rating_dict.keys()
    if item1_userID in item2_userIDs:
        item2_user_rating = item2_user_rating_dict[item1_userID]
        return item1_userID, (item1_user_rating, item2_user_rating)
    else:
        return None, (0.0, 0.0)


def item_item_similarity(sc, item_1_ratings_dict, item_2_ratings_dict):
    item_1_ratings_rdd = sc.parallelize(list(item_1_ratings_dict.items()))
    item_2_ratings_rdd = sc.parallelize(list(item_2_ratings_dict.items()))
    item_1_len = math.sqrt(item_1_ratings_rdd.map(lambda x: math.pow(x[1], 2)).sum())
    item_2_len = math.sqrt(item_2_ratings_rdd.map(lambda x: math.pow(x[1], 2)).sum())
    if item_1_len == 0 or item_2_len == 0:
        return 0.0
    two_items_intersection = item_1_ratings_rdd.map(lambda x, item_2_ratings_dict=item_2_ratings_dict:
                                                    join_two_items_map(x, item_2_ratings_dict))
    two_items_innerProduct = two_items_intersection.map(lambda x: x[1][0] * x[1][1]).sum()
    cosine_similarity = two_items_innerProduct / (item_1_len * item_2_len)
    return cosine_similarity


def sigmoid_2(x):
    return 1.0 / (1.0 + math.exp(-x-1.5))


def get_followee(userID, user_item_rating_dict):
    global trust_network_graph
    probability_distribution = []
    if trust_network_graph.has_node(userID):
        friends = [n for n in trust_network_graph[userID]]
    else:
        return None, 0
    if len(friends) > 0:
        friends_out_degree_dict = dict(trust_network_graph.out_degree(friends))
        rated_next_userIDs = {}  # 有給出評分過的friend
        for userID in friends_out_degree_dict.keys():
            if userID in user_item_rating_dict.keys():
                rated_next_userIDs[userID] = friends_out_degree_dict[userID]
            # 如果該friend沒給出過評分，則不考慮他
            else:
                friends.remove(userID)
        friends_out_degree_dict = rated_next_userIDs
        total_degree = sum(friends_out_degree_dict.values())
        if len(friends) == 0 or total_degree == 0:
            return None, 0
        for item in list(friends_out_degree_dict.values()):
            probability_distribution.append(item / total_degree)
        next_userID = choice(friends, size=1, p=probability_distribution)[0]
        to_user_prob = friends_out_degree_dict[next_userID] / total_degree
        return next_userID, to_user_prob
    else:
        return None, 0


def done_log(str):
    print('\t' + str)
    print('---------------------done-------------------------')


def keep_walking_log(most_sim_itemID_sim, most_sim_item_rating, to_user_prob, next_userID, system_prob, stop_prob):
    print('\tcurrent user max similarity: {:.2f}'.format(most_sim_itemID_sim))
    print('\tcurrent user rating: {:.2f}'.format(most_sim_item_rating))
    print('\tcurrent user friend numbers: {}'.format(int(1 / to_user_prob)))
    print('\tnext userID: {}'.format(next_userID))
    print('\tsystem prob: {:.2f}'.format(system_prob))
    print('\tstop prob: {:.2f}'.format(stop_prob))


def get_user_item_mean_rating(userID, itemID, ratings_mean, user_bias_dict, item_bias_dict):
    target_user_bias = user_bias_dict[userID] if userID in user_bias_dict.keys() else 0.0
    target_item_bias = item_bias_dict[itemID] if itemID in item_bias_dict.keys() else 0.0
    mean_rating = (round(ratings_mean + target_user_bias + target_item_bias) / 0.5) * 0.5
    return mean_rating


def get_return_rating(sc, all_steps_results, user_bias_dict, item_bias_dict, ratings_mean):
    # each step result: (userID, itemID, similarity, rating)
    all_steps_results_rdd = sc.parallelize(all_steps_results)
    ratings_weights_rdd = all_steps_results_rdd.map(lambda x: (x[3], [x[2]])).reduceByKey(lambda x, y: x + y).map(
        lambda x: (x[0], max(x[1])))
    ratings_weights_sum = ratings_weights_rdd.values().sum() + 1e-5
    norm_ratings_weights_rdd = ratings_weights_rdd.map(lambda x: (x[0], x[1] / ratings_weights_sum))
    max_weight_rating = norm_ratings_weights_rdd.sortBy(lambda x: -x[1]).collect()[0][0]  # descending
    max_weight_item_sim = ratings_weights_rdd.filter(lambda x: x[0] == max_weight_rating).collect()[0][1]
    avg_weight_rating = round(norm_ratings_weights_rdd.map(lambda x: x[0] * x[1]).sum() / 0.5) * 0.5
    if avg_weight_rating == 0.0:
        target_userID = all_steps_results[0][0]
        target_itemID = all_steps_results[0][0]
        mean_rating = get_user_item_mean_rating(target_userID, target_itemID, ratings_mean, user_bias_dict,
                                                item_bias_dict)
        return mean_rating, 0.0
    else:
        return max_weight_rating, max_weight_item_sim


def trustWalker(sc, followees_dict, user_item_rating_dict, item_user_rating_dict, userID, target_itemID, user_bias_dict,
                item_bias_dict, ratings_mean, k, all_steps_results,  system_prob=1.0):
    print('------------------------k={}------------------------'.format(k))
    global finished
    finished = False
    if target_itemID not in item_user_rating_dict.keys():  # no one has ratted on the target item, return
        item_sim = 0.0
        mean_rating = get_user_item_mean_rating(userID, target_itemID, ratings_mean, user_bias_dict, item_bias_dict)
        all_steps_results.append((userID, target_itemID, item_sim, mean_rating))
        rtn_str = 'no_ratted_item'
        done_log('No one has ratted on ITEM {}'.format(target_itemID))
        rtn_rating, rtn_sim = get_return_rating(sc, all_steps_results, user_bias_dict, item_bias_dict, ratings_mean)
        return system_prob, rtn_sim, rtn_rating, rtn_str
    if userID in user_item_rating_dict.keys():  # the user have ratted some items
        user_ratted_itemIDs = list(user_item_rating_dict[userID].keys())
        if target_itemID in user_ratted_itemIDs:  # target item has been ratted by current user, return
            item_sim = 1.0
            user_target_item_rating = user_item_rating_dict[userID][target_itemID]
            all_steps_results.append((userID, target_itemID, item_sim, user_target_item_rating))
            finished = True
            rtn_str = 'found_item'
            done_log('Found ITEM {} in USER {}, RETURN'.format(target_itemID, userID))
        else:  # target item hasn't been ratted by current user before
            most_sim_itemID_sim, most_sim_itemID = get_most_similar_itemID(sc, user_ratted_itemIDs, target_itemID,
                                                                           item_user_rating_dict)
            most_sim_itemID_rating = user_item_rating_dict[userID][most_sim_itemID]
            all_steps_results.append((userID, most_sim_itemID, most_sim_itemID_sim, most_sim_itemID_rating))
            stop_prob = min(sigmoid_2(5 * (most_sim_itemID_sim - 0.5)) + k * 0.02, 1.0)
            rand_num = random.uniform(0, 1)
            if rand_num < stop_prob or k == 6:  # have to stop
                system_prob = system_prob * stop_prob
                finished = True
                rtn_str = 'stop_prob'
                if k != 6:
                    done_log('Stopped by stop probability {:.2f}, RETURN'.format(stop_prob))
            else:  # get next user and decide if it needs to keep random walk
                next_userID, to_user_prob = get_followee(userID, user_item_rating_dict)
                system_prob = system_prob * (1 - stop_prob)
                if not next_userID:  # no trusted friends, return
                    finished = True
                    rtn_str = 'no_friend'
                    done_log('User {} has no friends, RETURN'.format(userID))
                else:  # keep random walk
                    keep_walking_log(most_sim_itemID_sim, most_sim_itemID_rating, to_user_prob, next_userID, system_prob,
                                     stop_prob)
                    system_prob = system_prob * to_user_prob
                    k += 1
                    return trustWalker(sc, followees_dict, user_item_rating_dict, item_user_rating_dict, next_userID,
                                       target_itemID, user_bias_dict, item_bias_dict, ratings_mean, k,
                                       all_steps_results, system_prob)
    else:  # the user doesn't have ratted items, just keep random walk
        next_userID, to_user_prob = get_followee(userID, user_item_rating_dict)
        print('\tUser {} has no ratted items'.format(userID))
        if not next_userID:  # no trusted friends, return
            finished = True
            item_sim = 0.0
            mean_rating = get_user_item_mean_rating(userID, target_itemID, ratings_mean, user_bias_dict, item_bias_dict)
            all_steps_results.append((userID, target_itemID, item_sim, mean_rating))
            rtn_str = 'no_ratted_and_friend'
            done_log('User {} has no friends, RETURN'.format(userID))
        else:
            k += 1
            rtn_str = 'no_ratted_user'
            trustWalker_rtn = trustWalker(sc, followees_dict, user_item_rating_dict, item_user_rating_dict, next_userID,
                               target_itemID, user_bias_dict, item_bias_dict, ratings_mean, k, all_steps_results,
                               system_prob)
            return trustWalker_rtn[0], trustWalker_rtn[1], trustWalker_rtn[2], rtn_str
    if finished:
        rtn_rating, rtn_sim = get_return_rating(sc, all_steps_results, user_bias_dict, item_bias_dict, ratings_mean)
        return system_prob, rtn_sim, rtn_rating, rtn_str


def trans_double_dict_to_flat_list(double_dict):
    flat_list = []
    for firstKey in double_dict.keys():
        for secondKey in double_dict[firstKey].keys():
            flat_list.append((firstKey, secondKey, double_dict[firstKey][secondKey]))
    return flat_list


def recommender(sc, userID, followees_dict, user_item_rating_dict, item_user_rating_dict, item_rating_mean_list
                , used_user_bias_dict, used_item_bias_dict, ratings_mean, sim_threshold, rating_threshold, rec_number):
    result = []
    recommend_movies = []
    all_items = [item[0] for item in item_rating_mean_list]
    have_seen_items = user_item_rating_dict[userID].keys()
    for item in all_items:
        if item in have_seen_items: continue
        r = (item, trustWalker(sc, followees_dict, user_item_rating_dict, item_user_rating_dict, userID, item,
                               used_user_bias_dict, used_item_bias_dict, ratings_mean, 0, []))
        _, (system_prob, similarity, rating, have_trusted_friends) = r
        print('(item={}, (system_prob={}, similarity={}, rating={}, have_friends={}))'.format(item, system_prob,
                                                                                              similarity, rating,
                                                                                              have_trusted_friends))
        result.append(r)
        if (r[1][1] >= sim_threshold) and (r[1][2] >= rating_threshold):
            recommend_movies.append((r[0], r[1][2]))
            if len(recommend_movies) >= rec_number:
                break
    recommend_movies.sort(key= lambda x: x[1], reverse=True)
    return recommend_movies


def main():
    # build pyspark
    conf = SparkConf().setMaster("local").setAppName("trustWalker")
    sc = SparkContext(conf=conf)

    # build environment
    test_model_with_csv_file = True  # if true, use "data/cv/*.csv" to predict, else, use "ft_ratings.txt"
    get_trust_network(TRUST_PATH)
    followers_dict, followees_dict = get_relations_dict(sc, TRUST_PATH)
    used_rating_paths, test_rating_path = get_split_rating_paths(test_model_with_csv_file, DATA_DIR, DATA_FILE_NUM)
    used_user_item_rating_dict = get_user_item_rating_dict(sc, used_rating_paths)  # for random walk
    # for item-item similarity
    used_item_user_rating_dict, used_item_norm_user_rating_dict, item_rating_mean_list = get_item_user_rating_dict(
        sc, used_rating_paths)
    used_ratings_mean = get_ratings_mean(sc, used_user_item_rating_dict)
    used_user_bias_dict = get_user_bias_dict(sc, used_user_item_rating_dict, used_ratings_mean)
    used_item_bias_dict = get_item_bias_dict(sc, used_item_user_rating_dict, used_ratings_mean)

    # loss measurement object
    abs_loss = lossAvgMeter(lambda x, y: abs(x - y))
    found_item_abs_loss = lossAvgMeter(lambda x, y: abs(x - y))
    stop_prob_abs_loss = lossAvgMeter(lambda x, y: abs(x - y))
    no_friend_abs_loss = lossAvgMeter(lambda x, y: abs(x - y))
    no_ratted_item_abs_loss = lossAvgMeter(lambda x, y: abs(x - y))
    no_ratted_user_abs_loss = lossAvgMeter(lambda x, y: abs(x - y))
    no_ratted_and_friend_abs_loss = lossAvgMeter(lambda x, y: abs(x - y))
    rand_abs_loss = lossAvgMeter(lambda x, y: abs(x - y))
    possible_ratings = list(np.arange(0.5, 4.5, 0.5))

    # start the algorithm
    if test_model_with_csv_file:
        test_user_item_rating_dict = get_user_item_rating_dict(sc, test_rating_path)
        test_user_item_rating_list = trans_double_dict_to_flat_list(test_user_item_rating_dict)
        # random.shuffle(test_user_item_rating_list)
        counts = 0
        for test_userID, test_itemID, true_rating in test_user_item_rating_list:
                system_prob, item_sim, pred_rating, return_reason = (
                    trustWalker(sc, followees_dict, used_user_item_rating_dict, used_item_norm_user_rating_dict, test_userID,
                                test_itemID, used_user_bias_dict, used_item_bias_dict, used_ratings_mean, 0, []))
                counts += 1
                print(counts)
                # compute mean loss and print log
                rand_rating = random.choice(possible_ratings)
                rand_abs_loss.add(true_rating, rand_rating)
                abs_loss.add(true_rating, pred_rating)
                if return_reason == 'no_ratted_item':
                    no_ratted_item_abs_loss.add(true_rating, pred_rating)
                elif return_reason == 'no_ratted_user':
                    no_ratted_user_abs_loss.add(true_rating, pred_rating)
                elif return_reason == 'found_item':
                    found_item_abs_loss.add(true_rating, pred_rating)
                elif return_reason == 'stop_prob':
                    stop_prob_abs_loss.add(true_rating, pred_rating)
                elif return_reason == 'no_friend':
                    no_friend_abs_loss.add(true_rating, pred_rating)
                elif return_reason == 'no_ratted_and_friend':
                    no_ratted_and_friend_abs_loss.add(true_rating, pred_rating)
                print("\t(user: {}, item: {}) : (true rating: {}, predict rating: {}, similarity {:.2f})  | "
                      " system prob: {:.2f})".format(test_userID, test_itemID, true_rating, pred_rating, item_sim,
                                                  system_prob))
                print("\t\tmae: {:.2f}, count: {}".format(abs_loss.mean, abs_loss.count))
                print("\t\tfound_item_mae: {:.2f}, count: {}".format(found_item_abs_loss.mean,
                                                                     found_item_abs_loss.count))
                print("\t\tstop_prob_mae: {:.2f}, count: {}".format(stop_prob_abs_loss.mean, stop_prob_abs_loss.count))
                print("\t\tno_friend_mae: {:.2f}, count: {}".format(no_friend_abs_loss.mean, no_friend_abs_loss.count))
                print("\t\tno_ratted_item_mae: {:.2f}, count: {}".format(no_ratted_item_abs_loss.mean,
                                                                         no_ratted_item_abs_loss.count))
                print("\t\tno_ratted_user_mae: {:.2f}, count: {}".format(no_ratted_user_abs_loss.mean,
                                                                         no_ratted_user_abs_loss.count))
                print("\t\tno_ratted_and_friend_mae: {:.2f}, count: {}".format(no_ratted_and_friend_abs_loss.mean,
                                                                               no_ratted_and_friend_abs_loss.count))
                print("\t\trand_mae: {:.2f}, count: {}".format(rand_abs_loss.mean, rand_abs_loss.count))
    else:
        userID = int(input("Input User ID: "))
        sim_threshold = 0.5
        rating_threshold = 3.5
        recommend_movie_number = 3
        recommend_movies = recommender(sc, userID, followees_dict, used_user_item_rating_dict,
                                       used_item_norm_user_rating_dict, item_rating_mean_list, used_user_bias_dict,
                                       used_item_bias_dict, used_ratings_mean, sim_threshold, rating_threshold,
                                       recommend_movie_number)

        print("Recommended Movies:")
        for movie in recommend_movies:
            print("movie: {}    rating: {}".format(movie[0], movie[1]))


class lossAvgMeter:
    def __init__(self, loss_func):
        self.sum = 0
        self.mean = 0
        self.count = 0
        self._loss_func = loss_func

    def add(self, item1, item2):
        self.count += 1
        loss = self._loss_func(item1, item2)
        self.sum += loss
        self.mean = float(self.sum / self.count)


if __name__ == "__main__":
    main()






