import json, os, random, re,csv
import argparse
import numpy as np
import time
import pandas as pd

def create_input_files4pheme_new(mid):
    all_maps_lists_needs = get_map_and_list_pheme(mid)
    create_train_dev_test(all_maps_lists_needs, [70,11,19])
    create_graph_txt(all_maps_lists_needs)
    create_original_adj(all_maps_lists_needs)


def get_map_and_list_pheme(mid):
    tweet_content_map = {}
    review_content_map = {}
    tweet_label_map = {}
    user_tweet_map = {}
    tweet_review_map = {}

    dir_path =  os.getcwd() + '/phemewithreactions/'
    process_rumor_norumor4pheme(dir_path, 'non_romor', mid,tweet_label_map, tweet_content_map,review_content_map, \
                                tweet_review_map,user_tweet_map)
    process_rumor_norumor4pheme(dir_path, 'rumor',mid, tweet_label_map, tweet_content_map, review_content_map,\
                                tweet_review_map,user_tweet_map)
    user_num = 0
    for k, v in tweet_review_map.items():
        user_num = user_num + len(v)
    with open(os.getcwd() + '/pheme_files/comment_content.json', 'w') as f:
        f.write(json.dumps(review_content_map, indent=4))
    with open(os.getcwd() + '/pheme_files/user_tweet.json', 'w') as f:
        f.write(json.dumps(user_tweet_map, indent=4))
    return [tweet_content_map, tweet_label_map, user_tweet_map, \
            tweet_review_map]

def process_rumor_norumor4pheme(d, filename,mid, tweet_label_map, tweet_content_map,review_content_map,
                                tweet_review_map, user_tweet_map):
    path = os.path.join(d, filename)
    files = os.listdir(path)
    for file in files:
        if file not in mid:continue
        tweet_label_map[file] = 'non-rumor' if filename == 'non_romor' else 'false'
        path2 = os.path.join(path, file, 'source-tweet')
        source = os.listdir(path2)[0]
        path2 = os.path.join(path2, source)
        with open(path2, 'r') as f:
            res = json.load(f)
            tweet_content_map[file] = res['text']
            user_id = res['user']['id']
            if len(user_tweet_map) < 100:
                if user_id not in user_tweet_map:
                    user_tweet_map[user_id] = []
                user_tweet_map[user_id].append(file)
        path2 = os.path.join(path, file, 'reactions')
        if not os.path.exists(path2):
            continue
        reactions = os.listdir(path2)
        tmp = []
        for i, reaction in enumerate(reactions):
            if i == 5:
                break
            if reaction.endswith(".json"):
                file_id = reaction.split(".")[0]
                if file_id == file:
                    continue
                path3 = os.path.join(path2, reaction)
                with open(path3, 'r') as f:
                    res = json.load(f)
                    if len(res['text']) < 130:continue
                    review_content_map[file_id] = res['text']
                tmp.append(file_id)
        if file not in tweet_review_map:
            tweet_review_map[file] = []
        tweet_review_map[file].extend(tmp)

def create_train_dev_test(all_maps_lists_needs: list, ratio):
    tweet_content_map, tweet_label_map, user_tweet_map, tweet_review_map = all_maps_lists_needs
    rumor = [(k, v) for k, v in tweet_content_map.items() if tweet_label_map[k] == 'false']
    nonrumor = [(k, v) for k, v in tweet_content_map.items() if tweet_label_map[k] == 'non-rumor']
    rumor_train, rumor_dev, rumor_test = split_dataset(ratio, rumor)
    nonrumor_train, nonrumor_dev, nonrumor_test = split_dataset(ratio, nonrumor)
    train = rumor_train + nonrumor_train
    dev = rumor_dev + nonrumor_dev
    test = rumor_test + nonrumor_test
    random.seed(666)
    random.shuffle(train)
    random.shuffle(dev)
    random.shuffle(test)
    write_train_dev_test_to_file(suffix='.train', dataset=train, tweet_label_map=tweet_label_map)
    write_train_dev_test_to_file(suffix='.dev', dataset=dev, tweet_label_map=tweet_label_map)
    write_train_dev_test_to_file(suffix='.test', dataset=test, tweet_label_map=tweet_label_map)

def split_dataset(ratio: list, dataset: list):
    random.seed(666)
    random.shuffle(dataset)
    n = len(dataset)
    train_num = int(n * ratio[0] / 100)
    dev_num = int(n * ratio[1] / 100)
    train = dataset[:train_num]
    dev = dataset[train_num:train_num + dev_num]
    test = dataset[train_num + dev_num:]
    return train, dev, test


def write_train_dev_test_to_file(suffix: str, dataset: list, tweet_label_map: dict):
    path = os.getcwd() + '/pheme_files/pheme' + suffix
    with open(path, 'w') as f:
        for item in dataset:
            id_ = item[0]
            content = re.sub(pattern='[\t\n]', repl='', string=item[1])
            label = tweet_label_map[id_]
            line = str(id_) + '\t' + content + '\t' + label + '\n'
            f.write(line)

def create_graph_txt(all_maps_lists_needs: list):
    tweet_content_map, tweet_label_map, user_tweet_map, tweet_review_map = all_maps_lists_needs
    path = os.getcwd() + '/pheme_files/' + 'pheme_graph.txt'
    with open(path, 'w') as f:
        for k, v in user_tweet_map.items():
            id_ = k
            line = str(id_) + '\t'
            for t in v:
                line = line + str(t) + ':1 '
            line = line.strip() + '\n'
            f.write(line)
        for k, v in tweet_review_map.items():
            id_ = k
            line = str(id_) + '\t'
            for t in v:
                line = line + str(t) + ':1 '
            line = line.strip() + '\n'
            f.write(line)


def create_original_adj(all_maps_lists_needs: list):
    tweet_content_map, tweet_label_map, user_tweet_map, tweet_review_map = all_maps_lists_needs
    path = os.getcwd() + '/pheme_files/' + 'original_adj'
    original_adj_dict = {}
    original_adj_dict.update(user_tweet_map)
    original_adj_dict.update(tweet_review_map)
    original_adj_dict_tmp = {}
    for i, v in original_adj_dict.items():
        if len(v) == 0:
            continue
        original_adj_dict_tmp[i] = v
    original_adj_dict = original_adj_dict_tmp
    with open(path, 'w') as f:
        json.dump(original_adj_dict, f)

def main():
    path = os.getcwd() + '/content.csv'
    with open(path, 'r') as f:
        reader = csv.reader(f)
        result = list(reader)[1:]
        mid_img = []
        for line in result:
            mid_img.append(line[1])
    create_input_files4pheme_new(mid_img)

if __name__ == '__main__':
    main()