import json, os, random, re
import argparse
import numpy as np
import time
import pandas as pd

maxlen = 50

def get_origin7rumor7non_rumor(data_root_dir='download/weibo2', min_review_length=150):
    weibo_content_map = {}
    review_content_map = {}
    user_weibo_map = {}
    origin_file_path = os.path.join(data_root_dir, 'original-microblog')
    origin_file_names = os.listdir(origin_file_path)
    origin_mid_json_map = {}
    for file in origin_file_names:
        if not file.endswith('.json'):
            continue
        path = os.path.join(origin_file_path, file)
        user = file.split("_")[-1].split('.')[0]
        weibo_name = path.split('_')[-2]
        if user not in user_weibo_map:
            user_weibo_map[user] = []
        user_weibo_map[user].append(weibo_name)
        with open(path, 'r', encoding='utf-8') as f:
            jsoncontent = json.load(f)
            origin_mid_json_map[weibo_name] = jsoncontent
            weibo_content_map[file] = jsoncontent['text']
    rumor_file_path = os.path.join(data_root_dir, 'rumor-repost')
    rumor_file_names = os.listdir(rumor_file_path)
    rumor_mid_reviews_map = {}
    for file in rumor_file_names:
        if not file.endswith('.json'):
            continue
        path = os.path.join(rumor_file_path, file)
        rumor_name = path.split('_')[-2]
        origin_mid_json_map[rumor_name]['label'] = 'false'
        with open(path, 'r', encoding='utf-8') as f:
            reivews = json.load(f)
            rumor_mid_reviews_map[rumor_name] = select_reviews(reivews, min_review_length,review_content_map)

    non_rumor_file_path = os.path.join(data_root_dir, 'non-rumor-repost')
    non_rumor_file_names = os.listdir(non_rumor_file_path)
    non_rumor_mid_reviews_map = {}
    for file in non_rumor_file_names:
        if not file.endswith('.json'):
            continue
        path = os.path.join(non_rumor_file_path, file)
        non_rumor_name = path.split('_')[-2]
        origin_mid_json_map[non_rumor_name]['label'] = 'non-rumor'
        with open(path, 'r', encoding='utf-8') as f:
            reviews = json.load(f)
            non_rumor_mid_reviews_map[non_rumor_name] = select_reviews(reviews,min_review_length,review_content_map)
    return origin_mid_json_map, rumor_mid_reviews_map, non_rumor_mid_reviews_map, user_weibo_map,\
            weibo_content_map,review_content_map


def select_reviews(reviews, min_review_length,review_content_map):
    review_list = []
    for review in reviews:
        if review['parent'] == '' and len(review['text']) > min_review_length:
                review_list.append(review)
                review_content_map[review['mid']] = review['text']
    return review_list


def create_input_files(data_root_dir='download/weibo2', ratio=[70, 10, 20], data_set_name='weibo2', min_review_length=150):
    origin, rumor, non_rumor, user_weibo_map_old, weibo_content_map,review_content_map = \
        get_origin7rumor7non_rumor(data_root_dir,min_review_length)
    origin_list = list(origin.keys())

    name_list = []
    name_list.extend(list(user_weibo_map_old.keys()))
    name_list.extend(list(origin.keys()))

    name_reviews_map = rumor.copy()
    name_reviews_map.update(non_rumor)

    review_num = 0
    for name, reviews in name_reviews_map.items():
        for review in reviews:
            name_list.append(review['mid'])
            review_num += 1

    with open(os.getcwd() + '/weibo_files/comment_content.json', 'w') as f:
        f.write(json.dumps(review_content_map, indent=4))
    with open(os.getcwd() + '/weibo_files/user_tweet.json', 'w') as f:
        f.write(json.dumps(user_weibo_map_old, indent=4))

    weibo_content_map = {name: contents['text'] for name, contents in origin.items()}
    weibo_label_map = {name: contents['label'] for name, contents in origin.items()}
    user_weibo_map = {user: [weibo for weibo in weibos] for user, weibos in
                      user_weibo_map_old.items()}
    weibo_review_map = {weibo_name: [review['mid'] for review in reviews] for
                        weibo_name, reviews in name_reviews_map.items()}
    adj_node_map = {}
    adj_node_map.update(user_weibo_map)
    adj_node_map.update(weibo_review_map)
    if not os.path.exists(data_set_name):
        os.mkdir(data_set_name)
    counts = len(origin_list)
    random.seed(666)
    random.shuffle(origin_list)
    train_num = int(counts * ratio[0] / 100.0)
    dev_num = int(counts * ratio[1] / 100.0)
    train = origin_list[:train_num]
    dev = origin_list[train_num:train_num + dev_num]
    test = origin_list[train_num + dev_num:]

    write2file(data_set_name,weibo_content_map, weibo_label_map, adj_node_map, name_list=train,
               suffix='.train')
    write2file(data_set_name, weibo_content_map, weibo_label_map, adj_node_map, name_list=dev,
               suffix='.dev')
    write2file(data_set_name, weibo_content_map, weibo_label_map, adj_node_map, name_list=test,
               suffix='.test')
    write2file(data_set_name, weibo_content_map, weibo_label_map, adj_node_map,
               name_list=list(adj_node_map.keys()),
               suffix='_graph.txt')
    path = os.getcwd() + '/weibo_files/original_adj'
    adj_node_map_tmp = {}
    for i, v in adj_node_map.items():
        if len(v) == 0:
            continue
        adj_node_map_tmp[i] = v
    adj_node_map = adj_node_map_tmp
    with open(path, 'w') as f:
        json.dump(adj_node_map, f)


def write2file(data_set_name, weibo_content_map, weibo_label_map, adj_node_map, name_list,
               suffix='.train'):
    path = os.getcwd() + '/weibo_files/' + data_set_name + suffix
    if suffix == '_graph.txt' :
        count = 0
        with open(os.path.join(path), 'w', encoding='utf-8') as f:
            for id_ in name_list:
                if len(adj_node_map[id_]) < 1:
                    count += 1
                    continue
                line = str(id_) + '\t'
                for node in adj_node_map[id_]:
                    line = line + str(node) + ':' + '1' + ' '
                line = line + '\n'
                f.write(line)
    else:
        with open(os.path.join(path), 'w', encoding='utf-8') as f:
            for e in name_list:
                id_ = e
                line = str(id_) + '\t' + re.sub(pattern=r'\t', repl='', string=weibo_content_map[id_]) + '\t' + \
                       weibo_label_map[id_] + '\n'
                f.write(line)

def main():
    data_dir = os.path.join(os.getcwd(),'weibocontentwithimage/')
    ratio = [70,10,20]
    create_input_files(data_root_dir=data_dir, ratio=ratio, data_set_name='weibo', min_review_length=80)

if __name__ == '__main__':
    main()
