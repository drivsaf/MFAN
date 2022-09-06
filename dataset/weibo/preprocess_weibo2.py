import os
import itertools
import re
from collections import Counter
import gensim
import numpy as np
import scipy.sparse as sp
import pickle
import jieba
import json

jieba.set_dictionary(os.getcwd() + '/weibo_files/dict.txt.big')
w2v_dim = 300
use_stopwords = True
dic = {
    'non-rumor': 0,
    'false': 1,
    'unverified': 2,
    'true': 3,
}
stopwords_path = os.getcwd() + '/weibo_files/stopwords.txt'
stopwords = []
with open(stopwords_path, 'r') as f:
    for line in f.readlines():
        stopwords.append(line.strip())

def clean_str_cut(string, task):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """

    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)

    words = list(
        jieba.cut(string.strip().lower(), cut_all=False)) if "weibo" in task else string.strip().lower().split()
    if use_stopwords:
        words = [w for w in words if w not in stopwords]
    return words


def build_symmetric_adjacency_matrix(edges, shape):
    def normalize_adj(mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv_sqrt = np.power(rowsum, -0.5).flatten()
        r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
        r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
        return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)
    adj = sp.coo_matrix(arg1=(edges[:, 2], (edges[:, 0], edges[:, 1])), shape=shape, dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))
    return adj.tocoo()


def read_corpus(root_path, file_name):
    X_tids = []
    X_uids = []
    old_id_post_map = {}
    with open(root_path + file_name + ".train", 'r', encoding='utf-8') as input:
        X_train_tid, X_train_content, y_train = [], [], []
        for line in input.readlines():
            tid, content, label = line.strip().split("\t")
            X_tids.append(tid)
            X_train_tid.append(tid)
            fenci_res = clean_str_cut(content, file_name)
            X_train_content.append(fenci_res)
            y_train.append(dic[label])
            old_id_post_map[tid] = fenci_res

    with open(root_path + file_name + ".dev", 'r', encoding='utf-8') as input:
        X_dev_tid, X_dev_content, y_dev = [], [], []
        for line in input.readlines():
            tid, content, label = line.strip().split("\t")
            X_tids.append(tid)
            X_dev_tid.append(tid)
            fenci_res = clean_str_cut(content, file_name)
            X_dev_content.append(fenci_res)
            y_dev.append(dic[label])
            old_id_post_map[tid] = fenci_res

    with open(root_path + file_name + ".test", 'r', encoding='utf-8') as input:
        X_test_tid, X_test_content, y_test = [], [], []
        for line in input.readlines():
            tid, content, label = line.strip().split("\t")
            X_tids.append(tid)
            X_test_tid.append(tid)
            fenci_res = clean_str_cut(content, file_name)
            X_test_content.append(fenci_res)
            y_test.append(dic[label])
            old_id_post_map[tid] = fenci_res

    with open(root_path + file_name + "_graph.txt", 'r', encoding='utf-8') as input:
        relation = []
        for line in input.readlines():
            tmp = line.strip().split()
            src = tmp[0]
            X_uids.append(src)

            for dst_ids_ws in tmp[1:]:
                dst, w = dst_ids_ws.split(":")
                X_uids.append(dst)
                relation.append([src, dst, w])

    with open(root_path + '/weibo_files/comment_content.json','r',encoding='utf-8') as input:
        old_id_comment_map = {}
        test_id_comment_map = json.load(input)
        for k,v in test_id_comment_map.items():
            fenci_res = clean_str_cut(v,file_name)
            old_id_comment_map[k] = fenci_res
    with open(root_path + '/weibo_files/user_tweet.json','r',encoding='utf-8') as input:
        old_user_post_map = json.load(input)

    X_id = list(set(X_tids + X_uids))
    num_node = len(X_id)
    X_id_dic = {id: i for i, id in enumerate(X_id)}
    with open(os.getcwd() + '/weibo_files/new_id_dic.json', 'w') as f:
        f.write(json.dumps(X_id_dic, indent=4))

    with open(os.getcwd() + "/weibo_files/original_adj", 'r', encoding='utf-8') as f:
        original_adj = {}
        original_adj_old = json.load(f)
        for i, v in original_adj_old.items():
            i = X_id_dic[i]
            original_adj[i] = []
            for j in v:
                j = X_id_dic[str(j)]
                original_adj[i].append(j)

    with open(os.getcwd() + "/weibo_files/original_adj", 'w', encoding='utf-8') as f:
        json.dump(original_adj, f)


    relation = np.array([[X_id_dic[tup[0]], X_id_dic[tup[1]], tup[2]] for tup in relation])
    relation = build_symmetric_adjacency_matrix(edges=relation, shape=(num_node, num_node))
    X_train_tid = np.array([X_id_dic[tid] for tid in X_train_tid])
    X_dev_tid = np.array([X_id_dic[tid] for tid in X_dev_tid])
    X_test_tid = np.array([X_id_dic[tid] for tid in X_test_tid])

    np.random.seed(666)
    model = gensim.models.KeyedVectors.load_word2vec_format(fname=os.getcwd()+"/weibo_files/weibo_w2v.bin", binary=True)
    node_embedding_matrix = np.random.uniform(-0.25, 0.25, (num_node, 300))

    postnum, commentnum, usernum = 0, 0, 0
    for i, words in old_id_post_map.items():
        new_id = X_id_dic[i]
        embedding = 0.0
        count = 0
        for word in words:
            if model.__contains__(word):
                embedding += model[word]
                count += 1
        if count > 0:
            embedding = embedding / count
            node_embedding_matrix[new_id,:] = embedding
            postnum += 1
    for i, words in old_id_comment_map.items():
        new_id = X_id_dic[i]
        embedding = 0.0
        count = 0
        for word in words:
            if model.__contains__(word):
                embedding += model[word]
                count += 1
        if count > 0:
            embedding = embedding / count
            node_embedding_matrix[new_id, :] = embedding
            commentnum += 1
    for u, posts in old_user_post_map.items():
        new_uid = X_id_dic[u]
        embedding = 0.0
        count = 0
        for post in posts:
            new_pid = X_id_dic[post]
            embedding += node_embedding_matrix[new_pid, :]
            count += 1
        if count > 0:
            embedding = embedding / count
            node_embedding_matrix[new_uid, :] = embedding
            usernum += 1

    pickle.dump([node_embedding_matrix],
                open(root_path + "/weibo_files/node_embedding.pkl", 'wb'))
    return X_train_tid, X_train_content, y_train, \
           X_dev_tid, X_dev_content, y_dev, \
           X_test_tid, X_test_content, y_test, \
           relation


def w2v_feature_extract(root_path, filename, w2v_path):
    X_train_tid, X_train, y_train, \
    X_dev_tid, X_dev, y_dev, \
    X_test_tid, X_test, y_test, relation = read_corpus(root_path, filename)

    print("text word2vec generation.......")
    vocabulary, word_embeddings = build_vocab_word2vec(X_train + X_dev + X_test, w2v_path=w2v_path)
    pickle.dump(vocabulary, open(root_path + "/weibo_files/vocab.pkl", 'wb'))
    print("Vocabulary size: " + str(len(vocabulary)))
    X_train = build_input_data(X_train, vocabulary)
    X_dev = build_input_data(X_dev, vocabulary)
    X_test = build_input_data(X_test, vocabulary)
    pickle.dump([X_train_tid, X_train, y_train, word_embeddings, relation], open(root_path + "/weibo_files/train.pkl", 'wb'))
    pickle.dump([X_dev_tid, X_dev, y_dev], open(root_path + "/weibo_files/dev.pkl", 'wb'))
    pickle.dump([X_test_tid, X_test, y_test], open(root_path + "/weibo_files/test.pkl", 'wb'))


def build_vocab_word2vec(sentences, w2v_path='numberbatch-en.txt'):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    vocabulary_inv = []
    word_counts = Counter(itertools.chain(*sentences))
    vocabulary_inv += [x[0] for x in word_counts.most_common() if x[1] >= 1]
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}

    print("embedding_weights generation.......")
    word2vec = vocab_to_word2vec(w2v_path, vocabulary)
    embedding_weights = build_word_embedding_weights(word2vec, vocabulary_inv)
    return vocabulary, embedding_weights

def vocab_to_word2vec(fname, vocab):
    """
    Load word2vec from Mikolov
    """
    np.random.seed(666)
    word_vecs = {}
    model = gensim.models.KeyedVectors.load_word2vec_format(fname, binary=True)
    count_missing = 0
    for word in vocab:
        if model.__contains__(word):
            word_vecs[word] = model[word]
        else:
            count_missing += 1
            word_vecs[word] = np.random.uniform(-0.25, 0.25, w2v_dim)

    return word_vecs

def build_word_embedding_weights(word_vecs, vocabulary_inv):
    """
    Get the word embedding matrix, of size(vocabulary_size, word_vector_size)
    ith row is the embedding of ith word in vocabulary
    """
    vocab_size = len(vocabulary_inv)
    embedding_weights = np.zeros(shape=(vocab_size + 1, w2v_dim), dtype='float32')
    embedding_weights[0] = np.zeros(shape=(w2v_dim,))

    for idx in range(1, vocab_size):
        embedding_weights[idx] = word_vecs[vocabulary_inv[idx]]
    print("Embedding matrix of size " + str(np.shape(embedding_weights)))
    return embedding_weights


def build_input_data(X, vocabulary):
    """
    Maps sentencs and labels to vectors based on a vocabulary.
    """
    x = [[vocabulary[word] for word in sentence if word in vocabulary] for sentence in X]
    x = pad_sequence(x, max_len=50)
    return x

def pad_sequence(X, max_len=50):
    X_pad = []
    for doc in X:
        if len(doc) >= max_len:
            doc = doc[:max_len]
        else:
            doc = [0] * (max_len - len(doc)) + doc
        X_pad.append(doc)
    return X_pad


if __name__ == "__main__":
    with open(os.getcwd() + '/preprocess_weibo.py') as f:
        exec(f.read())
    root_path = os.getcwd()
    filename = '/weibo_files/weibo'
    w2v_feature_extract(root_path=root_path, filename=filename, w2v_path=os.getcwd()+"/weibo_files/weibo_w2v.bin")
