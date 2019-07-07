#-*- coding:utf-8 -*-

"""Download Yelp dataset from https://www.kaggle.com/yelp-dataset/yelp-dataset
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys
import codecs
import re, io
import json

try:
    # Python3
    import _pickle as pickle
except ImportError:
    # Python2
    import cPickle as pickle

import gc
from collections import Counter

# pylint: disable=wrong-import-order
# from absl import app as absl_app
# from absl import flags
# from six.moves import urllib
import tensorflow as tf
import numpy as np

import tarfile
import sys
import random
from itertools import islice
import gc

from itertools import izip_longest
from collections import namedtuple

Example = namedtuple('Example', ['label', 'user_index', 'u_dense_feature', 'business_index', 'b_sparse_feature', 'b_dense_features', 'u_i_ids_sparse', 'i_u_ids_sparse'])
ExampleBatch = namedtuple('ExampleBatch', ['positive', 'negative'])

def read_file(filePath, encoding='utf-8'):
    file = codecs.open(filePath, encoding = encoding)
    lines = []
    cnt = 0
    for line in file:
        cnt += 1
        if cnt % 100000 == 10:
            print ("File Reading line %d" % cnt)
        line = line.strip().replace("\n", "")
        lines.append(line)
    return lines

def save_file(lines, filePath, encoding='utf-8'):
    output_writer = codecs.open(filePath, 'w',encoding = encoding)
    for line in lines:
        line = line.strip().replace("\n", "")
        output_writer.write(line + "\n")
    output_writer.close()

def read_json_line(json_path, encoding='utf-8', limit = None):
    """ Read Json File, one json object per line
    """
    f_in = codecs.open(json_path, encoding = encoding)
    json_obj_list = []
    cnt = 0
    for line in f_in:
        cnt += 1
        if cnt % 100000 == 10:
            print ("File Reading line %d" % cnt)
        if limit is not None:
            if cnt > limit:
                break
        line = line.strip().replace("\n", "")
        json_obj = json.loads(line)
        json_obj_list.append(json_obj)
    return json_obj_list

def read_json_batch(json_lines, encoding='utf-8'):
    """
    """
    json_obj_list = []
    for line in json_lines:
        json_obj = json.loads(line)
        json_obj_list.append(json_obj)
        #print(load_dict)
    return json_obj_list

def normalizer(value, max_value):
    """
    """
    if value is None:
        return 0.0
    if value >= max_value:
        return 1.0
    else:
        return float(value)/float(max_value)

def parse_business_feature(business_obj, sparse_id_dict):
    """ business feature extractor
        sparse_id_dict: contains category_id_dict and K,V sparse features
        K: business_id + cateogories + sparse_features, V: index
        Dataset Version: Yelp Dataset removes 'neighborhood' key
    """
    DEFAULT_INDEX = 0
    business_id = business_obj['business_id']
    
    # sparse_kv
    sparse_kv_features = []

    attributes = business_obj['attributes']
    state = business_obj['state'] if 'state' in business_obj else None
    city = business_obj['city'] if 'city' in business_obj else None
    hours = business_obj['hours'] if 'hours' in business_obj else None
    is_open = business_obj['is_open'] if 'is_open' in business_obj else None
    # latitude = business_obj['latitude']
    # longitude = business_obj['longitude']
    neighborhood = business_obj['neighborhood'] if 'neighborhood' in business_obj else None

    if state is not None:
        sparse_kv_features.append("state_" + state)
    if city is not None:
        sparse_kv_features.append("city_" + city)
    if is_open is not None:
        sparse_kv_features.append("is_open_" + str(is_open))
    if neighborhood is not None:
        sparse_kv_features.append("neighborhood_" + neighborhood)
    if attributes is not None:
        sparse_kv_features.extend(get_kv_feature(attributes))

    # Category
    categories_str = business_obj['categories']
    cate_list_raw = []
    if categories_str is not None:
        cate_list_raw = categories_str.split(",")
    cate_list = [c.strip()for c in cate_list_raw]
    for c in cate_list:
        sparse_kv_features.append(c)

    # Dense Feature
    dense_features = []
    review_count = business_obj['review_count']
    max_review_count = 5000
    stars = business_obj['stars']
    max_star_count = 5.0
    dense_features.append(normalizer(review_count, max_review_count))
    dense_features.append(normalizer(stars, max_star_count))

    ## 
    business_index = sparse_id_dict[business_id] if business_id in sparse_id_dict else DEFAULT_INDEX
    sparse_features = [sparse_id_dict[f] if f in sparse_id_dict else DEFAULT_INDEX for f in sparse_kv_features]
    return business_index, sparse_features, dense_features

DEFAULT_INDEX = 0
DEFAULT_UNK = "UNK"
def list_to_dict(l):
    d = {}
    d[DEFAULT_UNK] = DEFAULT_INDEX
    cnt = 0
    for item in l:
        cnt += 1
        d[item] = cnt
    return d

def get_sparse_kv_feature(feature_name, feature_values):
    """ feature_name + "_" + feature_values
    """
    feature_kv = []
    for value in feature_values:
        try:
            feature_kv.append(feature_name + "_" + str(value))
        except Exception as e:
            print (e)
    return feature_kv

def get_attributes_kv_feature(attributes):
    """ Generate the 1-level key value pair string
        e.g. u'OutdoorSeating': u'False'  to OutdoorSeating_False
            'BusinessParking': u"{'garage': False, 'street': False, 'validated': False, 'lot': True, 'valet': False}"
            to BusinessParking_{'garage': False, 'street': False, 'validated': False, 'lot': True, 'valet': False}
    """
    feature_kv_batch = []
    for attr_dict in attributes:
        feature_kv = get_kv_feature(attr_dict)
        feature_kv_batch.extend(feature_kv)
    # remove duplicate
    feature_kv_list = list(set(feature_kv_batch))
    return feature_kv_list

## recursive cal the features
def get_kv_feature(dic):
    if dic is None:
        return []
    if type(dic) is str:
        return []
    feature_kv = []
    for k, v in dic.items():
        if v is None:
            continue
        else:
            if (is_json_str(v)):
                v_clean = fix_json_str(v)
                v_dict = json.loads(v_clean)
                if (type(v_dict) is dict):
                    kv_pair = get_kv_feature(v_dict)
                    feature_kv.extend(kv_pair)
            else:
                kv_pair = k + "_" + v
                feature_kv.append(kv_pair)
    return feature_kv

def is_json_str(s):
    if s is None:
        return False
    else:
        if s[0] == '{' and s[len(s)-1] == '}':
            return True
        else:
            return False

def fix_json_str(s):
    """
    """
    s_clean = s
    if "'" in s_clean:
        s_clean = s_clean.replace("'", '"')
    if "False" in s_clean and '"False"' not in s_clean:
        s_clean = s_clean.replace("False", '"False"')
    if "True" in s_clean and '"True"' not in s_clean:
        s_clean = s_clean.replace("True", '"True"')
    return s_clean

def generate_business_feature(business_obj_list):
    """ Iterate over business_obj_list, calculate the features of business object
        # K: 'categories' V: Separated by separator ","
        'neighborhood' key is removed from Yelp dataset Version 9
    """
    business_id = [b['business_id'] for b in business_obj_list]
    attributes = [b['attributes'] for b in business_obj_list]
    categories = [b['categories'] for b in business_obj_list]
    categories_batch = []
    for cate_str in categories:
        if cate_str is not None:
            sub_cate_list = cate_str.split(",")
            categories_batch.extend([c.strip()for c in sub_cate_list])
    categories_list = list(set(categories_batch))

    # Location
    state = list(set([b['state'] if 'state' in b else "" for b in business_obj_list]))
    city = list(set([b['city'] if 'city' in b else "" for b in business_obj_list]))
    is_open = list(set([b['is_open'] if 'is_open' in b else "" for b in business_obj_list])) 
    # hours = [b['hours'] if 'hours' in b else "" for b in business_obj_list]      # dict, monday:5:00-22:00
    #latitude = [b['latitude'] if 'neighborhood' in b else "" for b in business_obj_list]
    #longitude = [b['longitude'] if 'longitude' in b else "" for b in business_obj_list]
    #neighborhood = list(set([b['neighborhood'] if "neighborhood" in b else "" for b in business_obj_list]))
    
    # Popularity
    review_count = [b['review_count'] for b in business_obj_list]
    stars = [b['stars'] for b in business_obj_list]

    print ("DEBUG: Total Business Instance Cnt: %d" % len(business_id))
    print ("DEBUG: categories_list: %d" % len(categories_list))
    print ("DEBUG: state size: %d" % len(set(state)))  # 
    print ("DEBUG: city: %d" % len(set(city)))
    print ("DEBUG: is_open: %d" % len(set(is_open)))
    print ("DEBUG: review_count max number: %d" % max(review_count))
    print ("DEBUG: stars size: %d" % len(set(stars)))  # 

    ## sparse_feature
    sparse_feature_list = []
    # json
    sparse_feature_list.extend(get_attributes_kv_feature(attributes))
    sparse_feature_list.extend(get_sparse_kv_feature("state", state))
    sparse_feature_list.extend(get_sparse_kv_feature("city", city))
    sparse_feature_list.extend(get_sparse_kv_feature("is_open", is_open))
    # sparse_feature_list.extend(get_sparse_kv_feature("neighborhood", list(set(neighborhood))))

    ## output
    print ("DEBUG: Saving Sparse Feature List to path %s" % "../data/yelp/yelp-dataset/yelp_sparse_feature_list.txt")
    print ("DEBUG: Saving Category Name List to path %s" % "../data/yelp/yelp-dataset/yelp_categories_list.txt") 
    save_file(sparse_feature_list, "../data/yelp/yelp-dataset/yelp_sparse_feature_list.txt")
    save_file(categories_list, "../data/yelp/yelp-dataset/yelp_categories_list.txt")
    return sparse_feature_list, categories_list

def parse_user_feature(user_obj, user_id_dict):
    """ user obj
    """ 
    user_id = user_obj['user_id']
    user_index = user_id_dict[user_id] if user_id in user_id_dict.keys() else DEFAULT_INDEX

    ## feature_cnt  
    average_stars = user_obj['average_stars']
    compliment_cool = user_obj['compliment_cool']
    compliment_cute = user_obj['compliment_cute']
    compliment_funny = user_obj['compliment_funny']
    compliment_hot = user_obj['compliment_hot']
    compliment_list = user_obj['compliment_list']
    compliment_more = user_obj['compliment_more']
    compliment_note = user_obj['compliment_note']
    compliment_photos = user_obj['compliment_photos']
    compliment_plain = user_obj['compliment_plain']
    compliment_profile = user_obj['compliment_profile']
    compliment_writer = user_obj['compliment_writer']
    cool = user_obj['cool']
    fans = user_obj['fans']
    funny = user_obj['funny']
    review_count = user_obj['review_count']
    useful = user_obj['useful']

    dense_features = []

    dense_features.append(normalizer(average_stars, 5.0))
    dense_features.append(normalizer(compliment_cool, 5000.0))
    dense_features.append(normalizer(compliment_cute, 5000.0))
    dense_features.append(normalizer(compliment_funny, 5000.0))
    dense_features.append(normalizer(compliment_hot, 5000.0))
    dense_features.append(normalizer(compliment_list, 5000.0))
    dense_features.append(normalizer(compliment_more, 5000.0))
    dense_features.append(normalizer(compliment_note, 5000.0))
    dense_features.append(normalizer(compliment_photos, 5000.0))
    dense_features.append(normalizer(compliment_plain, 5000.0))
    dense_features.append(normalizer(compliment_profile, 5000.0))
    dense_features.append(normalizer(compliment_writer, 5000.0))
    dense_features.append(normalizer(cool, 5000.0))
    dense_features.append(normalizer(fans, 5000.0))
    dense_features.append(normalizer(funny, 5000.0))
    dense_features.append(normalizer(review_count, 5000.0))
    dense_features.append(normalizer(useful, 5000.0))
    return user_index, dense_features

def calculate_user_feature(user_obj_list):
    """  average_stars: double

    """ 
    user_id = [user_obj['user_id'] for user_obj in user_obj_list]
    average_stars = [user_obj['average_stars'] for user_obj in user_obj_list]
    compliment_cool = [user_obj['compliment_cool'] for user_obj in user_obj_list]
    compliment_cute = [user_obj['compliment_cute'] for user_obj in user_obj_list]
    compliment_funny = [user_obj['compliment_funny'] for user_obj in user_obj_list]
    compliment_hot = [user_obj['compliment_hot'] for user_obj in user_obj_list]
    compliment_list = [user_obj['compliment_list'] for user_obj in user_obj_list]
    compliment_more = [user_obj['compliment_more'] for user_obj in user_obj_list]
    compliment_note = [user_obj['compliment_note'] for user_obj in user_obj_list]
    compliment_photos = [user_obj['compliment_photos'] for user_obj in user_obj_list]
    compliment_plain = [user_obj['compliment_plain'] for user_obj in user_obj_list]
    compliment_profile = [user_obj['compliment_profile'] for user_obj in user_obj_list]
    compliment_writer = [user_obj['compliment_writer'] for user_obj in user_obj_list]
    cool = [user_obj['cool'] for user_obj in user_obj_list]
    fans = [user_obj['fans'] for user_obj in user_obj_list]
    funny = [user_obj['funny'] for user_obj in user_obj_list]
    review_count =  [user_obj['review_count'] for user_obj in user_obj_list]
    useful = [user_obj['useful'] for user_obj in user_obj_list]

    print ("DEBUG: Total User Instance Cnt: %d" % len(user_id))
    print ("DEBUG: average_stars Cnt: %d and max value %f" % (len(set(average_stars)), max(average_stars)))
    print ("DEBUG: compliment_cool Cnt: %d and max value %f" % (len(set(compliment_cool)), max(compliment_cool)))
    print ("DEBUG: compliment_cute Cnt: %d and max value %f" % (len(set(compliment_cute)), max(compliment_cute)))
    print ("DEBUG: compliment_funny Cnt: %d and max value %f" % (len(set(compliment_funny)), max(compliment_funny)))
    print ("DEBUG: compliment_hot Cnt: %d and max value %f" % (len(set(compliment_hot)), max(compliment_hot)))
    print ("DEBUG: compliment_list Cnt: %d and max value %f" % (len(set(compliment_list)), max(compliment_list)))
    print ("DEBUG: compliment_more Cnt: %d and max value %f" % (len(set(compliment_more)), max(compliment_more)))
    print ("DEBUG: compliment_note Cnt: %d and max value %f" % (len(set(compliment_note)), max(compliment_note)))
    print ("DEBUG: compliment_photos Cnt: %d and max value %f" % (len(set(compliment_photos)), max(compliment_photos)))
    print ("DEBUG: compliment_plain Cnt: %d and max value %f" % (len(set(compliment_plain)), max(compliment_plain)))
    print ("DEBUG: compliment_profile Cnt: %d and max value %f" % (len(set(compliment_profile)), max(compliment_profile)))
    print ("DEBUG: compliment_writer Cnt: %d and max value %f" % (len(set(compliment_writer)), max(compliment_writer)))
    print ("DEBUG: cool Cnt: %d and max value %f" % (len(set(cool)), max(cool)))
    print ("DEBUG: fans Cnt: %d and max value %f" % (len(set(fans)), max(fans)))
    print ("DEBUG: funny Cnt: %d and max value %f" % (len(set(funny)), max(funny)))
    print ("DEBUG: review_count Cnt: %d and max value %f" % (len(set(review_count)), max(review_count)))
    print ("DEBUG: useful Cnt: %d and max value %f" % (len(set(useful)), max(useful)))

    # output
    save_file(user_id, "../data/yelp/yelp-dataset/user_id.txt")

def parse_review_feature(review_obj):
    review_id = review_obj['review_id']
    user_id = review_obj['user_id']
    business_id = review_obj['business_id']
    stars = review_obj['stars']

    useful = review_obj['useful']
    cool = review_obj['cool']
    funny = review_obj['funny']

    date = review_obj['date']
    text = review_obj['text']

def filter_valid_review_obj(review_obj_list, min_star = 4):
    """ Filter out reviews with no less than 4 starts
    """
    review_candidates = []
    for i, review_obj in enumerate(review_obj_list):
        if review_obj['stars'] >= min_star:
            review_candidates.append(review_obj)
    ## DEBUG
    # print ("DEBUG: review_candidates after filter size: %s" % len(review_candidates))
    return review_candidates

def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return izip_longest(*args, fillvalue=fillvalue)

def iter_file(filename, N = 1000):
    """ read and go over review dataset
    """
    with open(filename) as f:
         for lines in grouper(f, N, ''):
             assert len(lines) == N
             yield lines

def prepare_dataset(review_obj_list, user_min_review_cnt = 20, negative_sample_ratio = 50, ratio = 0.8):
    """ Prepare Business Review Dataset with 
        minimum review count: user_min_review_cnt
        negative sampling rate: negative_sample_ratio
        ratio: train and test split ratio 80%/20%
    """ 
    # user_id cnt
    user_id_list_raw = [obj['user_id'] for obj in review_obj_list]
    user_business_cnt_dict = dict(Counter(user_id_list_raw))
    # user filter list after min cnt
    user_id_filter_list = []
    for k, v in user_business_cnt_dict.items():
        if (v >= user_min_review_cnt):
            user_id_filter_list.append(k)
    print ("DEBUG: user_id_filter_list after filter size is: %d" % len(user_id_filter_list))

    user_business_tuple = []
    user_id_filter_set = set(user_id_filter_list)
    user_item_dict = {}
    cnt = 0
    for obj in review_obj_list:
        cnt += 1
        if cnt % 100000 == 0:
            print ("DEBUG: Processing Review Obj Number: %d" % cnt)
        user_id = obj['user_id']
        business_id = obj['business_id']
        if user_id in user_id_filter_set:
            user_business_tuple.append((user_id, business_id))
            #if user_id in user_id_filter_list:
            if user_id in user_item_dict.keys():
                user_item_dict[user_id].append(business_id)
            else:
                user_item_dict[user_id] = [business_id]
    print ("DEBUG: User-Business Interaction num is %d" % len(user_business_tuple))
    user_business_tuple_sorted = sorted(user_business_tuple, key = lambda x:x[0])
    
    # deine filtered user_id list and business_id_list
    user_id_list = list(user_id_filter_set)
    business_id_list = list(set([i for u, i in user_business_tuple_sorted]))
    interaction_list = [ u + "\t" + i for (u, i) in user_business_tuple_sorted]
    
    # Positive and Negative Label
    # convert (U, I) tuple to dataset dict
    user_business_interaction_dict = {}
    cur_user_id = user_business_tuple_sorted[0][0]
    cur_business_id = []
    cnt = 0
    for user_id, business_id in user_business_tuple_sorted:
        cnt += 1
        if cnt % 100000 == 0:
            print ("DEBUG: Processing line %d" % cnt)
        if cur_user_id != user_id:
            user_business_interaction_dict[cur_user_id] = cur_business_id
            cur_business_id = []
            cur_business_id.append(business_id)
            cur_user_id = user_id
        else:
            cur_business_id.append(business_id)
    ## Generate Positive and Negative Example
    dataset = {}
    cnt = 0
    for user_id in user_business_interaction_dict.keys():
        cnt += 1
        if cnt % 100000 == 0:
            print ("DEBUG: Generating Dataset line %d" % cnt)
        positive_sample = user_business_interaction_dict[user_id]
        negative_sample = None
        dataset[user_id] = {}
        dataset[user_id]['positive'] = positive_sample
        dataset[user_id]['negative'] = negative_sample

    # sample negative samples
    train_dataset, test_dataset = split_dataset(dataset, business_id_list, negative_sample_ratio, ratio)
    return dataset, train_dataset, test_dataset

def calculate_dataset_statistics(dataset):
    """ 
    """ 
    user_cnt = 0
    interaction_cnt = 0
    business_cnt = 0
    user_cnt = len(dataset.keys())
    business_list = []
    for uid in dataset.keys():
        positive_sample = dataset[uid]['positive']
        business_list.extend(positive_sample)
    interaction_cnt = len(business_list)
    business_cnt = len(set(business_list))
    print ("DEBUG: User Instance Cnt %d" % user_cnt)
    print ("DEBUG: Business Instance Cnt %d" % business_cnt)
    print ("DEBUG: Interaction Cnt %d" % interaction_cnt)

def sample_negative(negative_sample, number):
    return random.sample(negative_sample, number)

def split_dataset(dataset, business_id_list, negative_sample_ratio = 5, ratio = 0.8):
    # filter out business
    train_dataset, test_dataset = {}, {}
    index = 0
    business_id_set = set(business_id_list)
    
    for user_id in dataset.keys():
        index += 1
        if (index % 1000 == 0):
            print ("DEBUG: Split Dataset Lines %d" % index)
        positive_sample = dataset[user_id]['positive']  # positive interaction id
        negative_sample = (business_id_set - set(positive_sample))
        
        # random shuffle
        np.random.shuffle(positive_sample)
        split_size = int(ratio * len(positive_sample))

        train_dataset[user_id] = {}
        train_dataset[user_id]['positive'] = positive_sample[0:split_size]
        n_train_pos = split_size
        train_dataset[user_id]['negative'] = sample_negative(negative_sample, n_train_pos * negative_sample_ratio)

        test_dataset[user_id] = {}
        test_dataset[user_id]['positive'] = positive_sample[split_size:len(positive_sample)]
        n_test_pos = len(positive_sample) - split_size
        test_dataset[user_id]['negative'] = sample_negative(negative_sample, n_test_pos * negative_sample_ratio)
    return train_dataset, test_dataset

def split_array(feature_str, sep = ","):
    """ split the feature_str by separator
    """
    digits = feature_str.split(sep)
    digits_array = np.array([float(d.strip()) for d in digits])
    return digits_array

def get_user_obj_dict(datafolder):
    """ Read file yelp_academic_dataset_user.json user json object
    """
    user_json_path = os.path.join(datafolder, "yelp_academic_dataset_user.json")
    print ("DEBUG: Start Reading User Data from file %s" % user_json_path)
    user_obj_list = read_json_line(user_json_path)
    user_obj_dict = {}
    for u in user_obj_list:
        user_obj_dict[u['user_id']] = u
    print ("DEBUG: Finish Reading User Data lines %d" % len(user_obj_list))
    return user_obj_dict

def get_business_obj_dict(business_obj_list):
    """ Convert business_obj_list to business_obj_dict, with key: business_id, V: business_obj
    """
    business_obj_dict = {}
    for w in business_obj_list:
        business_obj_dict[w['business_id']] = w
    return business_obj_dict

def get_business_obj_list(datafolder):
    """ Read file yelp_academic_dataset_business.json business json object as list
    """    
    business_json_path = os.path.join(datafolder, "yelp_academic_dataset_business.json")
    print ("DEBUG: Start Reading Business Data from file %s" % business_json_path)
    business_obj_list = read_json_line(business_json_path)
    print ("DEBUG: Finish Reading Business Data lines %d" % len(business_obj_list))
    return business_obj_list

def get_user_obj_list(datafolder):
    """ Read file yelp_academic_dataset_user.json user json object as list
    """    
    user_json_path = os.path.join(datafolder, "yelp_academic_dataset_user.json")
    print ("DEBUG: Start Reading User Data from file %s" % user_json_path)
    user_obj_list = read_json_line(user_json_path)
    print ("DEBUG: Finish Reading User Data lines %d" % len(user_obj_list))
    return user_obj_list

def get_user_item_interaction(dataset):
    """
    """
    user_item_id_dict = {}
    item_user_id_dict = {}
    cnt = 0
    for uid in dataset.keys():
        cnt += 1
        if (cnt % 1000 == 0):
            print ("DEBUG: Output Cnt %d" % cnt)
        item_ids = dataset[uid]['positive']
        # uid-item_id
        user_item_id_dict[uid] = item_ids
        for item_id in item_ids:
            if item_id in item_user_id_dict.keys():
                item_user_id_dict[item_id].append(uid)
            else:
                item_user_id_dict[item_id] = [uid]
    print ("DEBUG: Generating User Item Id Dict Size %d" % len(user_item_id_dict))
    print ("DEBUG: Generating Item User Id Dict Size %d" % len(item_user_id_dict))
    return user_item_id_dict, item_user_id_dict

def generate_pretrain_examples(datafolder, train_dataset, dataset_name, user_obj_dict, business_obj_dict,
        sparse_id_dict, user_item_id_dict, item_user_id_dict):
    # Generate Pretrain Examples
    pretrain_dataset = []
    row = 0
    for uid in train_dataset.keys():
        row += 1
        if (row % 100 == 0):
            print ("DEBUG: Processing row %d" % row)
        positive_sample = train_dataset[uid]['positive']
        user_obj = user_obj_dict[uid]
        user_index, u_dense_feature = parse_user_feature(user_obj, sparse_id_dict)
        u_i_ids = user_item_id_dict.get(uid) if uid in user_item_id_dict else []
        u_i_ids_sparse = [sparse_id_dict.get(bid, DEFAULT_INDEX) for bid in u_i_ids]

        for business_id in positive_sample:
            business_obj = business_obj_dict[business_id]
            business_index, b_sparse_feature, b_dense_features = parse_business_feature(business_obj, sparse_id_dict)
            i_u_ids = item_user_id_dict.get(business_id) if business_id in item_user_id_dict else []
            i_u_ids_sparse = [sparse_id_dict.get(uid, DEFAULT_INDEX) for uid in i_u_ids]
            pretrain_dataset.append((user_index, u_dense_feature, business_index, b_sparse_feature, b_dense_features,
                u_i_ids_sparse, i_u_ids_sparse))
    
    example_path = os.path.join(datafolder, dataset_name)
    print ("DEBUG: Exporting model to below path %s" % example_path)
    with io.open(example_path, 'wb') as output_file:
        pickle.dump(pretrain_dataset, output_file)
    return pretrain_dataset

def generate_examples_batch(datafolder, dataset, dataset_name,
        user_obj_dict, business_obj_dict,
        sparse_id_dict,
        user_item_id_dict,
        item_user_id_dict,
        NS = 50, save_every_num = 10000):
    """ generate training examples features and testing examples features
        Return: list of tuples
        (label, user_index, u_dense_feature, business_index, b_sparse_feature, b_dense_features)
        sparse_id_dict: user_id, shop_id, category_id  dict
    """
    # training dataset
    examples = []
    row = 0
    part_id = 0
    example_cnt = 0
    for uid in dataset.keys():
        row += 1
        if (row % 100 == 0):
            print ("DEBUG: Processing %s Data Row %d" % (dataset_name, row))
        positive_sample = dataset[uid]['positive']
        negative_sample = dataset[uid]['negative']
        user_obj = user_obj_dict[uid]
        user_index, u_dense_feature = parse_user_feature(user_obj, sparse_id_dict)
        u_i_ids = user_item_id_dict.get(uid) if uid in user_item_id_dict else []
        u_i_ids_sparse = [sparse_id_dict.get(bid, DEFAULT_INDEX) for bid in u_i_ids]

        # Positive Samples
        for i, pos_bid in enumerate(positive_sample):
            example_cnt += 1
            pos_b_obj = business_obj_dict[pos_bid]
            pos_business_index, pos_b_sparse_feature, pos_b_dense_features = parse_business_feature(pos_b_obj, sparse_id_dict)
            i_u_ids = item_user_id_dict.get(pos_bid) if pos_bid in item_user_id_dict else []
            i_u_ids_sparse = [sparse_id_dict.get(uid, DEFAULT_INDEX) for uid in i_u_ids]

            # 1 positive_sample
            positive_example = Example(label = 1, user_index = user_index, 
                u_dense_feature = u_dense_feature,
                business_index = pos_business_index, 
                b_sparse_feature = pos_b_sparse_feature, 
                b_dense_features = pos_b_dense_features,
                u_i_ids_sparse = u_i_ids_sparse,
                i_u_ids_sparse = i_u_ids_sparse)
            # NS negative samples
            negative_sample_batch = negative_sample[i * NS: (i+1) * NS]
            negative_example_list = []
            for neg_bid in negative_sample_batch:
                neg_obj = business_obj_dict[neg_bid]
                neg_business_index, neg_b_sparse_feature, neg_b_dense_features = parse_business_feature(neg_obj, sparse_id_dict)
                i_u_ids = item_user_id_dict.get(neg_bid) if neg_bid in item_user_id_dict else []
                i_u_ids_sparse = [sparse_id_dict.get(uid, DEFAULT_INDEX) for uid in i_u_ids]

                negative_example = Example(label = 0, user_index = user_index, 
                    u_dense_feature = u_dense_feature,
                    business_index = neg_business_index, 
                    b_sparse_feature = neg_b_sparse_feature, 
                    b_dense_features = neg_b_dense_features,
                    u_i_ids_sparse = u_i_ids_sparse,
                    i_u_ids_sparse = i_u_ids_sparse)
                negative_example_list.append(negative_example)
            # Conbine to one ExampleBatch
            example_batch = ExampleBatch(positive = positive_example, negative = negative_example_list)
            examples.append(example_batch)
            # Check if need to save
            if (example_cnt % save_every_num == 0):
                part_id += 1
                print ("DEBUG: Generating Examples size %d at row %d and example cnt%d" 
                    % (len(examples), row, example_cnt))
                example_path = os.path.join(datafolder, dataset_name + "_" + str(part_id) + ".pkl")
                print ("DEBUG: Exporting model to below path %s" % example_path)
                with io.open(example_path, 'wb') as output_file:
                    pickle.dump(examples, output_file)
                # empty
                examples = []
    # Save the last batch
    part_id += 1
    example_path = os.path.join(datafolder, dataset_name + "_" + str(part_id) + ".pkl")
    print ("DEBUG: Exporting model to below path %s" % example_path)
    with io.open(example_path, 'wb') as output_file:
        pickle.dump(examples, output_file)
    return examples

def generate_examples(datafolder, train_dataset, test_dataset, 
        user_obj_dict, business_obj_dict,
        user_id_dict, business_id_dict, 
        sparse_feature_dict):
    """ generate training examples features and testing examples features
        Return: list of tuples
        (label, user_index, u_dense_feature, business_index, b_sparse_feature, b_dense_features)
    """
    # training dataset
    train_examples = []
    row = 0
    for uid in train_dataset.keys():
        row += 1
        if (row % 100 == 0):
            print ("DEBUG: Processing Train Data Row %d" % row)
        positive_sample = train_dataset[uid]['positive']
        negative_sample = train_dataset[uid]['negative']
        user_obj = user_obj_dict[uid]
        user_index, u_dense_feature = parse_user_feature(user_obj, user_id_dict)
        # Positive Samples
        for business_id in positive_sample:
            business_obj = business_obj_dict[business_id]
            business_index, b_sparse_feature, b_dense_features = parse_business_feature(business_obj, business_id_dict, sparse_feature_dict)
            train_examples.append((1, user_index, u_dense_feature, business_index, b_sparse_feature, b_dense_features))
        # Negative Samples
        for business_id in negative_sample:
            business_obj = business_obj_dict[business_id]
            business_index, b_sparse_feature, b_dense_features = parse_business_feature(business_obj, business_id_dict, sparse_feature_dict)
            train_examples.append((0, user_index, u_dense_feature, business_index, b_sparse_feature, b_dense_features))

    # training dataset
    test_examples = []
    row = 0
    for uid in test_dataset.keys():
        row += 1
        if (row % 10000 == 0):
            print ("DEBUG: Processing Test Data %d" % row)
        positive_sample = test_dataset[uid]['positive']
        negative_sample = test_dataset[uid]['negative']
        user_obj = user_obj_dict[uid]
        user_index, u_dense_feature = parse_user_feature(user_obj, user_id_dict)
        # Positive Samples
        for business_id in positive_sample:
            business_obj = business_obj_dict[business_id]
            business_index, b_sparse_feature, b_dense_features = parse_business_feature(business_obj, business_id_dict, sparse_feature_dict)
            test_examples.append((1, user_index, u_dense_feature, business_index, b_sparse_feature, b_dense_features))
        # Negative Samples
        for business_id in negative_sample:
            business_obj = business_obj_dict[business_id]
            business_index, b_sparse_feature, b_dense_features = parse_business_feature(business_obj, business_id_dict, sparse_feature_dict)
            test_examples.append((0, user_index, u_dense_feature, business_index, b_sparse_feature, b_dense_features))
    print ("DEBUG: Generating Training Examples size %d" % len(train_examples))
    print ("DEBUG: Generating Testing Examples size %d" % len(test_examples))
    
    train_example_path = os.path.join(datafolder, "yelp_train_examples.pkl")
    test_example_path = os.path.join(datafolder, "yelp_test_examples.pkl")
    with io.open(train_example_path, 'wb') as output_file:
        pickle.dump(train_examples, output_file)
    with io.open(test_example_path, 'wb') as output_file:
        pickle.dump(test_examples, output_file)        
    return

def pretrain_dataset_reader(pretrain_dataset_path):
    """ Read Existing pick files
    """
    file_exist = os.path.exists(pretrain_dataset_path)
    pretrain_dataset = []
    if file_exist:
        print ("DEBUG: Pretrain DataPath Exist Load data from %s" % pretrain_dataset_path)
        with io.open(pretrain_dataset_path, 'rb') as input_file:
            pretrain_dataset = pickle.load(input_file)
    else:
        print ("DEBUG: Pretrain DataPath doesn't exist...")
    return pretrain_dataset

def train_dataset_reader(datapath):
    file_exist = os.path.exists(datapath)
    dataset = []
    if file_exist:
        print ("DEBUG: Train DataPath Exist Load data from %s" % datapath)
        with io.open(datapath, 'rb') as input_file:
            dataset = pickle.load(input_file)
    else:
        print ("DEBUG: Train DataPath doesn't exist...")
    return dataset

def train_dataset_reader_list(datapath):
    file_exist = os.path.exists(datapath)
    dataset = []
    if file_exist:
        print ("DEBUG: Train DataPath Exist Load data from %s" % datapath)
        with io.open(datapath, 'rb') as input_file:
            dataset = pickle.load(input_file)
    else:
        print ("DEBUG: Train DataPath doesn't exist...")
    # Convert Dataset Reader to List
    example_features_list = []
    for i, batch in enumerate(dataset):
        if (i % 10000 == 0):
            print ("DEBUG: Converting to datalist %d" % i)
        example_list = [batch.positive] + batch.negative
        example_features = [(t.label, t.user_index, t.u_dense_feature, 
            t.business_index, t.b_sparse_feature, t.b_dense_features,
            t.u_i_ids_sparse, t.i_u_ids_sparse) for t in example_list]
        example_features_list.extend(example_features)
    print ("DEBUG: Total example_features_list size %d" % len(example_features_list))
    return example_features_list

def pad_to_fix_length(dim_2_list, pad_default, fix_length = None):
    """ pad to fix length of 2d list
    """
    max_len = 0
    if fix_length is None:
        len_list = [len(l) for l in dim_2_list]
        max_len = max(len_list)
    else:
        max_len = fix_length
    dim_2_list_new = []
    for l in dim_2_list:
        l_new = l + [pad_default] * max((max_len - len(l)), 0)
        dim_2_list_new.append(l_new)
    return dim_2_list_new

class PretrainDatasetIter(object):
    def __init__(self, data_path, batch_size, limit= None):
        self.data_path = data_path
        self.batch_size = batch_size
        self.limit = limit

    def __iter__(self):
        self.dataset = pretrain_dataset_reader(self.data_path)
        print ("DEBUG: PretrainDatasetIter Size %d" % len(self.dataset))
        size = len(self.dataset)
        total_batch_num = int(size/self.batch_size)
        for i in range(total_batch_num):
            curBatch = self.dataset[i*self.batch_size:(i+1)*self.batch_size]
            u_idx_batch = np.array([t[0] for t in curBatch])
            u_dense_feature_batch = np.array([t[1] for t in curBatch])
            b_idx_batch = np.array([t[2] for t in curBatch])
            # default pad to 0
            b_sparse_feature_batch = np.array(pad_to_fix_length([t[3] for t in curBatch], 0))
            b_dense_feature_batch = np.array([t[4] for t in curBatch])

            u_i_ids_sparse = np.array(pad_to_fix_length([t[5] for t in curBatch], 0))
            i_u_ids_sparse = np.array(pad_to_fix_length([t[6] for t in curBatch], 0))
            yield u_idx_batch, u_dense_feature_batch, b_idx_batch, b_sparse_feature_batch, b_dense_feature_batch, u_i_ids_sparse, i_u_ids_sparse

class TrainDatasetIter(object):
    """ Train Dataset Iterator
    """
    def __init__(self, datapath, batch_size, limit=None):
        self.datapath = datapath
        self.batch_size = batch_size
        self.limit = limit
        self.batch_size = batch_size
    
    def __iter__(self):
        # self.dataset = train_dataset_reader(self.datapath)
        self.dataset = train_dataset_reader_list(self.datapath)
        size = len(self.dataset)
        print ("DEBUG: Loading Train Dataset Size %d from path %s" % (size, self.datapath))
        total_batch_num = int(size/self.batch_size)
        for i in range(total_batch_num):
            curBatch = self.dataset[i*self.batch_size:(i+1)*self.batch_size]
            y_batch = np.array([(1 - t[0], t[0]) for t in curBatch])
            u_idx_batch = np.array([t[1] for t in curBatch])
            u_dense_feature_batch = np.array([t[2] for t in curBatch])
            b_idx_batch = np.array([t[3] for t in curBatch])
            # default pad to 0
            b_sparse_feature_batch = np.array(pad_to_fix_length([t[4] for t in curBatch], 0))
            b_dense_feature_batch = np.array([t[5] for t in curBatch])
            u_i_ids_sparse_batch = np.array(pad_to_fix_length([t[6] for t in curBatch], 0))
            i_u_ids_sparse_batch = np.array(pad_to_fix_length([t[7] for t in curBatch], 0))
            yield y_batch, u_idx_batch, u_dense_feature_batch, b_idx_batch, b_sparse_feature_batch, b_dense_feature_batch, u_i_ids_sparse_batch, i_u_ids_sparse_batch
    
    def remove(self):
        remove_file(self.datapath)

def group_by_user(dataset):
    """ (0, user_index, u_dense_feature, business_index, b_sparse_feature, b_dense_features)
    """
    dataset_sorted = sorted(dataset, key = lambda x:x[1])
    user_id = None
    user_batch = []
    user_batch_dict = {}
    for i, example in enumerate(dataset_sorted):
        if (i % 1000000 == 0):
            print ("DEBUG: Generating Group By Id User %d" % i)
        label = example[0]
        user_index = example[1]
        u_dense_feature = example[2]
        business_index = example[3]
        b_sparse_feature = example[4]
        b_dense_features = example[5]
        if user_index != user_id:
            user_batch_dict[user_id] = user_batch
            user_batch = []
            user_id = user_index
        else:
            user_batch.append(example)
    return user_batch_dict

class TestDatasetIter(object):
    """ Used for model Evaluation, Use  a ExampleBatch of positive and negative samples
    """
    def __init__(self, datapath, limit=None):
        self.datapath = datapath
        self.limit = limit
        
    def __iter__(self):
        self.dataset = train_dataset_reader(self.datapath)
        print ("DEBUG: TestDatasetIter Loading dataset size %d" % len(self.dataset))
        for exampleBatch in self.dataset:
            positive_sample = exampleBatch.positive
            negative_sample = exampleBatch.negative
            example_list = [positive_sample] + negative_sample
            y_batch = np.array([(1 - t.label, t.label) for t in example_list])
            u_idx_batch = np.array([t.user_index for t in example_list])
            u_dense_feature_batch = np.array([t.u_dense_feature for t in example_list])
            b_idx_batch = np.array([t.business_index for t in example_list])
            # default pad to 0
            b_sparse_feature_batch = np.array(pad_to_fix_length([t.b_sparse_feature for t in example_list], 0))
            b_dense_feature_batch = np.array([t.b_dense_features for t in example_list])
            u_i_ids_sparse_batch = np.array(pad_to_fix_length([t.u_i_ids_sparse for t in example_list], 0))
            i_u_ids_sparse_batch = np.array(pad_to_fix_length([t.i_u_ids_sparse for t in example_list], 0))
            yield y_batch, u_idx_batch, u_dense_feature_batch, b_idx_batch, b_sparse_feature_batch, b_dense_feature_batch, u_i_ids_sparse_batch, i_u_ids_sparse_batch

def unarchive_file(archive_file, target_path):
    tar = tarfile.open(archive_file)
    tar.extractall(path = target_path)
    tar.close()

def remove_file(file_path):
    if(os.path.exists(file_path)):
        os.remove(file_path)
        print ('DEBUG: File %s is removed' % file_path)
    else:
        print ('DEBUG: File %s does not exist' % file_path)

def test_examples():
    train_path = "/Users/dingxichen/Desktop/project/gitlab/context_research/tensorflow/data/yelp/yelp-dataset/train/yelp_train_examples_merge_1.pkl"
    dataset = train_dataset_reader_list(train_path)
    print ("DEBUG: Dataset Size %d" % len(dataset))
    train_dataset_iter = TrainDatasetIter(train_path, batch_size = 5)
    for i, batch in enumerate(train_dataset_iter):
        print (batch[1].shape)
        print (batch[1])
        print (batch[2])
        print (batch[3])
        print (batch[4])
        print (batch[5])
        break

## Yelp dataset 
class DatasetFolderIter(object):
    def __init__(self, folder, batch_size, limit=None):
        files=[]
        for root, dirs, cur_files in os.walk(folder):
            print (cur_files)
            root = root
            files.extend(cur_files)
        regex = r'[^\x20\r\n\f\t]*.pkl'
        self.folder = folder
        self.match_file_names = re.findall(regex, "\t".join(files))
        print ("DEBUG: Matched Files are:")
        print (self.match_file_names)
        self.batch_size = batch_size

    def __iter__(self):
        file_num = len(self.match_file_names)
        for i in range(file_num):
            pickle_file_name = os.path.join(self.folder, self.match_file_names[i])
            print ("DEBUG: pick file name %s" % pickle_file_name)
            dataset = TrainDatasetIter(pickle_file_name, batch_size=self.batch_size)
            yield dataset

def prepare_dataset_from_file(datafolder):
    """ Read .json file from Yelp dataset datapath
        review data path: ../data/yep/yelp-dataset/academic_dataset_review.json

        Output File:
            ../data/yelp/yelp-dataset/yelp_train_dataset.pkl
            ../data/yelp/yelp-dataset/yelp_test_dataset.pkl
            ../data/yelp/yelp-dataset/yelp_pretrain_dataset.pkl
            ../data/yelp/yelp-dataset/train/yelp_train_examples_*.pkl  * is the split part Id: 1-10
            ../data/yelp/yelp-dataset/test/yelp_test_examples_*.pkl    * is the split part Id
    """
    if_load_raw_review_dataset = True
    if if_load_raw_review_dataset:
        # Iterate Review Json Object
        review_json_path = os.path.join(datafolder, "yelp_academic_dataset_review.json")
        reviews_generator = iter_file(review_json_path, N = 10000)
        review_obj_list = []
        print ("DEBUG: Start Processing Business Review file %s" % "yelp_academic_dataset_review.json")
        for i, lines in enumerate(reviews_generator):
            if (i % 100 == 0):
                print ("DEBUG: Processing Review Data Batch %d, Size %d" % (i, len(lines)))
            ## DEBUG
            if (i >= 150):
                break
            try:
                review_obj_list.extend(read_json_batch(lines))
            except Exception as e:
                print ("DEBUG: Processing Review Data meet error")
                print (e)
        ## Filter Valid Review with 4 and 5 stars as positive examples
        review_obj_filter = filter_valid_review_obj(review_obj_list, 4)
        print ("DEBUG: Finish Review Object Filtering size %d" % len(review_obj_filter))
        ## Train/Test split and negative sampling
        print ("DEBUG: Start train/test splitting and negative sampling...")
        dataset, train_dataset, test_dataset = prepare_dataset(review_obj_filter, user_min_review_cnt = 20, negative_sample_ratio = 50, ratio = 0.8)
        calculate_dataset_statistics(dataset)
        ## output train and test dataset(user/item interaction file)
        train_dataset_path = os.path.join(datafolder, "yelp_train_dataset.pkl")
        test_dataset_path = os.path.join(datafolder, "yelp_test_dataset.pkl")
        with io.open(train_dataset_path, 'wb') as output_file:
            pickle.dump(train_dataset, output_file)
        with io.open(test_dataset_path, 'wb') as output_file:
            pickle.dump(test_dataset, output_file)      
        ## output user_item dictionary file
        user_item_id_dict, item_user_id_dict = get_user_item_interaction(dataset)
        user_item_id_dict_path = os.path.join(datafolder, "yelp_user_item_id_dict.pkl")
        item_user_id_dict_path = os.path.join(datafolder, "yelp_item_user_id_dict.pkl")
        with io.open(user_item_id_dict_path, 'wb') as output_file:
            pickle.dump(user_item_id_dict, output_file)
        with io.open(item_user_id_dict_path, 'wb') as output_file:
            pickle.dump(item_user_id_dict, output_file)     

        ## Read sparse feature id dictionary
        # load instance of businesses and users
        business_obj_list = get_business_obj_list(datafolder)
        business_obj_dict = get_business_obj_dict(business_obj_list)
        # user_obj_list = get_user_obj_list(datafolder)
        user_obj_dict = get_user_obj_dict(datafolder)

        ## Generate Sparse Feature Mapping Dictionary
        print ("DEBUG: Start Generating Sparse Features Mapping...")
        user_id_list = list(set([uid for uid in user_obj_dict.keys()]))
        business_id_list = list(set([bid for bid in business_obj_dict.keys()]))
        categories_list, sparse_feature_list = generate_business_feature(business_obj_list)
        print ("DEBUG: Finish Generating Sparse Features Mapping...")

        # Merge sparse_feature_list, categories_list, user_id_list and business_id_list 
        # into Sparse Feature Dict Mapping
        sparse_id_list = []
        sparse_id_list.extend(user_id_list)
        sparse_id_list.extend(business_id_list)
        sparse_id_list.extend(sparse_feature_list)
        sparse_id_list.extend(categories_list)
        # remove duplicate values
        sparse_id_list = list(set(sparse_id_list))
        sparse_id_dict = list_to_dict(sparse_id_list)
        print ("DEBUG: Saving Sparse Id Dict to file...")
        sparse_id_dict_path = os.path.join(datafolder, "yelp_sparse_feature_id_dict.pkl")
        with io.open(sparse_id_dict_path, 'wb') as sparse_output_file:
            pickle.dump(sparse_id_dict, sparse_output_file)

        # Release Memory
        del review_obj_list
        del review_obj_filter
        gc.collect()

    # Reload data from file
    # User Item ID Mapping
    user_item_id_dict, item_user_id_dict = {}, {}
    user_item_id_dict_path = os.path.join(datafolder, "yelp_user_item_id_dict.pkl")
    item_user_id_dict_path = os.path.join(datafolder, "yelp_item_user_id_dict.pkl")
    if os.path.exists(user_item_id_dict_path):
        with io.open(user_item_id_dict_path, 'rb') as input_file:
            user_item_id_dict = pickle.load(input_file)
    if os.path.exists(item_user_id_dict_path):
        with io.open(item_user_id_dict_path, 'rb') as input_file:
            item_user_id_dict = pickle.load(input_file) 

    ## Read train/test split dataset
    train_dataset, test_dataset = [], []
    train_dataset_path = os.path.join(datafolder, "yelp_train_dataset.pkl")
    test_dataset_path = os.path.join(datafolder, "yelp_test_dataset.pkl")
    if os.path.exists(train_dataset_path):
        with io.open(train_dataset_path, 'rb') as input_file:
            train_dataset = pickle.load(input_file)
    if os.path.exists(test_dataset_path):
        with io.open(test_dataset_path, 'rb') as input_file:
            test_dataset = pickle.load(input_file)   
    
    # generate prtrain dataset
    pretrain_dataset = generate_pretrain_examples(datafolder, train_dataset, "yelp_pretrain_dataset.pkl",
        user_obj_dict, business_obj_dict, sparse_id_dict, user_item_id_dict, item_user_id_dict)
    # generate training dataset
    train_examples = generate_examples_batch(datafolder, train_dataset, "train/yelp_train_examples",
        user_obj_dict, business_obj_dict, sparse_id_dict, user_item_id_dict, item_user_id_dict, NS = 50, save_every_num = 100000)
    test_examples = generate_examples_batch(datafolder, test_dataset, "test/yelp_test_examples", 
        user_obj_dict, business_obj_dict, sparse_id_dict, user_item_id_dict, item_user_id_dict, NS = 50, save_every_num = 150000)

def test_dataset_folder_iter(train_file_folder):
    train_dataset_iter = DatasetFolderIter(train_file_folder, batch_size = 256)
    for i, train_dataset in enumerate(train_dataset_iter):
        print ("DEBUG: Loading Batch size %d" % i)
        ## Train
        for j, data_batch in enumerate(train_dataset):
            if (j % 1000 == 0):
                print ("DEBUG: processing data_batch %d" % j)

def test_pretrain_iterator(data_path):
    """ u_idx_batch, u_dense_feature_batch, b_idx_batch, b_cate_batch, b_sparse_feature_batch, b_dense_feature_batch
    """
    dataset = PretrainDatasetIter(data_path, batch_size = 32)
    for i, group in enumerate(dataset):
        print (i)
        print (len(group))
        print ("DEBUG: Batch %d Group Feature Shape %s,%s,%s,%s,%s" % (i, str(group[0].shape), 
            str(group[1].shape), str(group[2].shape) , str(group[3].shape), 
            str(group[4].shape)))
        u_idx_batch = group[0]
        u_dense_feature_batch = group[1]
        b_idx_batch = group[2]
        b_sparse_feature_batch = group[3]
        b_dense_feature_batch = group[4]
        print ("DEBUG: User Index Batch...")
        print (u_idx_batch)
        print ("DEBUG: u_dense_feature_batch Batch...")
        print (u_dense_feature_batch)
        print ("DEBUG: Business Index Batch...")
        print (b_idx_batch)
        print ("DEBUG: Business Sparse Category Batch...")
        print (b_sparse_feature_batch)
        print ("DEBUG: Business Dense Feature Batch...")
        print (b_dense_feature_batch)
        break

if __name__ == '__main__':
    # Yelp dataset folder
    datafolder = "../data/yelp/yelp-dataset"
    prepare_dataset_from_file(datafolder)
    # Train folder
    train_file_folder="../../context_recommendation/data/yelp/yelp-dataset/train"
    test_dataset_folder_iter(train_file_folder)
    # Pretrain Dataset
    pretrain_datapath="../data/yelp/yelp-dataset/yelp_pretrain_dataset.pkl"
    test_pretrain_iterator(pretrain_datapath)
