# -*- coding: utf-8 -*- 

# 参数

import argparse
import random
from sklearn import metrics
import numpy as np
import re
from sklearn.svm import SVC
from sklearn.externals import joblibs

keywords_files = ['keywords_BLK.txt', 'keywords_EVT.txt', 'keywords_GOV.txt', 'keywords_IDX.txt', 'keywords_VIP.txt']
fold_num = 5
embed_size = 200
vocab_size = 7123

#import sys
#default_encoding="utf-8"
#if(default_encoding!=sys.getdefaultencoding()):
#    reload(sys)
#    sys.setdefaultencoding(default_encoding)


def getvec(words):
    assert len(words)==embed_size+1
    word = words[0]
    vec = []
    for i in range(embed_size):
        vec.append(float(words[i+1]))
    return str(word),vec

def read_vector(filepath):
    # 读取词向量
    fopen = open(filepath, 'r')
    word_vec = {}
    i=0
    for line in fopen:
        words = line.split()
        if i==0:
            vocab_size = int(words[0])
            embed_size = int(words[1])
        else:
            word, vec = getvec(words)
            word_vec[word]=vec
        i+=1
    return word_vec

def read_words(filepath):
    # 读取词典
    fopen = open(filepath, 'r')
    if fopen is None:
        raise IOError('%s file no exist!'%(filepath))
    words = []
    for line in fopen:
        words.append(line[0:-1])
    fopen.close()
    return words

def read_keywords_files(keywords_files):
    # 读取关键字
    keywords = []
    for filename in keywords_files:
        keyword = read_words(filename)
        keywords.extend(keyword)
    return keywords

def build_dictionary_id(words):
    idx = 0
    dictionary={}
    for word in words:
        if not word in dictionary.keys():
            dictionary[word]=idx
            idx+=1
    return dictionary

def array_sum(data, vec):
    for i in range(embed_size):
        data[i]+=vec[i]
    return data

def array_div(data, number):
    for i in range(embed_size):
        data[i]/=number
    return data
def read_data_with_label(filepath, label, wordvec_dict, keywords):
    # 读取某一标签数据
    fopen = open(filepath)
    if fopen is None:
        raise IOError('%s file no exist!'%(filepath))
    datas={'X':[], 'y':[]}
   
    for key in keywords:
        print(key)
    for line in fopen:

        words = line.strip()
        data = [0]*(embed_size+len(keywords))
        number = 0
        # normal words
        for word in words:
            if word in wordvec_dict.keys():
                data = array_sum(data, wordvec_dict[word])
                number+=1
        if number>0:
            data = array_div(data, number)
        else:
            print('zero')
        
        #keywords
        idx = embed_size
        for key in keywords:
            if key in line:
                data[idx]+=1
            idx+=1

        #添加
        datas['X'].append(data)
        datas['y'].append(label)
    fopen.close()
    return datas

def merge_and_shuffle(pos_datas, neg_datas):
    # merge and shuffle
   
    for data in neg_datas['X']:
        pos_datas['X'].append(data)
    for label in neg_datas['y']:
        pos_datas['y'].append(label)

    pos_datas['X'] = np.array(pos_datas['X'])
    pos_datas['y'] = np.array(pos_datas['y'])

    index = [i for i in range(len(pos_datas['X']))]
    random.shuffle(index)   
    pos_datas['X'] = pos_datas['X'][index]  
    pos_datas['y'] = pos_datas['y'][index]  
    return pos_datas

def show_result(real_y, predict_y):
    # summarize the fit of the model
    print('RESULT')
    print(metrics.classification_report(real_y, predict_y))
    print('CONFUSION MATRIX')
    print(metrics.confusion_matrix(real_y, predict_y))


def fold_data(data, index, fold_num):
    total_len = len(data['X'])
    one_fold_len = total_len//fold_num
    test_start_index = index*one_fold_len
    test_end_index = min(test_start_index+one_fold_len, total_len)

    test_X = data['X'][test_start_index:test_end_index]
    test_y = data['y'][test_start_index:test_end_index]
    
    train_X = []
    train_y = []
    train_X.extend(data['X'][0:test_start_index])
    train_y.extend(data['y'][0:test_start_index])
    train_X.extend(data['X'][test_end_index:total_len])
    train_y.extend(data['y'][test_end_index:total_len])

    return train_X, train_y, test_X, test_y
    
def train_model(datas, modelpath, C, kernel, probability, shrinking, tol, class_weight, verbose, max_iter):
    

    train_predict_y = []
    train_real_y=[]
    test_predict_y=[]
    test_real_y=[]
    for i in range(fold_num):
        train_X, train_y, test_X, test_y = fold_data(datas, i, fold_num)
        clf = SVC(C=C, kernel=kernel, shrinking=shrinking, probability=probability, tol=tol, verbose=verbose, max_iter=max_iter)
        clf.fit(train_X, train_y)
        
        train_predict_y.extend(clf.predict(train_X))
        train_real_y.extend(train_y)

        test_predict_y.extend(clf.predict(test_X))
        test_real_y.extend(test_y)


    show_result(train_real_y, train_predict_y)
    show_result(test_real_y, test_predict_y)

    clf.fit(datas['X'], datas['y'])
    joblib.dump(clf, modelpath)


def test_model(data, modelpath):
    clf = joblib.load(modelpath)
    predict_y = clf.predict(data['X'])
    
def main(args):
    # 获取文件列表
    posfilepath = args.posfilepath
    negfilepath = args.negfilepath
    vecfilepath = args.vecfilepath

    # 创建词典
    wordvec_dict = read_vector(vecfilepath)
    keywords = read_keywords_files(keywords_files)
 
    # 读取参数数据
    modelpath = args.modelpath
    flag = args.flag
    C = args.C
    kernel = args.kernel
    probability = args.probability
    shrinking = args.shrinking
    tol = args.tol
    class_weight = args.class_weight
    verbose = args.verbose
    max_iter = args.max_iter

    #读取数据
    pos_datas = read_data_with_label(posfilepath, 1, wordvec_dict, keywords)
    neg_datas = read_data_with_label(negfilepath, 0, wordvec_dict, keywords)

    datas = merge_and_shuffle(pos_datas, neg_datas)


    if flag=='train':
        train_model(datas, modelpath, C, kernel, probability, shrinking, tol, class_weight, verbose, max_iter)
    elif flag == 'test':
        test_model(pos_datas, modelpath)

if __name__ == '__main__':
    # 参数解析
    parser = argparse.ArgumentParser(usage="newsflash svm usage:", description="help instruction")
    parser.add_argument("-posfilepath", default="", help="the input data path.")
    parser.add_argument("-negfilepath", default="", help="the input data path.") 
    parser.add_argument("-vecfilepath", default="", help="the input data path.")
    parser.add_argument("-flag", choices=['train', 'test'], default="train", help="svm mode")
    parser.add_argument("-C", default=1.0, type=float, help="Penalty parameter C of the error term.")
    parser.add_argument("-kernel", choices=['rbf','linear','poly', 'sigmoid', 'precomputed'], default="rbf", help="kernel function")
    parser.add_argument("-probability", default=False, help="whether to enable probability estimates")
    parser.add_argument("-shrinking", default=True, help="Whether to use the shrinking heuristic")
    parser.add_argument("-tol", default=1e-3, help="Tolerance for stopping criterion")
    parser.add_argument("-class_weight", default='balanced', help="whether use balanced tree class weight")
    parser.add_argument("-verbose", default=False, help="Enable Verbose output")
    parser.add_argument("-max_iter", default=-1, help="Hard limit to iterations with solver")
    parser.add_argument("-modelpath", default="", help="the model path, save for train and use for test!")
    
    args = parser.parse_args()
     
    main(args)
