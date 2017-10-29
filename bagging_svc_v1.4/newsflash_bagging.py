# -*- coding: utf-8 -*- 

# 参数

import argparse
import random
from sklearn import metrics
import numpy as np
import re
from sklearn.svm import SVC
from sklearn.externals import joblib
import jieba
import sys
import time
import os

from sklearn.ensemble import RandomForestClassifier, BaggingClassifier 
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

keywords_files = ['keywords_BLK.txt','keywords_ORG.txt', 'keywords_PER.txt','keywords_GPE.txt',
	'keywords_EVT.txt', 'keywords_GOV.txt', 'keywords_IDX.txt', 'keywords_VIP.txt']
fold_num = 5
embed_size = 200
vocab_size = 7123

def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)

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

def read_keywords_file(filepath):
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
        keyword = read_keywords_file(filename)
        keywords.extend(keyword)
    return keywords


def show_bar(info):
    info = '\r '+info
    sys.stdout.write(info)
    sys.stdout.flush()

def read_raw_datas(filepath):
    datas = []
    fopen = open(filepath, 'r')
    if fopen is None:
        raise IOError("%s cannt open!"%(filepath))
    for line in fopen:
        datas.append(line)
    return datas

def write_label_result_to_file(raw_datas, predict_y, filename):
    fwrite = open(filename, 'w')
    if fwrite is None:
        raise IOError('%s file cannt open!'%(filename))

    fwrite.write('******************正样例:**************\n')
    i=0
    for predict in predict_y:
        if predict==1:
            fwrite.write("%s"%(raw_datas[i]))
        i+=1

    fwrite.write("\n\n******************负样例:****************\n")
    i=0
    for predict in predict_y:
        if predict == 0:
            fwrite.write("%s"%(raw_datas[i]))
        i+=1

def read_data_with_label(filepath, label, wordvec_dict, keywords):
    # 读取某一标签数据
    fopen = open(filepath)
    if fopen is None:
        raise IOError('%s file no exist!'%(filepath))
    datas={'X':[], 'y':[]}
    
    start = time.clock()
    line_num = 0
    for line in fopen:
        line_num+=1
        end = time.clock()
        if end- start>=1:
            show_bar('filename: %s, read line number: %d'%(filepath, line_num))
            start = time.clock()

        words = jieba.cut(line)
        data = []
        # normal words
        for word in words:
            if hasNumbers(word):
                word = 'NUMBER'
            if word in wordvec_dict.keys():
                data.append(wordvec_dict[word])
        if len(data)>0:
            data = np.mean(data, axis=0)
            data = np.append(data, [0]*len(keywords))
        else:
            data = np.zeros(embed_size+len(keywords))

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

def shuffle_datas(datas):
    data_num = len(datas['X'])
    datas['X'] = np.array(datas['X'])
    datas['y'] = np.array(datas['y'])

    index = [i for i in range(data_num)]
    random.shuffle(index)
          
    datas['X'] = datas['X'][index]
    datas['y'] = datas['y'][index]
    return datas

def pick_datas(datas, number):
    pick_datas={}
    remain_datas = {}

    number = min(number, len(datas['X']))

    pick_datas['X'] = datas['X'][0:number]
    pick_datas['y'] = datas['y'][0:number]
    remain_datas['X'] = datas['X'][number:]
    remain_datas['y'] = datas['y'][number:]
    return pick_datas, remain_datas

def merge_and_shuffle(pos_datas, neg_datas):
    # merge and shuffle
    
    datas={}
    datas['X'] = np.concatenate((pos_datas['X'], neg_datas['X']),axis=0)
    datas['y'] = np.concatenate((pos_datas['y'], neg_datas['y']),axis=0)


    index = [i for i in range(len(datas['X']))]
    random.shuffle(index)   
    datas['X'] = datas['X'][index]  
    datas['y'] = datas['y'][index]  
    return datas

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

def votes(predict_y):
    vote_num = len(predict_y)
    predict_y = np.array(predict_y)
    votes = np.mean(predict_y, axis=0)
    votes = [1 if vote>0.5 else 0 for vote in votes]
    return votes
    

def multitest_model(test_number, pos_datas, neg_datas, traindatarate, modelpath, modelnumber, C, kernel, shrinking, probability, tol, class_weight, verbose, max_iter):
    real_ys =[]
    predict_ys = []
    for i in range(test_number):
        print('\nround:%d'%(i+1))
        real_y, predict_y = train_model(pos_datas, neg_datas, traindatarate, modelpath, modelnumber, C, kernel, shrinking,probability, tol, class_weight, verbose, max_iter)
        
        real_ys.extend(real_y)
        predict_ys.extend(predict_y)

    show_result(real_ys, predict_ys)

def create_base_estimators(modelnumber, kernel, verbose):
    estimators = []
    clf1 = GradientBoostingClassifier()
    clf2 = RandomForestClassifier(random_state=1)
    clf3 = LogisticRegression(random_state=1)
    clf4 = GaussianNB()

    for i in range(modelnumber):
        clf = SVC(kernel=kernel)
        estimators.append(('svc%d'%(i), clf))
    return estimators

def train_model(pos_datas, neg_datas, traindatarate, modelpath, modelnumber, C, kernel, shrinking, probability, tol, class_weight, verbose, max_iter):
    

    pos_datas = shuffle_datas(pos_datas)
    neg_datas = shuffle_datas(neg_datas)

    train_num = int(len(pos_datas['y'])*traindatarate)
    test_num = int(len(pos_datas['y'])*(1-traindatarate))

    test_pos_datas, train_pos_datas = pick_datas(pos_datas, test_num)
    test_neg_datas, train_neg_datas = pick_datas(neg_datas, test_num)

    test_datas = merge_and_shuffle(test_pos_datas, test_neg_datas)
    train_datas = merge_and_shuffle(train_pos_datas, train_neg_datas)
    
    predict_y = []
    real_y = test_datas['y']

    #estimators = create_base_estimators(modelnumber, kernel, verbose)

    clf = SVC(kernel=kernel, class_weight={1:class_weight})

    eclf = BaggingClassifier(base_estimator=clf, n_estimators=modelnumber, max_samples=0.2, max_features=1.0, bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False, n_jobs=-1, random_state=None, verbose=1)
    
    eclf.fit(train_datas['X'], train_datas['y'])
   
    joblib.dump(eclf, modelpath)

    predict_y = eclf.predict(test_datas['X'])
    return real_y, predict_y

def test_model(data, modelpath):
    clf = joblib.load(modelpath)
    predict_y = clf.predict(data['X'])

def unlabel_test(datas, raw_datas, modeldir, outputfilepath):
    
    predict_y = []

    parents = os.listdir(modeldir)
    for parent in parents:
        modelpath = os.path.join(modeldir, parent)
        clf = joblib.load(modelpath)
        predict_y.append(clf.predict(datas['X']))
    
    predict_y = votes(predict_y)

    write_label_result_to_file(raw_datas, predict_y, outputfilepath)

def main(args):
    # 获取文件列表
    posfilepath = args.posfilepath
    negfilepath = args.negfilepath
    vecfilepath = args.vecfilepath
    unlabelfilepath = args.unlabelfilepath
    outputfilepath = args.outputfilepath

    # 创建词典
    wordvec_dict = read_vector(vecfilepath)
    keywords = read_keywords_files(keywords_files)
 
    # 读取参数数据
   
    flag = args.flag
    modeldir = args.modeldir
    modelname = args.modelname
    modelnumber = args.modelnumber
    traindatarate = args.traindatarate
    test_number = args.test_number

    C = args.C
    kernel = args.kernel
    probability = args.probability
    shrinking = args.shrinking
    tol = args.tol
    class_weight = args.class_weight
    verbose = args.verbose
    max_iter = args.max_iter

    #读取数据

    if flag == 'unlabeltest':
        unlabel_datas = read_data_with_label(unlabelfilepath, 1, wordvec_dict, keywords)
        raw_datas = read_raw_datas(unlabelfilepath)
    else:
        pos_datas = read_data_with_label(posfilepath, 1, wordvec_dict, keywords) 
        #neg_datas = read_data_with_label(posfilepath, 0, wordvec_dict, keywords) 
        neg_datas = read_data_with_label(negfilepath, 0, wordvec_dict, keywords)
        class_weight = len(neg_datas['X'])//len(pos_datas['X'])
        print('class_weight:%d'%(class_weight))

    if flag=='train':
        real_y, predict_y = train_model(pos_datas, neg_datas, traindatarate, modeldir+modelname, modelnumber, C, kernel, shrinking,probability, tol, class_weight, verbose, max_iter)
        show_result(real_y, predict_y)
    elif flag=='multitest':
        multitest_model(test_number, pos_datas, neg_datas, traindatarate, modeldir+modelname, modelnumber, C, kernel, shrinking,probability, tol, class_weight, verbose, max_iter)
    elif flag == 'test':
        test_model(pos_datas, neg_datas, modeldir)
    elif flag == 'unlabeltest':
        unlabel_test(unlabel_datas, raw_datas, modeldir, outputfilepath)

if __name__ == '__main__':
    # 参数解析
    parser = argparse.ArgumentParser(usage="sample: python newsflash_vec_SVM.py -posfilepath newsflash.pos -negfilepath newsflash.neg -vecfilepath newsflash_wordvectors.bin -unlabelfilepath unlabel.txt -outputfilepath label.result -modeldir ./models/ -modelname svc_linear -modelnumber 10 -traindatarate 0.9 -test_number 10 -kernel linear -flag train", description="help instruction")
    parser.add_argument("-posfilepath", default="newsflash.pos", help="the input data path.")
    parser.add_argument("-negfilepath", default="newsflash.neg", help="the input data path.") 
    parser.add_argument("-vecfilepath", default="newsflash_wordvectors.bin", help="the input data path.")
    parser.add_argument("-unlabelfilepath", default="unlabel.txt", help="the input data path.")
    parser.add_argument("-outputfilepath", default="label.result", help="the input data path.")
    parser.add_argument("-flag", choices=['train', 'test', 'multitest', 'unlabeltest'], default="train", help="svm mode")
   
    parser.add_argument("-modeldir", default="./models/", help="the model path, save for train and use for test!")
    parser.add_argument("-modelname", default="bagging.model", help="the model path, save for train and use for test!")
    parser.add_argument("-modelnumber", default=40, type=int, help="the model path, save for train and use for test!")
    parser.add_argument("-traindatarate", default=0.9, type=float, help="the model path, save for train and use for test!")
    parser.add_argument("-test_number", default=10, type=int, help="the model path, save for train and use for test!")
   
    parser.add_argument("-C", default=1.0, type=float, help="Penalty parameter C of the error term.")
    parser.add_argument("-kernel", choices=['rbf','linear','poly', 'sigmoid', 'precomputed'], default="rbf", help="kernel function")
    parser.add_argument("-probability", default=False, help="whether to enable probability estimates")
    parser.add_argument("-shrinking", default=True, type=bool, help="Whether to use the shrinking heuristic")
    parser.add_argument("-tol", default=1e-3, type=float, help="Tolerance for stopping criterion")
    parser.add_argument("-class_weight", default=15,type=int, help="whether use balanced tree class weight")
    parser.add_argument("-verbose", default=False, type=bool, help="Enable Verbose output")
    parser.add_argument("-max_iter", default=-1, type=int, help="Hard limit to iterations with solver")
    
    
    args = parser.parse_args()
     
    main(args)
