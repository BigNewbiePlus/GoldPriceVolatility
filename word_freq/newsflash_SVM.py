# -*- coding: utf-8 -*- 

# 参数

import argparse
import random
from sklearn import metrics
from sklearn.externals import joblib
import numpy as np
import re
from sklearn.svm import SVC
from sklearn.feature_selection import chi2
import jieba
import sys
import collections
from operator import itemgetter
from sklearn.ensemble import VotingClassifier
import math
import feature_selection


min_freq = 1
ctl_num = 500
kernel = 'linear'
C=1.0
gamma=1e-3

keywords_files = ['keywords_BLK.txt', 'keywords_EVT.txt', 'keywords_GOV.txt', 'keywords_IDX.txt', 'keywords_VIP.txt']

def hasNumbers(word):
    # 判断是否包含数字
    return any(char.isdigit() for char in word)

def showbar(info):
    # 显示进度
    info = '\r '+info
    sys.stdout.write(info)
    sys.stdout.flush()

def read_words(filepath):
    fopen = open(filepath, 'r')
    if fopen is None:
        raise IOError('%s file no exist!'%(filepath))

    words = []
    print('ctl:%d'%ctl_num)
    num = 0
    for line in fopen: 
        num+=1
        if num>ctl_num:
            break
        line = line.replace('\n', '')
        seg_list = jieba.cut(line)
        if num%100==0:
            showbar('read file %s %d'%(filepath, num))
        for word in seg_list:
            if hasNumbers(word):
                word = 'NUMBER'
            words.append(word)
    fopen.close()
    print('')
    
    return words

def read_data_with_label(filepath, label, dictionary, local_fun, global_fun):
    # 读取某一标签数据
    fopen = open(filepath)
    if fopen is None:
        raise IOError('%s file no exist!'%(filepath))
    
    datas = []
    labels = []

    dict_size = len(dictionary)
    idf_cnt = np.ones(dict_size)

    num=0
    print(ctl_num)
    for line in fopen:
        num+=1
        if num>ctl_num:
            break
        if num%100==0:
            showbar('read file %s:%d'%(filepath, num))
        data=np.zeros(dict_size)

        # normal words
        line = line.replace('\n', '')
        words = jieba.cut(line)
        for word in words:
            if hasNumbers(word):
                word = 'NUMBER'
            if word in dictionary.keys():
                if local_fun == 'one':
                    data[dictionary[word]]=1
                else:
                    data[dictionary[word]]+=1
        
        if global_fun == 'idf':
            for i in range(dict_size):
                if data[i]>0:
                    idf_cnt[i]+=1
        #keywords
        #for key in keywords:
        #    if key in line:
        #        data[dictionary[key]]+=1
        #添加
        datas.append(data)
        labels.append(label)
    fopen.close()
    print('')
    return datas, labels, idf_cnt

def read_data_with_label_and_weight(filepath, label, dictionary, idf, local_fun, global_fun):
    # 读取某一标签数据
    fopen = open(filepath)
    if fopen is None:
        raise IOError('%s file no exist!'%(filepath))
    
    datas = {'X':[], 'y':[]}

    dict_size = len(dictionary) 
 
    if global_fun == 'one':
        idf = np.ones(dict_size)

    num=0
    print(ctl_num)
    for line in fopen:
        num+=1
        if num>ctl_num:
            break
        if num%100==0:
            showbar('read file %s:%d'%(filepath, num))
        data=np.zeros(dict_size)

        # normal words
        line = line.replace('\n', '')
        words = jieba.cut(line)
        for word in words:
            if hasNumbers(word):
                word = 'NUMBER'
            if word in dictionary.keys():
                if local_fun == 'one':
                    data[dictionary[word]]=1
                else:
                    data[dictionary[word]]+=1
        # tf*idf
        data = [data[i]*idf[i] for i in range(dict_size)]
        #keywords
        #for key in keywords:
        #    if key in line:
        #        data[dictionary[key]]+=1
        #添加
        datas['X'].append(data)
        datas['y'].append(label)
    fopen.close()
    print('')
    return datas

def read_keywords_file(filepath):
    # 读取关键字
    keywords = []
    fopen = open(filepath)
    if fopen is None:
        raise IOError('%s cannt open'%(filepath))
    num=0
    for line in fopen:
        num+=1
        if num%100==0:
            showbar('read file %s:%d'%(filepath,num))
        line = line.replace('\n', '')
        keywords.append(line)

    print('')
    fopen.close()
    return keywords

# 创建词典
def create_dictionary(posfile, negfile, dicfile, use_stopwords, stopwordsfile, use_chi2_select, local_fun, global_fun):
    stopwords = []
    if use_stopwords:
        stopwords = read_keywords_file(stopwordsfile)

    fwrite = open(dicfile, 'w')
    if fwrite is None:
        raise IOError('%s cannt open'%(dicfile))

    # 读取所有文件内的词
    words = []
    words.extend(read_words(posfile))
    words.extend(read_words(negfile))

    # 统计词频
    cnts = collections.Counter(words).most_common()

    print('total vocab:%d'%len(cnts))

    # 去除停用词+低词频
    dictionary = {}
    idx = 0
    reverse_dic = []
    for cnt in cnts:
        if cnt[1]>=min_freq and cnt[0] not in stopwords:
            dictionary[cnt[0]] = idx
            reverse_dic.append(cnt[0])
            idx+=1
    
    print('total vocab after stop and min_req :%d'%idx)

    # 卡方检验抽取chi2_rate的词，并计算global_fun
    if not use_chi2_select:#使用卡方检验
        return

    posdata, poslabel, posidf = read_data_with_label(posfile, 1, dictionary, local_fun, global_fun)
    negdata, neglabel, negidf = read_data_with_label(negfile, -1, dictionary, local_fun, global_fun)
    
    # 文档数
    D = len(posdata) + len(negdata)

    # 获取idf
    idf = np.log(D/(posidf+negidf))

    datas = posdata
    labels = poslabel
    datas.extend(negdata)
    labels.extend(neglabel)
    
    global C
    global kernel
    global gamma

    dim_k, C, kernel, gamma, scores, pvals = feature_selection.feature_selection(datas, labels)


    # chi2值，p值, 单词, idf合并
    chi2info = zip(scores, pvals, reverse_dic, idf)

    chi2info = sorted(chi2info, key=itemgetter(0), reverse=True)

    vocab_size = dim_k
    print('total vocab after chi2:%d'%vocab_size)
    for i in range(vocab_size):
        fwrite.write('%lf\t%lf\t%s\t%lf\n'%(chi2info[i][0], chi2info[i][1], chi2info[i][2], chi2info[i][3]))
    fwrite.close()

def read_dictionary(filepath):
    # 读取词典
    fopen = open(filepath, 'r')
    if fopen is None:
        raise IOError('%s file no exist!'%(filepath))
    
    dictionary = {}
    idx=0
    idf = []
    for line in fopen:
        line = line.replace('\n', '')
        line = line.split('\t')
        dictionary[line[2]] = idx
        idf.append(float(line[3]))
        idx+=1
    fopen.close()
    return dictionary, idf

def read_keywords_files(keywords_files):
    # 读取关键字
    keywords = []
    for filename in keywords_files:
        keyword = read_words(filename)
        keywords.extend(keyword)
    return keywords

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

def create_base_estimators(modelnumber, kernel, class_weight, verbose):
    estimators = []
    for i in range(modelnumber):
        clf = SVC(kernel=kernel, C=C, gamma=gamma, class_weight={1:class_weight}, verbose=verbose)
        estimators.append(('svc%d'%(i), clf))    
    return estimators

def multitest_model(test_number, pos_datas, neg_datas, traindatarate, modelpath, modelnumber, shrinking, probability, tol, verbose, max_iter):
    real_ys =[]
    predict_ys = []
    for i in range(test_number):
        print('\nround:%d'%(i+1))
        real_y, predict_y = train_model(pos_datas, neg_datas, traindatarate, modelpath, modelnumber,shrinking,probability, tol, verbose, max_iter)
        
        real_ys.extend(real_y)
        predict_ys.extend(predict_y)

    show_result(real_ys, predict_ys)

def train_model(pos_datas, neg_datas, traindatarate, modelpath, modelnumber, shrinking, probability, tol, verbose, max_iter):
    

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
	
    class_weight = int(len(neg_datas['X'])/len(pos_datas['X']))

    estimators = create_base_estimators(modelnumber, kernel, class_weight, verbose) 
    eclf = VotingClassifier(estimators=estimators, voting='hard', n_jobs=-1)

    
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

    global min_freq
    global ctl_num

    # 获取文件列表
    posfile = args.posfile
    negfile = args.negfile
    dicfile = args.dicfile

    min_freq = args.min_freq
    use_stopwords = args.use_stopwords
    stopwordsfile = args.stopwordsfile
    use_chi2_select = args.use_chi2_select

    local_fun = args.local_fun
    global_fun =args.global_fun
    recreate_dic = args.recreate_dic
    # 读取参数数据
    modeldir = args.modeldir
    modelname = args.modelname
    modelnumber = args.modelnumber

    flag = args.flag
    C = args.C
    kernel = args.kernel
    probability = args.probability
    shrinking = args.shrinking
    tol = args.tol
    traindatarate = args.traindatarate
    verbose = args.verbose
    max_iter = args.max_iter
    test_number = args.test_number
    ctl_num = args.ctl_num
    
    #创建词典并保存，每行一个词，按词频由大到小排序
    if recreate_dic:
        create_dictionary(posfile, negfile, dicfile, use_stopwords, stopwordsfile, use_chi2_select, local_fun, global_fun)
    
    #读取词典，并赋予id（从0开始递增）
    dictionary, idf = read_dictionary(dicfile)

    # 创建词典
    #keywords=[]
    ''':
    keywords = read_keywords_files(keywords_files)
    for key in keywords:
        words.append(key)
    '''
 


    #读取数据
    pos_datas = read_data_with_label_and_weight(posfile, 1, dictionary, idf, local_fun, global_fun)
    neg_datas = read_data_with_label_and_weight(negfile, -1, dictionary, idf, local_fun, global_fun)

    if flag=='train':
        real_y, predict_y = train_model(pos_datas, neg_datas, traindatarate, modeldir+modelname, modelnumber,shrinking,probability, tol, verbose, max_iter)
        show_result(real_y, predict_y)
    elif flag=='multitest':
        multitest_model(test_number, pos_datas, neg_datas, traindatarate, modeldir+modelname, modelnumber, shrinking, probability, tol , verbose, max_iter)
    elif flag == 'test':
        test_model(pos_datas, neg_datas, modeldir)
    elif flag == 'unlabeltest':
        unlabel_test(unlabel_datas, raw_datas, modeldir, outputfilepath)

if __name__ == '__main__':
    # 参数解析
    parser = argparse.ArgumentParser(usage="python newsflash_SVM.py -dicfile dic.txt -use_stopwords False -use_chi2 False -chi2_rate 0.6 -local_fun tf -global_fun idf -recreate_dic True -ctl_num 500 -flag multitest -kernel linear -modelname svm.model -modelnumber 10 -traindatarate 0.9 -test_number 10", description="help instruction")
    parser.add_argument("-posfile", default="newsflash.pos", help="the input data path.")
    parser.add_argument("-negfile", default="newsflash.neg", help="the input data path.") 
    parser.add_argument("-dicfile", default="dic.txt", help="the input data path.")
    parser.add_argument("-min_freq", default=1, type=int, help="the input data path.")
    
    parser.add_argument("-use_stopwords", default=False, type=bool, help="the input data path.")
    parser.add_argument("-stopwordsfile", default='stopwords.txt', help="the input data path.")

    parser.add_argument("-use_chi2_select", default=False, type=bool, help="the input data path.")
    
    parser.add_argument('-local_fun', choices=['tf', 'one'], default='tf',)
    parser.add_argument("-global_fun", choices=['idf', 'one'], default='idf', help="the input data path.")
    
    parser.add_argument("-recreate_dic",  default=False, type=bool, help="the input data path.")
    parser.add_argument("-ctl_num",  default=500, type=int, help="the input data path.")
    
    parser.add_argument("-flag", choices=['train', 'test', 'multitest'], default="train", help="svm mode")
    parser.add_argument("-C", default=1.0, type=float, help="Penalty parameter C of the error term.")
    parser.add_argument("-kernel", choices=['rbf','linear','poly', 'sigmoid', 'precomputed'], default="linear", help="kernel function")
    parser.add_argument("-probability", default=False, help="whether to enable probability estimates")
    parser.add_argument("-shrinking", default=True, help="Whether to use the shrinking heuristic")
    parser.add_argument("-tol", default=1e-3, help="Tolerance for stopping criterion")
    parser.add_argument("-verbose", default=False, type=bool, help="Enable Verbose output")
    parser.add_argument("-max_iter", default=-1, help="Hard limit to iterations with solver")
    parser.add_argument("-modeldir", default="./models/", help="the model path, save for train and use for test!")
    parser.add_argument("-modelname", default="svm.model", help="the model path, save for train and use for test!")
    parser.add_argument("-modelnumber", default=2, type=int, help="the model path, save for train and use for test!")
    parser.add_argument("-traindatarate", default=0.9, type=float, help="the model path, save for train and use for test!")
    parser.add_argument("-test_number", default=10, type=int, help="the model path, save for train and use for test!")
    
    args = parser.parse_args()
     
    main(args)
