# -*- coding: utf-8 -*- 

import collections
import string
import re
import argparse

pos_mixed_keys = []
neg_mixed_keys = []

neg_keys = []
pos_keys = []

pos_mixed_files = ['mixed_keywords_POS.txt']
neg_mixed_files = ['mixed_keywords_NEG.txt']

pos_files = ['keywords_EVT.txt', 'keywords_GOV.txt', 'keywords_IDX.txt', 'keywords_VIP.txt', 'keywords_POS.txt']
neg_files = ['keywords_BLK.txt', 'keywords_ORG.txt', 'keywords_PER.txt', 'keywords_GPE.txt', 'keywords_NEG.txt']

def contain_key(line, keys):
    for key in keys:
        if key in line:
#            print(key)
#            print(line)
            return True;
    return False

def contain_mixed_key(line, mixed_keys):
    for mixed_key in mixed_keys:
        keys = mixed_key.split('+')
        mixed = True
        for key in keys:
            if not key in line:
                mixed=False
                break
        if mixed:
            return True
    return False


def read_keywords_file(filepath):
    # 读取词典
    fopen = open(filepath, 'r')
    if fopen is None:
        raise IOError('%s file no exist!'%(filepath))
    words = []
    for line in fopen:
        line = line.replace('\n','')
        words.append(line)
    fopen.close()
    return words

def read_keywords_files(keywords_files):
    # 读取关键字
    keywords = []
    for filename in keywords_files:
        keyword = read_keywords_file(filename)
        keywords.extend(keyword)
    return keywords


def mark_news_based_rule(one_news):
	# 输入：一条新闻资讯
	# 输出：正样例输出True, 否则False
	if contain_key(one_news, neg_keys): 
       return False
    elif contain_mixed_key(line, pos_mixed_keys):
        return True
	elif contain_mixed_key(line, neg_mixed_keys):
        return False
    elif contain_key(line, pos_keys):
        return False
    else:
        return False # 保留未识别新闻
	
def mark_data_based_rule(inputfile, pos_mixed_keys, neg_mixed_keys, pos_keys, neg_keys, outputfile):
    
    fopen = open(inputfile, 'r')
    fwrite = open(outputfile, 'w')
    
    unsure_pos = []
    sure_pos = []
    sure_neg = []

    if fopen is None or fwrite is None :
        raise IOError('%s , %s or %s cannt open!'%(inputfile, outputfile))

    num = 0
    import time
    startTime = time.time()

    for line in fopen:
        num+=1
        if contain_key(line, neg_keys): 
            sure_neg.append(line)
 #           print('fu')
 #           print(line)
        elif contain_mixed_key(line, pos_mixed_keys):
            sure_pos.append(line)
        elif contain_mixed_key(line, neg_mixed_keys):
            sure_neg.append(line)
        elif contain_key(line, pos_keys):
            unsure_pos.append(line)
        else:
            sure_neg.append(line)
    endTime = time.time()
    print('ave time:%f\n'%((endTime-startTime)/num))
    fwrite.write('正样例\n')
    for pos in sure_pos:
        fwrite.write(pos)
    
    fwrite.write('\n负样例\n')

    for neg in unsure_pos:
        fwrite.write(neg)

    for neg in sure_neg:
        fwrite.write(neg)

    fopen.close()
    fwrite.close()


def main(args):
    # 获取文件列表
    inputfile = args.input
    outputfile = args.output

    pos_mixed_keys.extend(read_keywords_files(pos_mixed_files)) 
    neg_mixed_keys.extend(read_keywords_files(neg_mixed_files))

    pos_keys.extend(read_keywords_files(pos_files))
    neg_keys.extend(read_keywords_files(neg_files))
    
    mark_data_based_rule(inputfile, pos_mixed_keys, neg_mixed_keys, pos_keys, neg_keys, outputfile)

if __name__ == '__main__':
    # 参数解析
    parser = argparse.ArgumentParser(usage="python3 rule_based.py -input unlabel.txt -output label.result", description="help instruction")
    parser.add_argument("-input", default="unlabel.txt", help="the input data path.")
    parser.add_argument("-output", default="label.result", help="the input data path.") 
    
    args = parser.parse_args()
     
    main(args)
