# -*- coding: utf-8 -*- 

import collections
import string
import re
import argparse

pos_mixed_keys = ['黄金', '白银','欧元兑美元','欧元/美元', '美元兑日元','美元/日元', '英镑/美元', '恐慌指数VIX', '道指', '美联储+通胀率'
'美国+核心PCE物价指数同比', '美国+cpi同比', '美国+CPI同比', '美国+核心cpi同比', '美国+核心CPI同比','美国+核心CPI环比','美国+核心cpi环比', '美国+非农就业人口变动', '美国+私营部门就业人口变动', 
'美国+失业率', '美国+平均每小时工资同比', '美国+平均每周工时', '美国+劳动参与率', '美国+个人收入环比', '美国+个人消费支出','美国+ism制造业指数', '美国+ISM制造业',
 '美国+PMI', '美国+耐用品订单环比', '美国+就业人数变动', '美国+周首次申请失业救济人数', '美国+JOLTS职位空缺', '美国+工业产出环比', '美国+核心PCE物价指数',
'美国+新屋开工环比', '美国+新屋销售环比','美国+国债','美国+ADP', '美国+adp','美股+暴涨', '美股+暴跌', '美元指数', '美国+耐用品订单', '美国+GDP', '美联储+加息', '美联储+减息',
'欧元区+失业率', '欧元区+cpi同比初值', '欧元区+核心cpi同比初值', '欧元区+零售销售环比','欧元区+工业产出环比', '欧元区+制造业pmi初值', '欧元区+服务业pmi初值','欧元区+综合pmi初值',
'欧洲央行+通胀', '欧洲央行+会议纪要', '美联储+会议纪要','人民币兑美元',
'英国+脱欧', '英国+退欧','伊朗', 
'Coeure：', 'Fischer：', 'Constancio：', 'Jack Lew：','耶伦：', 'Bullard：', '德拉吉：', 'Powell：', 'Villeroy：', 'Kaplan：', '野村：', 'Evans：', 'Lockhart：', 'Katainen：',
'Praet：', 'Knot：','黑田东彦：', 'Lacker：', 'Mnuchin：','Mersch：', 'Lael Brainard：', '容克：', 'Harker：', 'Preat：', 'Dombrovskis：', 'Nouy：', '美联储理事', 'Mester：', ]

neg_mixed_keys = ['英国', '欧元区', '美国+新屋', '美国+成屋', '美国+ISM非制造业', '美国+工厂订单','欧洲央行+GDP', '欧洲央行+QE', '新债王',
	'日本+CPI','日本+国债收益率', '日本+失业率', '日本+零售销售', '日本+工业产出', '日本+PMI','日本+GDP']

neg_keys = ['钢', '油','豆','【提醒】','棉花','铜','费城联储', '法国', '中国', '在岸人民币', '离岸人民币', '拍卖', '德国', '墨西哥', '纽约联储', '时代周刊',
 '芝加哥', '亚特兰大联储', '达拉斯联储', '里士满联储', '克利夫兰联储', '堪萨斯联储', '上证', '前美联储', '医保', '全国']
pos_keys = []

pos_files = ['keywords_EVT.txt', 'keywords_GOV.txt', 'keywords_IDX.txt', 'keywords_VIP.txt']
neg_files = ['keywords_BLK.txt', 'keywords_ORG.txt', 'keywords_PER.txt', 'keywords_GPE.txt']

def contain_key(line, keys):
    for key in keys:
        if key in line:
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


def mark_data(inputfile, pos_keys, neg_keys, outputposfile, outputnegfile):
    
    fopen = open(inputfile, 'r')
    fwritepos = open(outputposfile, 'w')
    fwriteneg = open(outputnegfile, 'w')
    
    sure_pos = []
    unsure_pos = []

    if fopen is None or fwritepos is None or fwriteneg is None:
        raise IOError('%s , %s or %s cannt open!'%(inputfile, outputposfile, outputnegfile))

    pos_num = 0
    neg_num = 0
    unsure_pos_num = 0
    for line in fopen:
        if contain_key(line, neg_keys):
            fwriteneg.write(line)
            neg_num+=1
        elif contain_mixed_key(line, pos_mixed_keys):
            sure_pos.append(line)
            pos_num+=1
        elif contain_mixed_key(line, neg_mixed_keys):
            fwriteneg.write(line)
            neg_num+=1
        elif contain_key(line, pos_keys):
            unsure_pos.append(line)
            pos_num+=1
            unsure_pos_num+=1
        else:
            fwriteneg.write(line)
            neg_num+=1

    for unsure in unsure_pos:
        fwriteneg.write(unsure)
    #fwritepos.write('********************************************************************\n')

    for sure in sure_pos:
        fwritepos.write(sure)

    print("unsure pos number:%d, pos number:%d, neg number:%d"%(unsure_pos_num, pos_num, neg_num))
    
    fopen.close()
    fwritepos.close()
    fwriteneg.close()


def main(args):
    # 获取文件列表
    inputfile = args.inputfile
    outputposfile = args.outputposfile
    outputnegfile = args.outputnegfile

    pos_keys.extend(read_keywords_files(pos_files))
    neg_keys.extend(read_keywords_files(neg_files))
    
    mark_data(inputfile, pos_keys, neg_keys, outputposfile, outputnegfile)

if __name__ == '__main__':
    # 参数解析
    parser = argparse.ArgumentParser(usage="sample: python3 markdataBykeywords.py -inputfile newsflash_all.txt -outputposfile newsflash.pos -outputnegfile newsflash.neg", description="help instruction")
    parser.add_argument("-inputfile", default="newsflash_all.txt", help="the input data path.")
    parser.add_argument("-outputposfile", default="newsflash.pos", help="the input data path.") 
    parser.add_argument("-outputnegfile", default="newsflash.neg", help="the input data path.") 
    
    args = parser.parse_args()
     
    main(args)
