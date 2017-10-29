# -*- coding: utf-8 -*- 

import collections
import string
import re
import argparse

# 全匹配删除
fullmatch_filter_keys = ['报', '台', '日本', '党内', '玻', '公司', '基金', '美方', '港', '日方', '汽', '欧委会', '夏普', '监察部', '穆斯林','教育部长','保守派','国防部长','工作组','大使馆','常委会', '众院', 
#GPE
'北美','日高','永定','俄美','美俄','美韩','美林', '美加','俄方','美东','东亚','川','德法','俄罗','德美','多哈','俄国','欧洲大陆','保税区','非本县','淡水','南通','新英','法德',
'高淳', '西南', '阿美', '永丰',
#ORG
'中央', '政府', '农业部', '外交部', '议会', '环保部', '欧佩克', '国土资源部', '国土部', '工会', '伊斯兰国极端组织', '国土安全部', '自民党', '卫计委', '财险', '人大', '科大', '卡夫', '奥迪', '礼来', '中源', 
'麦迪科技', '南玻', '民进', '安联', '彭博', '贝克休斯', '安邦系', '绿党', '国新', '德勤', '天津中煤民生', '总统府', '希腊政府', '正大', '卫生部长', '营业部', '能源部', '国投', '公安部', '反对党', '北大',
'联席会', '国内汽', '劳工部长', '立法会', '国土局', '穆迪', '国集团', '公安局', '大通', '政治局', '环保局', '上议院', '美林', '下议院', '美军', '社民党', '参议院', 
# PER
'洪崎', '卡尔', '布拉德', '蒂勒', '彭斯', '伦齐', '勒庞', '菲永', '布伦特', '拉加德', '吴晓灵', '伯克希尔', '拉夫罗夫', '标普', '布伦', '管清友', '易会满', '惠勒', '考克', '国海', '何昕', '陈琳', '蔡昉',
'嘉凯城', '华生', '安新', '标普纳', '伯克', '莱特', '瑞安', '钟正', '陆奇', '何平', '贝克', '连平', '银隆', '水汝庆', '韦伯', '张生', '陶然', '于利', '戈尔', '御准', '亨利', '大明', '王毅韧', '朱隽', 
'欧绩优', '韩军', '淡马', '沃尔克', 
]

all_filter_files = ['keywords_EVT.txt', 'keywords_GOV.txt', 'keywords_IDX.txt', 'keywords_VIP.txt']
all_filter_keys = []

org_keep_keys = ['银行', '集团', '公司', '证券', '基金' , '股份', '中国', '院', '厅', '台', '局', '网', '协', '学', '社','所','署','军', '警','世贸组织', '经济部长']
org_keep_files = ['keywords_BLK.txt']

org_filter_keys = ['白宫', '联合国', 'CFTC', '标普', 'IPSOS', '欧元区', '美国','欧盟','英国','英央行','欧洲', '特朗普', '委员会','伊斯兰国际极端组织','金银', '富时泛欧绩优', '占比会', '本季度',
'上涨', '例会', '司法部长', '通报', '隔夜', '内部', '工商界', '内改', '纳斯达克', '自由党团', '交易', '法国极右翼','经济部','周六证监会','联邦法院']

person_filter_keys = ['周涨', '周三', '周一', '约翰','区块链', '日元','比特币', '普通胀', '注册函','本周', '被刑拘', '若无法', '金涨超', '英镑刷', '钢现涨', '柴油价', '上证报', '高薪阶',
'很高兴', '周成交量', '本次汽', '密歇根', '欧洲央', '布油涨', '琼斯', '涨拉升', '次新股', '下周一', '沪铅', '西雅', '欧洲的', '普选择', '高开约', '沙特正', '防范金', '大卫', '威廉', '曲线', '反特',
'斯诺', '特郎普', '完奥巴', '特里', '热朗普', '白花钱','盘中','加强监', '金融强', '韩美日','伦铜','可实现']

gpe_filter_keys = ['中', '英美', '美英', '美朝','韩美', '欧美','美日', '美中', '美邦', '美军', '西欧', '北韩', '朝核', '美伊','日美', '京','华', '巴', '墨','袁', '美国华盛顿', '美国总部大楼', '野村', '第四季度银行', '访美', '美利坚', '美元美', '美股美', '日纽约', '美国日本', '巴克莱', '日英国', '英欧', '汉',  '以', '伊斯兰', '英国硬脱欧']

def full_match_key(name, keys):
    for key in keys:
        if key==name:
            return True
    return False

def has_letter(name):
    match = re.match(r'.*[a-zA-Z].*', name)
    return bool(match)

def contain_key(name, keys):
    for key in keys:
        if key in name or name in key:
            return True;
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

def get_list(all_names, min_freq):
    names = []

    cnts = collections.Counter(all_names).most_common(len(all_names))
    for cnt in cnts:
        if cnt[1]>min_freq:
            names.append(cnt[0])
    return names

def filter_ner(nerfilename, all_filter_keys, org_keep_keys):
    fopen = open(nerfilename, 'r')
    if fopen is None:
        raise IOError('%s cannt open!'%(nerfilename))

    keep_orgs = []
    eng_orgs = []
    unsure_orgs = []
    eng_pers = []
    keep_pers = []
    gpes = []

    r1 = u'[0-9’!"#$%&\'()（ ）*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'#用户也可以在此进行自定义过滤字符
 
    for eachLine in fopen:
        words = eachLine.split('\t')
        if len(words)>1 and words[1] != 'MISC':
            
            words[0]=re.sub(r1, '', words[0]) #过滤内容中的各种标点符号
            #words[0] = ''.join(words[0].split())
            if words[0]=='' or full_match_key(words[0], fullmatch_filter_keys):
                continue
            if words[1]=='ORGANIZATION':
                if contain_key(words[0], org_filter_keys):
                    continue
                elif contain_key(words[0], org_keep_keys):
                    keep_orgs.append(words[0])
                elif not contain_key(words[0], all_filter_keys):
                    if has_letter(words[0]):
                        eng_orgs.append(words[0])
                    else:
                        unsure_orgs.append(words[0])
            elif words[1]=='PERSON':                
                if len(words[0])>1 and not contain_key(words[0], all_filter_keys) and not contain_key(words[0], person_filter_keys):
                    if has_letter(words[0]):
                        eng_pers.append(words[0])
                    else:
                        keep_pers.append(words[0])
            elif words[1]=='GPE' or words[1]=='FACILITY' or words[1]=='LOCATION':
                if not contain_key(words[0], all_filter_keys) and not contain_key(words[0], gpe_filter_keys):
                    gpes.append(words[0])

    keep_orgs_unique = get_list(keep_orgs, 1)
    unsure_orgs_unique = get_list(unsure_orgs, 1)
    eng_orgs_unique = get_list(eng_orgs, 5)
    eng_orgs_unique.extend(unsure_orgs_unique)
    eng_pers_unique = get_list(eng_pers, 5)
    unsure_orgs_unique = get_list(unsure_orgs, 1)

    keep_pers_unique = get_list(keep_pers, 1)
    gpes_unique = get_list(gpes, 1)

    fopen.close()
    return eng_orgs_unique, keep_orgs_unique, eng_pers_unique, keep_pers_unique, gpes_unique



def save_ner(orgfile, personfile, gpefile, unsure_orgs, keep_orgs, eng_pers, keep_pers, gpes):
    fwrite_org = open(orgfile, 'w')
    fwrite_person = open(personfile, 'w')
    fwrite_gpe = open(gpefile, 'w')

    if fwrite_org is None or fwrite_person is None or fwrite_gpe is None:
        raise IOError("%s, %s or %s cannt open"%(orgfile, personfile, gpefile))

    for org in unsure_orgs:
        fwrite_org.write("%s\n"%org)

    for org in keep_orgs:
        fwrite_org.write("%s\n"%org)

    for eng_per in eng_pers:
        fwrite_person.write("%s\n"%eng_per)

    for keep_per in keep_pers:
        fwrite_person.write("%s\n"%keep_per)

    for gpe in gpes:
        fwrite_gpe.write("%s\n"%gpe)

    fwrite_org.close()
    fwrite_person.close()
    fwrite_gpe.close()


def main(args):
    # 获取文件列表
    nerfile = args.nerfile
    personfile = args.personfile
    orgfile = args.orgfile
    gpefile = args.gpefile
    

    all_filter_keys.extend(read_keywords_files(all_filter_files))
    org_keep_keys.extend(read_keywords_files(org_keep_files))
    
    unsure_orgs, keep_orgs, eng_pers, keep_pers, gpes= filter_ner(nerfile, all_filter_keys, org_keep_keys)

    print('unsure_orgs:%d,keep_orgs:%d, eng_per:%d, person:%d, gpe:%d'%(len(unsure_orgs), len(keep_orgs), len(eng_pers), len(keep_pers), len(gpes)))
    save_ner(orgfile, personfile, gpefile, unsure_orgs, keep_orgs, eng_pers, keep_pers, gpes)

if __name__ == '__main__':
    # 参数解析
    parser = argparse.ArgumentParser(usage="newsflash svm usage:", description="help instruction")
    parser.add_argument("-nerfile", default="./newsflash_ner.txt", help="the input data path.")
    parser.add_argument("-personfile", default="./keywords_PER.txt", help="the input data path.") 
    parser.add_argument("-orgfile", default="./keywords_ORG.txt", help="the input data path.") 
    parser.add_argument("-gpefile", default="./keywords_GPE.txt", help="the input data path.") 
    
    args = parser.parse_args()
     
    main(args)
