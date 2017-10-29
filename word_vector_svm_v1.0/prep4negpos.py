#encoding=utf-8

###
# 预处理数字替换为NUMBER，为word2vec做准备
# fdh 2017/8/22
###


def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)

def process(filename, output):

    fopen = open(filename, 'r')
    fwrite = open(output, 'w')

    if fopen is None or fwrite is None:
        raise IOError('%s or %s cannt open!'%(filename, output))

    for sentence in fopen:
        words = sentence.split()
        for word in words:
            if word == '':
                continue
            if hasNumbers(word):
                word = 'NUMBER'
            fwrite.write("%s "%(word))
        fwrite.write('\n')

    fopen.close()
    fwrite.close()




process('newsflash_seg.neg', 'newsflash_seg_nonumber.neg')
process('newsflash_seg.pos', 'newsflash_seg_nonumber.pos')
