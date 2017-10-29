#encoding=utf-8

###
# 预处理数字替换为NUMBER，为word2vec做准备
# fdh 2017/8/22
###

filename = 'newsflash_segments.txt'

output = 'newsflash_seg_nonumber.txt'

fopen = open(filename, 'r')
fwrite = open(output, 'w')

if fopen is None or fwrite is None:
    raise IOError('%s or %s cannt open!'%(filename, output))

def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)

for sentence in fopen:
    words = sentence.split()
    for word in words:
        if word == '':
            continue
        if hasNumbers(word):
            word = 'NUMBER'
        fwrite.write("%s "%(word))

fopen.close()
fwrite.close()



