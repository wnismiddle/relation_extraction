# -*- coding: utf-8 -*-

import re
import codecs
from gensim.models import word2vec

def get_re_result():
    with open('sentences.txt', 'r') as f:
        for line in f.readlines():
            print(re.sub("</?[A-Z]+>", "", line))    # regex_clean_simple
            # print re.findall("</?[A-Z]+>", line)  # regex_clean_simple
            # print re.findall("</[A-Z]+>|<[A-Z]+ url=[^>]+>", line)    # regex_clean_linked
            # print re.findall('<[A-Z]+>[^<]+</[A-Z]+>', line)  # regex_simple
            # print re.findall('<[A-Z]+ url=[^>]+>[^<]+</[A-Z]+>', line)  # regex_linked
            # print re.findall('<[A-Z]+>([^<]+)</[A-Z]+>', line)  # regex_entity_text_simple
            # print re.findall('<[A-Z]+ url=[^>]+>([^<]+)</[A-Z]+>', line)  # regex_entity_text_linked
            # print re.findall('<([A-Z]+)', line)  # regex_entity_type
            # print re.findall('</?[A-Z]+>', line)  # tags_regex

def generate_word2vec_pre():
    with codecs.open('sentences.txt', 'r', 'UTF-8') as f:
        with codecs.open('sentences_corpus.txt', 'w', 'UTF-8') as f_pro:
            for line in f:
                content = ''
                for idx in range(len(line)):
                    content += line[idx] + ' '
                f_pro.write(content)

def generate_word2vec():
    # sentences = word2vec.Text8Corpus(u"pre_word_vec.txt")
    # model = word2vec.Word2Vec(sentences, size=10)
    # model.save('zh')

    sentences = word2vec.Text8Corpus(u"pre_word_vec.txt")
    model = word2vec.Word2Vec(sentences, size=100)
    model.save('afp_apw_xin_embeddings.bin')

def test():
    # sentence = '上海钢联电子商务股份有限公司,type=ORG'
    sentence = 'BOB新浪科技讯3月16日晚间消息，<上海钢联电子商务股份有限公司,type=ORG>今日晚间公告称，拟购买北京知行锐景科技有限公司全部股权，以收购其旗下的中关村在线以及中关村商城网站资产EOE。'

    print(re.findall('<\w.*>', sentence))
    # print re.findall('(.*?)<\w+>.*', sentence)
    # print re.findall('(.*?)<\w+>.*', sentence)

if __name__ == '__main__':
    generate_word2vec()
    # get_re_result()
    # test()