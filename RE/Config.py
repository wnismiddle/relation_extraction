#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re
import json
import codecs

from nltk.corpus import stopwords
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from RE.ReVerb import ReVerb
import nltk
nltk.download("stopwords")

class Seed(object):
    def __init__(self, _e1, _e2):
        self.e1 = _e1
        self.e2 = _e2

    def __hash__(self):
        return hash(self.e1) ^ hash(self.e2)

    def __eq__(self, other):
        return self.e1 == other.e1 and self.e2 == other.e2


class Relationship(object):
    def __init__(self, rel_config):
        self.name = rel_config["name"]
        self.e1_type = rel_config["e1_type"]
        self.e2_type = rel_config["e2_type"]
        self.positive_seed = set([Seed(seed[0], seed[1]) for seed in rel_config["positive_seed"]])
        self.negative_seed = set([Seed(e1, e2) for e1, e2 in rel_config["negative_seed"]])


class Config(object):
    def __init__(self, config_file):
        config = json.load(open(config_file, 'r', encoding='utf-8'))

        # 词性过滤和tag map
        self.filter_pos = config['filter_pos']
        # 正则表达式 提取
        self.regex_clean_simple = re.compile(config['re']['regex_clean_simple'], re.U)  # 找到所有的开始、结束标签，放入list    结果如['<ORG>', '</ORG>', '<LOC>', '</LOC>']
        self.regex_clean_linked = re.compile(config['re']['regex_clean_linked'], re.U)  # 找到所有的结束标签，放入list      结果如['</ORG>', '</LOC>']
        self.regex_simple = re.compile(config['re']['regex_simple'], re.U)  # 找到所有实体标签对，放入list       结果如['<ORG>Citibank</ORG>', '<LOC>Athens</LOC>']
        self.regex_linked = re.compile(config['re']['regex_linked'], re.U)  # 找到标签带url属性的实体标签对，放入list       结果如['<LOC url="www.baidu.com">B</LOC>']
        self.regex_entity_text_simple = re.compile(config['re']['regex_entity_text_simple'])    # 找到所有标签内的实体值，放入list       结果如['Citibank', 'Athens']
        self.regex_entity_text_linked = re.compile(config['re']['regex_entity_text_linked'])    # 找到标签带url属性的实体标签对中的实体值，放入list       结果如['B']
        self.regex_entity_type = re.compile(config['re']['regex_entity_type'])                  # 提取标签值     结果如['ORG', 'LOC']
        self.tags_regex = re.compile(config['re']['tags_regex'], re.U)              # 找到所有的开始、结束标签，放入list    结果如['<ORG>', '</ORG>', '<LOC>', '</LOC>']

        # 待抽取的关系描述，包含关系的两个实体类型

        self.relationship = Relationship(config["relationship"])    # 提取关系对，正例种子，负例种子

        # hyper-parameters
        self.wUpdt = config['hyper_parameters']['wUpdt']
        self.wUnk = config['hyper_parameters']['wUnk']
        self.wNeg = config['hyper_parameters']['wNeg']
        self.number_iterations = config['hyper_parameters']['number_iterations']
        self.min_pattern_support = config['hyper_parameters']['min_pattern_support']
        self.max_tokens_away = config['hyper_parameters']['max_tokens_away']
        self.min_tokens_away = config['hyper_parameters']['min_tokens_away']

        self.alpha = config['context_weight']['alpha']
        self.beta = config['context_weight']['beta']
        self.gamma = config['context_weight']['gamma']
        self.tag_type = config['hyper_parameters']['tag_type']
        self.context_window_size = config['hyper_parameters']['context_window_size']

        self.similarity = config['similarity']['similarity']
        self.confidence = config['similarity']['confidence']

        # word2vec 模型导入
        self.word2vec_path = config['hyper_parameters']['word2vec_path']
        self.word2vec = None
        self.vec_dim = None
        self.read_word2vec()

        # ReVerb词性抽取
        self.reverb = ReVerb()

        # stopwords
        self.stopwords = self.load_stopwords('en')

    def read_word2vec(self):
        print("loading word2vec model ...\n")
        self.word2vec = Word2Vec.load(self.word2vec_path)   # 读取.bin后缀的词向量文件
        # self.word2vec = KeyedVectors.load_word2vec_format(self.word2vec_path, binary=True)    # 针对无后缀的词向量文件读取有效
        '''
        self.word2vec = Word2Vec.load_word2vec_format(self.word2vec_path, binary=True)
        原代码报错：DeprecationWarning: Deprecated. Use gensim.models.KeyedVectors.load_word2vec_format instead.
        弃用警告：已弃用。 请改用gensim.models.KeyedVectors.load_word2vec_format。
        '''
        self.vec_dim = self.word2vec.layer1_size
        print(self.vec_dim, "dimensions")

    @staticmethod
    def load_stopwords(language='en'):
        if language == 'en':
            return stopwords.words('english')
        if language == 'cn':
            with codecs.open('../data/stopwords.zh.txt') as f:
                return [word for word in f.readlines()]

