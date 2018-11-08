# -*- coding: utf-8 -*-
from numpy import zeros


class Tuple(object):
    def __init__(self, _sentence, _e1, _e2, _before, _between, _after, config):
        self.sentence = _sentence

        self.e1 = _e1
        self.e2 = _e2

        self.confidence = 0

        # context的词序列和词的标签
        self.bef_tags = _before     # 实体1的上文词（list）
        self.bet_tags = _between    # 实体1与实体2的中间词（list）
        self.aft_tags = _after      # 实体2的下文词（list）

        # _before 等的格式 [("I", "ADV"),("am", "V")]
        self.bef_words = " ".join([token for token, _ in _before])      # 实体1的上文词（str）
        self.bet_words = " ".join([token for token, _ in _between])     # 实体1与实体2的中间词（str）
        self.aft_words = " ".join([token for token, _ in _after])       # 实体2的下文词（str）

        # vector是一定维度的用来表示词语义的向量
        self.bef_vector = None
        self.bet_vector = None
        self.aft_vector = None

        self.construct_vector(_between, config)
        return

    def __str__(self):
        return str(self.e1 + '\t' +
                   self.e2 + '\t' + self.bef_words + '\t' + self.bet_words + '\t' + self.aft_words).encode("utf8")

    def __hash__(self):
        return hash(self.e1) ^ hash(self.e2) ^ hash(self.bef_words) ^ \
               hash(self.bet_words) ^ hash(self.aft_words)

    def __eq__(self, other):
        return (self.e1 == other.e1 and self.e2 == other.e2 and
                self.bef_words == other.bef_words and
                self.bet_words == other.bet_words and
                self.aft_words == other.aft_words)

    def __cmp__(self, other):
        if other.confidence > self.confidence:
            return -1
        elif other.confidence < self.confidence:
            return 1
        else:
            return 0

    def construct_vector(self, between, config):
        """
        
        :param between: 
        :param config: 
        :return: 
        """
        reverb_pattern = config.reverb.extract_reverb_patterns_tagged_ptb(between)
        bet_words = reverb_pattern if len(reverb_pattern) > 0 else between

        bet_filtered = [token for token, tag in bet_words if
                        token.lower() not in config.stopwords and tag not in config.filter_pos]

        self.bet_vector = self.context2vector(bet_filtered, config)

        bef_no_tags = [t[0] for t in self.bef_tags if t[0].lower() not in config.stopwords]
        aft_no_tags = [t[0] for t in self.aft_tags if t[0].lower() not in config.stopwords]
        self.bef_vector = self.context2vector(bef_no_tags, config)
        self.aft_vector = self.context2vector(aft_no_tags, config)

        return

    @staticmethod
    def context2vector(tokens, config):
        """
        token列表的word2vec值的加权平均
        :param tokens: 
        :param config: 
        :return: 
        """
        vector = zeros(config.vec_dim)
        for token in tokens:
            try:
                vector += config.word2vec[token.strip()]        # 若词不在词向量中，则改词为全0向量
            except KeyError:
                continue

        return vector
