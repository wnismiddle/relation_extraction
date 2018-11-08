# -*- coding: utf-8 -*-

import time
import sys

import pickle
import os


import codecs
from numpy import dot
from nltk.data import load
from gensim import matutils
from collections import defaultdict

from RE.Sentence import Sentence
from RE.Config import Config, Seed
from RE.Pattern import Pattern
from utils import get_logger
import nltk
# nltk.download('maxent_treebank_pos_tagger')
# nltk.download('punkt')

logger = get_logger('data/extract.log')

class AUTORE(object):
    def __init__(self, config_file):
        self.curr_iteration = 0
        self.patterns = list()
        self.processed_tuples = list()
        self.candidate_tuples = defaultdict(list)
        self.config = Config(config_file)

    def extract_tuples(self, sentence_file):
        try:
            # 先查找是否有提取出的用Pickle备份好的Tuple，如果有则从本地导入tuples，调试时使用能减少时间。
            f = open("../data/processed_tuples.pkl", 'rb')
            # f = open("data/processed_tuples.pkl", 'rb')
            self.processed_tuples = pickle.load(f, encoding='utf-8')
            logger.info("load tuples finished with %d tuples" % (len(self.processed_tuples)))
            f.close()

        except IOError:
            # 如果没有将Tuple保存好，则从给定的文件中按照给定的方法提取。
            with open(sentence_file, 'r', encoding='utf-8') as f:
                begin = time.time()
                count = 0
                tagger = load('taggers/maxent_treebank_pos_tagger/english.pickle')
                for line in f.readlines():
                    count += 1
                    if count % 1000 == 0:
                        sys.stdout.write('.')

                    sentence = Sentence(sentence=line.strip(), config=self.config, pos_tagger=tagger)

                    for tup in sentence.tuples:
                        self.processed_tuples.append(tup)
            logger.info("extract tuples finished with %d tuples " % (len(self.processed_tuples)))

            with open('data/processed_tuples.pkl', 'wb') as f:
                pickle.dump(self.processed_tuples, f)
            # f.close()
            print("dump tuples successfully")
            print(time.time() - begin)
        return

    def extract_matched_tuples(self):
        """
        在候选Tuple中找到与种子(POSITIVE)匹配的Tuple，并统计每一个Tuple出现的次数
        :return: matched_tuples : 是list of tuples counts : 是[e1,e2]为key的字典，value是该实体对出现的次数。
        """
        matched_tuples = list()
        counts = dict()
        for tup in self.processed_tuples:
            for s in self.config.relationship.positive_seed:
                if tup.e1 == s.e1 and tup.e2 == s.e2:
                    matched_tuples.append(tup)
                    try:
                        counts[(tup.e1, tup.e2)] += 1
                    except KeyError:
                        counts[(tup.e1, tup.e2)] = 1
        logger.info("extract matched tuples finished with %d tuples" % (len(matched_tuples)))
        # print "extract matched tuples finished with %d tuples" % (len(matched_tuples))
        return matched_tuples, counts

    def similarity_tuple_tuple(self, t1, t2):
        """
        计算两个Tuple之间的相似度，主要e1之前的部分，e1和e2之间的部分，e2之后的部分，三部分相似度的加权和
        :param t1: 
        :param t2: 
        :return: 
        """
        (bef, bet, aft) = (0, 0, 0)
        if t1.bef_vector is not None and t2.bef_vector is not None:
            bef = dot(matutils.unitvec(t1.bef_vector), matutils.unitvec(t2.bef_vector))
        if t1.bet_vector is not None and t2.bet_vector is not None:
            bet = dot(matutils.unitvec(t1.bet_vector), matutils.unitvec(t2.bet_vector))
        if t1.aft_vector is not None and t2.aft_vector is not None:
            aft = dot(matutils.unitvec(t1.aft_vector), matutils.unitvec(t2.aft_vector))

        return self.config.alpha * bef + self.config.beta * bet + self.config.gamma * aft

    def similarity_tuple_pattern(self, tup, pattern):
        """
        比较一个Tuple和一个pattern的相似度，主要是和pattern中每个tuple进行比较
        :param tup: 
        :param pattern: 
        :return: 
        """
        good = 0
        bad = 0
        max_similarity = 0

        for p in list(pattern.tuples):
            score = self.similarity_tuple_tuple(tup, p)
            max_similarity = score if score > max_similarity else max_similarity
            if score >= self.config.similarity:
                good += 1
            else:
                bad += 1
        if good >= bad:
            return True, max_similarity
        else:
            return False, 0.0

    def extract_patterns(self, matched_tuples):
        """
        利用查找到的与种子匹配的Tuple来提取一些Pattern。
        :param matched_tuples: 
        :return: 
        """
        if len(self.patterns) == 0:
            c1 = Pattern(matched_tuples[0])
            self.patterns.append(c1)

        count = 0
        for t in matched_tuples:
            count += 1
            if count % 1000 == 0:
                sys.stdout.write(".")
                sys.stdout.flush()
            max_similarity = 0
            max_similarity_cluster_index = 0

            for i in range(0, len(self.patterns), 1):
                pattern = self.patterns[i]
                accept, score = self.similarity_tuple_pattern(t, pattern)
                if accept is True and score > max_similarity:
                    max_similarity = score
                    max_similarity_cluster_index = i

            if max_similarity < self.config.similarity:
                c = Pattern(t)
                self.patterns.append(c)
            else:
                self.patterns[max_similarity_cluster_index].add_tuple(t)

        self.patterns = [p for p in self.patterns if len(p.tuples) > self.config.min_pattern_support]
        logger.info("extract patterns finished with %d patterns and pattern distribution %s" % (
            len(self.patterns), str([len(patt.tuples) for patt in self.patterns])))
        # print "extract patterns finished with %d patterns and pattern distribution %s" % (
        #     len(self.patterns), str([len(patt.tuples) for patt in self.patterns]))
        return 0

    def extract_candidate_tuples(self):
        """
        挑选符合关系的tuple
        :return: 
        """
        for t in self.processed_tuples:
            sim_best = 0
            # 第一步是用所有的pattern对该Tuple打分，并记录最高分和对应的pattern
            for pattern in self.patterns:
                accept, score = self.similarity_tuple_pattern(t, pattern)
                if accept is True:
                    pattern.update_selectivity(t, self.config)
                    if score > sim_best:
                        sim_best = score
                        pattern_best = pattern
            # 如果最高分高于设定的阈值，则保存对应的pattern
            if sim_best >= self.config.similarity:
                patterns = self.candidate_tuples[t]
                if patterns is not None:
                    if pattern_best not in [x[0] for x in patterns]:
                        self.candidate_tuples[t].append((pattern_best, sim_best))
                else:
                    self.candidate_tuples[t].append((pattern_best, sim_best))
        logger.info("extract candidate tuples finished with %d" % (len(self.candidate_tuples)))
        # print "extract candidate tuples finished with %d" % (len(self.candidate_tuples))
        return 0

    def update_pattern_confidence(self):
        for p in self.patterns:
            p.update_confidence(self.config)
        return 0

    def update_candidate_tuples(self):
        """
        根据更新的pattern得分来更新候选Tuple的得分
        :return: 
        """
        for t in self.candidate_tuples.keys():
            confidence = 1
            for p in self.candidate_tuples.get(t):
                confidence *= 1 - (p[0].confidence * p[1])
            t.confidence = 1 - confidence

        return 0

    def update_seeds(self):
        for t in self.candidate_tuples.keys():
            if t.confidence >= self.config.confidence:
                seed = Seed(t.e1, t.e2)
                self.config.relationship.positive_seed.add(seed)

    def bootstrap(self, sentence_file):
        """
        进行半自动bootstrap的迭代过程
        :return: 
        """
        self.extract_tuples(sentence_file)

        # curr_iteration 当前轮次   number_iterations 迭代多少轮
        while self.curr_iteration <= self.config.number_iterations:
            logger.info("-------------正在进行第 %d 轮抽取-----------" % (self.curr_iteration+1))
            matched_tuples, counts = self.extract_matched_tuples()  # 找到与原始积极种子匹配的三元组，统计数量

            # 如果匹配三元组数量为0，程序结束
            if len(matched_tuples) == 0:
                print("No seed matches found")
                return False

            self.extract_patterns(matched_tuples)
            self.extract_candidate_tuples()
            self.update_pattern_confidence()
            self.update_candidate_tuples()
            self.curr_iteration += 1

        # 获取候选三元组
        tuples = []
        with open('result.txt', 'w', encoding='utf-8') as f:
            for t in self.candidate_tuples:
                # print(type(t.e1))
                # print(type(t.e2))
                # print(type(t.bet_words))
                # print(t.e1, t.bet_words, t.e2)
                tuple = []
                tuple.append(t.e1)
                tuple.append(t.bet_words)
                tuple.append(t.e2)
                tuples.append(tuple)
                f.write(str(tuple)+'\n')
        logger.info(str(tuples)+'\n')
        return tuples


def main():
    re = AUTORE(config_file='parameter.json')
    tuples = re.bootstrap(sentence_file='sentences.txt')
    pass


if __name__ == "__main__":
    main()
