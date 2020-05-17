#coding:utf-8
# __author__ = 'HubertLiu'


#load data
def read_corpus(file):
    with open(file,'r',encoding='utf8',errors='ignore') as f:
        list = []
        lines = f.readlines()
        for i in lines:
            list.append(i)
    return list

questions = read_corpus('./Q.txt')
answers = read_corpus('./A.txt')

print('Example:')
print('Question',questions[0])
print('Answer',answers[0])

import re
import jieba
# from sklearn.feature_extraction.text import CountVectorizer
from feature_extractors import bow_extractor, tfidf_extractor
import gensim

def filter_out_category(input):
    new_input = re.sub('[\u4e00-\u9fa5]{2,5}\\/','',input)
    return new_input

def filter_out_punctuation(input):
    new_input = re.sub('([a-zA-Z0-9])','',input)
    new_input = ''.join(e for e in new_input if e.isalnum())
    return new_input

def word_segmentation(input):
    new_input = ','.join(jieba.cut(input))
    return new_input

def preprocess_text(data):
    new_data = []
    for q in data:
        q = filter_out_category(q)
        q = filter_out_punctuation(q)
        q = word_segmentation(q)
        new_data.append(q)
    return new_data

qlist = preprocess_text(questions)   # 更新后的
print('questions after preprocess',qlist[0:3])



# # 词袋模型特征
# bow_vectorizer, bow_train_features = bow_extractor(norm_train_corpus)
# bow_test_features = bow_vectorizer.transform(norm_test_corpus)

# # tfidf 特征
# tfidf_vectorizer, tfidf_train_features = tfidf_extractor(norm_train_corpus)
# tfidf_test_features = tfidf_vectorizer.transform(norm_test_corpus)


# # 词袋模型特征
def conver2BOW(data):
    new_data = []
    for q in data:
        new_data.append(q)
    bow_vectorizer, bow_X = bow_extractor(new_data)
    return bow_vectorizer, bow_X
bow_vectorizer, bow_X = conver2BOW(qlist)

# print('BOW model')
# print('vectorizer',bow_vectorizer.get_feature_names())
# print('vector of text',bow_X[0:3].toarray())


# # tfidf 特征
def conver2tfidf(data):
    new_data = []
    for q in data:
        new_data.append(q)
    tfidf_vectorizer, tfidf_X = tfidf_extractor(new_data)
    return tfidf_vectorizer, tfidf_X
tfidf_vectorizer, tfidf_X = conver2tfidf(qlist)

# print('TFIDF model')
# print('vectorizer',tfidf_vectorizer.get_feature_names())
# print('vector of text',tfidf_X[0:3].toarray())


import numpy as np
def idx_for_largest_cosine_sim(input, questions):
    list = []
    input = (input.toarray())[0]
    for question in questions:
        question = question.toarray()
        num = float(np.matmul(question, input))
        denom = np.linalg.norm(question) * np.linalg.norm(input)

        if denom ==0:
            cos = 0.0
        else:
            cos = num / denom

        list.append(cos)

    best_idx = list.index(max(list))
    return best_idx

def answer_bow(input):
    input = filter_out_punctuation(input)
    input = word_segmentation(input)
    bow = bow_vectorizer.transform([input])
    best_idx = idx_for_largest_cosine_sim(bow, bow_X)
    return answers[best_idx]

def answer_tfidf(input):
    input = filter_out_punctuation(input)
    input = word_segmentation(input)
    bow = tfidf_vectorizer.transform([input])
    best_idx = idx_for_largest_cosine_sim(bow, tfidf_X)
    return answers[best_idx]

print('bow model',answer_bow("煤的元素分析成分有哪些"))
print('tfidf model',answer_tfidf("煤的元素分析成分有哪些"))
