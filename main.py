import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score

import get_text
import text_to_vector
import re
import jieba
"""
@author: zyf
"""
min_freq = 5
train_file = './movie_data/training.tsv'   # 训练集
test_file = './movie_data/test.tsv'     # 测试集
valid_file = './movie_data/validation.tsv'  # 验证集
demo_train = './movie_data/demo_train.txt'
demo_valid = './movie_data/demo_validation.txt'
demo_test = './movie_data/demo_test.txt'


def split_text(sentences):
    """
    将原始文本分词
    中文jieba分词，英文直接用空格切分句子即可
    """
    for i in range(len(sentences)):
        sentences[i] = sentences[i].split(' ')

    print("split")
    return sentences


def vectorizer(train_data, valid_data, test_data):
    """
        将原始文本转化为向量
        方法:
        1. one hot 方法, 2. tf-idf方法，3. 词向量或者其他方法
        可用的包：
        CountVectorizer, TfidfVectorizer可以实现one-hot和tfd-idf编码
        使用词向量也可

        返回分别对应train, valid, test的向量化结果，每一个结果都是一个二维的list, 每一个元素都是int or float
        :param text :  一个list, 每个元素是一句话
        :return: 3个二维list
    """
    all_sentence = train_data + valid_data + test_data
    train_length = len(train_data)
    valid_length = len(valid_data)

    onehot_vector = text_to_vector.text_to_vector(all_sentence)
    train_vec = onehot_vector[:train_length]
    valid_vec = onehot_vector[train_length:valid_length+train_length]
    test_vec = onehot_vector[valid_length+train_length:]

    """
    for i in range(len(test_data)):

        print("index=", i)
        print("source_sentence", test_data[i])
        print("onvector", test_vec[i])
        print("===")
    """
    print("vec end")

    return train_vec, valid_vec, test_vec


def train_valid(train_vec, train_label, valid_vec, valid_label):
    """
    开始训练模型并使用验证集验证效果
    方法：
    1. 朴素贝叶斯: 可用包 BernoulliNB,
    2. LogisticRegression 可用包 sklearn中的LogisticsRegression
    """
    # 定义模型
    model = LogisticRegression()
    # 开始训练
    model.fit(train_vec, train_label)
    # 训练完毕，预测一下验证集
    prediction = model.predict(valid_vec)

    # 评估一下真正的结果和预测的结果的差异
    # average参数有五种：(None, ‘micro’, ‘macro’, ‘weighted’, ‘samples’)
    acc_score = accuracy_score(valid_label, prediction)
    rec_score = recall_score(valid_label, prediction)
    f_score = f1_score(valid_label, prediction)
    pre_score = precision_score(valid_label, prediction, average='macro')
    # 打印预测的结果和得分
    print("prediction:", prediction)
    print("accuracy score: {:.4f}".format(acc_score))
    print("recall score: {:.4f}".format(rec_score))
    print("f1 score: {:.4f}".format(f_score))
    print("precision score: {:.4f}".format(pre_score))
    return model


def predict(model, test_data):
    """
    使用训练好的模型预测测试数据
    返回预测的标签，为list
    """
    result = model.predict(test_data)
    return result


def run_step():
    # 读文件
    
    train_data, train_label = get_text.read_train_valid(train_file)
    valid_data, valid_label = get_text.read_train_valid(valid_file)
    test_data, test_ids = get_text.read_test(test_file)
    '''
    train_data, train_label = get_text.read_train_valid(demo_train)
    valid_data, valid_label = get_text.read_train_valid(demo_valid)
    test_data, test_ids = get_text.read_test(demo_test)
    '''
    # 将原始文本分词：
    train_data = split_text(train_data)
    valid_data = split_text(valid_data)
    test_data = split_text(test_data)

    # 将分词后的文本变成向量：
    train_vec, valid_vec, test_vec = vectorizer(train_data, valid_data, test_data)

    # 开始训练模型，使用train_data, train_label训练，valid_data, valid_label验证
    model = train_valid(train_vec, train_label, valid_vec, valid_label)

    # 使用训练好的模型训练，得到预测结果
    result = predict(model, test_vec)

    # 打印预测结果
    print("test result:", result)

    # 写入文件：
    with open("submit.txt", "w", encoding='utf-8') as f:
        f.write("id\tlabels\n")
        for i in range(len(result)):
            idx = str(test_ids[i])
            lable = str(result[i])
            f.write(idx + '\t' + lable + '\n')


if __name__ == '__main__':
    run_step()

