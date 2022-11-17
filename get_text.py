import re
import tqdm

# 从数据集中获取文本数据

def readFile(filename):
    # 读文件， filename是文件名，‘r’表示读，encoding='utf-8'表示以‘utf-8’打开文件
    # 返回结果是一个类似字符串，即整篇文章可以当一个str处理
    content = open(filename, 'r', encoding='utf-8').read()

    # 除去HTML标签
    re_br = re.compile('<br\s*?/?>')  # 处理换行
    re_h = re.compile('</?\w+[^>]*>')  # HTML标签
    content = re_br.sub(' ', content)  # 将br转换为空格
    content = re_h.sub('', content)  # 去掉HTML 标签

    # 切分，按照换行符‘\n’切分成一句一句的
    # 结果是一个list， 每一句里面都是一个str对象
    sentences = content.split('\n')

    # 遍历所有的句子，并打印编号和句子内容
    # for i in range(len(sentences)):
    # print(sentences[i])
    return sentences


"""
    读取训练或者验证文件
    :param filename: 训练集/验证集的文件名字
    :return:text,labels
    返回训练集的文本和标签
    其中文本是一个list, 标签是一个list(每个元素为int)
"""
def read_train_valid(filename):
    sentences = readFile(filename)  # 读取训练集
    labels = []
    text = []

    for i in tqdm.tqdm(range(len(sentences))):
        if i == 0:  # 第一行是表头，不用管, 从第二行开始读
            continue
        sentence = sentences[i]

        # 对每一个句子按照'\t'切开为标签和文本
        sentence = sentence.split('\t')

        if len(sentence) != 3:
            continue

        # 第二个元素为sentiment，转换成int型
        label = int(sentence[1])

        # 第三个元素为句子文本，分词并将每一个词变为str对象,最终得到一个列表
        data = sentence[2]

        # 将每一次读到的label添加到最终的labels列表后面
        labels.append(label)
        # 将每一次读到的分词后的文本添加到最终的text列表后面
        text.append(data)

    # 遍历所有的句子，并打印编号和句子内容
    """
    for i in range(len(text)):
        print("index=", i)
        print("sentence:", text[i])
        print("label:", labels[i])
        print("====")
    """

    print("read ", filename)
    return text, labels


"""
    读取测试文件
    :param filename: 测试集文件名字
    :return:text,test_ids
    返回测试集文本和对应的id编号
    其中文本是一个list, id就是文件中的id
"""
def read_test(filename):
    sentences = readFile(filename)  # 读取测试集
    test_ids = []
    text = []

    for i in range(len(sentences)):
        if i == 0:  # 第一行是表头，不用管, 从第二行开始读
            continue
        sentence = sentences[i]

        # 对每一个句子按照'\t'切开为标签和文本
        sentence = sentence.split('\t')
        if len(sentence) != 2:
            continue

        # 第一个元素为id
        test_id = sentence[0]

        # 第二个元素为句子文本，分词并将每一个词变为str对象,最终得到一个列表
        sentence = sentence[1]

        # 将每一次读到的label添加到最终的labels列表后面
        test_ids.append(test_id)
        # 将每一次读到的分词后的文本添加到最终的sentences_splited列表后面
        text.append(sentence)

    print("read ", filename)
    return text, test_ids
