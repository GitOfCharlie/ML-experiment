import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score

def read_files(mail_dir_path):
    file_paths = []
    for one_dir in os.listdir(mail_dir_path):
        if one_dir:
            sub_path = os.path.join(mail_dir_path, one_dir, 'ham')
            file_paths += [os.path.join(sub_path, file) for file in os.listdir(sub_path)]
    ham_mail_count = len(file_paths)
    for one_dir in os.listdir(mail_dir_path):
        if one_dir:
            sub_path = os.path.join(mail_dir_path, one_dir, 'spam')
            file_paths += [os.path.join(sub_path, file) for file in os.listdir(sub_path)]
    spam_mail_count = len(file_paths) - ham_mail_count

    mail_matrix = np.ndarray((len(file_paths)), dtype=object)
    ptr = 0
    for file in file_paths:
        with open(file, 'r', errors="ignore") as f:
            next(f)
            content = f.read().replace('\n', ' ')
            mail_matrix[ptr] = content
            ptr += 1
    # 标签：正例ham为1，反例spam为0，与混淆矩阵位置相一致
    labels = np.append(np.ones(ham_mail_count), np.zeros(spam_mail_count))
    return mail_matrix, labels

if __name__ == '__main__':
    mail_dir_path = 'resources' #  当前目录resources文件夹，由于压缩包过大因此不打包给助教了O(∩_∩)O
    mail_matrix, mail_labels = read_files(mail_dir_path)
    # 分割训练测试集
    mail_train, mail_test, mail_train_label, mail_test_label \
        = train_test_split(mail_matrix, mail_labels, test_size=0.2, random_state=1)
    # TF-IDF将文本特征提取为向量
    count_vec = CountVectorizer(stop_words='english', max_df=0.6, decode_error='ignore', binary=False)
    count_train = count_vec.fit_transform(mail_train)
    tfidfTransformer = TfidfTransformer()
    tfidf_train = tfidfTransformer.fit_transform(count_train)
    # 训练模型
    svn_model = LinearSVC()
    svn_model.fit(tfidf_train, mail_train_label)

    # 对测试数据也进行TF-IDF预处理
    count_vec_test = CountVectorizer(vocabulary=count_vec.vocabulary_, stop_words='english', max_df=0.6, decode_error='ignore', binary=False)
    count_test = count_vec_test.fit_transform(mail_test)
    tfidf_test = tfidfTransformer.fit_transform(count_test)

    predict_label = svn_model.predict(tfidf_test)
    '''
    混淆矩阵：   预测正例(1)    预测反例(0)
    真实正例(1)     TP            FN
    真实反例(0)     FP            TN
    设矩阵的labels字段为[1, 0]定义顺序
    '''
    confusion_m = pd.DataFrame(confusion_matrix(mail_test_label, predict_label, labels=[1, 0]),
                               index=['actual_ham', 'actual_spam'], columns=['predicted_ham', 'predicted_spam'])
    print(confusion_m)
    print('精确率Precision=TP/(TP+FP)，预测为正类的实例中真正预测正确的比例：', precision_score(mail_test_label, predict_label, pos_label=1))
    print('准确率Accuracy=(TP+TN)/(TP+FN+FP+TN)，总体预测结果正确率：', accuracy_score(mail_test_label, predict_label))
    print('召回率Recall=TP/(TP+FN)，指对于所有正类实例的预测正确率：', recall_score(mail_test_label, predict_label, pos_label=1))