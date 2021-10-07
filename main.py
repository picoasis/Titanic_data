#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 17:15:51 2021

@author: picoasis
"""

'''
预测测试集中乘客的生还可能性，并给出准确率
'''
import pandas as pd
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

#探索数据--训练数据
print(train_data.info())#基本信息
print('-'*30)
print(train_data.describe())#统计信息
print('-'*30)
print(train_data.describe(include=['O']))#字符串数据信息
print('-'*30)
print(train_data.head())
print('-'*30)
print(train_data.tail())

#探索数据--测试数据
print(test_data.info())#基本信息
print('-'*30)
print(test_data.describe())#统计信息
print('-'*30)
print(test_data.describe(include=['O']))#字符串数据信息
print('-'*30)
print(test_data.head())
print('-'*30)
print(test_data.tail())


#数据处理
#空数据
# train数据集891：age 714，carbin 204，embarked 889  有null
# test数据集418： age 332，carbin 91， Fare 417，有null

# 数值型空数据，可以用（平均值）补齐 ：train_age，test_age, test_fare
train_data['Age'].fillna(train_data['Age'].mean(),inplace=True)

test_data['Age'].fillna(test_data['Age'].mean(),inplace=True)
test_data['Fare'].fillna(test_data['Fare'].mean(),inplace=True)

# 字符串类型空数据，
# carbin，缺失超过3/4，无法补全
# train_embark丢失了2个数据，可以尝试补齐
    #观察embark数据特征
print(train_data['Embarked'].value_counts())
#发现共3个港口，S，C，Q，S的最多，可以用S补齐空缺数据
train_data['Embarked'].fillna('S',inplace = True)

#特征选择
#丢弃没用的字段：PassengerID，Cabin，Name，TicketNumber（杂乱无章无规律）
features = ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']
train_features = train_data[features]
test_features = test_data[features]
train_labels = train_data['Survived']


#将字符串数据转换为数值数据 DictVectorizer类，将符号转成数字0/1表示
from sklearn.feature_extraction import DictVectorizer
dvec = DictVectorizer(sparse=False) #是否生成稀疏矩阵
train_features = dvec.fit_transform(train_features.to_dict(orient = 'records'))
test_features = dvec.fit_transform(test_features.to_dict(orient = 'records'))
#https://www.jb51.net/article/141481.htm   ==》 pandas.DataFrame.to_dict
#orient = 'record' 整体构成一个列表，内层是将原始数据的每行提取出来形成字典
print(dvec.feature_names_)
#['Age', 'Embarked=C', 'Embarked=Q', 'Embarked=S', 'Fare', 'Parch', 'Pclass', 'Sex=female', 'Sex=male', 'SibSp']
#原本是一列的 Embarked，变成了“Embarked=C”“Embarked=Q”“Embarked=S”三列。
#Sex 列变成了“Sex=female”“Sex=male”两列。
#train_features是一个ndarray，可以通过shape，dtype查看其大小类型，无法使用DataFrame的info describe等属性
print('train_features.shape:',train_features.shape,'train_features.dtype:',train_features.dtype)

#决策树模型
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion='entropy')
dtc.fit(train_features,train_labels)
#进行预测
pred_labels = dtc.predict(test_features)
#得到准确率自带的score函数
acc_decision_tree = round(dtc.score(train_features,train_labels),6)
print(u'score准确率为 %.4lf' % acc_decision_tree)

#0.9820 准确率接近100%，并不准确，因为用的训练数据做的准确率判定
#有什么办法，来统计决策树分类起的准确率呢？
# 方法1:K折交叉验证 :每次选取 K 分之一的数据作为验证，其余作为训练。轮流 K 次，取平均值。
import numpy as np
from sklearn.model_selection import cross_val_score
acc_dtc_cross_val_score = np.mean(cross_val_score(dtc, train_features,train_labels,cv=10))
print(u'cross_val_score 准确率为 %.4lf' % acc_dtc_cross_val_score)


#决策树可视化——Graphviz可视化工具 提前安装
from sklearn.tree import export_graphviz
import graphviz

#dot_data = export_graphviz(dtc)
#graph = graphviz.Source(dot_data)
# 生成 Source.gv.pdf 文件，并打开graph.view()
# 导出 titanic.dot 文件
with open("tree.dot", 'w') as f:
    f = export_graphviz(dtc, out_file=f)
#生成pdf：在命令行输入 dot -Tpdf tree.dot -o filename.pdf
#生成png：在命令行输入 dot -Tpng tree.dot -o filename.png


