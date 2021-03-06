---
layout: article
title: 吴恩达机器学习之初识机器学习
tags: Machine-Learning
article_header:
  type: overlay
typora-root-url: ..
---



<!--more-->

# 									初识机器学习



[TOC]





## 符号定义：

m ： 训练样本的数量。

x ： 输入变量（或者说特征）。

y ： 输出变量（目标变量）。

$(x,y)$ ： 表示一个训练样本。

$(x^{(i)},y^{(i)})$ ：表示第i个样本。

h：假设函数。

n：特征值的数量。

$x^{(i)}$ 第 $i$ 个训练样本的特征值。（特征值可能有多个）

$x^{(i)}_j$ 第 $i$ 个训练样本中第 $j$ 个特征量的值。




## 一、绪论：初识机器学习



### 	1、机器学习的定义

> ​	Arthur Samuel：在没有明确设置的情况下，使计算机具有学习能力的研究领域。（相对陈旧，不正式）

>   ​	==Tom Mitchell：计算机程序从经验E中学习，解决某一任务T，进行某一性能度量P，通过P测定在T上的表现因经验E而提高。==
>
>   

### 2、机器学习学习算法的分类

​			机器学习中学习算法最主要的两类是监督学习和无监督学习。



#### 		2.1 监督学习

​		==监督学习是指，我们给算法一个数据集，其中包含了正确答案，算法的目的就是给出更多的正确答案。==

##### 2.1.1 回归问题

回归问题的输出是连续型变量，是一种定量输出（预测明天的气温是多少度）。

![](/pic/MachineLearning/回归学习.png)

此图中的问题便是监督学习中的回归问题，通过给定英尺与价格的数据集对房价进行预测（给定英尺预测多少钱）。

##### 2.1.2 分类问题

分类问题的输出是离散型变量（如+1、-1），是一种定性输出（预测明天是）。

![](/pic/MachineLearning/分类学习.png)

此图中的问题便是监督学习中的分类问题，通过给定肿瘤尺寸与是否为恶性肿瘤相对应的数据集，对肿瘤进行预测（给定尺寸的肿瘤为恶性还是良性）



#### 2.2 无监督学习

​	==无监督学习是指数据集中没有标记（或都为一种标记），算法需要自行寻找数据中的结构。==

##### 聚类算法

​	将没有标记（或都为一种标记）的数据集分成不同的集合（簇）。

![](/pic/MachineLearning/聚类算法.png)



一个聚类算法典型的应用就是谷歌新闻：

![](/pic/MachineLearning/谷歌新闻（聚类应用）.png)

![](/pic/MachineLearning/谷歌新闻2.png)

谷歌新闻每天自动检索数以万计的新闻，并自动将他们聚合成一个个专题，每个专题由不同网站（大概率）的同类新闻组成。

##### 鸡尾酒会算法

![](/pic/MachineLearning/鸡尾酒会算法.png)

有这样一个场景，有一场两个人的鸡尾酒会，两个人在同时说话。同时有两个位置不同的麦克风，那么两个麦克风中所录的音频就会不同（距离不同音量不同），通过鸡尾酒会算法可以将通过两个音频，将两个人的声音分别提取出来，并去除其他杂音。
