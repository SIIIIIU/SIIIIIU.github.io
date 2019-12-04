---
layout: article
title: 吴恩达机器学习之线性代数回顾
tags: Machine-Learning
article_header:
  type: overlay
typora-root-url: ..

---



<!--more-->

# 线性代数回顾

[TOC]



## 三、线性代数回顾

### 1、矩阵和向量

![](/pic/MachineLearning/矩阵定义.png)

矩阵是指由数字组成的矩形阵列，并写在方括号内。

矩阵可以说是二维数组的另一种说法。

矩阵的维数写作矩阵的行数乘以列数。（左边便是一个 $4\times 2$ 的矩阵）

也有这样一种写法：

将左边的矩阵写作 $R^{4\times2}$ ，将该矩阵称为集合 $R^{4\times2}$ 的元素，也就是说， $R^{4\times2}$ 代表所有$4\times2$ 的矩阵的集合。		



![](/pic/MachineLearning/向量定义.png)

向量是指列数为1的矩阵。

如图 y 表示一个向量，向量的维数由向量的行数决定，y 便是一个4维向量。

$R^4$ 是指一个4维向量的集合。

$y_1$ 表示向量 y 中的第一个元素。

一般有下标从1开始和从0开始两个写法，机器学习的应用中一般用从0开始，本课程默认从1开始。

按惯例来说，用大写字母表示矩阵，用小写字母表示向量。



### 2、加法和标量乘法

![](/pic/MachineLearning/矩阵的加法.png)

两矩阵相加就是两矩阵对应位置的元素相加，因此，只有维度相同的两矩阵可以相加，维度不同的矩阵相加没有意义。

![](/pic/MachineLearning/矩阵标量乘法.png)

标量乘法中标量是一个复杂的结构，可能是一个数或者实数。标量乘法就是用这个标量乘矩阵中的每个元素。



### 3、矩阵向量乘法

矩阵和向量相乘：

![](/pic/MachineLearning/矩阵和向量相乘.png)

![](/pic/MachineLearning/矩阵向量相乘的应用.png)

在实际应用中，可以如上图这样使用。

有一组数据集（关于房价的）和一个假设函数 $h_{\theta}(x)$ ，那么计算预测值可以如图中这样列为一个矩阵和一个向量相乘。

### 4、矩阵乘法

![](/pic/MachineLearning/矩阵相乘.png)

假设有一个数据集，三个假设函数，那么计算预测值可以如下图列矩阵：

![](/pic/MachineLearning/矩阵乘法_2.png)

### 5、矩阵乘法的特性

（1）、矩阵乘法不满足交换律：

​                                            $A \times B \ne B \times A$ 

（2）、矩阵乘法满足结合律：

​                                           $A \times B \times C = (A \times B) \times C = A \times (B \times C)$ 

（3）、矩阵乘以单位矩阵等于原矩阵：

​                                          $A \times I = I \times A = A $      （ $I$ 为单位矩阵）

### 6、逆和转置

（1）、矩阵的逆：

如果 $A$ 是一个 $m \times m$ 的矩阵，如果 $A$ 有逆矩阵，那么

​                                         $AA^{-1} = A^{-1}A = I $ 

（2）、矩阵的转置运算：

​                                        $A = \begin{bmatrix}1&2&0\\3&5&9\end{bmatrix} \quad \quad \quad A^T = \begin{bmatrix}1&3\\2&5\\0&9\end{bmatrix}$     

定义：

假设 $A$ 是一个 $m \times n$ 的矩阵，并设矩阵 $B$ 等于 $A$ 的转置，那么 $B$ 是一个 $n \times m$ 的矩阵，并且 $B_{ij} = A_{ji}$ 。