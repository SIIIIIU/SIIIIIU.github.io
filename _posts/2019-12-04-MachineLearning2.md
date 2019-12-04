---
layout: article
title: 吴恩达机器学习之单变量线性回归
tags: Machine-Learning
article_header:
  type: overlay
typora-root-url: ..
---



<!--more-->

# 									单变量线性回归



[TOC]









## 二、单变量线性回归

只含有一个特征（输入变量）的线性回归问题。（两个变量之间存在一次方函数关系，就称它们之间存在线性关系）



### 1、监督学习算法的工作原理

![](/pic/MachineLearning/单变量线性回归模型.png)

首先，我们向学习算法提供训练集（比如我们的房价训练集），学习算法的任务是输出一个==函数h（假设函数）==，假设函数（hypothesis）的作用是把房子的大小作为输入变量把它作为x的值，而它会试着输出相应房子的预测y值，==h是一个引导从x得到y的函数==。接下来要决定怎么表示这个假设函数h。假设图中的假设函数表示为：

​                                                              $h_{\theta}(x) = \theta_0 + \theta_1x$ 

（函数$h_{\theta}(x)$有时会缩写为$h(x)$）

这就意味着我们要预测的 y 是一个关于 x 的线性函数。

==这种模型被称为线性回归模型。==

这个例子是一个一元线性回归，被称为==单变量线性回归模型==。



### 2、代价函数

![](/pic/MachineLearning/模型参数.png)

![](/pic/MachineLearning/模型参数_2.png)

![](/pic/MachineLearning/模型参数_3.png)

假设有一个数据集表示房屋的面积与房价（如图3所示），若模型是一个单变量线性回归模型，表示为：

​                                                             $h_{\theta}(x) = \theta_0 + \theta_1x$ 

那么我们要做的，就是选择合适的 $\theta_0$ 和 $\theta_1$ ，是模型与所给的函数尽可能的拟合，以保证以后预测数据的准确性。

线性回归问题便成了一个最小化的问题，求合适的 $\theta_0$ 和 $\theta_1$ ，使得

​                                                             $\frac{1}{2m}\sum\limits_{i=1}^{m}(h_{\theta}(x^{(i)})-y^{(i)})^2$ 

的值最小，这将是我的线性回归的整体目标函数。

按照惯例，我们要定义一个代价函数：

​                                               $ J(\theta_0,\theta_1) =\frac{1}{2m}\sum\limits_{i=1}^{m}(h_{\theta}(x^{(i)})-y^{(i)})^2$ 

求适当的 $\theta_0$ 和 $\theta_1$ ，使得代价函数的值最小。

代价函数也被称为平方误差函数。

之所以要求误差的平方和，是因为误差平方代价函数对于大多数问题，特别是回归问题，都是一个合理的选择。还有其他的代价函数也能很好的发挥作用，但平方误差代价函数可能是解决线性回归问题最常用的手段。



![](/pic/MachineLearning/代价函数.png)

假设使 $\theta_0=0$  来简化假设函数 $h_{\theta}(x)$ ，那么它的代价函数 $J(\theta_0,\theta_1)$ 如右边所示，是一个碗状。而若 $\theta_0!=0 $ ，那么它的代价函数 $J(\theta_0,\theta_1)$ 也是一个碗状，如下图所示：

![](/pic/MachineLearning/代价函数_2.png)

通常为了方便也使用等高线图来表示：

![](/pic/MachineLearning/代价函数_3.png)



### 3、梯度下降

梯度下降是一种可以将代价函数 $J$ 最小化的算法。它不仅可以对线性回归模型的代价函数进行最小化，还被广泛应用于机器学习的其他领域，对其他函数进行最小化。

以下便是梯度下降的应用场景：

![](/pic/MachineLearning/梯度下降.png)

有这样一个函数 $J(\theta_0,\theta_1)$ （泛指的函数，不一定非要是代价函数），想要求得函数 $J(\theta_0,\theta_1)$ 的最小值。

然后便是梯度下降的步骤，梯度下降以一个初始值为开始，一般将 $\theta_0 = 0 , \theta_1 = 0$ ，然后对 $\theta_0,\theta_1$ 来进行变化以减少函数 $J$ 的值，不断进行这一步直到收敛到了最小值或局部最小值。

下面表示梯度下降的过程：

![](/pic/MachineLearning/梯度下降_2.png)

![](/pic/MachineLearning/梯度下降_3.png)

如图所示，梯度下降有这样的一个特点，一开始选取不同的初始值，可能最后收敛于不同的局部最小值。



![](/pic/MachineLearning/梯度下降_4.png)

梯度下降的具体算法如图所示：

​                                                   $\theta_j:=\theta_j-\alpha\frac{\partial}{\partial\theta_j}J(\theta_0,\theta_1)  \quad(for \quad j=0 \quad and \quad j=1)$ 

其中 $:=$ 的意思就是平常编程语言中的等号，是一个赋值符号，而 = 就是日常中的判断符号。

 $\alpha$ 称为学习速率，用来控制以多大的幅度更新这个参数 $\theta_j$ （梯度下降的速度）， $\alpha $ 的值越大，梯度下降的速度越快。

其中这个更新是对$\theta_0$ 和 $\theta_1$ 两个参数都要更新，而且是要同时更新，如图中下方左边所示。右边的不是同时更新，在某些数据集或者模型中不一定错误，但已经称不上梯度下降算法了。   

下面直观的了解下偏导项 $\frac{\partial}{\partial\theta_1}J(\theta_0,\theta_1)$:

假设 $\theta_0 = 0$ ，那么代价函数 $J$ 的函数如下图所示： 

![](/pic/MachineLearning/梯度下降_5.png)

那么偏导项就变成了 $\frac{\partial}{\partial\theta_1}J(\theta_1)$ ，而偏导项其实就是函数在 $\theta_1$ 那一点的斜率，假设 $\theta_1$ 这一点在最低点的右边，那么导数的值是一个正数，也就是一个正斜率，那么根据公式 $\theta_1:=\theta_j-\alpha\frac{\partial}{\partial\theta_1}J(\theta_1)$ ，$\theta_1$ 会向左移动，正是最低点的方向。同理若是在最低点的左边，那么偏导项是一个负数，也就是一个负斜率，那么根据公式 $\theta_1$ 会向右移动，也正是最低点的方向。

![](/pic/MachineLearning/梯度下降_6.png)

而 $\alpha$ 的作用就是控制每次下降的幅度，若 $\alpha$ 数值过小，那么下降的速度欧就会很慢。而若 $\alpha$ 的数值过大，那么最坏的情况就是可能会无法收敛甚至发散。

![](/pic/MachineLearning/梯度下降_7.png)

而若到了最低点，因为最低点的导数为 0 ，所以 $\theta_1 = \theta_1-\alpha0 = 0$ ，也就是不会变化。 

![](/pic/MachineLearning/梯度下降_8.png)

随着梯度的下降，斜率不断减小，下降的速度也会不断减小。



下面将梯度下降应用到之前所学的线性回归模型中的代价函数中：

![](/pic/MachineLearning/梯度下降_9.png)

那么重点要求的就是偏导项，学过高数就可以轻松推出下图求导后的公式：

![](/pic/MachineLearning/梯度下降_10.png)



![](/pic/MachineLearning/梯度下降_11.png)

本节讲的梯度下降算法因为每一步都使用了所有的数据集，所以又称为 Batch梯度下降算法。









[^1]: 参考自https://blog.csdn.net/ariessurfer/article/details/41310525