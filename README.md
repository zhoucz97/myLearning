> 个人学习历程，持续更新中......

# 1. 刷题

- [labuladong的算法小抄](https://github.com/labuladong/fucking-algorithm)。100多道题，分门别类，先讲某个算法，再讲该算法下对应的力扣题目，Github标星已经80多k了。
- [liweiwei1419](https://github.com/liweiwei1419)。威威哥，长期活跃在力扣社区，为力扣官方贡献了大量的优质题解。[liweiwei个人博客](https://liweiwei1419.gitee.io/leetcode-algo/about/)，[liweiwei力扣主页](https://leetcode-cn.com/u/liweiwei1419/)
- [个人笔记之LeetCode题解与算法（语雀）](https://www.yuque.com/zhcz/leetcode)
- [知乎问题—如何理解和掌握KMP算法](https://www.zhihu.com/question/21923021/answer/281346746)

# 2. 机器学习

- CS229

  - [CS229 课程讲义中文翻译](https://github.com/Kivy-CN/Stanford-CS-229-CN)
  - [个人整理的229资料](https://github.com/zhoucz97/CS229)，主要是08年和18年吴恩达的。
- 《统计学习方法》

  - [手写实现李航《统计学习方法》书中全部算法](https://github.com/Dod-o/Statistical-Learning-Method_Code)，非常厉害且详细，每一章都有博客讲解和对应代码，且代码力求每一行都有注释，重要部分注明公式来源。

- 《神经网络与深度学习》
  - https://nndl.github.io/
- 《动手学深度学习》
  - [Tensorflow2版本](https://trickygo.github.io/Dive-into-DL-TensorFlow2.0/#/)
  - [pytorch版本](https://tangshusen.me/Dive-into-DL-PyTorch/#/)
- 深度学习圣经—**花书**
  - 英文版原书：https://www.deeplearningbook.org/
  - 中文版翻译：https://github.com/exacity/deeplearningbook-chinese
  - 花书数学推导、原理剖析与源码级别代码实现：https://github.com/MingchaoZhu/DeepLearning
  - 深度之眼花书啃书指导：https://www.bilibili.com/video/BV1kE4119726
  - 花书各章笔记：https://zhuanlan.zhihu.com/p/38431213
- [超容易理解的CRF讲解](https://zhuanlan.zhihu.com/p/44042528)
- [Bert使用的激活函数：gelu---高斯误差线性单元](https://blog.csdn.net/eunicechen/article/details/84774047?utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromMachineLearnPai2%7Edefault-3.control&dist_request_id=1332024.6353.16189739587076911&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromMachineLearnPai2%7Edefault-3.control)
- [GELU 激活函数](https://blog.csdn.net/liruihongbob/article/details/86510622)
- [详解准确率、精确率、召回率、F1值含义](https://blog.csdn.net/weixin_41753316/article/details/109382129)

- 过拟合问题

  - 解决方式：加大数据量、降低模型复杂度、Dropout、正则化、早停策略、warmup学习率、模型集成
  - [Early Stopping - 简书 (jianshu.com)](https://www.jianshu.com/p/9ab695d91459)
  - [深度学习技巧之Early Stopping（早停法） | 数据学习者官方网站(Datalearner)](https://www.datalearner.com/blog/1051537860479157)
  - [生动形象告诉你神经网络的Dropout为何有效_qiuzitao的博客-CSDN博客_dropout为什么有效](https://blog.csdn.net/qiuzitao/article/details/105370129)
  - [为什么正则化能减少模型过拟合程度_ybdesire的博客-CSDN博客](https://blog.csdn.net/ybdesire/article/details/79068603)
  - [L1正则化与L2正则化的区别_ybdesire的博客-CSDN博客_l1正则化和l2正则化的区别](https://blog.csdn.net/ybdesire/article/details/84946128)

- 数据归一化

  - [如何理解归一化（normalization）? - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/424518359)

  - [机器学习笔记：为什么要对数据进行归一化处理？ - 不说话的汤姆猫 - 博客园 (cnblogs.com)](https://www.cnblogs.com/silence-tommy/p/7113498.html)  

  - 不同评价指标会具有不同的量纲和量纲单位，会影响数据分析的结果；也会让模型寻找最优解的过程变得不够平滑。

  - 归一化可以消除奇异样本导致的不良影响，也能够加快梯度下降求最优解的速度，使模型训练过程更加平滑。

  - 归一化方法

    - 线性比例变换法 $x' = \frac{x}{max(x)}$

    - 极差变换法，即最大最小标准化 

      $x' = \frac{x-min(x)}{max(x) - min(x)}$

      - 适用于数值比较集中的情况，受max和min的影响较大；

    - z-score标准化  

      $x' = \frac{x- \mu}{\sigma}$

      - 将数据归一化为标准正态分布
      - 适用于需要使用距离来度量相似性的时候，或者使用PCA降维的时候。
    
  - 需要归一化的算法和不需要归一化的算法

    - 需要：基于距离计算的模型KNN；梯度下降求解的模型线性回归神经网络等。
    - 不需要：决策树，随机森林等。
  
- 为什么PCA不被推荐用来避免过拟合？

  - 因为PCA是无监督的降维方法。
  - [为什么PCA不被推荐用来避免过拟合_嘀嗒嘀嘀嗒嘀的博客-CSDN博客](https://blog.csdn.net/ACBattle/article/details/80011808)
  - [(5 封私信) 为什么PCA不被推荐用来避免过拟合？ - 知乎 (zhihu.com)](https://www.zhihu.com/question/47121788)

- SVM

  - [SVM算法 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/29862011)
  - [机器学习算法（一）SVM_yaoyz105的博客-CSDN博客_svm](https://blog.csdn.net/qq_31347869/article/details/88071930)

# 3. 深度学习

- CNN, RNN, LSTM，GRU
  - [一文看懂卷积神经网络-CNN（基本原理+独特价值+实际应用）- 产品经理的人工智能学习库 (easyai.tech)](https://easyai.tech/ai-definition/cnn/)
  - [一文看懂循环神经网络 RNN（2种优化算法+5个实际应用） (easyai.tech)](https://easyai.tech/ai-definition/rnn/)
  - [一文看懂 LSTM - 长短期记忆网络（基本概念+核心思路） (easyai.tech)](https://easyai.tech/ai-definition/lstm/)
  - [深度学习面试题37：LSTM Networks原理(Long Short Term Memory networks) - 黎明程序员 - 博客园 (cnblogs.com)](https://www.cnblogs.com/itmorn/p/13303155.html)
  - [经典必读：门控循环单元GRU的基本概念与原理](https://www.jiqizhixin.com/articles/2017-12-24#:~:text=在本文中，我们将讨论相当简单且可理解的神经网络模型：门控循环单元（GRU）。 根据 Cho, et al. 在 2014,年的介绍，GRU 旨在解决标准 RNN 中出现的梯度消失问题。 GRU 也可以被视为 LSTM 的变体，因为它们基础的理念都是相似的，且在某些情况能产生同样出色的结果。)
- Adam
  - [简单认识Adam优化器 - 简书 (jianshu.com)](https://www.jianshu.com/p/aebcaf8af76e)
- 主动学习
  - [主动学习(Active Learning)，看这一篇就够了 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/377045943)
  - [主动学习（Active Learning）概述及最新研究 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/422180658)
- 持续/增量学习
  - [增量学习(Incremental Learning)小综述 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/353273834)
- 未登陆词问题OOV
  - [自然语言处理1：分词 - 知乎](https://zhuanlan.zhihu.com/p/109054674)
  - 未登录词识别，NER
  - word2vec中，给未登录词一个随机初始化的向量；
  - 预训练语言模型中，统一当成【UNK】来处理；
- 如何在预训练模型中融入KG知识
  - 清华ernie：T-encoder，K-encoder；T-encoder与bert相同，K-encoder用来融入知识图谱信息。
    - 用NER技术识别出输入序列中的实体，并与知识图谱中的实体进行对应。采用TransE将其转化为向量表示，与原本的输入序列的向量表示拼接送入K-encoder。
  - KG与预训练模型结合的问题
    - 结构化文本与非结构化文本；
    - 异构特征空间的对齐；
    - 知识噪声的解决。
- 强化学习
  - [深度强化学习（Deep Reinforcement Learning）入门 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/25239682)

- 对比学习

  - [一文梳理2020年大热的对比学习模型](https://mp.weixin.qq.com/s/6qqFAQBaOFuXtaeRSmQgsQ)

  - [我分析了ACL21论文列表，发现对比学习已经... ](https://mp.weixin.qq.com/s?__biz=MzIwNzc2NTk0NQ==&mid=2247517437&idx=1&sn=9302ef9e05ad98f83eb177040d4fc7b0&chksm=970fac2ba078253dcb02f4e4ef6646961fa7e0d71d569ae0b6bf7806bbc4aa2bbe981467fc31&mpshare=1&scene=1&srcid=07267nxbZo75Fftv4595oFaR&sharer_sharetime=1627309284524&sharer_shareid=bce0786e4f1449b9738bd32da860598c&exportkey=AcNbTBQ+dsgT1D+cpQ29LSI=&pass_ticket=JLjWK5/tXz8xsSnYVv38ZkVzzOU20fUoGj+eFPLa/Lqykmsms/xwhs/t7W4wRLGW&wx_header=0#rd)





# 4. NLP

## Text Classification

- NLP入门项目之中文文本分类（入门必备）
  - [Chinese-Text-Classification-Pytorch](https://github.com/zhoucz97/Chinese-Text-Classification-Pytorch)
  - [Bert-Chinese-Text-Classification-Pytorch](https://github.com/zhoucz97/Bert-Chinese-Text-Classification-Pytorch)

## Transformer

- [the illstrusted Transformer](http://jalammar.github.io/illustrated-transformer/)
- [transformer详解](https://wmathor.com/index.php/archives/1438/)
- [哈佛transformer代码实现(pytorch)](http://nlp.seas.harvard.edu/2018/04/03/attention.html)
- [RoBERTa模型原理总结](https://zhuanlan.zhihu.com/p/347861417)
- [从BERT, XLNet, RoBERTa到ALBERT](https://zhuanlan.zhihu.com/p/84559048)
- [图解BERT模型](https://zhuanlan.zhihu.com/p/318495113)
- [超细节的BERT/Transformer知识点](https://zhuanlan.zhihu.com/p/132554155)



## BERT

- 关于BERT
  - [图解BERT模型 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/318495113)
  - [【深度学习】BERT详解 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/130913995)
  - BERT缺点
    - 预训练和微调之间的GAP；
    - 假设MASK掉的token是不相关的；
    - 速度慢
  - BERT参数量计算：
    - [How is the number of BERT model parameters calculated? · Issue #656 · google-research/bert (github.com)](https://github.com/google-research/bert/issues/656)
    - [BERT参数量如何计算 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/357353536)



## GNN

- [知乎—如何理解GCN？](https://www.zhihu.com/search?type=content&q=GCN)
- [GCN (Graph Convolutional Network) 图卷积网络解析](https://blog.csdn.net/weixin_36474809/article/details/89316439)
- [图注意力网络(GAT) ICLR2018, Graph Attention Network论文详解](https://blog.csdn.net/weixin_36474809/article/details/89401552)
- [GCN作者博客解释GCN](http://tkipf.github.io/graph-convolutional-networks/)
- [谱聚类原理总结](https://www.cnblogs.com/pinard/p/6221564.html)
- [Graph Convolutional Networks in PyTorch](https://github.com/tkipf/pygcn)
- [Pytorch Graph Attention Network](https://github.com/Diego999/pyGAT)

## Sentiment Analysis

- [2021属性级情感分析新进展](https://github.com/zhoucz97/myLearning/blob/main/第一届情感计算大会/CCAC2021夏睿属性级情感分析新进展.pdf)







# 其他

## Python

- [python中logging日志模块详解](https://www.cnblogs.com/xianyulouie/p/11041777.html)

## Git

- [廖雪峰的官方网站之Git](https://www.liaoxuefeng.com/wiki/896043488029600)

## Docker

- [datawhale组队学习Docker](https://github.com/datawhalechina/team-learning-program/tree/master/Docker)
- [《Docker从入门到实战》](https://vuepress.mirror.docker-practice.com/)
- [b站狂神说之Docker详细版教程](https://www.bilibili.com/video/BV1og4y1q7M4)

可以跟着狂神的视频一步步做，非常详细，从零开始。有点Linux基础且不愿看视频的可以直接去[datawhale组队学习Docker](https://github.com/datawhalechina/team-learning-program/tree/master/Docker)学习，这是[《Docker从入门到实战》](https://vuepress.mirror.docker-practice.com/)的简化版。

学Docker最好买个服务器或者配个Linux系统。

## Conda

- [Conda常用命令合集](https://zhuanlan.zhihu.com/p/363904808)