> 个人学习历程，持续更新中......
>
> 主要与NLP相关。



### Contents

- [1.教程](#1-教程)
- [2.基础知识](#2-基础知识)
- [3.AI](#3-AI)
- [4.论文写作](#4-科研)
- [5.其他](#5-其他)





# 1. 教程

- **【机器学习必看】**—吴恩达Stanford CS229
  - [CS229: Machine Learning (stanford.edu)](https://cs229.stanford.edu/)
  - [CS229 课程讲义中文翻译](https://github.com/Kivy-CN/Stanford-CS-229-CN)
  - [个人整理的229资料](https://github.com/zhoucz97/CS229)，主要是08年和18年吴恩达的。
  
- **【强推】**李宏毅人工智能课
  - [Bilibili-2021/2022春机器学习课程](https://www.bilibili.com/video/BV1Wv411h7kN/?p=1&vd_source=a0cdcca7d567da626e54d50523fa551c)
  - [李宏毅老师YouTube频道](https://www.youtube.com/c/HungyiLeeNTU)，包括机器学习、自然语言处理、GAN、强化学习等一系列课程。

- **【机器学习经典书籍】**—《统计学习方法》
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
- **【NLP必看】**—Stanford CS224N
  - [Stanford CS 224N | Natural Language Processing with Deep Learning](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1214/)
  - 作业：[parachutel/cs224n-stanford-winter2021: Stanford Winter 2021 (github.com)](https://github.com/parachutel/cs224n-stanford-winter2021)


# 2. 基础知识

- **激活函数**

  - [Bert使用的激活函数：gelu---高斯误差线性单元](https://blog.csdn.net/eunicechen/article/details/84774047?utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromMachineLearnPai2%7Edefault-3.control&dist_request_id=1332024.6353.16189739587076911&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromMachineLearnPai2%7Edefault-3.control)
  - [GELU 激活函数](https://blog.csdn.net/liruihongbob/article/details/86510622)

- **评价指标**

  - [精确率、准确率、召回率、F1值含义及sklearn调用 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/530885866)
  - [三种F1指标](https://blog.csdn.net/weixin_48185819/article/details/108195540)
  - [BLEU - Wikipedia](https://en.wikipedia.org/wiki/BLEU)
  - [ROUGE (metric) - Wikipedia](https://en.wikipedia.org/wiki/ROUGE_(metric))

- **过拟合问题**

  - 解决方式：加大数据量、降低模型复杂度、Dropout、正则化、早停策略、warmup学习率、模型集成
  - [Early Stopping - 简书 (jianshu.com)](https://www.jianshu.com/p/9ab695d91459)
  - [深度学习技巧之Early Stopping（早停法） | 数据学习者官方网站(Datalearner)](https://www.datalearner.com/blog/1051537860479157)
  - [生动形象告诉你神经网络的Dropout为何有效_qiuzitao的博客-CSDN博客_dropout为什么有效](https://blog.csdn.net/qiuzitao/article/details/105370129)
  - [为什么正则化能减少模型过拟合程度_ybdesire的博客-CSDN博客](https://blog.csdn.net/ybdesire/article/details/79068603)
  - [L1正则化与L2正则化的区别_ybdesire的博客-CSDN博客_l1正则化和l2正则化的区别](https://blog.csdn.net/ybdesire/article/details/84946128)
  - **为什么PCA不被推荐用来避免过拟合？**
    - 因为PCA是无监督的降维方法。
    - [为什么PCA不被推荐用来避免过拟合_嘀嗒嘀嘀嗒嘀的博客-CSDN博客](https://blog.csdn.net/ACBattle/article/details/80011808)
    - [(5 封私信) 为什么PCA不被推荐用来避免过拟合？ - 知乎 (zhihu.com)](https://www.zhihu.com/question/47121788)

- **数据归一化**

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

- **条件随机场CRF**

  - [超容易理解的CRF讲解](https://zhuanlan.zhihu.com/p/44042528)

- **支持向量机SVM**

  - [SVM算法 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/29862011)
  - [机器学习算法（一）SVM_yaoyz105的博客-CSDN博客_svm](https://blog.csdn.net/qq_31347869/article/details/88071930)

- **CNN, RNN, LSTM, GRU**

  - [一文看懂卷积神经网络-CNN（基本原理+独特价值+实际应用）- 产品经理的人工智能学习库 (easyai.tech)](https://easyai.tech/ai-definition/cnn/)
  - [一文看懂循环神经网络 RNN（2种优化算法+5个实际应用） (easyai.tech)](https://easyai.tech/ai-definition/rnn/)
  - [一文看懂 LSTM - 长短期记忆网络（基本概念+核心思路） (easyai.tech)](https://easyai.tech/ai-definition/lstm/)
  - [深度学习面试题37：LSTM Networks原理(Long Short Term Memory networks) - 黎明程序员 - 博客园 (cnblogs.com)](https://www.cnblogs.com/itmorn/p/13303155.html)
  - [经典必读：门控循环单元GRU的基本概念与原理](https://www.jiqizhixin.com/articles/2017-12-24#:~:text=在本文中，我们将讨论相当简单且可理解的神经网络模型：门控循环单元（GRU）。 根据 Cho, et al. 在 2014,年的介绍，GRU 旨在解决标准 RNN 中出现的梯度消失问题。 GRU 也可以被视为 LSTM 的变体，因为它们基础的理念都是相似的，且在某些情况能产生同样出色的结果。)

- **优化器**

  - [简单认识Adam优化器 - 简书 (jianshu.com)](https://www.jianshu.com/p/aebcaf8af76e)

- **主动学习**

  - [主动学习(Active Learning)，看这一篇就够了 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/377045943)
  - [主动学习（Active Learning）概述及最新研究 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/422180658)

- **持续/增量学习**

  - [增量学习(Incremental Learning)小综述 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/353273834)

- **未登陆词问题OOV**

  - [自然语言处理1：分词 - 知乎](https://zhuanlan.zhihu.com/p/109054674)
  - 未登录词识别，NER
  - word2vec中，给未登录词一个随机初始化的向量；
  - 预训练语言模型中，统一当成【UNK】来处理；

- **预训练模型中融入KG知识**

  - 清华ernie：T-encoder，K-encoder；T-encoder与bert相同，K-encoder用来融入知识图谱信息。
    - 用NER技术识别出输入序列中的实体，并与知识图谱中的实体进行对应。采用TransE将其转化为向量表示，与原本的输入序列的向量表示拼接送入K-encoder。
  - KG与预训练模型结合的问题
    - 结构化文本与非结构化文本；
    - 异构特征空间的对齐；
    - 知识噪声的解决。

- **强化学习**

  - [深度强化学习（Deep Reinforcement Learning）入门 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/25239682)

- **对比学习**
  
  - [图解SimCLR框架，用对比学习得到一个好的视觉预训练模型-CSDN博客](https://blog.csdn.net/u011984148/article/details/106233313/)
  
  - [一文梳理2020年大热的对比学习模型](https://mp.weixin.qq.com/s/6qqFAQBaOFuXtaeRSmQgsQ)
  
  - [我分析了ACL21论文列表，发现对比学习已经... ](https://mp.weixin.qq.com/s?__biz=MzIwNzc2NTk0NQ==&mid=2247517437&idx=1&sn=9302ef9e05ad98f83eb177040d4fc7b0&chksm=970fac2ba078253dcb02f4e4ef6646961fa7e0d71d569ae0b6bf7806bbc4aa2bbe981467fc31&mpshare=1&scene=1&srcid=07267nxbZo75Fftv4595oFaR&sharer_sharetime=1627309284524&sharer_shareid=bce0786e4f1449b9738bd32da860598c&exportkey=AcNbTBQ+dsgT1D+cpQ29LSI=&pass_ticket=JLjWK5/tXz8xsSnYVv38ZkVzzOU20fUoGj+eFPLa/Lqykmsms/xwhs/t7W4wRLGW&wx_header=0#rd)
  
- **Normalization**

  - BatchNorm:
    - [聊聊Batch Normalization在网络结构中的位置_炼丹笔记的博客-CSDN博客](https://blog.csdn.net/m0_52122378/article/details/117221003)
    - [BatchNorm原理以及PyTorch实现_ffiirree的博客-CSDN博客_batchnorm pytorch](https://blog.csdn.net/ice__snow/article/details/121472283)
    - [Pytorch中的BatchNorm_小王同学w的博客-CSDN博客_pytorch中的batchnorm](https://blog.csdn.net/a171232886/article/details/121489828?spm=1001.2101.3001.6650.1&utm_medium=distribute.pc_relevant.none-task-blog-2~default~CTRLIST~default-1-121489828-blog-121472283.pc_relevant_aa&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2~default~CTRLIST~default-1-121489828-blog-121472283.pc_relevant_aa&utm_relevant_index=1)
    - [batch normlization问题总结 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/271677545)
    - [对 BatchNormalization 中 Internal Convariate Shift 的理解_wzg2016的博客-CSDN博客_协变量漂移](https://blog.csdn.net/strive_for_future/article/details/108323634)
    
  - BatchNorm和LayerNorm区别:
    - **BatchNorm是对一个batch-size样本内的每个特征做归一化，LayerNorm是对每个样本的所有特征做归一化。**
    - [BatchNorm与LayerNorm的异同 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/428620330)

  - [各种Normalization - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/86765356)
  
- **霍普菲尔德网络**

  - [Hopfield网络的基本 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/116908556)
  - [最简单的神经网络：霍普菲尔德神经网络 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/370000751)
  
- 对神经网络的结果做**显著性检验**

  - [如何对一个神经网络分类器做显著性检验（例如t-test）？ - 知乎 (zhihu.com)](https://www.zhihu.com/question/528045984/answer/2488535331)




# 3. AI

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

## Sentiment Analysis

- [2021属性级情感分析新进展](https://github.com/zhoucz97/myLearning/blob/main/第一届情感计算大会/CCAC2021夏睿属性级情感分析新进展.pdf)
- [情感分析论文笔记 - 知乎 (zhihu.com)](https://www.zhihu.com/column/c_1308370575622688768)
- 对话情绪识别paper reading list: [zhoucz97/ERC-Reading-List](https://github.com/zhoucz97/ERC-Reading-List)
- 情绪溯源paper reading list: [zhoucz97/ECA-Reading-List](https://github.com/zhoucz97/ECA-Reading-List)
- [CCAC 前沿趋势报告-文本情感计算新进展 (qq.com)](https://mp.weixin.qq.com/s/NNiqjoPzC9Eozpy0LGVgkw)
- [CCAC 2022 讲习班-情感分析基础与前沿 (qq.com)](https://mp.weixin.qq.com/s/u7tSBF2Im4IAabJpzJNicA)
- **[情感词库构建代表工作详解](https://mp.weixin.qq.com/s/mefUYQnTn8vdWV78c9lRBw)**



## LLM

### 1. ChatGPT-OpenAI

参见：[zhoucz97/awesome-ChatGPT: ChatGPT相关资源汇总 (github.com)](https://github.com/zhoucz97/awesome-ChatGPT)

### 2. Bard-Google

- Bard发布：[Google AI updates: Bard and new AI features in Search (blog.google)](https://blog.google/technology/ai/bard-google-ai-search-updates/)
- 

### 3. YouChat

- [who are you - AI Search Based Chat | AI for Search Engines](https://you.com/search?q=who+are+you&tbm=youchat&cfr=chat)



### Alpa

- [alpa-projects/alpa: Training and serving large-scale neural networks (github.com)](https://github.com/alpa-projects/alpa)
- 文档：[Alpa Documentation — Alpa 0.2.2.dev12 documentation](https://alpa.ai/)
- demo页面：[Serving OPT-175B Language Model with Alpa](https://opt.alpa.ai/)
- 

### 智源悟道

- 官网：[悟道 (wudaoai.cn)](https://wudaoai.cn/home)
- demo：[悟道通用图灵测试 (aminer.cn)](https://models.aminer.cn/turing-test/v2/)
- CPM-Live：[CPM-Live ｜ OpenBMB](https://live.openbmb.org/home)
- OpenBMB官网：[OpenBMB - 让大模型飞入千家万户](https://www.openbmb.org/home)



## GNN

- [知乎—如何理解GCN？](https://www.zhihu.com/search?type=content&q=GCN)
- [GCN (Graph Convolutional Network) 图卷积网络解析](https://blog.csdn.net/weixin_36474809/article/details/89316439)
- [图注意力网络(GAT) ICLR2018, Graph Attention Network论文详解](https://blog.csdn.net/weixin_36474809/article/details/89401552)
- [GCN作者博客解释GCN](http://tkipf.github.io/graph-convolutional-networks/)
- [谱聚类原理总结](https://www.cnblogs.com/pinard/p/6221564.html)
- [Graph Convolutional Networks in PyTorch](https://github.com/tkipf/pygcn)
- [Pytorch Graph Attention Network](https://github.com/Diego999/pyGAT)
- **[Pytorch-Geometric(PyG)官方文档](https://mp.weixin.qq.com/s/mefUYQnTn8vdWV78c9lRBw)**

## 扩散模型

- [从VAE到扩散模型：一文解读以文生图新范式 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/519415802)



# 4. 科研

1. **Latex写论文**：https://www.overleaf.com/
2. LaTeX教程：
   1. [Latex使用心得 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/518143774)

3. [MLNLP-World/Paper-Writing-Tips: 该仓库是MLNLP社区用来帮助大家避免论文投稿小错误的整理仓库。 Paper Writing Tips (github.com)](https://github.com/MLNLP-World/Paper-Writing-Tips)
4. **LaTeX数学公式**：[Online Equation Editor - standalone (codecogs.com)](https://www.codecogs.com/latex/eqneditor.php)
5. **画图表**：
   1. Excel：
      1. [Tip: Excel图表导出为PDF图像 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/260991352)，然后利用工具裁剪导出后的PDF，比如WPS（开会员）。
      2. [用Excel制作Origin科研论文图（分组柱状图） - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/527514961)
      3. [如何将论文图表做得漂亮？ - 知乎 (zhihu.com)](https://www.zhihu.com/question/26983018)
   2. Python：
      1. [Python常用画图代码（折线图、柱状图、饼图） - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/345841672)
      2. [Python 绘图，我只用 Matplotlib（三）—— 柱状图 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/47679593)
   3. [MLNLP-World/Paper-Picture-Writing-Code: MLNLP: Paper Picture Writing Code (github.com)](https://github.com/MLNLP-World/Paper-Picture-Writing-Code)
6. **画模型图**：draw.io
7. **设置字体颜色**：[LaTeX知识分享|如何设置字体颜色 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/426780029?utm_id=0)
8. **数学公式加粗**：[如何在LaTeX数学模式中更好地使用粗体？ - 知乎 (zhihu.com)](https://www.zhihu.com/question/25290041)
9. **对号和叉号**：[latex中的对与错(对号√与叉号×)](https://blog.csdn.net/m0_61899108/article/details/126585359)
10. 科研小技巧：
    1. [我的收藏 - 收藏夹 - 知乎 (zhihu.com)](https://www.zhihu.com/collection/700323211)




# 5. 其他



## 算法题

- [labuladong的算法小抄](https://github.com/labuladong/fucking-algorithm)。100多道题，分门别类，先讲某个算法，再讲该算法下对应的力扣题目，Github标星已经80多k了。
- [liweiwei1419](https://github.com/liweiwei1419)。威威哥，长期活跃在力扣社区，为力扣官方贡献了大量的优质题解。[liweiwei个人博客](https://liweiwei1419.gitee.io/leetcode-algo/about/)，[liweiwei力扣主页](https://leetcode-cn.com/u/liweiwei1419/)
- [个人笔记之LeetCode题解与算法（语雀）](https://www.yuque.com/zhcz/leetcode)
- [知乎问题—如何理解和掌握KMP算法](https://www.zhihu.com/question/21923021/answer/281346746)

## Python

- Logging：[python中logging日志模块详解](https://www.cnblogs.com/xianyulouie/p/11041777.html)
- 调试器Pdb：[10分钟教程掌握Python调试器pdb - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/37294138)



## Pytorch

- 冻结参数
  - [ pytorch网络冻结的三种方法区别：detach、requires_grad、with_no_grad_shuttle6的博客-CSDN博客_pytorch冻结网络层](https://blog.csdn.net/weixin_42855362/article/details/127284573)
  - [pytorch 冻结某些层参数不训练 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/65105409)
- **手动下载Pytorch**
  - 官方源：https://download.pytorch.org/whl/torch_stable.html
  - 清华源：[Index of /anaconda/cloud/pytorch/linux-64/ | 清华大学开源软件镜像站 | Tsinghua Open Source Mirror](https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/linux-64/)
  - [Pytorch手动下载安装教程 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/591788841)




## 服务器

- **tmux**

  -  [一文助你打通 tmux - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/102546608)
  - [Tmux 使用教程 - 阮一峰的网络日志](http://www.ruanyifeng.com/blog/2019/10/tmux.html)

- **nohup**

  - [服务器后台跑程序的一种方法（nohup）_Laiguibing521的博客-CSDN博客](https://blog.csdn.net/laiguibing521/article/details/90316703)

- [Ubuntu登录SSH后显示欢迎消息 - 知乎](https://zhuanlan.zhihu.com/p/390518917)

- **conda**
  - [Conda常用命令合集](https://zhuanlan.zhihu.com/p/363904808)
  
- **Nvidia-smi**

  - [windos中查看gpu信息，以及NVIDIA-SMI命令详解_tanlangqie的博客-CSDN博客_查看gpu](https://blog.csdn.net/tanlangqie/article/details/82967296)

- **frp内网穿透**
  - [frp (gofrp.org)](https://gofrp.org/)
  - [frp实现内网穿透访问内网多台Linux服务器 - JasonCeng - 博客园 (cnblogs.com)](https://www.cnblogs.com/JasonCeng/p/14375087.html)
  - [详细内网穿透原理（转载） - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/444415996)
  

## Git

- [廖雪峰的官方网站之Git](https://www.liaoxuefeng.com/wiki/896043488029600)

## Docker

- [datawhale组队学习Docker](https://github.com/datawhalechina/team-learning-program/tree/master/Docker)
- [《Docker从入门到实战》](https://vuepress.mirror.docker-practice.com/)
- [b站狂神说之Docker详细版教程](https://www.bilibili.com/video/BV1og4y1q7M4)

- [datawhale组队学习Docker](https://github.com/datawhalechina/team-learning-program/tree/master/Docker)
- [《Docker从入门到实战》](https://vuepress.mirror.docker-practice.com/)
