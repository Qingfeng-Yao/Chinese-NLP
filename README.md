## NLP-Research
### 如何表示文本+如何进行计算
- 词袋模型(通过统计不同单词的频次来形成文本向量表示)或N-grams+浅层机器学习
  - 特征工程：基于规则的特征或统计特征
  - 丢失文本的词序信息以及单词之间的联系(稀疏性、多义性、同义性、忽略不同词的语义相关性)
  - NB、LR、SVM
- 根据单词的共现学习词向量(语义相似或联系紧密的词在向量空间中距离更小)+深度学习[20]
  - 一般词嵌入(non-contextual，即不考虑词的上下文，为特定单词提供相同的向量)：word2vec[13]、GloVe[14]、Fast-Text[15][16]
  - 文本编码：RNN、CNN(将文本类比图像，编码形成矩阵表示)、GNN
- 预训练语言模型(利用自监督学习给单词或文本赋予上下文敏感的、多层的语义向量表示)+transformer
  - 动态词嵌入(contextualised vectors)：ELMo[3]、BERT[17] --> finetune(有标注数据)
  - 在目标领域和任务上要继续进行预训练，效果还会有明显提升[24]
    - 当我们所执行任务的标注数据较少，所属的领域与初始预训练语料越不相关，而又能获取到充分的、任务相关的无标注数据时，那就不要停止领域预训练和任务预训练
    - 领域自适应预训练：在领域相关的大规模无标注语料继续进行预训练，然后再对特定任务进行finetune
    - 任务自适应预训练：在任务相关的无标注语料继续进行预训练，然后再对特定任务进行finetune
  
### 文本分类
- 多标签`multi-label`和多类`multi-class`
- EXAM[21]
  - 基于深度学习的方法主要依赖于文本级表示(聚合词向量，FC层参数矩阵即为一组类表示[22]，文本属于一个类的概率很大程度上取决于其整体匹配分数)，忽略细粒度的分类线索(词和类之间的匹配信号，词级匹配信号将为分类提供明确的信号)
  - 解决方案：引入交互机制[23]，显式计算词和类之间的匹配分数，从而将词级匹配信号考虑到文本分类任务中
    - 词级eocoder：将文本映射成词级表示
    - 交互层：创建交互矩阵
    - 聚合层：聚合匹配分数

### 中文
- 小写、jieba分词(可自定义用户词典)、(去停用词)、将数字和符号一般化

### 对抗分类[27]
- 数据被对手主动操纵，试图使分类器产生假负类-->修改分类器-->又会有新的出现
- 解决方案：将分类看成是分类器和对手之间的博弈，然后根据对手的最优策略生成一个最优分类器，并能自动调整分类器以适应对手不断变化的操作
  - 首先将问题形式化为一个成本敏感的分类器和一个成本敏感的对手之间的博弈
  - 着重于email spam detection

### 中文垃圾邮件检测
- StoneSkipping[1]
  - 现有的基于关键词的垃圾邮件检测方法 --> 问题1：伪装逃脱检测(汉字的字形和发音的变异)
  - 数据驱动方法(如基于字符的CNN) --> 问题2：在不可见数据上表现差(一直有新的变异模式)
  - 解决方案：
    - 构建中文字符变异图，进行中文字符表示学习(抽取汉字变异知识) --> 解决问题1，破解汉字伪装
    - 为了解决问题2，即模型能够预测新的变异 --> 借鉴文献[2]的思路，为每一个字符分配一个变异分布，通过学习这些分布，能够预测不可见的变异模式
    - 优化：为了使汉字表示编码进语义信息，文章设计了一个门函数，融合了汉字图表示和文本表示；为了使汉字表示编码进上下文信息，作者借鉴文献[3]的思路，引入一个聚合学习函数，聚合门函数输出和一个双向语言模型的输出
- SIGNAL[4]
  - 将主动学习方法应用于中文垃圾邮件检测
    - 问题1: 伪装(同[1])
    - 问题2: 不平衡问题，即垃圾文本的数量明显小于普通文本数量
    - 问题3: 效率问题，即经典的主动学习是通过两两比较有标签样本和无标签样本来确定要标注的样本的，这样计算复杂度太高
  - 解决方案：问题1的解决思路同[1]。要解决问题2、3，作者通过借鉴文献[5]的思路，引入一个”self-diversity”标准来确定要标注的样本。这样的标准具有定位重要样本和降低计算复杂度的能力，尤其在中文垃圾检测背景下垃圾候选样本更可能获得一个更大的“self-diversity”
  - 可提供一个黑名单词汇(可自动更新)

### web spam detection
- 基于内容的随机游走算法[28]: 
  - 对检索结果产生负面影响
  - 根据页面的相关性和它们的spam likelihood获得页面的排序
    - 考虑网页内容来刻画网页特征，并获得网页spam likelihood的先验估计
    - 根据文本内容和图中关系为图中每个节点计算两个分数，从而来判断网页是否是spam-like
    - 比基于link的方法要好
  - 两类spam：
    - self-promotion: 主要基于内容，在页面中插入可见或不可见关键词的填充，以提高最常见查询的页面检索级别
    - mutual-promotion: 建立在多个网站合作的基础上，或者创建大量页面形成一个链接场，即大量页面相互指向另一个页面，以便通过增加链接数量来提高它们的分数，该方法对利用页面间的共引作为特征(pagerank)的搜索引擎是有效的
  
### 垃圾评论检测
- 分析垃圾评论的语言模式、分析用户的评论行为、基于图方法
- 基于评论图的在线商店spammer检测[29]: 
  - 构建异质评论图：捕捉评论者、评论和评论者所评论的商店之间的关系，探索节点之间如何交互以产生spam
  - 有效计算方法：定量评论者的可信度、评论的真实性和商店的可信度
  - 没有使用文本信息
- 联合垃圾评论检测[26]: 现有的方法只是单独利用了语言线索、行为足迹或者评论系统中代理人之间的关系
  - 解决方案：连接评论网络(用户、评论和产品)和metadata(文本、时间戳、评分)，对所有用户、评论和产品进行排名
- 中文垃圾评论检测[25]: 电子商务中的假评论-->在线app评论-->对抗的垃圾评论(将近一半的评论是垃圾，误导分类器)+人工标注成本高
  - 三类对抗垃圾评论形式：1)伪装：添加特殊的符号、复制正常用户评论、转变汉字形式；2)不相关：如广告、产品信息、卖药、敏感词；3)众包
  - 解决方案：1)分析评论特征(包括内容特征和用户行为特征，并分别进行核密度估计)，针对每个特征确定适当的阈值，以此获得最初的有标签数据；2)先单独训练两个模块：BBM(基于行为)和CBM(基于内容)，再联合训练(在每个epoch，直到收敛)[26]
  - 可创建黑白名单词汇：训练模型word2vec
  
### social spam detection
- 社交书签网站(社交标签系统)[30]: 
  - 需要管理员的时间和精力来手动过滤或删除spam
  - 支持标签系统的数据结构称为folksonomy，表示为超图，一个三元组的集合(三元组指用户u用标签t标注资源r，资源是网址；三元组可表示为连接用户、资源和标签的超边)，一个post可有多个标签
  - spammer会使用流行的标签标注，这些标签互相之间没什么关系且与网址无关
  - 提出六类不同特征(social spam的不同属性)，每个特征都提供一个有用信号来区分spammer和合法用户；之后这些特征用于各种机器学习算法做分类
- social spam detection framework[31]: 
  - 每个社交网络都需要建立自己的spam过滤器，并支持一个spam小组来跟进预防spam的最新进展
  - social spam：低质量信息
  - 框架可用于所有社交网络(新的社交网络可很容易插入系统中)，在一个社交网络中检测到新的spam，可在整个社交网络中快速识别
  - 框架分为三个部分：
    - 映射和装配: 其中映射是指将一个社交网络特定对象转换成一个标准模型(如profile model、message model、webpage model)；如果可以基于这个对象获取关联对象，则在这里组装
    - 预先过滤：用fast-path技术(如黑名单/IP、URL或相似度匹配/哈希、shingling)根据已知spam对象检查传入对象
    - 分类：有监督机器学习用于分类传入对象和关联对象，将贝叶斯技术与分类结果结合得到spam和non-spam
- 社交系统中的信息质量[32]: 
  - 社交网络、社交媒体网站、大规模信息共享社区、crowd-based资助服务、网络规模的众包系统
  - social spam、campaigns、misinformation、crowdturfing
    - social spam: 如何检测可疑的URL；讨论社会资本家和spammers之间的关系，以及如何惩罚spammers和这些社会资本家；有监督和无监督spam detection方法；Social Honeypot[33]的提出是为了监测spammers的行为并收集他们的信息；利用群体智慧识别social spammers
    - campaigns: 基于图的social spam campaign detection；内容驱动的campaign detection；使用分类方法检测和跟踪社交媒体中的政治campaigns；基于行为模型的频繁itemset挖掘方法检测虚假评论者群体
    - misinformation: 利用群体力量的分类方法来衡量社交媒体上的信息可信度；中国领先的微博服务提供商新浪微博的自动谣言检测方法；识别飓风桑迪期间Twitter上的假图片；紧急情况下信息可信度的方法(该方法包括无监督方法和有监督方法来检测消息可信度)
    - crowdturfing: 介绍新闻媒体报道的实例；了解在众包网站上有什么样的众包任务；了解东西方众包网站的crowdturfing市场规模；追踪并揭示社交媒体的众包操控，特别是关注西方的众包网站，并概述如何在社交媒体上发现众包者crowdturfers
  - social spam与传统的spam(比如email和web spam)的不同
    - 开放性，任何人都可以创建一个社交帐户，方便与其他用户联系
    - URL黑名单在识别新的威胁方面太慢了，使得超过90%的访问者能够在页面被列入黑名单之前查看该页面
    - 用于模糊处理的URL缩短服务
    - 使用API自动控制机器人程序
  - 改善信息质量：揭示和检测恶意参与者(如social spammers、内容污染者、众包者)和低质量内容
- 无监督的spam detection[34]: 
  - 需及时检测spam --> 无监督可节省训练消耗
  - 现有无监督检测方案严重依赖于不断变化的spamming模式以避免被检测
  - 提出一个基于sybli防御的spam detection方案SD2，该方案通过考虑社交网络关系，显著优于现有方案
  - 为了使其在面对日益严重的spam攻击时具有很强的鲁棒性，进一步设计了一种无监督spam detection方案UNIK
    - 不是直接检测spammers，而是故意从网络中删除non-spammers，同时利用社交图和用户链接图
    - 虽然spammers不断改变其模式来逃避检测，non-spammers不必这样做，因此具有相对non-valatile的模式
    - 基于UNIK的检测结果，进一步分析在这个社交网站上发现的几个spam campaigns

### campaign detection
- social spam campaigns detection[35]
- 内容驱动的campaigns detection[36][37]
  - 研究从大型基于消息的图中分离出一致的campaigns的图挖掘技术
  - 从实验中检测到5类campaigns：spam、promotion、template、news、celebrity campaigns

  
### 无监督文本分类
- Keyword enrichment(KE)[6]
  - 有个具体的实际场景，即银行业中的风险事件分类，原先这些事件已经映射到大约20个风险类中，但更多的风险类有助于更好地捕捉事件的细微差别并进行相关比较，所以就诞生了一个新的分类法，该分类法由264个类组成。这样就需要将所有事件映射到这264个类中。由于这是一个新的分类法，没有可用的有标签的数据，且鉴于该领域的特殊性和专家的缺乏，不可能为每一类别获得许多有标签的例子 --> 问题：给定一组已知的类别和若干未标记的文档，将所有未标记的文档映射到它所属的类别中去，且分类效果与有监督方法相当
  - 解决方案：利用文档词和标签关键词之间的文本相似性实现文本分类(得到文档和标签的向量表示/LSA向量，利用余弦相似度确定文档所属的标签)，该方法的新奇之处在于结合专家知识和语言模型对标签关键词的扩充
- TIGAN[7]
  - 将Infogan模型[8]应用于无监督文本分类
    - 问题1: mode collapse
    - 问题2: 词袋向量导致GAN训练失败
  - 解决方案：
    - 对于问题1，对损失进行裁剪
    - 对于问题2，作者通过借鉴文献[9]的思路，使用WGAN-gp来解决
    - 优化：通过借鉴文献[10]、[11]的思路，将infoGAN和auto-encoder相组合并轮流训练；通过借鉴文献[12]的思路，引入预训练词向量来帮助模型发现更一致的主题

### 无监督文本异常检测
- CVDD[18]
  - one-class分类 --> 问题：词袋表示
  - 解决方案：
    - 利用预训练模型
    - 优化：通过多头自注意力机制[19]将词嵌入的变长序列映射成固定长度的文本表示；将文本表示和一个上下文向量集合一起训练，能够捕捉到多个正常模式，如可能对应于一个不同但非异常的主题集合；解决manifold collapse(容易收敛到退化解，其中数据被转换为小流形或单点)


## 参考文献
- [1] 2019 | EMNLP | Detect Camouflaged Spam Content via StoneSkipping: Graph and Text Joint Embedding for Chinese Character Variation Representation
- [2] 2015 | AAAI | Topical word embeddings
- [3] 2018 | ACL | Deep contextualized word representations
- [4] 2020 | ACL | Camouflaged Chinese Spam Content Detection with Semi-supervised Generative Active Learning
- [5] 2017 | CVPR | Fine-tuning convolutional neural networks for biomedical image analysis: actively and incrementally
- [6] 2019 | ACL | Towards Unsupervised Text Classification Leveraging Experts and Word Embeddings
- [7] 2020 | Learning Interpretable and Discrete Representations with Adversarial Training for Unsupervised Text Classification
- [8] 2016 | NIPS | Infogan: Interpretable representation learning by information maximizing generative adversarial nets
- [9] 2017 | Wasserstein gan
- [10] 2016 | ICML | Autoencoding beyond pixels using a learned similarity metric
- [11] 2018 | NIPS | Introvae: Introspective variational autoencoders for photographic image synthesis
- [12] 2015 | TACL | Improving topic models with latent feature word representations
- [13] 2013 | NIPS | Distributed representations of words and phrases and their compositionality | Mikolov et al.
- [14] 2014 | EMNLP | Glove: Global vectors for word representation
- [15] 2017 | TACL | Enriching word vectors with subword information
- [16] 2017 | ACL | Bag of tricks for efficient text classification
- [17] 2018 | BERT: Pre-training of deep bidirectional transformers for language understanding
- [18] 2019 | ACL | Self-Attentive, Multi-Context One-Class Classification for Unsupervised Anomaly Detection on Text
- [19] 2017 | ICLR | A structured self-attentive sentence embedding
- [20] 2017 | ACL | Neural semantic encoders
- [21] 2019 | AAAI | Explicit Interaction Model towards Text Classification
- [22] 2017 | ACL | Using the output embedding to improve language models
- [23] 2016 | ACL | Learning natural language inference with LSTM
- [24] 2020 | ACL | Don’t Stop Pretraining: Adapt Language Models to Domains and Tasks
- [25] 2020 | WWW | Analyzing and Detecting Adversarial Spam on a Large-scale Online APP Review System
- [26] 2015 | KDD | Collective opinion spam detection: Bridging review networks and metadata 
- [27] 2004 | KDD | Adversarial Classification
- [28] 2010 | SMUC | Spam Detection with a Content-based Random-walk Algorithm
- [29] 2011 | ICDM | Review Graph based Online Store Review Spammer Detection
- [30] 2009 | Social Spam Detection
- [31] 2011 | A Social-Spam Detection Framework
- [32] 2014 | WWW | [Social Spam, Campaigns, Misinformation and Crowdturfing](https://web.cs.wpi.edu/~kmlee/tutorial/www2014.html)
- [33] 2011 | AAAI | Seven Months with the Devils: A Long-Term Study of Content Polluters on Twitter
- [34] 2013 | CIKM | UNIK: Unsupervised Social Network Spam Detection
- [35] 2010 | Detecting and Characterizing Social Spam Campaigns
- [36] 2011 | CIKM | Content-Driven Detection of Campaigns in Social Media
- [37] 2013 | Campaign Extraction from Social Media
