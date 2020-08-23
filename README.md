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
  
### 垃圾评论检测
- 分析垃圾评论的语言模式、分析用户的评论行为、基于图方法
- 联合垃圾评论检测[26]: 现有的方法只是单独利用了语言线索、行为足迹或者评论系统中代理人之间的关系
  - 解决方案：连接评论网络(用户、评论和产品)和metadata(文本、时间戳、评分)
- 中文垃圾评论检测[25]: 电子商务中的假评论-->在线app评论-->对抗的垃圾评论(将近一半的评论是垃圾，误导分类器)+人工标注成本高
  - 三类对抗垃圾评论形式：1)伪装：添加特殊的符号、复制正常用户评论、转变汉字形式；2)不相关：如广告、产品信息、卖药、敏感词；3)众包
  - 解决方案：1)分析评论特征(包括内容特征和用户行为特征，并分别进行核密度估计)，针对每个特征确定适当的阈值，以此获得最初的有标签数据；2)先单独训练两个模块：BBM(基于行为)和CBM(基于内容)，再联合训练(在每个epoch，直到收敛)[26]
  
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
