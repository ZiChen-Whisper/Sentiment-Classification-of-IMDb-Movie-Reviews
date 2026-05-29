# 人工智能期末实验 - IMDB情感分类实验执行计划

## 实验概述
- **任务**：IMDB电影评论情感二分类（正向/负向）
- **数据**：train.csv (40000条), valid.csv, test.csv，每条包含 text 和 label (0=负向, 1=正向)
- **目标**：构建至少7个模型，完成数据分析、预处理、建模、评测和报告

---

## 阶段一：数据分析

### 步骤1.1：数据集基本信息统计
- 读取 train.csv, valid.csv, test.csv
- 统计各数据集的样本数量
- 统计训练集中正向(label=1)和负向(label=0)情感的数量及比例

### 步骤1.2：词频分析
- 对训练集文本进行分词（tokenization）
- 分别统计正向和负向情感中**频率前十的词**
- 计算PMI（点互信息），找出正向和负向情感中**PMI前十大的词**
- 分析训练集中的用词特点、词性分布，是否表达情感信息

**输出**：数据分析结果表格和文字描述

---

## 阶段二：数据预处理

### 步骤2.1：文本清洗
- 使用 BeautifulSoup 去除HTML标签（如 `<br />`）
- 使用正则表达式去除标点符号和特殊字符
- 统一转为小写

### 步骤2.2：停用词处理
- 使用 NLTK 的英文停用词表去除停用词

### 步骤2.3：词元化与词形还原
- 使用 NLTK 进行分词（word_tokenize）
- 使用 WordNetLemmatizer 进行词形还原（lemmatization）

### 步骤2.4：保存预处理结果
- 将清洗后的文本保存，供后续模型使用

**参考代码**：`代码示例/1_preprocessing and feature extraction.ipynb`

---

## 阶段三：模型构建（至少7个模型）

### 模型1：基于规则的产生式系统
- **方法**：利用 SentiWordNet 计算每条评论的情感得分
- **流程**：
  1. 对预处理后的文本进行词性标注（nltk.pos_tag）
  2. 将Penn Treebank词性标签映射为WordNet词性（penn_to_wn）
  3. 查询每个词在SentiWordNet中的正向/负向得分
  4. 累加所有词的情感得分，根据总分判断正/负向
- **工具**：NLTK, WordNet, SentiWordNet

**参考代码**：`代码示例/1_preprocessing and feature extraction.ipynb` 中的特征提取部分

### 模型2：朴素贝叶斯（词频特征）
- **方法**：使用 CountVectorizer 将文本转为词频向量，训练 MultinomialNB
- **流程**：
  1. 使用 CountVectorizer 对预处理后的文本向量化
  2. 训练 MultinomialNB 分类器
  3. 在验证集和测试集上预测

**参考代码**：`代码示例/2_naive_bayes.ipynb`

### 模型3：逻辑回归（词频特征）
- **方法**：使用 CountVectorizer + LogisticRegression
- **流程**：
  1. 使用 CountVectorizer 向量化
  2. 训练 LogisticRegression（注意设置 max_iter 足够大）
  3. 在验证集和测试集上预测

**参考代码**：`代码示例/3_logistic_regression.ipynb`

### 模型4：逻辑回归（TF-IDF特征）
- **方法**：使用 TfidfVectorizer + LogisticRegression
- **流程**：
  1. 使用 TfidfVectorizer 向量化
  2. 训练 LogisticRegression
  3. 在验证集和测试集上预测

**参考代码**：`代码示例/3_logistic_regression.ipynb` 中的 TF-IDF 部分

### 模型5-6：逻辑回归 + 特征选择（至少2种不同特征选择结果）
- **方法**：在逻辑回归基础上进行特征选择
- **流程**：
  1. 使用 TF-IDF 或 CountVectorizer 生成特征
  2. 使用 SelectKBest + chi2 选择最好的前K个特征（如K=200, K=2000等）
  3. 训练 LogisticRegression
  4. 对比不同K值的性能差异
- **要求**：至少完成两个不同特征选择结果（如 top-200 和 top-2000）

**参考代码**：`代码示例/3_logistic_regression.ipynb` 中的特征选择部分

### 模型7+：逻辑回归 + 新特征设计（至少2个新特征）
- **方法**：在逻辑回归基础上设计新特征
- **可选新特征**：
  - 评论长度（字符数/词数）
  - 情感词比例（使用SentiWordNet计算正向/负向词占比）
  - 感叹号数量
  - 大写字母比例
  - 否定词数量
  - 平均词长
- **流程**：
  1. 提取新特征并与TF-IDF特征拼接
  2. 训练 LogisticRegression
  3. 对比加入新特征前后的性能变化
- **要求**：至少设计2个新特征，训练新模型

### 模型8：大语言模型
- **方法**：使用Prompt设计策略，调用大语言模型进行情感分类
- **要求1**：至少使用一种Prompt设计策略（如 few-shot, chain-of-thought, role-playing 等），在一种大语言模型上测试至少200条测试集数据
- **要求2**：设计prompt使大语言模型产生结构化输出（包含主题、情感等信息），用框架表示法表示，测试至少20条数据
- **可选模型**：DeepSeek, GPT, Claude 等

---

## 阶段四：结果分析

### 步骤4.1：评测指标计算
- 对至少6个模型（除大语言模型外）在测试集上计算：
  - Accuracy
  - Precision
  - Recall
  - F1 Score
- 使用 sklearn.metrics.classification_report

### 步骤4.2：可视化比较
- 使用柱状图比较7个模型（除大语言模型结构化输出外）的性能
- 比较维度：Accuracy, Macro F1, Weighted F1
- 使用 matplotlib 绘制

**参考代码**：`代码示例/3_logistic_regression.ipynb` 中的可视化部分

### 步骤4.3：SentiWordNet分析
- 对除产生式系统外的所有模型的输出结果进行分析
- 分析内容：如正向情感词数量与预测为正向的相关程度
- 方法：
  1. 对测试集每条文本用SentiWordNet计算情感词得分
  2. 对比模型预测结果与SentiWordNet情感得分的一致性
  3. 分析模型误分类样本的情感词分布

### 步骤4.4：大语言模型结构化输出分析
- 观察大语言模型的结构化输出
- 用语言总结发现（如主题提取是否准确、情感判断是否一致等）

---

## 阶段五：实验报告撰写

### 报告结构
1. **引言**：定义任务及其意义
2. **实验数据**：2.1 数据描述 / 2.2 数据预处理
3. **基于规则的情感分析模型**
4. **基于机器学习的情感分析模型**
5. **基于大语言模型的情感分析模型**
6. **实验结果和分析**
7. **结论**
8. **附录**

### 报告要求
- 使用AI辅助撰写，附录说明如何使用
- 不超过8页
- 非必要不贴代码、不截图

---

## 阶段六：答辩准备

- 10分钟演讲 + 5分钟提问
- 重点展示与众不同的地方：
  - 特征选择策略和结果
  - 新特征设计思路
  - Prompt设计策略
  - 框架式表示设计
- 分享发现（数据层面、模型层面的洞察）

---

## 执行顺序总结

| 序号 | 任务 | 依赖 |
|------|------|------|
| 1 | 数据分析（统计、词频、PMI） | 无 |
| 2 | 数据预处理（清洗、停用词、词形还原） | 无 |
| 3 | 模型1：基于规则的产生式系统 | 步骤2 |
| 4 | 模型2：朴素贝叶斯（词频） | 步骤2 |
| 5 | 模型3：逻辑回归（词频） | 步骤2 |
| 6 | 模型4：逻辑回归（TF-IDF） | 步骤2 |
| 7 | 模型5-6：逻辑回归+特征选择 | 步骤6 |
| 8 | 模型7+：逻辑回归+新特征 | 步骤6 |
| 9 | 模型8：大语言模型 | 无 |
| 10 | 评测指标计算与可视化 | 步骤3-8 |
| 11 | SentiWordNet分析 | 步骤3-8 |
| 12 | 大语言模型结构化输出分析 | 步骤9 |
| 13 | 撰写实验报告 | 步骤1-12 |
| 14 | 答辩准备 | 步骤13 |

---

## 关键依赖库

```
pandas, numpy, nltk, bs4 (BeautifulSoup), re
sklearn (CountVectorizer, TfidfVectorizer, MultinomialNB, LogisticRegression, SelectKBest, chi2, classification_report)
matplotlib
sentiwordnet, wordnet (via nltk)
tqdm
```

## 数据路径

- 训练集：`data/train.csv`
- 验证集：`data/valid.csv`
- 测试集：`data/test.csv`
