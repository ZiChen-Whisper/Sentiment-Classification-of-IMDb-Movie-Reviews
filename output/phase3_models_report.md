# 阶段三：模型构建报告

## 模型概览

| 编号 | 模型名称 | 特征方式 | 说明 |
|------|---------|---------|------|
| 1 | 基于规则的产生式系统 | SentiWordNet情感得分 | 无需训练，基于词典规则 |
| 2 | 朴素贝叶斯 | CountVectorizer词频 | MultinomialNB |
| 3 | 逻辑回归 | CountVectorizer词频 | LogisticRegression |
| 4 | 逻辑回归 | TfidfVectorizer | TF-IDF加权特征 |
| 5 | 逻辑回归+特征选择 | TF-IDF + SelectKBest(K=200) | 卡方检验特征选择 |
| 6 | 逻辑回归+特征选择 | TF-IDF + SelectKBest(K=2000) | 卡方检验特征选择 |
| 7 | 逻辑回归+新特征 | TF-IDF + 8个手工特征 | 特征拼接 |

---

## 模型1：基于规则的产生式系统（SentiWordNet）

### 方法
1. 对预处理后的文本进行词性标注（nltk.pos_tag）
2. 将Penn Treebank词性标签映射为WordNet词性（penn_to_wn）
3. 查询每个词在SentiWordNet中的正向/负向得分
4. 累加所有词的情感得分，根据总分判断正/负向

### 测试集结果

### 模型1_SentiWordNet

| 指标 | 值 |
|------|-----|
| Accuracy | 0.6192 |
| Macro Precision | 0.6654 |
| Macro Recall | 0.6187 |
| Macro F1 | 0.5900 |
| Weighted F1 | 0.5902 |

### 模型2_朴素贝叶斯_Count

| 指标 | 值 |
|------|-----|
| Accuracy | 0.8642 |
| Macro Precision | 0.8645 |
| Macro Recall | 0.8642 |
| Macro F1 | 0.8642 |
| Weighted F1 | 0.8642 |

### 模型3_逻辑回归_Count

| 指标 | 值 |
|------|-----|
| Accuracy | 0.8890 |
| Macro Precision | 0.8891 |
| Macro Recall | 0.8890 |
| Macro F1 | 0.8890 |
| Weighted F1 | 0.8890 |

### 模型4_逻辑回归_TFIDF

| 指标 | 值 |
|------|-----|
| Accuracy | 0.8954 |
| Macro Precision | 0.8955 |
| Macro Recall | 0.8954 |
| Macro F1 | 0.8954 |
| Weighted F1 | 0.8954 |

### 模型5_逻辑回归_SelectK200

| 指标 | 值 |
|------|-----|
| Accuracy | 0.8434 |
| Macro Precision | 0.8443 |
| Macro Recall | 0.8433 |
| Macro F1 | 0.8433 |
| Weighted F1 | 0.8433 |

### 模型6_逻辑回归_SelectK2000

| 指标 | 值 |
|------|-----|
| Accuracy | 0.8868 |
| Macro Precision | 0.8870 |
| Macro Recall | 0.8868 |
| Macro F1 | 0.8868 |
| Weighted F1 | 0.8868 |

### 模型7_逻辑回归_新特征

| 指标 | 值 |
|------|-----|
| Accuracy | 0.8936 |
| Macro Precision | 0.8937 |
| Macro Recall | 0.8936 |
| Macro F1 | 0.8936 |
| Weighted F1 | 0.8936 |

---

## 特征选择分析

### SelectKBest (K=200) 选中的前20个特征词
```
1010, 110, 210, 310, 410, 710, 810, 910, acting, actually, also, always, amateurish, amazing, annoying, anything, asleep, atrocious, attempt, avoid
```

### SelectKBest (K=2000) 选中的前30个特征词
```
010, 1010, 110, 15, 1945, 1947, 20th, 210, 30, 3000, 310, 410, 45, 710, 747, 810, 90, 910, aamir, abomination, absolutely, absorbing, absurd, abysmal, accent, accessible, achievement, act, acting, actor
```

### 特征选择对性能的影响
| K值 | 特征维度 | Accuracy | Macro F1 | Weighted F1 |
|-----|---------|----------|----------|-------------|
| 50000 | 50000 | 0.8954 | 0.8954 | 0.8954 |
| 200 | 200 | 0.8434 | 0.8433 | 0.8433 |
| 2000 | 2000 | 0.8868 | 0.8868 | 0.8868 |

---

## 新特征设计分析

### 设计的8个新特征

| 特征名 | 说明 |
|--------|------|
| word_count | 评论词数 |
| avg_word_len | 平均词长 |
| exclamation_count | 感叹号数量 |
| upper_ratio | 大写字母比例 |
| negation_count | 否定词数量 |
| sentiment_ratio | 情感词比例 |
| pos_sentiment_ratio | 正向情感词比例 |
| neg_sentiment_ratio | 负向情感词比例 |

### 新特征在逻辑回归中的系数
- **word_count**: 0.2145 (正向相关)
- **avg_word_len**: -0.0528 (负向相关)
- **exclamation_count**: 0.0292 (正向相关)
- **upper_ratio**: -0.0283 (负向相关)
- **negation_count**: -0.2467 (负向相关)
- **sentiment_ratio**: 0.0157 (正向相关)
- **pos_sentiment_ratio**: 0.1658 (正向相关)
- **neg_sentiment_ratio**: -0.2602 (负向相关)

### 新特征对性能的影响

| 模型 | Accuracy | Macro F1 | Weighted F1 |
|------|----------|----------|-------------|
| 逻辑回归(TF-IDF) | 0.8954 | 0.8954 | 0.8954 |
| 逻辑回归(TF-IDF+新特征) | 0.8936 | 0.8936 | 0.8936 |
| 逻辑回归(仅新特征) | 0.6254 | 0.6251 | 0.6251 |

---

## 模型对比总表（测试集）

| 模型 | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 |
|------|----------|----------------|-------------|----------|-------------|
| 模型1_SentiWordNet | 0.6192 | 0.6654 | 0.6187 | 0.5900 | 0.5902 |
| 模型2_朴素贝叶斯_Count | 0.8642 | 0.8645 | 0.8642 | 0.8642 | 0.8642 |
| 模型3_逻辑回归_Count | 0.8890 | 0.8891 | 0.8890 | 0.8890 | 0.8890 |
| 模型4_逻辑回归_TFIDF | 0.8954 | 0.8955 | 0.8954 | 0.8954 | 0.8954 |
| 模型5_逻辑回归_SelectK200 | 0.8434 | 0.8443 | 0.8433 | 0.8433 | 0.8433 |
| 模型6_逻辑回归_SelectK2000 | 0.8868 | 0.8870 | 0.8868 | 0.8868 | 0.8868 |
| 模型7_逻辑回归_新特征 | 0.8936 | 0.8937 | 0.8936 | 0.8936 | 0.8936 |
