"""
阶段一：数据分析
- 步骤1.1：数据集基本信息统计
- 步骤1.2：词频分析（频率前十、PMI前十、用词特点）
"""

import pandas as pd
import numpy as np
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import math
import json

# ============================================================
# 步骤1.1：数据集基本信息统计
# ============================================================
print("=" * 60)
print("步骤1.1：数据集基本信息统计")
print("=" * 60)

train_df = pd.read_csv('data/train.csv')
valid_df = pd.read_csv('data/valid.csv')
test_df = pd.read_csv('data/test.csv')

stats = {}
for name, df in [('训练集', train_df), ('验证集', valid_df), ('测试集', test_df)]:
    total = len(df)
    pos = (df['label'] == 1).sum()
    neg = (df['label'] == 0).sum()
    stats[name] = {'total': total, 'positive': int(pos), 'negative': int(neg),
                   'pos_ratio': round(pos / total * 100, 2), 'neg_ratio': round(neg / total * 100, 2)}
    print(f"\n{name}：")
    print(f"  样本数量：{total}")
    print(f"  正向(label=1)：{pos} ({pos/total*100:.2f}%)")
    print(f"  负向(label=0)：{neg} ({neg/total*100:.2f}%)")

# 计算文本长度统计
print("\n训练集文本长度统计：")
train_df['text_len'] = train_df['text'].apply(len)
train_df['word_count'] = train_df['text'].apply(lambda x: len(x.split()))
print(f"  平均字符长度：{train_df['text_len'].mean():.1f}")
print(f"  中位字符长度：{train_df['text_len'].median():.1f}")
print(f"  最短字符长度：{train_df['text_len'].min()}")
print(f"  最长字符长度：{train_df['text_len'].max()}")
print(f"  平均词数：{train_df['word_count'].mean():.1f}")
print(f"  中位词数：{train_df['word_count'].median():.1f}")

# 按正负向统计文本长度
for label_val, label_name in [(0, '负向'), (1, '正向')]:
    subset = train_df[train_df['label'] == label_val]
    print(f"\n  {label_name}评论：")
    print(f"    平均字符长度：{subset['text_len'].mean():.1f}")
    print(f"    平均词数：{subset['word_count'].mean():.1f}")


# ============================================================
# 步骤1.2：词频分析
# ============================================================
print("\n" + "=" * 60)
print("步骤1.2：词频分析")
print("=" * 60)

# 基础预处理：去HTML标签、去标点、转小写、去停用词
stop_words = set(stopwords.words('english'))

def basic_preprocess(text):
    """基础预处理：去HTML、去标点、转小写"""
    text = re.sub(r'<[^>]+>', ' ', text)  # 去HTML标签
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)  # 只保留字母
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize_and_filter(text, remove_stopwords=True):
    """分词并可选去除停用词"""
    tokens = word_tokenize(text)
    if remove_stopwords:
        tokens = [t for t in tokens if t not in stop_words and len(t) > 1]
    else:
        tokens = [t for t in tokens if len(t) > 1]
    return tokens

# 对训练集进行分词
print("\n正在对训练集进行分词...")
pos_texts = train_df[train_df['label'] == 1]['text'].tolist()
neg_texts = train_df[train_df['label'] == 0]['text'].tolist()

pos_tokens_list = []  # 每条评论的token列表（不去停用词，用于PMI）
neg_tokens_list = []
pos_all_tokens = []   # 所有正向token（去停用词，用于词频统计）
neg_all_tokens = []   # 所有负向token（去停用词）
pos_all_tokens_raw = []  # 不去停用词
neg_all_tokens_raw = []

for text in pos_texts:
    cleaned = basic_preprocess(text)
    tokens_raw = tokenize_and_filter(cleaned, remove_stopwords=False)
    tokens_filtered = tokenize_and_filter(cleaned, remove_stopwords=True)
    pos_tokens_list.append(tokens_raw)
    pos_all_tokens.extend(tokens_filtered)
    pos_all_tokens_raw.extend(tokens_raw)

for text in neg_texts:
    cleaned = basic_preprocess(text)
    tokens_raw = tokenize_and_filter(cleaned, remove_stopwords=False)
    tokens_filtered = tokenize_and_filter(cleaned, remove_stopwords=True)
    neg_tokens_list.append(tokens_raw)
    neg_all_tokens.extend(tokens_filtered)
    neg_all_tokens_raw.extend(tokens_raw)

# ---- 频率前十的词 ----
print("\n--- 正向情感中频率前十的词（去除停用词）---")
pos_freq = Counter(pos_all_tokens)
pos_top10 = pos_freq.most_common(10)
for word, count in pos_top10:
    print(f"  {word}: {count}")

print("\n--- 负向情感中频率前十的词（去除停用词）---")
neg_freq = Counter(neg_all_tokens)
neg_top10 = neg_freq.most_common(10)
for word, count in neg_top10:
    print(f"  {word}: {count}")

# ---- PMI分析 ----
print("\n--- PMI（点互信息）分析 ---")

# 构建全局词频和文档频率
all_tokens_raw = pos_all_tokens_raw + neg_all_tokens_raw
total_tokens = len(all_tokens_raw)
global_freq = Counter(all_tokens_raw)

# 文档级别统计（每条评论视为一个文档）
N_docs = len(train_df)  # 总文档数
pos_docs = len(pos_texts)
neg_docs = len(neg_texts)

# 计算每个词在正/负向文档中出现的文档数
pos_doc_freq = Counter()  # 词在正向文档中出现的文档数
neg_doc_freq = Counter()  # 词在负向文档中出现的文档数

for tokens in pos_tokens_list:
    unique_tokens = set(tokens)
    for t in unique_tokens:
        pos_doc_freq[t] += 1

for tokens in neg_tokens_list:
    unique_tokens = set(tokens)
    for t in unique_tokens:
        neg_doc_freq[t] += 1

# 计算PMI: PMI(word, positive) = log(P(word, pos) / (P(word) * P(pos)))
# 使用文档频率计算
# P(word, pos) = doc_freq(word, pos) / N_docs
# P(word) = (pos_doc_freq[word] + neg_doc_freq[word]) / N_docs
# P(pos) = pos_docs / N_docs

def compute_pmi(word, doc_freq_class, class_docs, N_docs):
    """计算PMI值"""
    p_word_class = doc_freq_class[word] / N_docs
    total_doc_freq = pos_doc_freq[word] + neg_doc_freq[word]
    p_word = total_doc_freq / N_docs
    p_class = class_docs / N_docs
    if p_word_class == 0 or p_word == 0:
        return -float('inf')
    return math.log(p_word_class / (p_word * p_class))

# 只考虑出现次数>=50的词（过滤低频噪声）
min_freq = 50
candidate_words = set()
for word, freq in global_freq.items():
    if freq >= min_freq and len(word) > 1:
        candidate_words.add(word)

# 计算正向PMI
pos_pmi = {}
for word in candidate_words:
    pmi_val = compute_pmi(word, pos_doc_freq, pos_docs, N_docs)
    pos_pmi[word] = pmi_val

pos_pmi_top10 = sorted(pos_pmi.items(), key=lambda x: x[1], reverse=True)[:10]
print("\n正向情感中PMI前十大的词：")
for word, pmi in pos_pmi_top10:
    print(f"  {word}: PMI={pmi:.4f} (正向文档频次={pos_doc_freq[word]}, 负向文档频次={neg_doc_freq[word]})")

# 计算负向PMI
neg_pmi = {}
for word in candidate_words:
    pmi_val = compute_pmi(word, neg_doc_freq, neg_docs, N_docs)
    neg_pmi[word] = pmi_val

neg_pmi_top10 = sorted(neg_pmi.items(), key=lambda x: x[1], reverse=True)[:10]
print("\n负向情感中PMI前十大的词：")
for word, pmi in neg_pmi_top10:
    print(f"  {word}: PMI={pmi:.4f} (正向文档频次={pos_doc_freq[word]}, 负向文档频次={neg_doc_freq[word]})")


# ============================================================
# 保存分析结果为JSON，供后续生成Markdown报告
# ============================================================
results = {
    'stats': stats,
    'text_length': {
        'avg_char_len': round(train_df['text_len'].mean(), 1),
        'median_char_len': round(train_df['text_len'].median(), 1),
        'min_char_len': int(train_df['text_len'].min()),
        'max_char_len': int(train_df['text_len'].max()),
        'avg_word_count': round(train_df['word_count'].mean(), 1),
        'median_word_count': round(train_df['word_count'].median(), 1),
        'pos_avg_char_len': round(train_df[train_df['label']==1]['text_len'].mean(), 1),
        'pos_avg_word_count': round(train_df[train_df['label']==1]['word_count'].mean(), 1),
        'neg_avg_char_len': round(train_df[train_df['label']==0]['text_len'].mean(), 1),
        'neg_avg_word_count': round(train_df[train_df['label']==0]['word_count'].mean(), 1),
    },
    'pos_top10': pos_top10,
    'neg_top10': neg_top10,
    'pos_pmi_top10': [(w, round(p, 4), int(pos_doc_freq[w]), int(neg_doc_freq[w])) for w, p in pos_pmi_top10],
    'neg_pmi_top10': [(w, round(p, 4), int(pos_doc_freq[w]), int(neg_doc_freq[w])) for w, p in neg_pmi_top10],
}

with open('output_phase1_data.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("\n分析结果已保存到 output_phase1_data.json")
