"""
阶段三：模型构建（模型1-7）
- 模型1：基于规则的产生式系统（SentiWordNet）
- 模型2：朴素贝叶斯（词频特征）
- 模型3：逻辑回归（词频特征）
- 模型4：逻辑回归（TF-IDF特征）
- 模型5：逻辑回归 + 特征选择（SelectKBest chi2, K=200）
- 模型6：逻辑回归 + 特征选择（SelectKBest chi2, K=2000）
- 模型7：逻辑回归 + 新特征设计
"""

import pandas as pd
import numpy as np
import json
import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from scipy.sparse import hstack
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 下载必要的NLTK资源
# ============================================================
print("检查并下载NLTK资源...")
for resource in ['punkt', 'punkt_tab', 'averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng',
                 'stopwords', 'wordnet', 'omw-1.4', 'sentiwordnet']:
    try:
        if resource in ['stopwords', 'wordnet', 'omw-1.4', 'sentiwordnet']:
            nltk.data.find(f'corpora/{resource}')
        elif 'punkt' in resource:
            nltk.data.find(f'tokenizers/{resource}')
        else:
            nltk.data.find(f'taggers/{resource}')
    except LookupError:
        print(f"  下载 {resource}...")
        nltk.download(resource, quiet=True)

# ============================================================
# 读取预处理后的数据
# ============================================================
print("\n读取预处理后的数据...")
train_df = pd.read_csv('data/train_processed.csv')
valid_df = pd.read_csv('data/valid_processed.csv')
test_df = pd.read_csv('data/test_processed.csv')
print(f"  训练集: {len(train_df)} 条")
print(f"  验证集: {len(valid_df)} 条")
print(f"  测试集: {len(test_df)} 条")

# 准备数据
X_train_text = train_df['text_cleaned'].fillna('')
X_valid_text = valid_df['text_cleaned'].fillna('')
X_test_text = test_df['text_cleaned'].fillna('')

X_train_raw = train_df['text_cleaned_raw'].fillna('')
X_valid_raw = valid_df['text_cleaned_raw'].fillna('')
X_test_raw = test_df['text_cleaned_raw'].fillna('')

y_train = train_df['label']
y_valid = valid_df['label']
y_test = test_df['label']

# 存储所有模型结果
all_results = {}

# ============================================================
# 模型1：基于规则的产生式系统（SentiWordNet）
# ============================================================
print("\n" + "=" * 60)
print("模型1：基于规则的产生式系统（SentiWordNet）")
print("=" * 60)

def penn_to_wn(tag):
    """将Penn Treebank词性标签映射为WordNet词性"""
    if tag.startswith('J'):
        return wn.ADJ
    elif tag.startswith('N'):
        return wn.NOUN
    elif tag.startswith('R'):
        return wn.ADV
    elif tag.startswith('V'):
        return wn.VERB
    return None

def sentiwordnet_predict(text):
    """使用SentiWordNet计算文本情感得分，返回预测标签和得分"""
    tokens = word_tokenize(str(text))
    tagged = nltk.pos_tag(tokens)

    pos_score = 0.0
    neg_score = 0.0
    pos_word_count = 0
    neg_word_count = 0

    for word, tag in tagged:
        wn_tag = penn_to_wn(tag)
        if wn_tag is None:
            continue
        synsets = wn.synsets(word, pos=wn_tag)
        if not synsets:
            continue
        # 取第一个同义词集
        synset = synsets[0]
        swn_synset = swn.senti_synset(synset.name())
        p = swn_synset.pos_score()
        n = swn_synset.neg_score()
        pos_score += p
        neg_score += n
        if p > n:
            pos_word_count += 1
        elif n > p:
            neg_word_count += 1

    # 根据总分判断情感
    prediction = 1 if pos_score >= neg_score else 0
    return prediction, pos_score, neg_score, pos_word_count, neg_word_count

# 对验证集和测试集预测
print("  预测验证集...")
valid_preds_swn = []
valid_swn_scores = []
for text in tqdm(X_valid_raw, desc="  验证集"):
    pred, pos_s, neg_s, pos_w, neg_w = sentiwordnet_predict(text)
    valid_preds_swn.append(pred)
    valid_swn_scores.append({'pos_score': pos_s, 'neg_score': neg_s, 'pos_word_count': pos_w, 'neg_word_count': neg_w})

print("  预测测试集...")
test_preds_swn = []
test_swn_scores = []
for text in tqdm(X_test_raw, desc="  测试集"):
    pred, pos_s, neg_s, pos_w, neg_w = sentiwordnet_predict(text)
    test_preds_swn.append(pred)
    test_swn_scores.append({'pos_score': pos_s, 'neg_score': neg_s, 'pos_word_count': pos_w, 'neg_word_count': neg_w})

valid_preds_swn = np.array(valid_preds_swn)
test_preds_swn = np.array(test_preds_swn)

# 评估
print("\n  验证集结果:")
print(classification_report(y_valid, valid_preds_swn, target_names=['负向', '正向']))
print("  测试集结果:")
print(classification_report(y_test, test_preds_swn, target_names=['负向', '正向']))

all_results['模型1_SentiWordNet'] = {
    'valid': classification_report(y_valid, valid_preds_swn, output_dict=True),
    'test': classification_report(y_test, test_preds_swn, output_dict=True)
}

# 保存SentiWordNet得分供后续分析
swn_analysis = pd.DataFrame({
    'label': y_test,
    'pred': test_preds_swn,
    'pos_score': [s['pos_score'] for s in test_swn_scores],
    'neg_score': [s['neg_score'] for s in test_swn_scores],
    'pos_word_count': [s['pos_word_count'] for s in test_swn_scores],
    'neg_word_count': [s['neg_word_count'] for s in test_swn_scores]
})
swn_analysis.to_csv('output/sentiwordnet_test_scores.csv', index=False)
print("  SentiWordNet测试集得分已保存到 output/sentiwordnet_test_scores.csv")

# ============================================================
# 模型2：朴素贝叶斯（词频特征）
# ============================================================
print("\n" + "=" * 60)
print("模型2：朴素贝叶斯（词频特征）")
print("=" * 60)

count_vec = CountVectorizer(max_features=50000)
X_train_count = count_vec.fit_transform(X_train_text)
X_valid_count = count_vec.transform(X_valid_text)
X_test_count = count_vec.transform(X_test_text)
print(f"  词表大小: {len(count_vec.vocabulary_)}")

nb_model = MultinomialNB()
nb_model.fit(X_train_count, y_train)

valid_preds_nb = nb_model.predict(X_valid_count)
test_preds_nb = nb_model.predict(X_test_count)

print("\n  验证集结果:")
print(classification_report(y_valid, valid_preds_nb, target_names=['负向', '正向']))
print("  测试集结果:")
print(classification_report(y_test, test_preds_nb, target_names=['负向', '正向']))

all_results['模型2_朴素贝叶斯_Count'] = {
    'valid': classification_report(y_valid, valid_preds_nb, output_dict=True),
    'test': classification_report(y_test, test_preds_nb, output_dict=True)
}

# ============================================================
# 模型3：逻辑回归（词频特征）
# ============================================================
print("\n" + "=" * 60)
print("模型3：逻辑回归（词频特征）")
print("=" * 60)

lr_count = LogisticRegression(max_iter=1000, random_state=42)
lr_count.fit(X_train_count, y_train)

valid_preds_lr_count = lr_count.predict(X_valid_count)
test_preds_lr_count = lr_count.predict(X_test_count)

print("\n  验证集结果:")
print(classification_report(y_valid, valid_preds_lr_count, target_names=['负向', '正向']))
print("  测试集结果:")
print(classification_report(y_test, test_preds_lr_count, target_names=['负向', '正向']))

all_results['模型3_逻辑回归_Count'] = {
    'valid': classification_report(y_valid, valid_preds_lr_count, output_dict=True),
    'test': classification_report(y_test, test_preds_lr_count, output_dict=True)
}

# ============================================================
# 模型4：逻辑回归（TF-IDF特征）
# ============================================================
print("\n" + "=" * 60)
print("模型4：逻辑回归（TF-IDF特征）")
print("=" * 60)

tfidf_vec = TfidfVectorizer(max_features=50000)
X_train_tfidf = tfidf_vec.fit_transform(X_train_text)
X_valid_tfidf = tfidf_vec.transform(X_valid_text)
X_test_tfidf = tfidf_vec.transform(X_test_text)
print(f"  TF-IDF特征维度: {X_train_tfidf.shape[1]}")

lr_tfidf = LogisticRegression(max_iter=1000, random_state=42)
lr_tfidf.fit(X_train_tfidf, y_train)

valid_preds_lr_tfidf = lr_tfidf.predict(X_valid_tfidf)
test_preds_lr_tfidf = lr_tfidf.predict(X_test_tfidf)

print("\n  验证集结果:")
print(classification_report(y_valid, valid_preds_lr_tfidf, target_names=['负向', '正向']))
print("  测试集结果:")
print(classification_report(y_test, test_preds_lr_tfidf, target_names=['负向', '正向']))

all_results['模型4_逻辑回归_TFIDF'] = {
    'valid': classification_report(y_valid, valid_preds_lr_tfidf, output_dict=True),
    'test': classification_report(y_test, test_preds_lr_tfidf, output_dict=True)
}

# ============================================================
# 模型5：逻辑回归 + 特征选择（SelectKBest chi2, K=200）
# ============================================================
print("\n" + "=" * 60)
print("模型5：逻辑回归 + 特征选择（SelectKBest chi2, K=200）")
print("=" * 60)

selector_200 = SelectKBest(chi2, k=200)
X_train_sel200 = selector_200.fit_transform(X_train_tfidf, y_train)
X_valid_sel200 = selector_200.transform(X_valid_tfidf)
X_test_sel200 = selector_200.transform(X_test_tfidf)
print(f"  特征选择后维度: {X_train_sel200.shape[1]}")

lr_sel200 = LogisticRegression(max_iter=1000, random_state=42)
lr_sel200.fit(X_train_sel200, y_train)

valid_preds_sel200 = lr_sel200.predict(X_valid_sel200)
test_preds_sel200 = lr_sel200.predict(X_test_sel200)

print("\n  验证集结果:")
print(classification_report(y_valid, valid_preds_sel200, target_names=['负向', '正向']))
print("  测试集结果:")
print(classification_report(y_test, test_preds_sel200, target_names=['负向', '正向']))

all_results['模型5_逻辑回归_SelectK200'] = {
    'valid': classification_report(y_valid, valid_preds_sel200, output_dict=True),
    'test': classification_report(y_test, test_preds_sel200, output_dict=True)
}

# 输出选中的特征词
selected_features_200 = np.array(tfidf_vec.get_feature_names_out())[selector_200.get_support()]
print(f"\n  选中的前20个特征词: {selected_features_200[:20].tolist()}")

# ============================================================
# 模型6：逻辑回归 + 特征选择（SelectKBest chi2, K=2000）
# ============================================================
print("\n" + "=" * 60)
print("模型6：逻辑回归 + 特征选择（SelectKBest chi2, K=2000）")
print("=" * 60)

selector_2000 = SelectKBest(chi2, k=2000)
X_train_sel2000 = selector_2000.fit_transform(X_train_tfidf, y_train)
X_valid_sel2000 = selector_2000.transform(X_valid_tfidf)
X_test_sel2000 = selector_2000.transform(X_test_tfidf)
print(f"  特征选择后维度: {X_train_sel2000.shape[1]}")

lr_sel2000 = LogisticRegression(max_iter=1000, random_state=42)
lr_sel2000.fit(X_train_sel2000, y_train)

valid_preds_sel2000 = lr_sel2000.predict(X_valid_sel2000)
test_preds_sel2000 = lr_sel2000.predict(X_test_sel2000)

print("\n  验证集结果:")
print(classification_report(y_valid, valid_preds_sel2000, target_names=['负向', '正向']))
print("  测试集结果:")
print(classification_report(y_test, test_preds_sel2000, target_names=['负向', '正向']))

all_results['模型6_逻辑回归_SelectK2000'] = {
    'valid': classification_report(y_valid, valid_preds_sel2000, output_dict=True),
    'test': classification_report(y_test, test_preds_sel2000, output_dict=True)
}

# 输出选中的特征词
selected_features_2000 = np.array(tfidf_vec.get_feature_names_out())[selector_2000.get_support()]
print(f"\n  选中的前30个特征词: {selected_features_2000[:30].tolist()}")

# ============================================================
# 模型7：逻辑回归 + 新特征设计
# ============================================================
print("\n" + "=" * 60)
print("模型7：逻辑回归 + 新特征设计")
print("=" * 60)

# 设计新特征
print("  提取新特征...")

def extract_new_features(df):
    """提取新特征：评论词数、情感词比例、感叹号数量、大写字母比例、否定词数量、平均词长"""
    features = []
    negation_words = {'not', 'no', 'never', 'neither', 'nobody', 'nothing',
                      'nowhere', 'nor', "n't", 'cannot', 'hardly', 'scarcely'}

    for _, row in tqdm(df.iterrows(), total=len(df), desc="  提取特征"):
        text = str(row['text'])  # 使用原始文本
        cleaned = str(row['text_cleaned_raw'])  # 清洗后文本

        tokens = cleaned.split()

        # 特征1：评论词数
        word_count = len(tokens)

        # 特征2：平均词长
        avg_word_len = np.mean([len(w) for w in tokens]) if tokens else 0

        # 特征3：感叹号数量（从原始文本）
        exclamation_count = text.count('!')

        # 特征4：大写字母比例（从原始文本）
        upper_count = sum(1 for c in text if c.isupper())
        upper_ratio = upper_count / len(text) if len(text) > 0 else 0

        # 特征5：否定词数量
        neg_count = sum(1 for t in tokens if t in negation_words)

        # 特征6：情感词比例（使用SentiWordNet快速计算）
        pos_word_count = 0
        neg_word_count = 0
        sentiment_word_count = 0
        tagged = nltk.pos_tag(tokens[:100])  # 限制词数以加速
        for word, tag in tagged:
            wn_tag = penn_to_wn(tag)
            if wn_tag is None:
                continue
            synsets = wn.synsets(word, pos=wn_tag)
            if synsets:
                s = swn.senti_synset(synsets[0].name())
                if s.pos_score() > s.neg_score():
                    pos_word_count += 1
                    sentiment_word_count += 1
                elif s.neg_score() > s.pos_score():
                    neg_word_count += 1
                    sentiment_word_count += 1

        sentiment_ratio = sentiment_word_count / len(tokens) if len(tokens) > 0 else 0
        pos_sentiment_ratio = pos_word_count / len(tokens) if len(tokens) > 0 else 0
        neg_sentiment_ratio = neg_word_count / len(tokens) if len(tokens) > 0 else 0

        features.append({
            'word_count': word_count,
            'avg_word_len': avg_word_len,
            'exclamation_count': exclamation_count,
            'upper_ratio': upper_ratio,
            'negation_count': neg_count,
            'sentiment_ratio': sentiment_ratio,
            'pos_sentiment_ratio': pos_sentiment_ratio,
            'neg_sentiment_ratio': neg_sentiment_ratio
        })

    return pd.DataFrame(features)

train_new_features = extract_new_features(train_df)
valid_new_features = extract_new_features(valid_df)
test_new_features = extract_new_features(test_df)

# 标准化新特征
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
train_new_scaled = scaler.fit_transform(train_new_features)
valid_new_scaled = scaler.transform(valid_new_features)
test_new_scaled = scaler.transform(test_new_features)

print(f"\n  新特征列: {list(train_new_features.columns)}")
print(f"  新特征示例（训练集前5条）:")
print(train_new_features.head())

# 拼接TF-IDF特征和新特征
X_train_combined = hstack([X_train_tfidf, train_new_scaled])
X_valid_combined = hstack([X_valid_tfidf, valid_new_scaled])
X_test_combined = hstack([X_test_tfidf, test_new_scaled])
print(f"  拼接后特征维度: {X_train_combined.shape[1]} (TF-IDF: {X_train_tfidf.shape[1]} + 新特征: {train_new_scaled.shape[1]})")

lr_combined = LogisticRegression(max_iter=1000, random_state=42)
lr_combined.fit(X_train_combined, y_train)

valid_preds_combined = lr_combined.predict(X_valid_combined)
test_preds_combined = lr_combined.predict(X_test_combined)

print("\n  验证集结果:")
print(classification_report(y_valid, valid_preds_combined, target_names=['负向', '正向']))
print("  测试集结果:")
print(classification_report(y_test, test_preds_combined, target_names=['负向', '正向']))

all_results['模型7_逻辑回归_新特征'] = {
    'valid': classification_report(y_valid, valid_preds_combined, output_dict=True),
    'test': classification_report(y_test, test_preds_combined, output_dict=True)
}

# 分析新特征的贡献（仅使用新特征的逻辑回归）
lr_new_only = LogisticRegression(max_iter=1000, random_state=42)
lr_new_only.fit(train_new_scaled, y_train)
test_preds_new_only = lr_new_only.predict(test_new_scaled)
print("\n  仅使用新特征的测试集结果（对比用）:")
print(classification_report(y_test, test_preds_new_only, target_names=['负向', '正向']))

all_results['模型7b_仅新特征'] = {
    'test': classification_report(y_test, test_preds_new_only, output_dict=True)
}

# 新特征系数分析
print("\n  新特征在逻辑回归中的系数:")
feature_names = list(train_new_features.columns)
for name, coef in zip(feature_names, lr_combined.coef_[0][-len(feature_names):]):
    print(f"    {name}: {coef:.4f}")

# ============================================================
# 保存所有模型结果
# ============================================================
print("\n" + "=" * 60)
print("保存结果...")
print("=" * 60)

# 保存JSON结果
with open('output_phase3_models.json', 'w', encoding='utf-8') as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)
print("  模型结果已保存到 output_phase3_models.json")

# 保存特征选择结果
feature_selection_results = {
    'K=200_选中特征词': selected_features_200.tolist(),
    'K=2000_选中特征词前50': selected_features_2000[:50].tolist()
}
with open('output_feature_selection.json', 'w', encoding='utf-8') as f:
    json.dump(feature_selection_results, f, ensure_ascii=False, indent=2)
print("  特征选择结果已保存到 output_feature_selection.json")

# 保存新特征数据
train_new_features.to_csv('output/train_new_features.csv', index=False)
test_new_features.to_csv('output/test_new_features.csv', index=False)
print("  新特征数据已保存到 output/train_new_features.csv 和 output/test_new_features.csv")

# ============================================================
# 生成Markdown报告
# ============================================================
print("\n生成阶段三报告...")

report = """# 阶段三：模型构建报告

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
"""

# 添加各模型测试集结果
for model_name, results in all_results.items():
    if model_name == '模型7b_仅新特征':
        continue  # 这个是辅助对比，不单独列出
    test_r = results['test']
    acc = test_r['accuracy']
    macro_f1 = test_r['macro avg']['f1-score']
    weighted_f1 = test_r['weighted avg']['f1-score']
    precision = test_r['macro avg']['precision']
    recall = test_r['macro avg']['recall']

    report += f"""
### {model_name}

| 指标 | 值 |
|------|-----|
| Accuracy | {acc:.4f} |
| Macro Precision | {precision:.4f} |
| Macro Recall | {recall:.4f} |
| Macro F1 | {macro_f1:.4f} |
| Weighted F1 | {weighted_f1:.4f} |
"""

# 添加特征选择分析
report += """
---

## 特征选择分析

### SelectKBest (K=200) 选中的前20个特征词
"""
report += f"```\n{', '.join(selected_features_200[:20].tolist())}\n```\n"

report += """
### SelectKBest (K=2000) 选中的前30个特征词
"""
report += f"```\n{', '.join(selected_features_2000[:30].tolist())}\n```\n"

report += """
### 特征选择对性能的影响
| K值 | 特征维度 | Accuracy | Macro F1 | Weighted F1 |
|-----|---------|----------|----------|-------------|
"""
for k, model_key in [(50000, '模型4_逻辑回归_TFIDF'), (200, '模型5_逻辑回归_SelectK200'), (2000, '模型6_逻辑回归_SelectK2000')]:
    r = all_results[model_key]['test']
    report += f"| {k} | {k} | {r['accuracy']:.4f} | {r['macro avg']['f1-score']:.4f} | {r['weighted avg']['f1-score']:.4f} |\n"

# 新特征分析
report += """
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
"""
feature_names = list(train_new_features.columns)
coefs = lr_combined.coef_[0][-len(feature_names):]
for name, coef in zip(feature_names, coefs):
    direction = "正向相关" if coef > 0 else "负向相关"
    report += f"- **{name}**: {coef:.4f} ({direction})\n"

report += """
### 新特征对性能的影响

| 模型 | Accuracy | Macro F1 | Weighted F1 |
|------|----------|----------|-------------|
"""
r_tfidf = all_results['模型4_逻辑回归_TFIDF']['test']
r_combined = all_results['模型7_逻辑回归_新特征']['test']
r_new_only = all_results['模型7b_仅新特征']['test']
report += f"| 逻辑回归(TF-IDF) | {r_tfidf['accuracy']:.4f} | {r_tfidf['macro avg']['f1-score']:.4f} | {r_tfidf['weighted avg']['f1-score']:.4f} |\n"
report += f"| 逻辑回归(TF-IDF+新特征) | {r_combined['accuracy']:.4f} | {r_combined['macro avg']['f1-score']:.4f} | {r_combined['weighted avg']['f1-score']:.4f} |\n"
report += f"| 逻辑回归(仅新特征) | {r_new_only['accuracy']:.4f} | {r_new_only['macro avg']['f1-score']:.4f} | {r_new_only['weighted avg']['f1-score']:.4f} |\n"

# 模型对比总表
report += """
---

## 模型对比总表（测试集）

| 模型 | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 |
|------|----------|----------------|-------------|----------|-------------|
"""
for model_name, results in all_results.items():
    if model_name == '模型7b_仅新特征':
        continue
    r = results['test']
    report += f"| {model_name} | {r['accuracy']:.4f} | {r['macro avg']['precision']:.4f} | {r['macro avg']['recall']:.4f} | {r['macro avg']['f1-score']:.4f} | {r['weighted avg']['f1-score']:.4f} |\n"

with open('output/phase3_models_report.md', 'w', encoding='utf-8') as f:
    f.write(report)
print("  阶段三报告已保存到 output/phase3_models_report.md")

print("\n" + "=" * 60)
print("阶段三：模型构建（模型1-7）完成！")
print("=" * 60)
