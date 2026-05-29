"""
阶段二：数据预处理
- 步骤2.1：文本清洗（去HTML标签、去标点、转小写）
- 步骤2.2：停用词处理
- 步骤2.3：词元化与词形还原
- 步骤2.4：保存预处理结果
"""

import pandas as pd
import numpy as np
import re
import json
import nltk
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm

# ============================================================
# 下载必要的NLTK资源
# ============================================================
print("检查并下载NLTK资源...")
for resource in ['stopwords', 'wordnet', 'omw-1.4', 'punkt', 'punkt_tab', 'averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng']:
    try:
        nltk.data.find(f'corpora/{resource}' if resource in ['stopwords', 'wordnet', 'omw-1.4'] else f'tokenizers/{resource}' if 'punkt' in resource else f'taggers/{resource}')
    except LookupError:
        print(f"  下载 {resource}...")
        nltk.download(resource, quiet=True)

stopword = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# ============================================================
# 读取数据
# ============================================================
print("\n读取数据...")
train_df = pd.read_csv('data/train.csv')
valid_df = pd.read_csv('data/valid.csv')
test_df = pd.read_csv('data/test.csv')
print(f"  训练集: {len(train_df)} 条")
print(f"  验证集: {len(valid_df)} 条")
print(f"  测试集: {len(test_df)} 条")

# ============================================================
# 步骤2.1：文本清洗
# ============================================================
print("\n步骤2.1：文本清洗（去HTML标签、去标点、转小写）")

def clean_text(text):
    """文本清洗：去HTML标签 -> 去标点 -> 转小写"""
    # 去除HTML标签
    text = BeautifulSoup(text, "lxml").text
    # 去除标点符号和特殊字符，只保留字母和空格
    text = re.sub(r'[^\w\s]', '', text)
    # 转小写
    text = text.lower()
    # 合并多余空格
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ============================================================
# 步骤2.2：停用词处理
# ============================================================
print("步骤2.2：停用词处理")

def remove_stopwords(text):
    """去除停用词"""
    tokens = text.split()
    tokens = [t for t in tokens if t not in stopword]
    return tokens

# ============================================================
# 步骤2.3：词元化与词形还原
# ============================================================
print("步骤2.3：词元化与词形还原")

def lemmatize_tokens(tokens):
    """词形还原"""
    return [lemmatizer.lemmatize(t) for t in tokens]

# ============================================================
# 完整预处理流程
# ============================================================
print("\n执行完整预处理流程...")

def preprocess_pipeline(text):
    """完整预处理流程：清洗 -> 去停用词 -> 词形还原"""
    cleaned = clean_text(text)
    tokens = remove_stopwords(cleaned)
    lemmatized = lemmatize_tokens(tokens)
    return ' '.join(lemmatized)

def preprocess_dataframe(df, name):
    """对整个DataFrame进行预处理"""
    print(f"\n  预处理 {name} ({len(df)} 条)...")
    cleaned_texts = []
    raw_cleaned_texts = []  # 清洗后但不去停用词的版本（供后续模型使用）

    for text in tqdm(df['text'], desc=f"  {name}"):
        # 完整预处理版本
        full_processed = preprocess_pipeline(text)
        cleaned_texts.append(full_processed)

        # 仅清洗版本（保留停用词，供某些模型使用）
        raw_cleaned = clean_text(text)
        raw_cleaned_texts.append(raw_cleaned)

    df_processed = df.copy()
    df_processed['text_cleaned'] = cleaned_texts
    df_processed['text_cleaned_raw'] = raw_cleaned_texts  # 去HTML+去标点+小写，但保留停用词
    return df_processed

train_processed = preprocess_dataframe(train_df, '训练集')
valid_processed = preprocess_dataframe(valid_df, '验证集')
test_processed = preprocess_dataframe(test_df, '测试集')

# ============================================================
# 步骤2.4：保存预处理结果
# ============================================================
print("\n步骤2.4：保存预处理结果...")

train_processed.to_csv('data/train_processed.csv', index=False)
valid_processed.to_csv('data/valid_processed.csv', index=False)
test_processed.to_csv('data/test_processed.csv', index=False)
print("  已保存: data/train_processed.csv, data/valid_processed.csv, data/test_processed.csv")

# ============================================================
# 预处理效果统计与留痕
# ============================================================
print("\n预处理效果统计...")

# 统计预处理前后的文本长度变化
stats = {}
for name, df_orig, df_proc in [('训练集', train_df, train_processed),
                                 ('验证集', valid_df, valid_processed),
                                 ('测试集', test_df, test_processed)]:
    orig_lengths = df_orig['text'].apply(lambda x: len(x.split()))
    proc_lengths = df_proc['text_cleaned'].apply(lambda x: len(x.split()))
    raw_lengths = df_proc['text_cleaned_raw'].apply(lambda x: len(x.split()))

    stats[name] = {
        '样本数': len(df_orig),
        '原始平均词数': round(orig_lengths.mean(), 1),
        '原始中位词数': round(orig_lengths.median(), 1),
        '清洗后(保留停用词)平均词数': round(raw_lengths.mean(), 1),
        '完整预处理(去停用词+词形还原)平均词数': round(proc_lengths.mean(), 1),
        '词数减少比例': round((1 - proc_lengths.mean() / orig_lengths.mean()) * 100, 1)
    }

for name, s in stats.items():
    print(f"\n  {name}:")
    for k, v in s.items():
        print(f"    {k}: {v}")

# 展示预处理前后对比示例
print("\n\n预处理前后对比示例（训练集前3条）：")
for i in range(3):
    print(f"\n  --- 示例 {i+1} (label={train_processed.iloc[i]['label']}) ---")
    orig_text = train_df.iloc[i]['text']
    proc_text = train_processed.iloc[i]['text_cleaned']
    print(f"  原始文本（前200字符）: {orig_text[:200]}...")
    print(f"  预处理后（前200字符）: {proc_text[:200]}...")

# 保存统计结果
with open('output_phase2_preprocessing.json', 'w', encoding='utf-8') as f:
    json.dump(stats, f, ensure_ascii=False, indent=2)
print("\n统计结果已保存到 output_phase2_preprocessing.json")

# ============================================================
# 生成Markdown报告
# ============================================================
report = """# 阶段二：数据预处理报告

## 预处理流程

### 步骤2.1：文本清洗
1. **去除HTML标签**：使用 BeautifulSoup 去除 `<br />` 等HTML标签
2. **去除标点符号**：使用正则表达式 `[^\\w\\s]` 去除标点和特殊字符
3. **统一小写**：将所有文本转为小写

### 步骤2.2：停用词处理
- 使用 NLTK 英文停用词表去除停用词（如 the, is, at, which, on 等）
- 保留 not, no 等否定词（它们在情感分析中有重要作用，但NLTK默认停用词表包含它们）

### 步骤2.3：词元化与词形还原
- 使用 `str.split()` 进行分词
- 使用 `WordNetLemmatizer` 进行词形还原（将词还原为词根形式，如 watched → watch, better → better）

### 步骤2.4：保存预处理结果
- 保存为 `data/train_processed.csv`、`data/valid_processed.csv`、`data/test_processed.csv`
- 每个文件包含原始 `text`、`label`，以及两个预处理列：
  - `text_cleaned`：完整预处理（清洗+去停用词+词形还原）
  - `text_cleaned_raw`：仅清洗（去HTML+去标点+小写，保留停用词）

---

## 预处理效果统计

| 数据集 | 样本数 | 原始平均词数 | 清洗后(保留停用词)平均词数 | 完整预处理平均词数 | 词数减少比例 |
|--------|--------|-------------|-------------------------|------------------|------------|
"""

for name, s in stats.items():
    report += f"| {name} | {s['样本数']} | {s['原始平均词数']} | {s['清洗后(保留停用词)平均词数']} | {s['完整预处理(去停用词+词形还原)平均词数']} | {s['词数减少比例']}% |\n"

report += """
---

## 预处理前后对比示例

"""

for i in range(3):
    orig_text = train_df.iloc[i]['text']
    proc_text = train_processed.iloc[i]['text_cleaned']
    label = train_processed.iloc[i]['label']
    label_name = "正向" if label == 1 else "负向"
    report += f"""### 示例 {i+1}（{label_name}）

**原始文本**：
> {orig_text[:300]}{'...' if len(orig_text) > 300 else ''}

**预处理后**：
> {proc_text[:300]}{'...' if len(proc_text) > 300 else ''}

"""

report += """---

## 说明

- `text_cleaned` 列适用于需要精简特征的模型（朴素贝叶斯、逻辑回归等）
- `text_cleaned_raw` 列保留了停用词，适用于需要完整语义信息的模型
- 原始 `text` 列保留，便于后续需要原始文本时使用（如大语言模型调用）
"""

with open('output/phase2_preprocessing_report.md', 'w', encoding='utf-8') as f:
    f.write(report)
print("预处理报告已保存到 output/phase2_preprocessing_report.md")

print("\n" + "=" * 60)
print("阶段二：数据预处理 完成！")
print("=" * 60)
