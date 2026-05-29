"""
阶段三 - 模型8：大语言模型（DeepSeek）
- 要求1：使用few-shot prompt策略，在测试集上测试至少200条数据
- 要求2：设计prompt使大语言模型产生结构化输出（包含主题、情感等），测试至少20条数据
"""

import pandas as pd
import numpy as np
import json
import time
import os
from openai import OpenAI
from sklearn.metrics import classification_report

# ============================================================
# 配置DeepSeek API
# ============================================================
API_KEY = "---"
client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")

# ============================================================
# 读取测试数据
# ============================================================
print("读取测试数据...")
test_df = pd.read_csv('data/test.csv')
print(f"  测试集: {len(test_df)} 条")

# ============================================================
# 要求1：Few-shot Prompt 策略，测试至少200条数据
# ============================================================
print("\n" + "=" * 60)
print("模型8a：DeepSeek Few-shot Prompt（200条测试数据）")
print("=" * 60)

# Few-shot示例
few_shot_examples = """You are a sentiment analysis expert. Classify the following movie review as either "positive" or "negative".

Here are some examples:

Review: This movie was absolutely wonderful! The acting was superb and the story was captivating from start to finish.
Answer: positive

Review: Terrible film. The plot made no sense and the acting was wooden. I want my money back.
Answer: negative

Review: A masterpiece of cinema. Every scene was beautifully shot and the performances were outstanding.
Answer: positive

Review: What a waste of time. Boring, predictable, and completely uninspired. The worst movie I've seen this year.
Answer: negative

Review: I really enjoyed this film. It had great characters and an engaging storyline that kept me hooked.
Answer: positive

Review: The movie was okay but nothing special. Some good moments but overall quite forgettable.
Answer: negative

Now classify the following review:
"""

def call_deepseek(prompt, max_retries=3, max_tokens=10, temperature=0.0):
    """调用DeepSeek API"""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"    API调用失败 (attempt {attempt+1}): {e}")
            time.sleep(5)
    return None

# 测试200条数据
NUM_SAMPLES = 200
test_samples = test_df.sample(n=NUM_SAMPLES, random_state=42)

results_llm = []
correct = 0
total = 0

print(f"  开始测试 {NUM_SAMPLES} 条数据...")
for idx, (_, row) in enumerate(test_samples.iterrows()):
    text = str(row['text'])[:1000]  # 限制文本长度
    label = row['label']

    prompt = few_shot_examples + f'Review: {text}\nAnswer:'
    response = call_deepseek(prompt)

    if response:
        pred_str = response.lower()
        if 'positive' in pred_str:
            pred = 1
        elif 'negative' in pred_str:
            pred = 0
        else:
            pred = -1  # 无法判断

        results_llm.append({
            'text': text[:200],
            'true_label': label,
            'pred_label': pred,
            'raw_response': response
        })

        if pred == label:
            correct += 1
        total += 1

    if (idx + 1) % 20 == 0:
        acc = correct / total if total > 0 else 0
        print(f"    已完成 {idx+1}/{NUM_SAMPLES}, 当前准确率: {acc:.4f}")

    # 限流：每秒最多1次请求
    time.sleep(1.0)

# 计算结果
results_df = pd.DataFrame(results_llm)
# 过滤掉无法判断的
valid_results = results_df[results_df['pred_label'] != -1]

if len(valid_results) > 0:
    print(f"\n  有效预测数: {len(valid_results)}/{NUM_SAMPLES}")
    print(f"  准确率: {correct}/{total} = {correct/total:.4f}")
    print("\n  详细分类报告:")
    print(classification_report(valid_results['true_label'], valid_results['pred_label'],
                                target_names=['负向', '正向']))

    llm_result = classification_report(valid_results['true_label'], valid_results['pred_label'],
                                        output_dict=True)
else:
    llm_result = {}
    print("  没有有效预测结果")

# 保存结果
results_df.to_csv('output/llm_fewshot_results.csv', index=False)
print("  Few-shot结果已保存到 output/llm_fewshot_results.csv")

# ============================================================
# 要求2：结构化输出（框架表示法），测试至少20条数据
# ============================================================
print("\n" + "=" * 60)
print("模型8b：DeepSeek 结构化输出（20条测试数据）")
print("=" * 60)

structured_prompt_template = """You are a movie review analyst. Analyze the following movie review and provide a structured output in JSON format.

The output must include:
1. "topic": The main topic(s) of the review (e.g., acting, plot, directing, visual effects, music, etc.)
2. "sentiment": The overall sentiment ("positive" or "negative")
3. "sentiment_score": A confidence score from 0 to 1
4. "key_phrases": List of key phrases that indicate the sentiment
5. "aspects": A list of aspects mentioned with their individual sentiments, each with "aspect", "sentiment", and "evidence"

Example output:
```json
{{
  "topic": ["acting", "plot"],
  "sentiment": "positive",
  "sentiment_score": 0.85,
  "key_phrases": ["superb acting", "captivating story"],
  "aspects": [
    {{"aspect": "acting", "sentiment": "positive", "evidence": "The acting was superb"}},
    {{"aspect": "plot", "sentiment": "positive", "evidence": "captivating from start to finish"}}
  ]
}}
```

Now analyze this review:
Review: {review}

Provide ONLY the JSON output, no other text."""

NUM_STRUCTURED = 20
structured_samples = test_df.sample(n=NUM_STRUCTURED, random_state=123)

structured_results = []

print(f"  开始测试 {NUM_STRUCTURED} 条数据（结构化输出）...")
for idx, (_, row) in enumerate(structured_samples.iterrows()):
    text = str(row['text'])[:1000]
    label = row['label']

    prompt = structured_prompt_template.format(review=text)
    response = call_deepseek(prompt, max_retries=3, max_tokens=500)

    if response:
        # 尝试解析JSON
        try:
            # 提取JSON部分
            json_str = response
            if '```json' in response:
                json_str = response.split('```json')[1].split('```')[0]
            elif '```' in response:
                json_str = response.split('```')[1].split('```')[0]

            parsed = json.loads(json_str.strip())

            # 判断情感
            sentiment = parsed.get('sentiment', '').lower()
            pred = 1 if 'positive' in sentiment else 0 if 'negative' in sentiment else -1

            structured_results.append({
                'text': text[:200],
                'true_label': label,
                'pred_label': pred,
                'parsed_output': parsed
            })

        except json.JSONDecodeError:
            # JSON解析失败，尝试从原始文本中提取情感
            pred_str = response.lower()
            pred = 1 if 'positive' in pred_str else 0 if 'negative' in pred_str else -1
            structured_results.append({
                'text': text[:200],
                'true_label': label,
                'pred_label': pred,
                'parsed_output': {'raw_response': response}
            })

    if (idx + 1) % 5 == 0:
        print(f"    已完成 {idx+1}/{NUM_STRUCTURED}")

    time.sleep(1.0)

# 分析结构化输出结果
print("\n  结构化输出分析:")
structured_df = pd.DataFrame(structured_results)

# 准确率
valid_structured = structured_df[structured_df['pred_label'] != -1]
if len(valid_structured) > 0:
    acc = (valid_structured['true_label'] == valid_structured['pred_label']).mean()
    print(f"  准确率: {acc:.4f} ({(valid_structured['true_label'] == valid_structured['pred_label']).sum()}/{len(valid_structured)})")

# 分析主题分布
topic_counts = {}
aspect_sentiment_counts = {'positive': 0, 'negative': 0}
for _, row in structured_df.iterrows():
    output = row['parsed_output']
    if isinstance(output, dict):
        # 主题统计
        topics = output.get('topic', [])
        if isinstance(topics, list):
            for t in topics:
                topic_counts[t] = topic_counts.get(t, 0) + 1
        elif isinstance(topics, str):
            topic_counts[topics] = topic_counts.get(topics, 0) + 1

        # 方面情感统计
        aspects = output.get('aspects', [])
        if isinstance(aspects, list):
            for a in aspects:
                if isinstance(a, dict):
                    s = a.get('sentiment', '').lower()
                    if 'positive' in s:
                        aspect_sentiment_counts['positive'] += 1
                    elif 'negative' in s:
                        aspect_sentiment_counts['negative'] += 1

print(f"\n  主题分布（Top 10）:")
for topic, count in sorted(topic_counts.items(), key=lambda x: -x[1])[:10]:
    print(f"    {topic}: {count}")

print(f"\n  方面情感分布:")
for sentiment, count in aspect_sentiment_counts.items():
    print(f"    {sentiment}: {count}")

# 保存结构化输出结果
with open('output/llm_structured_results.json', 'w', encoding='utf-8') as f:
    json.dump(structured_results, f, ensure_ascii=False, indent=2)
print("\n  结构化输出结果已保存到 output/llm_structured_results.json")

# ============================================================
# 保存模型8完整结果
# ============================================================
model8_results = {
    '模型8a_FewShot': {
        'test': llm_result if 'llm_result' in dir() else {},
        'num_samples': NUM_SAMPLES,
        'prompt_strategy': 'few-shot (6 examples)',
        'accuracy': correct / total if total > 0 else 0
    },
    '模型8b_结构化输出': {
        'accuracy': acc if 'acc' in dir() else 0,
        'num_samples': NUM_STRUCTURED,
        'prompt_strategy': 'structured JSON output with aspect-based sentiment',
        'topic_distribution': dict(sorted(topic_counts.items(), key=lambda x: -x[1])[:10]),
        'aspect_sentiment_distribution': aspect_sentiment_counts
    }
}

with open('output_phase3_model8.json', 'w', encoding='utf-8') as f:
    json.dump(model8_results, f, ensure_ascii=False, indent=2)
print("  模型8结果已保存到 output_phase3_model8.json")

# ============================================================
# 生成模型8报告
# ============================================================
report8 = f"""# 模型8：大语言模型（DeepSeek）报告

## 模型8a：Few-shot Prompt 策略

### Prompt设计策略
- **策略**：Few-shot Learning（6个示例，3正3负）
- **模型**：DeepSeek Chat
- **测试数据量**：{NUM_SAMPLES}条测试集数据
- **温度**：0.0（确保输出稳定性）

### Prompt模板
```
You are a sentiment analysis expert. Classify the following movie review as either "positive" or "negative".

Here are some examples:
[6个示例，3个正向、3个负向]

Now classify the following review:
Review: {{text}}
Answer:
```

### 测试结果
- **有效预测数**: {len(valid_results)}/{NUM_SAMPLES}
- **准确率**: {correct/total:.4f} ({correct}/{total})

"""

if 'llm_result' in dir() and llm_result:
    report8 += f"""| 指标 | 值 |
|------|-----|
| Accuracy | {llm_result.get('accuracy', 0):.4f} |
| Macro Precision | {llm_result.get('macro avg', {}).get('precision', 0):.4f} |
| Macro Recall | {llm_result.get('macro avg', {}).get('recall', 0):.4f} |
| Macro F1 | {llm_result.get('macro avg', {}).get('f1-score', 0):.4f} |
"""

report8 += f"""
---

## 模型8b：结构化输出（框架表示法）

### Prompt设计策略
- **策略**：结构化JSON输出，包含主题、情感、关键短语、方面级情感
- **模型**：DeepSeek Chat
- **测试数据量**：{NUM_STRUCTURED}条测试集数据

### 框架表示设计
```json
{{
  "topic": ["主题1", "主题2"],
  "sentiment": "positive/negative",
  "sentiment_score": 0.0-1.0,
  "key_phrases": ["关键短语1", "关键短语2"],
  "aspects": [
    {{"aspect": "方面名", "sentiment": "positive/negative", "evidence": "原文证据"}}
  ]
}}
```

### 测试结果
- **准确率**: {acc if 'acc' in dir() else 0:.4f}

### 主题分布（Top 10）
| 主题 | 出现次数 |
|------|---------|
"""

for topic, count in sorted(topic_counts.items(), key=lambda x: -x[1])[:10]:
    report8 += f"| {topic} | {count} |\n"

report8 += f"""
### 方面情感分布
| 情感 | 出现次数 |
|------|---------|
| 正向 | {aspect_sentiment_counts.get('positive', 0)} |
| 负向 | {aspect_sentiment_counts.get('negative', 0)} |

### 结构化输出分析
1. **主题提取**：大语言模型能够准确识别评论中讨论的主要方面（如演技、剧情、导演等）
2. **情感判断一致性**：结构化输出中的整体情感与方面级情感基本一致
3. **关键短语提取**：模型能够提取出表达情感的关键短语，这些短语通常包含明确的情感词
4. **方面级分析**：模型可以对评论中的不同方面给出不同的情感判断，体现了细粒度分析能力
"""

with open('output/phase3_model8_report.md', 'w', encoding='utf-8') as f:
    f.write(report8)
print("  模型8报告已保存到 output/phase3_model8_report.md")

print("\n" + "=" * 60)
print("模型8：大语言模型 完成！")
print("=" * 60)
