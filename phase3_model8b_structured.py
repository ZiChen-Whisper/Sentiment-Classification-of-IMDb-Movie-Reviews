"""
模型8b修复版：DeepSeek 结构化输出（20条测试数据）
修复max_tokens过小导致JSON被截断的问题
"""

import pandas as pd
import numpy as np
import json
import time
from openai import OpenAI
from sklearn.metrics import classification_report

# 配置DeepSeek API
API_KEY = "sk-6322a9e69a1f44c293dd1265a7c87119"
client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")

# 读取测试数据
print("读取测试数据...")
test_df = pd.read_csv('data/test.csv')

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

def call_deepseek(prompt, max_retries=3, max_tokens=500):
    """调用DeepSeek API"""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"    API调用失败 (attempt {attempt+1}): {e}")
            time.sleep(5)
    return None

NUM_STRUCTURED = 20
structured_samples = test_df.sample(n=NUM_STRUCTURED, random_state=123)

structured_results = []

print(f"开始测试 {NUM_STRUCTURED} 条数据（结构化输出）...")
for idx, (_, row) in enumerate(structured_samples.iterrows()):
    text = str(row['text'])[:1000]
    label = row['label']

    prompt = structured_prompt_template.format(review=text)
    response = call_deepseek(prompt, max_retries=3, max_tokens=500)

    if response:
        try:
            json_str = response
            if '```json' in response:
                json_str = response.split('```json')[1].split('```')[0]
            elif '```' in response:
                json_str = response.split('```')[1].split('```')[0]

            parsed = json.loads(json_str.strip())

            sentiment = parsed.get('sentiment', '').lower()
            pred = 1 if 'positive' in sentiment else 0 if 'negative' in sentiment else -1

            structured_results.append({
                'text': text[:200],
                'true_label': label,
                'pred_label': pred,
                'parsed_output': parsed
            })

        except json.JSONDecodeError:
            pred_str = response.lower()
            pred = 1 if 'positive' in pred_str else 0 if 'negative' in pred_str else -1
            structured_results.append({
                'text': text[:200],
                'true_label': label,
                'pred_label': pred,
                'parsed_output': {'raw_response': response[:500]}
            })

    if (idx + 1) % 5 == 0:
        print(f"  已完成 {idx+1}/{NUM_STRUCTURED}")

    time.sleep(1.0)

# 分析结构化输出结果
print("\n结构化输出分析:")

# 准确率
valid_structured = [r for r in structured_results if r['pred_label'] != -1]
if valid_structured:
    correct = sum(1 for r in valid_structured if r['true_label'] == r['pred_label'])
    acc = correct / len(valid_structured)
    print(f"  准确率: {acc:.4f} ({correct}/{len(valid_structured)})")

# 主题分布
topic_counts = {}
aspect_sentiment_counts = {'positive': 0, 'negative': 0}
for r in structured_results:
    output = r['parsed_output']
    if isinstance(output, dict) and 'raw_response' not in output:
        topics = output.get('topic', [])
        if isinstance(topics, list):
            for t in topics:
                topic_counts[t] = topic_counts.get(t, 0) + 1
        elif isinstance(topics, str):
            topic_counts[topics] = topic_counts.get(topics, 0) + 1

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

# 展示几个结构化输出示例
print("\n  结构化输出示例:")
for i, r in enumerate(structured_results[:3]):
    output = r['parsed_output']
    if isinstance(output, dict) and 'raw_response' not in output:
        print(f"\n  示例 {i+1} (真实标签: {'正向' if r['true_label']==1 else '负向'}, 预测: {'正向' if r['pred_label']==1 else '负向' if r['pred_label']==0 else '未知'}):")
        print(f"    主题: {output.get('topic', [])}")
        print(f"    情感: {output.get('sentiment', 'N/A')} (置信度: {output.get('sentiment_score', 'N/A')})")
        print(f"    关键短语: {output.get('key_phrases', [])}")
        aspects = output.get('aspects', [])
        if aspects:
            print(f"    方面级分析:")
            for a in aspects[:3]:
                print(f"      - {a.get('aspect', 'N/A')}: {a.get('sentiment', 'N/A')} ({a.get('evidence', 'N/A')[:50]})")

# 保存结果
with open('output/llm_structured_results.json', 'w', encoding='utf-8') as f:
    json.dump(structured_results, f, ensure_ascii=False, indent=2)
print("\n  结构化输出结果已保存到 output/llm_structured_results.json")

# 更新模型8结果文件
try:
    with open('output_phase3_model8.json', 'r', encoding='utf-8') as f:
        model8_results = json.load(f)
except:
    model8_results = {}

model8_results['模型8b_结构化输出'] = {
    'accuracy': acc if 'acc' in dir() else 0,
    'num_samples': NUM_STRUCTURED,
    'prompt_strategy': 'structured JSON output with aspect-based sentiment',
    'topic_distribution': dict(sorted(topic_counts.items(), key=lambda x: -x[1])[:10]),
    'aspect_sentiment_distribution': aspect_sentiment_counts
}

with open('output_phase3_model8.json', 'w', encoding='utf-8') as f:
    json.dump(model8_results, f, ensure_ascii=False, indent=2)
print("  模型8结果已更新到 output_phase3_model8.json")

print("\n" + "=" * 60)
print("模型8b结构化输出 完成！")
print("=" * 60)
