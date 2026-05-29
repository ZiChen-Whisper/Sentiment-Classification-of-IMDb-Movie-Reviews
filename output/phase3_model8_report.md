# 模型8：大语言模型（DeepSeek）报告

## 模型8a：Few-shot Prompt 策略

### Prompt设计策略
- **策略**：Few-shot Learning（6个示例，3正3负）
- **模型**：DeepSeek Chat
- **测试数据量**：200条测试集数据
- **温度**：0.0（确保输出稳定性）

### Prompt模板
```
You are a sentiment analysis expert. Classify the following movie review as either "positive" or "negative".

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
Review: {text}
Answer:
```

### 测试结果
- **有效预测数**: 195/200（5条因API连接问题未获取结果）
- **准确率**: 0.9538 (186/195)

| 指标 | 值 |
|------|-----|
| Accuracy | 0.9538 |
| Macro Precision | 0.9554 |
| Macro Recall | 0.9547 |
| Macro F1 | 0.9538 |
| Weighted F1 | 0.9538 |

### 分析
- Few-shot策略在200条测试数据上取得了95.38%的准确率，远超所有传统机器学习方法
- 正向评论的Precision高达0.99，说明模型对正向判断非常准确
- 负向评论的Recall高达0.99，说明模型很少遗漏负向评论
- 部分API调用出现连接错误，但重试机制保证了大部分数据的成功获取

---

## 模型8b：结构化输出（框架表示法）

### Prompt设计策略
- **策略**：结构化JSON输出，包含主题、情感、关键短语、方面级情感
- **模型**：DeepSeek Chat
- **测试数据量**：20条测试集数据

### 框架表示设计
```json
{
  "topic": ["主题1", "主题2"],
  "sentiment": "positive/negative",
  "sentiment_score": 0.0-1.0,
  "key_phrases": ["关键短语1", "关键短语2"],
  "aspects": [
    {"aspect": "方面名", "sentiment": "positive/negative", "evidence": "原文证据"}
  ]
}
```

### 测试结果
- **准确率**: 0.9000 (18/20)

### 主题分布（Top 10）
| 主题 | 出现次数 |
|------|---------|
| plot | 17 |
| acting | 14 |
| directing | 5 |
| visual effects | 3 |
| music | 3 |
| casting | 2 |
| character development | 2 |
| visual style | 2 |
| historical accuracy | 2 |
| overall quality | 2 |

### 方面情感分布
| 情感 | 出现次数 |
|------|---------|
| 正向 | 40 |
| 负向 | 41 |

### 结构化输出示例

**示例1**（真实标签: 正向, 预测: 负向 - 误分类）：
- 主题: educational content, production values, characters, merchandise, live show
- 情感: negative (置信度: 0.4)
- 关键短语: not much educational content, average show, don't buy into the merchandise
- 方面级分析:
  - characters: positive (interesting, vibrant with primary colours)
  - educational content: negative (not much educational content)
  - production values: positive (well produced with high production values)

**示例2**（真实标签: 正向, 预测: 正向）：
- 主题: acting, plot, story adaptation, characterization, realism
- 情感: positive (置信度: 0.92)
- 关键短语: great performances, exceptional good adaptation, must see movie
- 方面级分析:
  - acting: positive (great performances from the star cast)
  - plot: positive (The adaptation of the story was exceptional)
  - characterization: positive (characterization was exceptional good)

### 结构化输出分析

1. **主题提取**：大语言模型能够准确识别评论中讨论的主要方面。plot（剧情）和acting（演技）是最常被提及的主题，这与电影评论的特点一致。

2. **情感判断一致性**：结构化输出中的整体情感与方面级情感基本一致。当评论中多数方面为正向时，整体情感通常为正向。

3. **方面级细粒度分析**：模型可以对评论中的不同方面给出不同的情感判断。例如，某条评论对角色设计给出正面评价，但对教育内容给出负面评价，体现了细粒度分析能力。

4. **置信度校准**：模型的sentiment_score与预测准确性有一定关联，高置信度的预测通常更准确。低置信度（如0.4）的样本更容易误分类。

5. **关键短语提取**：模型能够提取出表达情感的关键短语，这些短语通常包含明确的情感词或评价性表达。

6. **误分类分析**：少数误分类样本通常是因为评论中混合了正负情感（如"角色设计好但教育内容不足"），导致整体情感判断困难。
