# 阶段二：数据预处理报告

## 预处理流程

### 步骤2.1：文本清洗
1. **去除HTML标签**：使用 BeautifulSoup 去除 `<br />` 等HTML标签
2. **去除标点符号**：使用正则表达式 `[^\w\s]` 去除标点和特殊字符
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
| 训练集 | 40000 | 231.3 | 226.4 | 119.9 | 48.2% |
| 验证集 | 5000 | 228.9 | 224.0 | 118.6 | 48.2% |
| 测试集 | 5000 | 231.9 | 227.0 | 120.3 | 48.1% |

---

## 预处理前后对比示例

### 示例 1（负向）

**原始文本**：
> I grew up (b. 1965) watching and loving the Thunderbirds. All my mates at school watched. We played "Thunderbirds" before school, during lunch and after school. We all wanted to be Virgil or Scott. No one wanted to be Alan. Counting down from 5 became an art form. I took my children to see the movie...

**预处理后**：
> grew b 1965 watching loving thunderbird mate school watched played thunderbird school lunch school wanted virgil scott one wanted alan counting 5 became art form took child see movie hoping would get glimpse loved child bitterly disappointing high point snappy theme tune could compare original score...

### 示例 2（负向）

**原始文本**：
> When I put this movie in my DVD player, and sat down with a coke and some chips, I had some expectations. I was hoping that this movie would contain some of the strong-points of the first movie: Awsome animation, good flowing story, excellent voice cast, funny comedy and a kick-ass soundtrack. But, ...

**预处理后**：
> put movie dvd player sat coke chip expectation hoping movie would contain strongpoints first movie awsome animation good flowing story excellent voice cast funny comedy kickass soundtrack disappointment found atlantis milo return read review first might let following paragraph directed seen first mo...

### 示例 3（负向）

**原始文本**：
> Why do people who do not know what a particular time in the past was like feel the need to try to define that time for others? Replace Woodstock with the Civil War and the Apollo moon-landing with the Titanic sinking and you've got as realistic a flick as this formulaic soap opera populated entirely...

**预处理后**：
> people know particular time past like feel need try define time others replace woodstock civil war apollo moonlanding titanic sinking youve got realistic flick formulaic soap opera populated entirely lowlife trash kid young allowed go woodstock failed grade school composition ill show old meany ill ...

---

## 说明

- `text_cleaned` 列适用于需要精简特征的模型（朴素贝叶斯、逻辑回归等）
- `text_cleaned_raw` 列保留了停用词，适用于需要完整语义信息的模型
- 原始 `text` 列保留，便于后续需要原始文本时使用（如大语言模型调用）
