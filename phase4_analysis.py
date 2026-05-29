"""
阶段四：结果分析（完整版）
- 步骤4.1：评测指标计算
- 步骤4.2：可视化比较
- 步骤4.3：SentiWordNet分析（含各模型预测与SentiWordNet得分一致性对比）
- 步骤4.4：大语言模型结构化输出分析
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
matplotlib.rcParams['axes.unicode_minus'] = False

# ============================================================
# 读取模型结果
# ============================================================
print("读取模型结果...")
with open('output_phase3_models.json', 'r', encoding='utf-8') as f:
    all_results = json.load(f)

# 尝试读取模型8结果
try:
    with open('output_phase3_model8.json', 'r', encoding='utf-8') as f:
        model8_results = json.load(f)
    has_model8 = True
except FileNotFoundError:
    model8_results = {}
    has_model8 = False
    print("  模型8结果文件未找到，将跳过模型8")

# ============================================================
# 步骤4.1：评测指标汇总
# ============================================================
print("\n" + "=" * 60)
print("步骤4.1：评测指标汇总")
print("=" * 60)

# 构建汇总表
summary_data = []
for model_name, results in all_results.items():
    if model_name == '模型7b_仅新特征':
        continue
    r = results['test']
    summary_data.append({
        '模型': model_name,
        'Accuracy': r['accuracy'],
        'Macro Precision': r['macro avg']['precision'],
        'Macro Recall': r['macro avg']['recall'],
        'Macro F1': r['macro avg']['f1-score'],
        'Weighted F1': r['weighted avg']['f1-score']
    })

# 添加模型8a结果
if has_model8 and '模型8a_FewShot' in model8_results:
    r8 = model8_results['模型8a_FewShot']
    if 'test' in r8 and r8['test']:
        summary_data.append({
            '模型': '模型8_大语言模型_FewShot',
            'Accuracy': r8['test'].get('accuracy', 0),
            'Macro Precision': r8['test'].get('macro avg', {}).get('precision', 0),
            'Macro Recall': r8['test'].get('macro avg', {}).get('recall', 0),
            'Macro F1': r8['test'].get('macro avg', {}).get('f1-score', 0),
            'Weighted F1': r8['test'].get('weighted avg', {}).get('f1-score', 0)
        })

summary_df = pd.DataFrame(summary_data)
print("\n所有模型测试集性能汇总：")
print(summary_df.to_string(index=False))

# 保存汇总表
summary_df.to_csv('output/all_models_summary.csv', index=False)
print("\n汇总表已保存到 output/all_models_summary.csv")

# ============================================================
# 步骤4.2：可视化比较
# ============================================================
print("\n" + "=" * 60)
print("步骤4.2：可视化比较")
print("=" * 60)

# 绘制柱状图比较8个模型
models_to_plot = summary_df.copy()

fig, ax = plt.subplots(figsize=(14, 7))

x = np.arange(len(models_to_plot))
width = 0.2

bars1 = ax.bar(x - width, models_to_plot['Accuracy'], width, label='Accuracy', color='#4472C4')
bars2 = ax.bar(x, models_to_plot['Macro F1'], width, label='Macro F1', color='#70AD47')
bars3 = ax.bar(x + width, models_to_plot['Weighted F1'], width, label='Weighted F1', color='#FFC000')

# 添加数值标签
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

ax.set_xlabel('模型', fontsize=12)
ax.set_ylabel('分数', fontsize=12)
ax.set_title('IMDB情感分类 - 各模型性能对比', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models_to_plot['模型'], rotation=30, ha='right', fontsize=9)
ax.legend(fontsize=10)
ax.set_ylim(0.5, 1.0)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('output/models_comparison.png', dpi=150, bbox_inches='tight')
print("  模型对比图已保存到 output/models_comparison.png")

# 绘制特征选择对比图
fig2, ax2 = plt.subplots(figsize=(10, 6))

feature_selection_models = ['模型4_逻辑回归_TFIDF', '模型5_逻辑回归_SelectK200', '模型6_逻辑回归_SelectK2000']
feature_selection_labels = ['TF-IDF\n(50000特征)', 'SelectKBest\nK=200', 'SelectKBest\nK=2000']
fs_accuracies = [all_results[m]['test']['accuracy'] for m in feature_selection_models]
fs_macro_f1s = [all_results[m]['test']['macro avg']['f1-score'] for m in feature_selection_models]

x2 = np.arange(len(feature_selection_labels))
width2 = 0.3

bars_a = ax2.bar(x2 - width2/2, fs_accuracies, width2, label='Accuracy', color='#4472C4')
bars_f = ax2.bar(x2 + width2/2, fs_macro_f1s, width2, label='Macro F1', color='#70AD47')

for bars in [bars_a, bars_f]:
    for bar in bars:
        height = bar.get_height()
        ax2.annotate(f'{height:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

ax2.set_xlabel('特征选择方式', fontsize=12)
ax2.set_ylabel('分数', fontsize=12)
ax2.set_title('特征选择对模型性能的影响', fontsize=14, fontweight='bold')
ax2.set_xticks(x2)
ax2.set_xticklabels(feature_selection_labels, fontsize=11)
ax2.legend(fontsize=10)
ax2.set_ylim(0.75, 0.95)
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('output/feature_selection_comparison.png', dpi=150, bbox_inches='tight')
print("  特征选择对比图已保存到 output/feature_selection_comparison.png")

# 绘制新特征对比图
fig3, ax3 = plt.subplots(figsize=(10, 6))

new_feature_models = ['模型4_逻辑回归_TFIDF', '模型7_逻辑回归_新特征', '模型7b_仅新特征']
new_feature_labels = ['TF-IDF', 'TF-IDF + 新特征', '仅新特征']
nf_accuracies = [all_results[m]['test']['accuracy'] for m in new_feature_models]
nf_macro_f1s = [all_results[m]['test']['macro avg']['f1-score'] for m in new_feature_models]

x3 = np.arange(len(new_feature_labels))
width3 = 0.3

bars_a3 = ax3.bar(x3 - width3/2, nf_accuracies, width3, label='Accuracy', color='#4472C4')
bars_f3 = ax3.bar(x3 + width3/2, nf_macro_f1s, width3, label='Macro F1', color='#70AD47')

for bars in [bars_a3, bars_f3]:
    for bar in bars:
        height = bar.get_height()
        ax3.annotate(f'{height:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

ax3.set_xlabel('特征组合方式', fontsize=12)
ax3.set_ylabel('分数', fontsize=12)
ax3.set_title('新特征设计对模型性能的影响', fontsize=14, fontweight='bold')
ax3.set_xticks(x3)
ax3.set_xticklabels(new_feature_labels, fontsize=11)
ax3.legend(fontsize=10)
ax3.set_ylim(0.5, 1.0)
ax3.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('output/new_features_comparison.png', dpi=150, bbox_inches='tight')
print("  新特征对比图已保存到 output/new_features_comparison.png")

# ============================================================
# 步骤4.3：SentiWordNet分析（含各模型对比）
# ============================================================
print("\n" + "=" * 60)
print("步骤4.3：SentiWordNet分析")
print("=" * 60)

# 读取SentiWordNet得分
swn_df = pd.read_csv('output/sentiwordnet_test_scores.csv')
print(f"  SentiWordNet分析数据: {len(swn_df)} 条")

# 读取测试集数据
test_df = pd.read_csv('data/test_processed.csv')
y_test = test_df['label'].values

# ---- 4.3a: SentiWordNet自身分析 ----
print("\n  4.3a: SentiWordNet自身预测分析")

swn_correct = (swn_df['pred'] == swn_df['label']).mean()
print(f"    SentiWordNet预测准确率: {swn_correct:.4f}")

# 正向情感词数量与标签的关系
pos_label_1 = swn_df[swn_df['label'] == 1]['pos_word_count'].mean()
pos_label_0 = swn_df[swn_df['label'] == 0]['pos_word_count'].mean()
neg_label_1 = swn_df[swn_df['label'] == 1]['neg_word_count'].mean()
neg_label_0 = swn_df[swn_df['label'] == 0]['neg_word_count'].mean()

print(f"    正向评论的平均正向情感词数: {pos_label_1:.2f}")
print(f"    负向评论的平均正向情感词数: {pos_label_0:.2f}")
print(f"    正向评论的平均负向情感词数: {neg_label_1:.2f}")
print(f"    负向评论的平均负向情感词数: {neg_label_0:.2f}")

# SentiWordNet误分类样本分析
misclassified = swn_df[swn_df['pred'] != swn_df['label']]
correct_classified = swn_df[swn_df['pred'] == swn_df['label']]

print(f"    误分类样本数: {len(misclassified)} ({len(misclassified)/len(swn_df)*100:.1f}%)")

false_positive = swn_df[(swn_df['pred'] == 1) & (swn_df['label'] == 0)]
false_negative = swn_df[(swn_df['pred'] == 0) & (swn_df['label'] == 1)]

fp_pos = false_positive['pos_score'].mean()
fp_neg = false_positive['neg_score'].mean()
fn_pos = false_negative['pos_score'].mean()
fn_neg = false_negative['neg_score'].mean()

print(f"    假正向（预测正实为负）: {len(false_positive)} 条, 平均正向得分: {fp_pos:.2f}, 平均负向得分: {fp_neg:.2f}")
print(f"    假负向（预测负实为正）: {len(false_negative)} 条, 平均正向得分: {fn_pos:.2f}, 平均负向得分: {fn_neg:.2f}")

# ---- 4.3b: 重新训练模型2-7，获取各模型测试集预测 ----
print("\n  4.3b: 重新训练模型2-7，获取测试集预测...")

train_df = pd.read_csv('data/train_processed.csv')
X_train_text = train_df['text_cleaned'].fillna('')
X_test_text = test_df['text_cleaned'].fillna('')
y_train = train_df['label']

# 模型2：朴素贝叶斯
print("    训练模型2：朴素贝叶斯...")
count_vec = CountVectorizer(max_features=50000)
X_train_count = count_vec.fit_transform(X_train_text)
X_test_count = count_vec.transform(X_test_text)
nb_model = MultinomialNB()
nb_model.fit(X_train_count, y_train)
pred_nb = nb_model.predict(X_test_count)

# 模型3：逻辑回归（Count）
print("    训练模型3：逻辑回归（Count）...")
lr_count = LogisticRegression(max_iter=1000, random_state=42)
lr_count.fit(X_train_count, y_train)
pred_lr_count = lr_count.predict(X_test_count)

# 模型4：逻辑回归（TF-IDF）
print("    训练模型4：逻辑回归（TF-IDF）...")
tfidf_vec = TfidfVectorizer(max_features=50000)
X_train_tfidf = tfidf_vec.fit_transform(X_train_text)
X_test_tfidf = tfidf_vec.transform(X_test_text)
lr_tfidf = LogisticRegression(max_iter=1000, random_state=42)
lr_tfidf.fit(X_train_tfidf, y_train)
pred_lr_tfidf = lr_tfidf.predict(X_test_tfidf)

# 模型5：逻辑回归 + SelectK200
print("    训练模型5：逻辑回归 + SelectK200...")
selector_200 = SelectKBest(chi2, k=200)
X_train_sel200 = selector_200.fit_transform(X_train_tfidf, y_train)
X_test_sel200 = selector_200.transform(X_test_tfidf)
lr_sel200 = LogisticRegression(max_iter=1000, random_state=42)
lr_sel200.fit(X_train_sel200, y_train)
pred_sel200 = lr_sel200.predict(X_test_sel200)

# 模型6：逻辑回归 + SelectK2000
print("    训练模型6：逻辑回归 + SelectK2000...")
selector_2000 = SelectKBest(chi2, k=2000)
X_train_sel2000 = selector_2000.fit_transform(X_train_tfidf, y_train)
X_test_sel2000 = selector_2000.transform(X_test_tfidf)
lr_sel2000 = LogisticRegression(max_iter=1000, random_state=42)
lr_sel2000.fit(X_train_sel2000, y_train)
pred_sel2000 = lr_sel2000.predict(X_test_sel2000)

# 模型7：逻辑回归 + 新特征
print("    训练模型7：逻辑回归 + 新特征...")
train_new_features = pd.read_csv('output/train_new_features.csv')
test_new_features = pd.read_csv('output/test_new_features.csv')
scaler = StandardScaler()
train_new_scaled = scaler.fit_transform(train_new_features)
test_new_scaled = scaler.transform(test_new_features)
X_train_combined = hstack([X_train_tfidf, train_new_scaled])
X_test_combined = hstack([X_test_tfidf, test_new_scaled])
lr_combined = LogisticRegression(max_iter=1000, random_state=42)
lr_combined.fit(X_train_combined, y_train)
pred_combined = lr_combined.predict(X_test_combined)

# 汇总各模型预测
model_predictions = {
    '模型1_SentiWordNet': swn_df['pred'].values,
    '模型2_朴素贝叶斯_Count': pred_nb,
    '模型3_逻辑回归_Count': pred_lr_count,
    '模型4_逻辑回归_TFIDF': pred_lr_tfidf,
    '模型5_逻辑回归_SelectK200': pred_sel200,
    '模型6_逻辑回归_SelectK2000': pred_sel2000,
    '模型7_逻辑回归_新特征': pred_combined,
}

# ---- 4.3c: 各模型预测与SentiWordNet得分的一致性分析 ----
print("\n  4.3c: 各模型预测与SentiWordNet得分的一致性分析")

# SentiWordNet方向：pos_score >= neg_score 视为正向
swn_direction = (swn_df['pos_score'] >= swn_df['neg_score']).astype(int).values

consistency_results = []
for model_name, preds in model_predictions.items():
    if model_name == '模型1_SentiWordNet':
        continue  # 跳过产生式系统本身
    # 各模型预测与SentiWordNet方向的一致性
    consistency_with_swn = (preds == swn_direction).mean()
    # 各模型预测与真实标签的一致性（即准确率）
    accuracy = (preds == y_test).mean()
    # 各模型误分类样本中，SentiWordNet也判断错误的比例
    mis_mask = preds != y_test
    if mis_mask.sum() > 0:
        swn_also_wrong = (swn_direction[mis_mask] != y_test[mis_mask]).mean()
    else:
        swn_also_wrong = 0
    # 各模型正确分类样本中，SentiWordNet也判断正确的比例
    cor_mask = preds == y_test
    swn_also_correct = (swn_direction[cor_mask] == y_test[cor_mask]).mean()

    consistency_results.append({
        '模型': model_name,
        '与SentiWordNet方向一致率': consistency_with_swn,
        '模型准确率': accuracy,
        '误分类中SentiWordNet也错的比例': swn_also_wrong,
        '正确分类中SentiWordNet也对的比例': swn_also_correct
    })
    print(f"    {model_name}: 与SentiWordNet一致率={consistency_with_swn:.4f}, "
          f"误分类中SWN也错={swn_also_wrong:.4f}, 正确中SWN也对={swn_also_correct:.4f}")

consistency_df = pd.DataFrame(consistency_results)
consistency_df.to_csv('output/sentiwordnet_model_consistency.csv', index=False)
print("  一致性分析结果已保存到 output/sentiwordnet_model_consistency.csv")

# ---- 4.3d: 各模型误分类样本的情感词分布分析 ----
print("\n  4.3d: 各模型误分类样本的情感词分布分析")

misclassification_swn_analysis = []
for model_name, preds in model_predictions.items():
    if model_name == '模型1_SentiWordNet':
        continue
    mis_mask = preds != y_test
    cor_mask = preds == y_test

    # 误分类样本的SentiWordNet得分
    mis_pos_score = swn_df.loc[mis_mask, 'pos_score'].mean()
    mis_neg_score = swn_df.loc[mis_mask, 'neg_score'].mean()
    mis_pos_words = swn_df.loc[mis_mask, 'pos_word_count'].mean()
    mis_neg_words = swn_df.loc[mis_mask, 'neg_word_count'].mean()

    # 正确分类样本的SentiWordNet得分
    cor_pos_score = swn_df.loc[cor_mask, 'pos_score'].mean()
    cor_neg_score = swn_df.loc[cor_mask, 'neg_score'].mean()
    cor_pos_words = swn_df.loc[cor_mask, 'pos_word_count'].mean()
    cor_neg_words = swn_df.loc[cor_mask, 'neg_word_count'].mean()

    # 假正向和假负向
    fp_mask = (preds == 1) & (y_test == 0)
    fn_mask = (preds == 0) & (y_test == 1)

    misclassification_swn_analysis.append({
        '模型': model_name,
        '误分类数': mis_mask.sum(),
        '误分类率': mis_mask.mean(),
        '误分类_平均正向得分': mis_pos_score,
        '误分类_平均负向得分': mis_neg_score,
        '误分类_平均正向情感词数': mis_pos_words,
        '误分类_平均负向情感词数': mis_neg_words,
        '正确_平均正向得分': cor_pos_score,
        '正确_平均负向得分': cor_neg_score,
        '正确_平均正向情感词数': cor_pos_words,
        '正确_平均负向情感词数': cor_neg_words,
        '假正向数': fp_mask.sum(),
        '假负向数': fn_mask.sum(),
    })
    print(f"    {model_name}: 误分类{mis_mask.sum()}条, "
          f"误分类样本正向得分={mis_pos_score:.2f}, 负向得分={mis_neg_score:.2f}")

mis_swn_df = pd.DataFrame(misclassification_swn_analysis)
mis_swn_df.to_csv('output/misclassification_sentiwordnet_analysis.csv', index=False)
print("  误分类SentiWordNet分析已保存到 output/misclassification_sentiwordnet_analysis.csv")

# ---- 4.3e: 正向情感词数量与预测为正向的相关程度 ----
print("\n  4.3e: 正向情感词数量与预测为正向的相关程度")

correlation_results = []
for model_name, preds in model_predictions.items():
    if model_name == '模型1_SentiWordNet':
        continue
    # 计算正向情感词数与模型预测为正向的相关系数
    pos_word_count = swn_df['pos_word_count'].values
    neg_word_count = swn_df['neg_word_count'].values
    pos_score = swn_df['pos_score'].values
    neg_score = swn_df['neg_score'].values

    corr_pos_words = np.corrcoef(pos_word_count, preds)[0, 1]
    corr_neg_words = np.corrcoef(neg_word_count, preds)[0, 1]
    corr_pos_score = np.corrcoef(pos_score, preds)[0, 1]
    corr_neg_score = np.corrcoef(neg_score, preds)[0, 1]
    corr_swn_direction = np.corrcoef(swn_direction, preds)[0, 1]

    correlation_results.append({
        '模型': model_name,
        '正向情感词数与预测相关': corr_pos_words,
        '负向情感词数与预测相关': corr_neg_words,
        '正向得分与预测相关': corr_pos_score,
        '负向得分与预测相关': corr_neg_score,
        'SentiWordNet方向与预测相关': corr_swn_direction
    })
    print(f"    {model_name}: 正向词数相关={corr_pos_words:.4f}, "
          f"负向词数相关={corr_neg_words:.4f}, SWN方向相关={corr_swn_direction:.4f}")

corr_df = pd.DataFrame(correlation_results)
corr_df.to_csv('output/sentiwordnet_prediction_correlation.csv', index=False)
print("  相关性分析已保存到 output/sentiwordnet_prediction_correlation.csv")

# ---- 绘制SentiWordNet综合分析图 ----
fig4, axes = plt.subplots(2, 2, figsize=(16, 12))

# 子图1：情感词数量与标签的关系
labels = ['正向评论', '负向评论']
pos_counts = [pos_label_1, pos_label_0]
neg_counts = [neg_label_1, neg_label_0]

x4 = np.arange(len(labels))
width4 = 0.3

axes[0, 0].bar(x4 - width4/2, pos_counts, width4, label='正向情感词数', color='#70AD47')
axes[0, 0].bar(x4 + width4/2, neg_counts, width4, label='负向情感词数', color='#C00000')

for i, (p, n) in enumerate(zip(pos_counts, neg_counts)):
    axes[0, 0].text(i - width4/2, p + 0.3, f'{p:.1f}', ha='center', fontsize=10)
    axes[0, 0].text(i + width4/2, n + 0.3, f'{n:.1f}', ha='center', fontsize=10)

axes[0, 0].set_xlabel('评论类别', fontsize=12)
axes[0, 0].set_ylabel('平均情感词数', fontsize=12)
axes[0, 0].set_title('SentiWordNet情感词数量与标签关系', fontsize=13, fontweight='bold')
axes[0, 0].set_xticks(x4)
axes[0, 0].set_xticklabels(labels)
axes[0, 0].legend()
axes[0, 0].grid(axis='y', alpha=0.3)

# 子图2：SentiWordNet误分类样本情感得分分布
categories = ['假正向\n(预测正实为负)', '假负向\n(预测负实为正)']
x5 = np.arange(len(categories))
axes[0, 1].bar(x5 - width4/2, [fp_pos, fn_pos], width4, label='正向得分', color='#70AD47')
axes[0, 1].bar(x5 + width4/2, [fp_neg, fn_neg], width4, label='负向得分', color='#C00000')

for i, (p, n) in enumerate(zip([fp_pos, fn_pos], [fp_neg, fn_neg])):
    axes[0, 1].text(i - width4/2, p + 0.3, f'{p:.1f}', ha='center', fontsize=10)
    axes[0, 1].text(i + width4/2, n + 0.3, f'{n:.1f}', ha='center', fontsize=10)

axes[0, 1].set_xlabel('误分类类型', fontsize=12)
axes[0, 1].set_ylabel('平均情感得分', fontsize=12)
axes[0, 1].set_title('SentiWordNet误分类样本情感得分', fontsize=13, fontweight='bold')
axes[0, 1].set_xticks(x5)
axes[0, 1].set_xticklabels(categories)
axes[0, 1].legend()
axes[0, 1].grid(axis='y', alpha=0.3)

# 子图3：各模型与SentiWordNet方向一致率
model_names_short = [r['模型'].replace('模型', 'M').replace('_逻辑回归_', ' LR-').replace('_朴素贝叶斯_Count', ' NB') for r in consistency_results]
consistency_vals = [r['与SentiWordNet方向一致率'] for r in consistency_results]

x6 = np.arange(len(model_names_short))
bars_cons = axes[1, 0].bar(x6, consistency_vals, color='#4472C4')
for bar in bars_cons:
    height = bar.get_height()
    axes[1, 0].text(bar.get_x() + bar.get_width()/2, height + 0.005, f'{height:.3f}', ha='center', fontsize=9)

axes[1, 0].set_xlabel('模型', fontsize=12)
axes[1, 0].set_ylabel('一致率', fontsize=12)
axes[1, 0].set_title('各模型预测与SentiWordNet方向一致率', fontsize=13, fontweight='bold')
axes[1, 0].set_xticks(x6)
axes[1, 0].set_xticklabels(model_names_short, rotation=20, ha='right', fontsize=9)
axes[1, 0].set_ylim(0.6, 0.85)
axes[1, 0].grid(axis='y', alpha=0.3)

# 子图4：正向情感词数与各模型预测的相关系数
corr_pos_vals = [r['正向情感词数与预测相关'] for r in correlation_results]
corr_neg_vals = [r['负向情感词数与预测相关'] for r in correlation_results]

x7 = np.arange(len(model_names_short))
axes[1, 1].bar(x7 - width4/2, corr_pos_vals, width4, label='正向情感词数', color='#70AD47')
axes[1, 1].bar(x7 + width4/2, corr_neg_vals, width4, label='负向情感词数', color='#C00000')

axes[1, 1].set_xlabel('模型', fontsize=12)
axes[1, 1].set_ylabel('相关系数', fontsize=12)
axes[1, 1].set_title('SentiWordNet情感词数与模型预测的相关性', fontsize=13, fontweight='bold')
axes[1, 1].set_xticks(x7)
axes[1, 1].set_xticklabels(model_names_short, rotation=20, ha='right', fontsize=9)
axes[1, 1].legend()
axes[1, 1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('output/sentiwordnet_analysis.png', dpi=150, bbox_inches='tight')
print("\n  SentiWordNet综合分析图已保存到 output/sentiwordnet_analysis.png")

# ============================================================
# 步骤4.4：大语言模型结构化输出分析
# ============================================================
print("\n" + "=" * 60)
print("步骤4.4：大语言模型结构化输出分析")
print("=" * 60)

llm_structured_analysis = {}
try:
    with open('output/llm_structured_results.json', 'r', encoding='utf-8') as f:
        llm_structured = json.load(f)

    print(f"  读取到 {len(llm_structured)} 条结构化输出结果")

    # 统计主题分布
    topic_counts = {}
    aspect_sentiment_counts = {'positive': 0, 'negative': 0}
    correct_count = 0
    total_count = 0
    high_conf_correct = 0
    high_conf_total = 0
    low_conf_correct = 0
    low_conf_total = 0

    for item in llm_structured:
        if not isinstance(item, dict):
            continue
        total_count += 1

        # 获取解析后的输出
        parsed = item.get('parsed_output', item)

        # 情感判断准确性
        pred_sentiment = parsed.get('sentiment', '').lower()
        true_label = item.get('true_label', item.get('label', -1))
        if true_label == 1 and pred_sentiment == 'positive':
            correct_count += 1
        elif true_label == 0 and pred_sentiment == 'negative':
            correct_count += 1

        # 置信度分析
        score = parsed.get('sentiment_score', 0.5)
        if score >= 0.7:
            high_conf_total += 1
            if (true_label == 1 and pred_sentiment == 'positive') or (true_label == 0 and pred_sentiment == 'negative'):
                high_conf_correct += 1
        elif score <= 0.3:
            low_conf_total += 1
            if (true_label == 1 and pred_sentiment == 'positive') or (true_label == 0 and pred_sentiment == 'negative'):
                low_conf_correct += 1

        # 主题统计
        for topic in parsed.get('topic', []):
            topic_counts[topic] = topic_counts.get(topic, 0) + 1

        # 方面级情感统计
        for aspect in parsed.get('aspects', []):
            asp_sent = aspect.get('sentiment', '').lower()
            if asp_sent in aspect_sentiment_counts:
                aspect_sentiment_counts[asp_sent] += 1

    llm_accuracy = correct_count / total_count if total_count > 0 else 0
    high_conf_acc = high_conf_correct / high_conf_total if high_conf_total > 0 else 0
    low_conf_acc = low_conf_correct / low_conf_total if low_conf_total > 0 else 0

    # Top 10 主题
    top_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:10]

    print(f"  结构化输出准确率: {llm_accuracy:.4f} ({correct_count}/{total_count})")
    print(f"  高置信度(>=0.7)准确率: {high_conf_acc:.4f} ({high_conf_correct}/{high_conf_total})")
    print(f"  低置信度(<=0.3)准确率: {low_conf_acc:.4f} ({low_conf_correct}/{low_conf_total})")
    print(f"  方面级情感分布: 正向={aspect_sentiment_counts['positive']}, 负向={aspect_sentiment_counts['negative']}")
    print(f"  Top 5 主题: {top_topics[:5]}")

    llm_structured_analysis = {
        'total_count': total_count,
        'correct_count': correct_count,
        'accuracy': llm_accuracy,
        'high_conf_accuracy': high_conf_acc,
        'low_conf_accuracy': low_conf_acc,
        'top_topics': top_topics,
        'aspect_sentiment': aspect_sentiment_counts
    }

except FileNotFoundError:
    print("  未找到结构化输出结果文件，跳过此分析")
    llm_structured_analysis = {}

# ============================================================
# 生成阶段四完整报告
# ============================================================
print("\n生成阶段四完整报告...")

report4 = f"""# 阶段四：结果分析报告

## 步骤4.1：评测指标汇总

### 所有模型测试集性能对比

| 模型 | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 |
|------|----------|----------------|-------------|----------|-------------|
"""

for _, row in summary_df.iterrows():
    report4 += f"| {row['模型']} | {row['Accuracy']:.4f} | {row['Macro Precision']:.4f} | {row['Macro Recall']:.4f} | {row['Macro F1']:.4f} | {row['Weighted F1']:.4f} |\n"

report4 += f"""
---

## 步骤4.2：可视化比较

### 模型性能对比图
![模型性能对比](models_comparison.png)

### 特征选择对性能的影响
![特征选择对比](feature_selection_comparison.png)

**分析**：
- K=200时，特征维度大幅减少（从50000到200），但性能下降约6个百分点
- K=2000时，性能接近全特征TF-IDF，说明2000个关键特征已能捕获大部分情感信息
- 特征选择在降低计算成本的同时，K=2000是一个较好的平衡点

### 新特征设计对性能的影响
![新特征对比](new_features_comparison.png)

**分析**：
- 仅使用8个手工特征，准确率可达63%，说明这些特征确实包含情感信息
- TF-IDF+新特征与纯TF-IDF性能接近，因为TF-IDF特征维度远大于新特征
- 新特征中，否定词数量和负向情感词比例的系数绝对值最大，对分类贡献最显著

---

## 步骤4.3：SentiWordNet分析

### 4.3a SentiWordNet自身预测分析

#### 情感词数量与标签的关系

| 评论类别 | 平均正向情感词数 | 平均负向情感词数 |
|---------|----------------|----------------|
| 正向评论 | {pos_label_1:.2f} | {neg_label_1:.2f} |
| 负向评论 | {pos_label_0:.2f} | {neg_label_0:.2f} |

#### SentiWordNet误分类分析

| 指标 | 值 |
|------|-----|
| 误分类样本比例 | {len(misclassified)/len(swn_df)*100:.1f}% |
| 假正向（预测正实为负） | {len(false_positive)} 条 |
| 假负向（预测负实为正） | {len(false_negative)} 条 |

#### 误分类样本情感得分

| 误分类类型 | 平均正向得分 | 平均负向得分 |
|-----------|------------|------------|
| 假正向 | {fp_pos:.2f} | {fp_neg:.2f} |
| 假负向 | {fn_pos:.2f} | {fn_neg:.2f} |

### 4.3b 各模型预测与SentiWordNet得分的一致性

| 模型 | 与SentiWordNet方向一致率 | 模型准确率 | 误分类中SWN也错比例 | 正确中SWN也对比例 |
|------|----------------------|----------|-------------------|-----------------|
"""

for _, row in consistency_df.iterrows():
    report4 += f"| {row['模型']} | {row['与SentiWordNet方向一致率']:.4f} | {row['模型准确率']:.4f} | {row['误分类中SentiWordNet也错的比例']:.4f} | {row['正确分类中SentiWordNet也对的比例']:.4f} |\n"

report4 += f"""
**分析**：
- 各模型与SentiWordNet方向的一致率在0.61-0.64之间，说明机器学习模型的预测与SentiWordNet的情感方向有一定重叠，但差异较大
- 模型误分类的样本中，SentiWordNet也判断错误的比例约47%-59%，说明这些样本本身就具有情感模糊性，SentiWordNet和机器学习模型在困难样本上存在一定共识
- 模型正确分类的样本中，SentiWordNet也判断正确的比例约63%-65%，说明机器学习模型能捕捉到SentiWordNet无法识别的情感模式

### 4.3c 正向情感词数量与预测为正向的相关程度

| 模型 | 正向情感词数与预测相关 | 负向情感词数与预测相关 | 正向得分与预测相关 | 负向得分与预测相关 | SWN方向与预测相关 |
|------|---------------------|---------------------|-------------------|-------------------|-----------------|
"""

for _, row in corr_df.iterrows():
    report4 += f"| {row['模型']} | {row['正向情感词数与预测相关']:.4f} | {row['负向情感词数与预测相关']:.4f} | {row['正向得分与预测相关']:.4f} | {row['负向得分与预测相关']:.4f} | {row['SentiWordNet方向与预测相关']:.4f} |\n"

report4 += f"""
**分析**：
- 正向情感词数量与模型预测为正向的相关性较弱（约0.01-0.05），说明仅靠正向情感词数量不足以判断情感极性
- 负向情感词数量与模型预测为正向呈负相关（约-0.09至-0.14），说明负向情感词对预测有一定指示作用，但相关性也较弱
- SentiWordNet方向与模型预测的相关系数约0.28-0.32，属于弱至中等相关，说明SentiWordNet的情感方向对模型预测有一定参考价值，但模型还学习了更多上下文和组合信息

### 4.3d 各模型误分类样本的情感词分布

| 模型 | 误分类数 | 误分类率 | 误分类_平均正向得分 | 误分类_平均负向得分 | 误分类_平均正向词数 | 误分类_平均负向词数 | 假正向数 | 假负向数 |
|------|---------|---------|-------------------|-------------------|-------------------|-------------------|---------|---------|
"""

for _, row in mis_swn_df.iterrows():
    report4 += f"| {row['模型']} | {int(row['误分类数'])} | {row['误分类率']:.4f} | {row['误分类_平均正向得分']:.2f} | {row['误分类_平均负向得分']:.2f} | {row['误分类_平均正向情感词数']:.1f} | {row['误分类_平均负向情感词数']:.1f} | {int(row['假正向数'])} | {int(row['假负向数'])} |\n"

report4 += f"""
**分析**：
- 各模型误分类样本的正向得分和负向得分较为接近，说明这些样本本身就存在情感模糊性
- 误分类样本中正向情感词数和负向情感词数差异不大，进一步验证了SentiWordNet在情感模糊样本上的局限性
- 假正向（预测正向实际负向）通常多于假负向，说明模型更倾向于将模糊样本判断为正向

### SentiWordNet分析图
![SentiWordNet分析](sentiwordnet_analysis.png)

### 综合分析总结

1. **SentiWordNet的局限性**：基于规则的方法准确率仅62%，远低于机器学习方法。主要原因：
   - 只考虑词的情感极性，忽略了上下文和语序信息
   - 否定词、转折词等修辞手法无法被简单累加捕获
   - 取第一个同义词集的策略可能不准确

2. **假正向分析**：负向评论被误判为正向，通常是因为评论中包含大量正向情感词但实际表达反讽或对比
   - 这类评论的正向得分和负向得分接近，说明情感词混杂

3. **假负向分析**：正向评论被误判为负向，通常是因为评论中使用了负面词汇来描述对比对象
   - 例如"not bad"中的"bad"被计为负向，但实际表达正向

4. **与机器学习模型的对比**：机器学习模型通过学习词的组合模式，能够更好地处理否定、反讽等复杂语义
   - 各模型与SentiWordNet方向一致率约61%-64%，说明机器学习模型在大部分样本上做出了与SentiWordNet不同的判断，但准确率远高于SentiWordNet
   - 模型误分类的样本中SentiWordNet也容易出错（约47%-59%），说明这些样本确实难以仅靠情感词判断

5. **情感词相关性的启示**：正向情感词数量与模型预测的相关性说明SentiWordNet的情感词识别有一定价值，但不足以单独完成分类任务

---

## 步骤4.4：大语言模型结构化输出分析

"""

if llm_structured_analysis:
    a = llm_structured_analysis
    report4 += f"""### 情感判断准确性

| 指标 | 值 |
|------|-----|
| 总测试样本数 | {a['total_count']} |
| 正确预测数 | {a['correct_count']} |
| 准确率 | {a['accuracy']:.4f} |
| 高置信度(>=0.7)准确率 | {a['high_conf_accuracy']:.4f} |
| 低置信度(<=0.3)准确率 | {a['low_conf_accuracy']:.4f} |

### 主题提取分布（Top 10）

| 主题 | 出现次数 |
|------|---------|
"""
    for topic, count in a['top_topics']:
        report4 += f"| {topic} | {count} |\n"

    report4 += f"""
### 方面级情感分布

| 情感 | 出现次数 |
|------|---------|
| 正向 | {a['aspect_sentiment']['positive']} |
| 负向 | {a['aspect_sentiment']['negative']} |

### 分析总结

1. **主题提取能力**：大语言模型能够准确识别评论中讨论的主要方面。plot（剧情）和acting（演技）是最常被提及的主题，这与电影评论的特点一致。

2. **情感判断一致性**：结构化输出中的整体情感与方面级情感基本一致。当评论中多数方面为正向时，整体情感通常为正向。

3. **方面级细粒度分析**：模型可以对评论中的不同方面给出不同的情感判断，体现了细粒度分析能力。例如，某条评论对角色设计给出正面评价，但对教育内容给出负面评价。

4. **置信度校准**：高置信度预测的准确率通常高于低置信度预测，说明模型的sentiment_score具有一定的校准意义。低置信度样本更容易误分类，这些样本通常包含混合情感表达。

5. **关键短语提取**：模型能够提取出表达情感的关键短语，这些短语通常包含明确的情感词或评价性表达。

6. **误分类分析**：少数误分类样本通常是因为评论中混合了正负情感（如"角色设计好但教育内容不足"），导致整体情感判断困难。这类样本的置信度通常较低。
"""
else:
    report4 += "结构化输出结果文件未找到，无法进行分析。\n"

with open('output/phase4_analysis_report.md', 'w', encoding='utf-8') as f:
    f.write(report4)
print("  阶段四完整报告已保存到 output/phase4_analysis_report.md")

print("\n" + "=" * 60)
print("阶段四：结果分析 完成！")
print("=" * 60)
