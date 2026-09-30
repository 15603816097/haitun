# 海豚杯：蛋白质功能预测

2026 全国大学生“海豚杯”数智分析赛道——蛋白质功能预测项目。

## 当前阶段

V0：建立可复现的数据审计、验证划分、Macro F1 评估和官方 RandomForest baseline。

## 数据放置

竞赛原始数据不提交到 GitHub。服务器克隆仓库后，将以下文件放在仓库根目录：

- `train.csv`
- `test.csv`
- `submit_template_v1.csv`
- `baseline-v2.ipynb`（可选，仅作官方 baseline 参考）

## 环境

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt
```

## V0 使用

数据审计：

```bash
python -m src.data_audit --train train.csv --test test.csv
```

生成验证索引：

```bash
python -m src.splits --train train.csv --test test.csv --output outputs/splits
```

复现官方 RandomForest baseline：

```bash
python -m src.baseline_rf \
  --train train.csv \
  --test test.csv \
  --output outputs/submissions/v0_rf_submission.csv
```

> 正式实验后续将按 V1（k-mer TF-IDF + SGD）、V2（OOF 阈值优化）、V3（ESM2）逐步迭代。

## GitHub / 服务器工作流

GitHub 只管理代码和配置；训练数据、缓存、模型、日志与提交结果只保存在服务器。

服务器更新代码：

```bash
git pull
```


## V1：3–5mer TF-IDF + SGD

先做 20 标签冒烟测试，确认服务器内存、特征缓存与训练流程正常：

```bash
git pull
python -m src.train_kmer --mode random --max-labels 20
```

冒烟测试通过后，运行完整 500 标签 random Group validation：

```bash
python -m src.train_kmer --mode random
```

然后运行 test-like tail validation。该分数与 random validation 不是同一口径，不能直接比较绝对值：

```bash
python -m src.train_kmer --mode tail
```

V1 会保存连续概率到 `outputs/oof/`，供 V2 做逐标签阈值优化。TF-IDF 特征缓存保存在 `cache/tfidf/`，两者均不会提交到 GitHub。

如果需要重新构建 TF-IDF 而不读取缓存：

```bash
python -m src.train_kmer --mode random --no-cache
```
