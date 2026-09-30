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
