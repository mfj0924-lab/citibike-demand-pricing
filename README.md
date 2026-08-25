# CitiBike Dynamic Pricing Strategy
# 城市共享单车动态定价策略

## 项目简介
大数据管理与应用课程设计 — Topic 2：城市共享单车动态定价策略。
基于纽约 CitiBike 2026年 1-5 月骑行数据（2700+ MB），使用 PySpark 分布式计算预测
各站点每小时的借车/还车需求，并基于供需缺口设计动态定价策略。

## 技术栈
- **分布式计算**: PySpark + PySpark MLlib
- **数据存储**: Parquet（Medallion Architecture: Bronze → Silver → Gold）
- **特征工程**: 时间周期编码（Cyclic Encoding）、假期标注、天气特征
- **降维**: PCA（PySpark ML） + t-SNE（sklearn 采样）
- **机器学习**: Random Forest / Gradient Boosted Trees（回归）
- **自动调参**: Hyperopt（贝叶斯优化）
- **API 部署**: FastAPI + Uvicorn + Pydantic
- **可视化**: PyEcharts
- **CI/CD**: GitHub Actions + flake8 + pytest

## 项目结构
```
├── config/               # Spark配置、模型配置、定价参数
├── data/
│   ├── raw/              # 原始CitiBike CSV + station_information.json
│   ├── processed/        # bronze/silver/gold 三层Parquet
│   └── unstructured/     # 非结构化数据（天气JSON等）
├── models/               # 训练好的PySpark ML模型
├── notebooks/            # Jupyter分析笔记
├── src/
│   ├── data_pipeline/    # 数据管道（Raw→Bronze→Silver→Gold）
│   ├── features/         # 特征工程 + PCA/t-SNE降维
│   ├── train_pipeline/   # 模型训练 + Hyperopt调参
│   ├── pricing/          # 动态定价策略引擎
│   ├── api/              # FastAPI推理服务
│   ├── visualization/    # PyEcharts可视化看板
│   └── utils/            # 工具函数（日志、IO）
├── tests/                # pytest单元测试
└── README.md
```

## 快速开始
已验证环境为 Python 3.10、JDK 17。完整数据约 2.7GB，不进入 Git 仓库。

```powershell
# Windows：创建并激活独立环境
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1

# 安装依赖
python -m pip install -r requirements.txt

# 设置本机 Java（请替换为自己的安装目录）
$env:JAVA_HOME = "<JDK_17_DIR>"

# 将官方下载并解压的 CSV 放入 data/raw 后，依次运行数据管道
python -m src.data_pipeline.data_ingestor
# Silver、Gold 阶段以 src/data_pipeline 中的模块和项目说明为准

# 训练管线由 src/train_pipeline/ 中的 RF、GBT 与调参模块组成；先运行测试确认环境
python -m pytest

# 启动API服务
python -m uvicorn src.api.main:app --reload
# 然后访问 API 文档、交互演示或分析大屏
# http://localhost:8000/docs
# http://localhost:8000/demo
# http://localhost:8000/dashboard
```

## API 与成果展示

| 内容 | 入口 | 作用 |
|---|---|---|
| 交互式分析大屏 | [`dashboard.html`](dashboard.html) | 无需启动服务即可查看 6 类分析图表 |
| 课程答辩展示 | [`showcase/presentation.html`](showcase/presentation.html) | 快速了解问题、数据管道、模型、定价策略和结果 |
| 完整项目报告 | [`reports/citibike_project_report.docx`](reports/citibike_project_report.docx) | 保留数据来源、建模过程、结果与边界 |
| FastAPI 文档 | `http://localhost:8000/docs` | 查看和试调用接口 |

服务启动后提供三个核心接口：

- `POST /api/v1/predict`：输入站点、时间和当前库存，返回需求预测与定价建议。
- `POST /api/v1/station/{station_id}/forecast`：返回指定站点未来 24 小时逐时预测。
- `GET /api/v1/hotspots`：返回当前建议涨价或降价的热点站点。

报告中的本机环境命令属于课程实施记录；公开仓库统一使用上面的可移植启动命令。

## 数据来源
- CitiBike 官方骑行数据：https://citibikenyc.com/system-data
- GBFS 站点信息：https://gbfs.citibikenyc.com/gbfs/en/station_information.json
- Open-Meteo 历史天气 API：https://open-meteo.com/

仓库提供代码、测试、说明、轻量展示资源和脱敏后的课程报告。原始 ZIP、Parquet
中间层、训练模型和第三方参考项目不上传；使用者需从官方来源自行获取数据。

## 结果边界

- API 页面中的规则推理用于产品演示；完整模型结果由离线训练流程生成。
- 动态价格是需求与供需缺口驱动的策略模拟，不代表 CitiBike 官方定价。
- 项目使用 AI 辅助代码生产与复核；业务问题、技术路线、指标口径、验收和最终结论由本人负责。
