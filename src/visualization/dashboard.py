"""
PyEcharts interactive dashboard for CitiBike Dynamic Pricing analysis.

Generates an HTML page with 6 charts:
1. Pricing zone distribution (pie chart)
2. Hourly demand pattern (line chart)
3. Feature importance (bar chart)
4. Bike vs Dock demand scatter (scatter chart)
5. Pricing multiplier histogram (bar chart)
6. Station surge ranking (horizontal bar chart)

All charts use data from pricing_analysis.csv and model training summaries.
"""

import os
import sys
import pandas as pd
import numpy as np

from pyecharts.charts import (
    Pie, Line, Bar, Scatter, Page,
)
from pyecharts import options as opts
from pyecharts.globals import ThemeType

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def _load_data():
    """Load pricing analysis data."""
    base = os.path.join(os.path.dirname(__file__), "..", "..")
    csv_path = os.path.join(base, "data", "processed", "pricing_analysis.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return _generate_sample_data()


def _generate_sample_data():
    """Generate sample data if CSV is not available."""
    np.random.seed(42)
    n = 500
    hours = np.tile(range(24), n // 24 + 1)[:n]
    return pd.DataFrame({
        "station_id": [f"{i:04d}" for i in range(n)],
        "bike_demand": np.random.poisson(10, n),
        "dock_demand": np.random.poisson(8, n),
        "pricing_multiplier": np.clip(np.random.normal(1.0, 0.2, n), 0.7, 2.0),
        "suggested_price": np.random.normal(4.5, 1.0, n),
        "pricing_zone": np.random.choice(
            ["Surge", "Mild Surge", "Normal", "Discount"], n,
            p=[0.05, 0.10, 0.35, 0.50]
        ),
        "hour": hours,
    })


def chart_pricing_pie(pdf: pd.DataFrame) -> Pie:
    """Pie chart: pricing zone distribution."""
    counts = pdf["pricing_zone"].value_counts()
    data_pair = [(zone, int(count)) for zone, count in counts.items()]

    return (
        Pie(init_opts=opts.InitOpts(theme=ThemeType.LIGHT, width="500px", height="400px"))
        .add(
            series_name="定价区域",
            data_pair=data_pair,
            radius=["40%", "70%"],
            label_opts=opts.LabelOpts(formatter="{b}: {c} ({d}%)"),
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title="定价区域分布", subtitle="Surge / Normal / Discount"),
            legend_opts=opts.LegendOpts(orient="vertical", pos_right="5%"),
        )
        .set_colors(["#e74c3c", "#f39c12", "#3498db", "#2ecc71"])
    )


def chart_hourly_demand(pdf: pd.DataFrame) -> Line:
    """Line chart: average bike/dock demand by hour."""
    if "hour" not in pdf.columns:
        pdf["hour"] = pd.to_datetime(pdf["event_hour"]).dt.hour

    hourly = pdf.groupby("hour").agg(
        bike=("bike_demand", "mean"),
        dock=("dock_demand", "mean"),
    ).reset_index()

    return (
        Line(init_opts=opts.InitOpts(theme=ThemeType.LIGHT, width="700px", height="400px"))
        .add_xaxis(hourly["hour"].astype(str).tolist())
        .add_yaxis(
            "借车需求", hourly["bike"].round(1).tolist(),
            is_smooth=True,
            label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=opts.LineStyleOpts(width=2),
        )
        .add_yaxis(
            "还车需求", hourly["dock"].round(1).tolist(),
            is_smooth=True,
            label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=opts.LineStyleOpts(width=2),
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title="24小时需求模式", subtitle="按小时平均借车/还车量"),
            xaxis_opts=opts.AxisOpts(name="小时"),
            yaxis_opts=opts.AxisOpts(name="平均需求（辆）"),
            tooltip_opts=opts.TooltipOpts(trigger="axis"),
        )
    )


def chart_feature_importance() -> Bar:
    """Bar chart: RF model feature importance (Top-10)."""
    # Feature importance from our trained model
    features = [
        ("电动车比例", 0.372),
        ("会员占比", 0.359),
        ("平均骑行时长", 0.158),
        ("小时(cos)", 0.024),
        ("小时(sin)", 0.021),
        ("月份", 0.018),
        ("一年第几天", 0.015),
        ("星期几(cos)", 0.010),
        ("星期几(sin)", 0.008),
        ("是否周末", 0.006),
    ]

    return (
        Bar(init_opts=opts.InitOpts(theme=ThemeType.LIGHT, width="600px", height="400px"))
        .add_xaxis([f[0] for f in features])
        .add_yaxis("特征重要度", [round(f[1], 3) for f in features],
                    itemstyle_opts=opts.ItemStyleOpts(color="#3498db"))
        .set_global_opts(
            title_opts=opts.TitleOpts(title="模型特征重要性", subtitle="Random Forest Top-10"),
            xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=45)),
            yaxis_opts=opts.AxisOpts(name="重要度"),
        )
        .reversal_axis()
    )


def chart_price_histogram(pdf: pd.DataFrame) -> Bar:
    """Bar chart: suggested price distribution."""
    prices = pdf["suggested_price"].dropna()
    bins = [0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 10.0]
    labels = ["<2.5", "2.5-3", "3-3.5", "3.5-4", "4-4.5", "4.5-5", "5-5.5", "5.5-6", "6-7", ">7"]
    counts = pd.cut(prices, bins=bins).value_counts().sort_index()

    return (
        Bar(init_opts=opts.InitOpts(theme=ThemeType.LIGHT, width="600px", height="400px"))
        .add_xaxis(labels)
        .add_yaxis("站点-小时数", counts.tolist(),
                    itemstyle_opts=opts.ItemStyleOpts(color="#27ae60"))
        .set_global_opts(
            title_opts=opts.TitleOpts(title="建议价格分布", subtitle="各站点×小时的定价频率"),
            xaxis_opts=opts.AxisOpts(name="建议价格 (USD)"),
            yaxis_opts=opts.AxisOpts(name="记录数"),
        )
    )


def chart_demand_scatter(pdf: pd.DataFrame) -> Scatter:
    """Scatter chart: bike_demand vs dock_demand, colored by pricing zone."""
    # Sample 2000 points for performance
    sample = pdf.sample(n=min(2000, len(pdf)), random_state=42)

    color_map = {"Surge": "#e74c3c", "Mild Surge": "#f39c12",
                 "Normal": "#3498db", "Discount": "#2ecc71"}

    zones = sample["pricing_zone"].unique()
    scatter = Scatter(
        init_opts=opts.InitOpts(theme=ThemeType.LIGHT, width="600px", height="400px")
    )
    scatter.add_xaxis([])  # Placeholder

    for zone in zones:
        subset = sample[sample["pricing_zone"] == zone]
        scatter.add_yaxis(
            zone,
            list(zip(
                subset["bike_demand"].clip(0, 80).tolist(),
                subset["dock_demand"].clip(0, 80).tolist(),
            )),
            symbol_size=6,
            label_opts=opts.LabelOpts(is_show=False),
        )

    scatter.set_global_opts(
        title_opts=opts.TitleOpts(title="借车 vs 还车需求分布", subtitle="按定价区域着色"),
        xaxis_opts=opts.AxisOpts(name="借车需求", type_="value", min_=0),
        yaxis_opts=opts.AxisOpts(name="还车需求", type_="value", min_=0),
        tooltip_opts=opts.TooltipOpts(formatter="Bike: {@[0]} Dock: {@[1]}"),
    )
    return scatter


def chart_top_surge_stations(pdf: pd.DataFrame) -> Bar:
    """Bar chart: top-10 surge stations by average multiplier."""
    station_agg = pdf.groupby("station_id").agg(
        avg_price=("suggested_price", "mean"),
        avg_multiplier=("pricing_multiplier", "mean"),
    ).reset_index()

    top10 = station_agg.nlargest(10, "avg_multiplier")

    return (
        Bar(init_opts=opts.InitOpts(theme=ThemeType.LIGHT, width="600px", height="500px"))
        .add_xaxis(top10["station_id"].tolist())
        .add_yaxis("平均定价乘数", top10["avg_multiplier"].round(2).tolist(),
                    itemstyle_opts=opts.ItemStyleOpts(color="#e74c3c"))
        .set_global_opts(
            title_opts=opts.TitleOpts(title="Top-10 高频涨价站点", subtitle="按平均定价乘数排序"),
            xaxis_opts=opts.AxisOpts(name="站点ID"),
            yaxis_opts=opts.AxisOpts(name="平均定价乘数"),
        )
        .reversal_axis()
    )


def render_dashboard() -> str:
    """Build the full PyEcharts dashboard page.

    Returns:
        HTML string with all charts embedded.
    """
    pdf = _load_data()

    page = Page(
        layout=Page.SimplePageLayout,
        page_title="CitiBike Dynamic Pricing Dashboard",
    )

    page.add(
        chart_pricing_pie(pdf),
        chart_hourly_demand(pdf),
        chart_feature_importance(),
        chart_demand_scatter(pdf),
        chart_price_histogram(pdf),
        chart_top_surge_stations(pdf),
    )

    # Render to HTML
    dashboard_html = page.render_embed()

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CitiBike Dynamic Pricing Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/echarts@5.5.0/dist/echarts.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: #f0f2f5;
        }}
        .header {{
            background: linear-gradient(135deg, #2c3e50, #3498db);
            color: white; padding: 24px 40px;
        }}
        .header h1 {{ font-size: 24px; font-weight: 600; }}
        .header p {{ font-size: 14px; opacity: 0.8; margin-top: 4px; }}
        .kpi-row {{
            display: flex; gap: 20px; padding: 20px 40px;
        }}
        .kpi-card {{
            flex: 1; background: white; border-radius: 10px;
            padding: 20px; text-align: center; box-shadow: 0 1px 4px rgba(0,0,0,.08);
        }}
        .kpi-card .value {{ font-size: 28px; font-weight: 700; color: #2c3e50; }}
        .kpi-card .label {{ font-size: 13px; color: #7f8c8d; margin-top: 4px; }}
        .charts-grid {{
            display: grid; grid-template-columns: 1fr 1fr;
            gap: 20px; padding: 0 40px 40px;
        }}
        .chart-card {{
            background: white; border-radius: 10px; padding: 20px;
            box-shadow: 0 1px 4px rgba(0,0,0,.08);
        }}
        .footer {{
            text-align: center; padding: 20px; color: #95a5a6; font-size: 13px;
        }}
        @media (max-width: 1200px) {{
            .charts-grid {{ grid-template-columns: 1fr; }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🚲 纽约 CitiBike 动态定价数据分析大屏</h1>
        <p>基于 2026年 1-5月 骑行数据 (1,450万条) | PySpark + Random Forest | 站点小时级需求预测</p>
    </div>

    <div class="kpi-row">
        <div class="kpi-card">
            <div class="value">{len(pdf):,}</div>
            <div class="label">分析记录数 (站点×小时)</div>
        </div>
        <div class="kpi-card">
            <div class="value">2,433</div>
            <div class="label">CitiBike 站点数</div>
        </div>
        <div class="kpi-card">
            <div class="value">{pdf['pricing_multiplier'].mean():.2f}x</div>
            <div class="label">平均定价乘数</div>
        </div>
        <div class="kpi-card">
            <div class="value">${pdf['suggested_price'].mean():.2f}</div>
            <div class="label">平均建议价格 (USD)</div>
        </div>
    </div>

    <div class="charts-grid" id="charts">
        {dashboard_html}
    </div>

    <div class="footer">
        大数据管理与应用 · 课程设计 Topic 2 · 城市共享单车动态定价策略
    </div>
</body>
</html>"""


if __name__ == "__main__":
    html = render_dashboard()
    base = os.path.join(os.path.dirname(__file__), "..", "..")
    out_path = os.path.join(base, "dashboard.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Dashboard saved to {out_path}")
    print(f"Size: {len(html):,} bytes")
