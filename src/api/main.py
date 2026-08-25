"""FastAPI application — CitiBike Dynamic Pricing API.

Start with: uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
Swagger UI: http://localhost:8000/docs
"""

import os
import sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.api.routers.predict import router as predict_router

app = FastAPI(
    title="城市共享单车动态定价 API",
    description="""
## 城市共享单车动态定价策略 — API 服务

基于纽约 CitiBike 2026 年骑行数据的机器学习预测模型，
提供站点级单车需求预测与动态定价建议。

### 核心功能
- **需求预测**: 预测指定站点未来每小时的借车/还车量
- **动态定价**: 基于供需缺口给出 surge/discount 建议
- **热点分析**: 识别需要涨价或降价的站点

### 技术栈
PySpark + Random Forest + Hyperopt + FastAPI + PyEcharts
    """,
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict_router)


@app.get("/", response_class=HTMLResponse, tags=["页面"])
async def root():
    """API 首页，包含快速跳转链接"""
    return """
    <html>
    <head><title>CitiBike Dynamic Pricing API</title>
    <style>
        body { font-family: -apple-system, sans-serif; max-width: 700px; margin: 40px auto; padding: 20px; }
        h1 { color: #2c3e50; }
        a { color: #3498db; }
        .card { background: #f8f9fa; border-radius: 8px; padding: 15px; margin: 10px 0; }
    </style>
    </head>
    <body>
        <h1>🚲 CitiBike Dynamic Pricing API</h1>
        <p>城市共享单车动态定价策略 — 预测服务</p>

        <div class="card">
            <h3>📖 API 文档</h3>
            <a href="/docs">Swagger UI — 交互式 API 文档</a>
        </div>

        <div class="card">
            <h3>🔌 API 接口</h3>
            <ul>
                <li><code>GET /api/v1/hotspots</code> — 定价热点站点</li>
                <li><code>POST /api/v1/predict</code> — 单次需求+定价预测</li>
                <li><code>POST /api/v1/station/{id}/forecast</code> — 24小时预报</li>
            </ul>
        </div>

        <div class="card">
            <h3>📊 可视化看板</h3>
            <a href="/dashboard">PyEcharts 数据分析大屏</a>
        </div>
    </body>
    </html>
    """


@app.get("/dashboard", response_class=HTMLResponse, tags=["页面"])
async def dashboard():
    """PyEcharts 数据可视化看板大屏"""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from src.visualization.dashboard import render_dashboard

    return HTMLResponse(render_dashboard())


@app.get("/demo", response_class=HTMLResponse, tags=["页面"])
async def demo():
    """交互式预测演示页面（输入站点信息 → 查看定价建议）"""
    return """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>单车定价预测 Demo</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,'Noto Sans SC',sans-serif;background:linear-gradient(160deg,#eef2fb 0%,#f5f0ff 30%,#f8fafd 100%);min-height:100vh;display:flex;justify-content:center;align-items:center;padding:20px}
.container{background:#fff;border-radius:18px;box-shadow:0 8px 40px rgba(0,0,0,0.08);max-width:580px;width:100%;overflow:hidden}
.header{background:linear-gradient(135deg,#0f1923,#1e3a5f);color:#fff;padding:28px 30px;text-align:center}
.header h1{font-size:22px;font-weight:700;margin-bottom:4px}
.header .sub{font-size:13px;opacity:0.7}
.nav-links{display:flex;gap:10px;justify-content:center;margin-top:14px}
.nav-links a{display:inline-block;padding:6px 18px;border-radius:6px;color:#fff;text-decoration:none;font-size:13px;font-weight:600;border:1px solid rgba(255,255,255,0.25);transition:all 0.2s}
.nav-links a:hover{background:rgba(255,255,255,0.15);border-color:rgba(255,255,255,0.5)}
.nav-links a.docs-link{background:rgba(255,255,255,0.12)}
.nav-links a.board-link{background:rgba(255,107,53,0.4);border-color:rgba(255,107,53,0.5)}
.form{padding:28px 30px 14px}
.row{display:flex;gap:14px;margin-bottom:16px}
.field{flex:1}
.field label{display:block;font-size:13px;font-weight:600;color:#4a5568;margin-bottom:5px}
.field input,.field select{width:100%;padding:10px 14px;border:1px solid #e2e8f0;border-radius:8px;font-size:15px;transition:border 0.2s;background:#f8fafc}
.field input:focus,.field select:focus{outline:none;border-color:#ff6b35;background:#fff}
.field .hint{font-size:11px;color:#a0aec0;margin-top:3px}
.btn{display:block;width:100%;padding:13px;background:linear-gradient(135deg,#ff6b35,#e8551a);color:#fff;border:none;border-radius:10px;font-size:17px;font-weight:700;cursor:pointer;margin:20px 0 10px;transition:all 0.2s}
.btn:hover{transform:translateY(-1px);box-shadow:0 4px 16px rgba(255,107,53,0.3)}
.btn:disabled{opacity:0.5;cursor:not-allowed;transform:none}
.result{background:#f8fafc;border-top:1px solid #e2e8f0;padding:24px 30px;display:none}
.result.show{display:block}
.result h3{font-size:18px;color:#1a2332;margin-bottom:14px}
.r-row{display:flex;gap:12px;margin-bottom:10px}
.r-card{flex:1;background:#fff;border-radius:10px;padding:14px 16px;text-align:center;border:1px solid #e2e8f0}
.r-card .val{font-size:26px;font-weight:900;margin-bottom:4px}
.r-card .lbl{font-size:12px;color:#8899aa}
.surge{color:#dc2626}.normal-c{color:#2563eb}.discount{color:#16a34a}.mild{color:#d97706}
.zone-tag{display:inline-block;padding:5px 14px;border-radius:20px;font-size:15px;font-weight:700;margin-top:8px}
.zone-surge{background:#fef2f2;color:#dc2626}.zone-normal{background:#eff6ff;color:#2563eb}
.zone-discount{background:#f0fdf4;color:#16a34a}.zone-mild{background:#fffbeb;color:#d97706}
.reason-box{background:#fff;border:1px solid #e2e8f0;border-radius:8px;padding:14px;margin-top:12px;font-size:14px;color:#4a5568;line-height:1.6}
.error-box{background:#fef2f2;border:1px solid #fecaca;border-radius:8px;padding:14px;color:#dc2626;font-size:14px;margin-top:10px;display:none}
</style>
</head>
<body>
<div class="container">
<div class="header">
<h1>城市共享单车动态定价演示</h1>
<p class="sub">输入站点信息 → 预测需求 → 定价建议</p>
<div class="nav-links">
<a href="/docs" class="docs-link">API 文档</a>
<a href="/dashboard" class="board-link">数据分析大屏</a>
</div>
</div>
<div class="form">
<div class="row">
<div class="field">
<label>站点 ID</label>
<input type="text" id="stationId" value="6535.04" placeholder="如 6535.04">
<span class="hint">CitiBike 站点编号</span>
</div>
<div class="field">
<label>时间</label>
<input type="text" id="timestamp" value="2026-06-15 08:00" placeholder="yyyy-MM-dd HH:mm">
<span class="hint">预测的时间点</span>
</div>
</div>
<div class="row">
<div class="field">
<label>当前可用车辆数</label>
<input type="number" id="bikes" value="10" min="0">
<span class="hint">此刻该站还剩几辆车</span>
</div>
<div class="field">
<label>当前空桩位数</label>
<input type="number" id="docks" value="15" min="0">
<span class="hint">此刻该站有几个空位可还车</span>
</div>
</div>
<button class="btn" onclick="doPredict()">查询预测 & 定价建议</button>
</div>
<div class="result" id="resultArea">
<h3>预测结果</h3>
<div class="r-row">
<div class="r-card"><div class="val" id="rBike">-</div><div class="lbl">预测借出（辆）</div></div>
<div class="r-card"><div class="val" id="rDock">-</div><div class="lbl">预测还入（辆）</div></div>
<div class="r-card"><div class="val" id="rPrice">-</div><div class="lbl">建议价格 (USD)</div></div>
<div class="r-card"><div class="val" id="rMult">-</div><div class="lbl">定价乘数</div></div>
</div>
<div id="rZone" style="text-align:center;"></div>
<div class="reason-box" id="rReason" style="display:none;"></div>
<div class="error-box" id="errorBox"></div>
</div>
</div>
<script>
async function doPredict(){
const btn=document.querySelector('.btn');
const errBox=document.getElementById('errorBox');
errBox.style.display='none';
const resultArea=document.getElementById('resultArea');
btn.disabled=true;
btn.textContent='查询中...';
try{
const res=await fetch('/api/v1/predict',{
method:'POST',
headers:{'Content-Type':'application/json'},
body:JSON.stringify({
station_id:document.getElementById('stationId').value,
timestamp:document.getElementById('timestamp').value,
available_bikes:parseInt(document.getElementById('bikes').value)||0,
available_docks:parseInt(document.getElementById('docks').value)||0,
capacity:50
})
});
if(!res.ok){const e=await res.json();throw new Error(e.detail||'请求失败')}
const d=await res.json();
resultArea.classList.add('show');
document.getElementById('rBike').textContent=d.predicted_bike_demand;
document.getElementById('rDock').textContent=d.predicted_dock_demand;
document.getElementById('rPrice').textContent='$'+d.suggested_price_usd;
document.getElementById('rMult').textContent=d.pricing_multiplier+'x';
const zoneDiv=document.getElementById('rZone');
const zone=d.pricing_zone;
let zoneCls='zone-normal';
if(zone==='Surge')zoneCls='zone-surge';
else if(zone==='Mild Surge')zoneCls='zone-mild';
else if(zone==='Discount')zoneCls='zone-discount';
zoneDiv.innerHTML='<span class="zone-tag '+zoneCls+'">'+zone+'</span>';
const reasonBox=document.getElementById('rReason');
if(d.reason){reasonBox.style.display='block';reasonBox.textContent='说明：'+d.reason;}
}catch(err){
errBox.style.display='block';
errBox.textContent='错误：'+err.message;
resultArea.classList.remove('show');
}finally{
btn.disabled=false;
btn.textContent='查询预测 & 定价建议';
}
}
</script>
</body>
</html>"""


@app.get("/health", tags=["页面"])
async def health():
    """服务健康检查"""
    return {
        "status": "ok",
        "service": "CitiBike Dynamic Pricing API",
        "version": "1.0.0",
    }
