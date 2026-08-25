"""FastAPI 请求/响应数据格式定义（Pydantic）"""

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    station_id: str = Field(..., example="6535.04", description="CitiBike 站点编号")
    timestamp: str = Field(
        ..., example="2026-06-15 08:00",
        description="预测时间点，格式 'yyyy-MM-dd HH:mm'"
    )
    is_holiday: bool = Field(False, description="该日期是否是美国联邦假日")
    available_bikes: int = Field(15, description="站点当前可用车辆数")
    available_docks: int = Field(15, description="站点当前空桩位数（可还车的位置数）")
    capacity: int = Field(50, description="站点总桩位容量")


class PredictionResponse(BaseModel):
    station_id: str = Field(..., description="站点编号")
    timestamp: str = Field(..., description="预测的时间点")
    predicted_bike_demand: float = Field(..., description="预测借出量（辆）")
    predicted_dock_demand: float = Field(..., description="预测还入量（辆）")
    pricing_multiplier: float = Field(..., description="定价乘数（1.0=原价, >1.0=涨价, <1.0=降价）")
    suggested_price_usd: float = Field(..., description="建议价格（美元）")
    pricing_zone: str = Field(..., description="定价区域：Surge(涨价)/Mild Surge(温和涨价)/Normal(原价)/Discount(降价)")
    reason: str = Field(..., description="定价决策的文字说明")


class StationForecastRequest(BaseModel):
    station_id: str = Field(..., example="6535.04", description="CitiBike 站点编号")
    capacity: int = Field(50, description="站点总桩位容量")
    available_bikes: int = Field(15, description="当前可用车辆数")
    available_docks: int = Field(15, description="当前空桩位数")


class ForecastPoint(BaseModel):
    timestamp: str = Field(..., description="该预测点的时间")
    predicted_bike_demand: float = Field(..., description="预测借出量")
    predicted_dock_demand: float = Field(..., description="预测还入量")
    pricing_multiplier: float = Field(..., description="定价乘数")
    suggested_price_usd: float = Field(..., description="建议价格")
    pricing_zone: str = Field(..., description="定价区域")


class StationForecastResponse(BaseModel):
    station_id: str = Field(..., description="站点编号")
    capacity: int = Field(..., description="站点桩位容量")
    forecast: list[ForecastPoint] = Field(..., description="未来 24 小时逐时预测列表")


class HotspotItem(BaseModel):
    station_id: str = Field(..., description="站点编号")
    avg_price: float = Field(..., description="该站平均建议价格")
    avg_multiplier: float = Field(..., description="该站平均定价乘数")
    surge_pct: float = Field(..., description="该站 Surge 时段占比（%）")


class HotspotResponse(BaseModel):
    surge_stations: list[HotspotItem] = Field(..., description="需要涨价的前 5 个站点")
    discount_stations: list[HotspotItem] = Field(..., description="需要降价的前 5 个站点")
