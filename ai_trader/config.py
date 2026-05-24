"""
AlphaMind AI Trader V2 - 配置管理模块
"""
import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class DataSourceConfig:
    """数据源配置"""
    name: str
    enabled: bool = True
    priority: int = 0  # 优先级，数字越小优先级越高
    weight: float = 1.0  # 交叉验证时的权重


@dataclass
class TradingConfig:
    """交易配置"""
    initial_balance: float = 100000.0  # 初始资金
    commission_rate: float = 0.0003  # 佣金费率（万分之三）
    stamp_tax_rate: float = 0.001  # 印花税率（千分之一，仅卖出时收取）
    min_trade_amount: float = 100.0  # 最小交易金额
    max_position_ratio: float = 0.3  # 单只股票最大仓位比例


@dataclass
class AppConfig:
    """应用配置"""
    # 数据源配置
    data_sources: List[DataSourceConfig] = field(default_factory=lambda: [
        DataSourceConfig(name="akshare", enabled=True, priority=1, weight=1.0),
        DataSourceConfig(name="tushare", enabled=False, priority=2, weight=0.3),
        DataSourceConfig(name="yfinance", enabled=False, priority=3, weight=0.2),
    ])
    
    # 交叉验证配置
    cross_validation: bool = True  # 是否启用交叉验证
    max_price_diff_ratio: float = 0.02  # 最大允许价格差异比例（2%）
    
    # 交易配置
    trading: TradingConfig = field(default_factory=TradingConfig)
    
    # Tushare Token（可选，免费版有调用限制）
    tushare_token: str = field(default="")
    
    @classmethod
    def from_env(cls) -> 'AppConfig':
        """从环境变量加载配置"""
        config = cls()
        
        # 从环境变量读取 Tushare Token
        tushare_token = os.getenv("TUSHARE_TOKEN", "")
        if tushare_token:
            config.tushare_token = tushare_token
        
        return config


# 全局配置实例
config = AppConfig.from_env()


# ---------------------------------------------------------------------------
# A-Share 智能推荐系统配置
# ---------------------------------------------------------------------------

@dataclass
class AShareConfig:
    """A股智能推荐系统配置。"""

    # 关键指数
    key_indices: dict = field(default_factory=lambda: {
        "上证综指": "sh000001",
        "深证成指": "sz399001",
        "创业板指": "sz399006",
        "科创50": "sh000688",
        "沪深300": "sh000300",
        "中证500": "sh000905",
    })

    # 用户关注板块关键词
    focus_keywords: list = field(default_factory=lambda: [
        "人工智能", "AI", "ChatGPT", "AIGC", "大模型", "智能",
        "半导体", "芯片", "集成电路", "封测", "光刻",
        "创新药", "生物医药", "CRO", "CXO", "医药",
        "算力", "服务器", "光模块", "CPO", "液冷",
        "存储", "HBM", "闪存", "内存",
        "消费电子", "机器人", "自动驾驶", "低空经济",
        "科创", "创业板", "科创板",
    ])

    # 无效板块过滤关键词
    exclude_keywords: list = field(default_factory=lambda: [
        "融资融券", "昨日涨停", "昨日跌停", "昨日连板", "昨日",
        "转债", "百元股", "基金重仓", "券商重仓", "机构重仓",
        "信托重仓", "保险重仓", "社保持仓", "QFII", "MSCI",
        "罗素", "行业龙头", "大盘", "小盘",
    ])

    # 板块打分权重
    weight_pct: float = 0.30
    weight_flow: float = 0.30
    weight_anomaly: float = 0.20
    weight_volume: float = 0.20

    # 市场状态阈值
    trend_up: float = 0.4
    trend_down: float = -0.4
    breadth_strong: float = 2.0
    breadth_weak: float = 0.5

    # 回踩检测
    pullback_min: float = -10.0
    pullback_max: float = -1.0
    max_dist_to_ma20: float = 5.0

    # 仓位参数
    alloc_attack: float = 0.22
    alloc_range: float = 0.15
    alloc_rebound: float = 0.12
    alloc_adjust: float = 0.10
    alloc_defense: float = 0.08
    max_single_alloc: float = 0.25

    # 止损
    stop_loss_pct: float = 0.065

    # 分批比例
    tranche_ratios: tuple = (0.50, 0.30, 0.20)
    tranche_offsets: tuple = (0.0, -0.02, -0.05)

    # 推荐数量
    max_recommendations: int = 3
    top_sectors_display: int = 30

    # 默认资金
    default_capital: float = 100000.0


ashare_config = AShareConfig()
