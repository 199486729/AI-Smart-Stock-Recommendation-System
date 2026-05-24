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
