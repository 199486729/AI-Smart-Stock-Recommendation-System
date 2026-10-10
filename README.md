# AI-Smart-Stock-Recommendation-System

面向 A 股市场的智能分析与推荐系统，提供市场状态判断、板块强弱排名、股票与 ETF 筛选、AI 对话、交易记录和推荐历史等功能。

## 主要功能

- 邮箱和密码注册登录，密码使用 bcrypt 哈希保存
- 登录或注册成功后自动获取一次市场分析
- 大盘、市场宽度、资金流、板块和个股行情分析
- 股票与 ETF 推荐及仓位建议
- AI 行情问答、用户偏好记忆和历史会话
- 模拟交易与真实交易记录
- 30 秒分析缓存及并发请求合并

## 环境要求

- Python 3.10+
- MySQL 8.0+
- Linux 服务器推荐使用项目自带的 `run.sh`

## 安装

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

编辑 `.env`，填写数据库连接、Flask 会话密钥和 AI 接口配置。生成随机会话密钥：

```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

如需更换分析页面地址，可把 `ASHARE_PATH` 设置为只包含字母、数字、下划线或连字符的私有路径；修改后原 `/ashare` 会返回 404。

初始化数据库：

```bash
python -m ai_trader.init_db
```

## 启动

开发环境：

```bash
python -m ai_trader.web_app
```

Linux 双实例运行：

```bash
chmod +x run.sh
./run.sh start
./run.sh status
```

默认访问地址为 `http://服务器地址:5001/login`。

## 安全说明

- `.env` 已被 Git 忽略，请勿提交真实密码、Token 或会话密钥。
- 生产环境应通过 Nginx 等反向代理提供 HTTPS。
- 系统生成的分析仅供研究参考，不构成投资建议。
