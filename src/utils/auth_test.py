import jwt
# 请确保已安装 PyJWT 库，如果未安装，请先运行 pip install pyjwt
# 请确保已安装 PyJWT 库，如果未安装，请先运行 pip install pyjwt
# 假设项目中使用的是 PyJWT 库，如果未安装，请先运行 pip install pyjwt
import datetime
from datetime import timedelta
from typing import Dict

# 假设这是从项目中获取的 secret key
SECRET_KEY = "your_secret_key_here"

# 模拟用户信息
username = "admin_zrj"
exp = datetime.datetime.utcnow() + timedelta(hours=1)  # token 过期时间

# 构造 payload
payload = {
    "user": username,
    "exp": exp
}

# 生成 token
try:
    token = jwt.encode(payload, SECRET_KEY, algorithm="HS256")
    print("Generated JWT Token:", token.decode("utf-8"))
except jwt.PyJWTError as e:
    print("JWT encoding error:", e)
except Exception as e:
    print("Other error:", e)