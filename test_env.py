from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv('OPENAI_API_KEY')
if api_key:
  print(" 环境变量设置成功")
  print(f"API Key 前缀: {api_key[:10]}...")
else:
  print(" 环境变量未设置")