import requests
import logging


# 配置日志，方便调试
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OllamaClient:
    def __init__(self, host="localhost", port=11434, model="llama3"):
        """
        host, port: Ollama 服务地址
        model: 在 ollama run 时指定的模型名
        """
        self.url = f"http://{host}:{port}/api/generate"
        self.model = model

    def generate(self, prompt: str, stream: bool = False, **kwargs) -> str:
        """
        调用 Ollama 服务生成回复。
        prompt: 用户输入或拼接好的上下文字符串
        stream: 是否使用流式输出（默认 False）
        kwargs: 额外参数可传入，比如 max_tokens, temperature
        返回: 模型完整响应文本
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": stream,
            **kwargs
        }
        try:
            resp = requests.post(self.url, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            return data.get("response", "")
        except Exception as e:
            logger.error(f"调用 Ollama 接口失败：{e}")
            return "[模型调用出错，请稍后重试]"

if __name__ == "__main__":
    client = OllamaClient()
    reply = client.generate("你好，Ollama！", max_tokens=50, temperature=0.7)
    print("模型回复：", reply)
