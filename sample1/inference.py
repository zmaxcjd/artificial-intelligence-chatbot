# inference.py
from transformers import AutoModelForCausalLM, AutoTokenizer
from model_client import OllamaClient

class ChatInference:
    def __init__(self, **client_kwargs):
        # 1. 加载分词器和模型
        #self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        #self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        #self.device = device
        self.client = OllamaClient(**client_kwargs)
    def generate_reply(self, prompt: str, history: list[str], max_tokens=100) -> str:
        # 拼接上下文
        if history:
            full_prompt = "\n".join(history) + "\n" + prompt
        else:
            full_prompt = prompt

        # 调用 OllamaClient
        reply = self.client.generate(
            full_prompt,
            max_tokens=max_tokens,
            temperature=0.8
        )
        return reply
if __name__ == "__main__":
    bot = ChatInference(device="cpu")
    print(bot.generate_reply("你好！", history=[]))
