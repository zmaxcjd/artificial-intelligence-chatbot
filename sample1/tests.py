from inference import ChatInference

bot = ChatInference()
history = []
for _ in range(10):
    user = input("你说：")
    reply = bot.generate_reply(user, history, max_tokens=60)
    print("机器人：", reply)
    history.extend([user, reply])
