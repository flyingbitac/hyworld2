import base64
from openai import OpenAI

img = "/workspace/hyworld2/examples/worldgen/case000/panorama.png"
b64 = base64.b64encode(open(img, "rb").read()).decode()
client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": [
        {"type": "text", "text": "Judge whether this panoramic image is a 'indoor' or 'outdoor' scene. Return only 'indoor' if the image shows indoor scenes, or 'outdoor' if it does not."},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
    ]},
]
r = client.chat.completions.create(model="Qwen/Qwen3-VL-8B-Instruct", messages=messages, max_tokens=1024, temperature=0.0, seed=1024)
print("ANSWER:", repr(r.choices[0].message.content.strip()))
