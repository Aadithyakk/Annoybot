import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

resp = client.responses.create(
    model="gpt-4.1-mini",
    input="Reply with exactly: OK",
    max_output_tokens=32,
)

# Collect output text
out = []

for item in resp.output:
    if item.type == "message":
        for c in item.content:
            if c.type == "output_text":
                out.append(c.text)

print("RESULT:", " ".join(out).strip())
