import os
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
client = OpenAI()

resp = client.responses.create(
    model="gpt-4.1-mini",
    input="Reply with exactly: OK",
    max_output_tokens=32,
)

# Print output text
out = []
for item in resp.output:
    if item.type == "message":
        for c in item.content:
            if c.type == "output_text":

print("RESULT:", " ".join(out).strip())
