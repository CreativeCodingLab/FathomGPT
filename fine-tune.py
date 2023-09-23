import os
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.File.create(
  file=open("without-observation.jsonl", "rb"),
  purpose='fine-tune'
)