from .constants import LLM_TYPE
from .chatgpt import run_prompt_chatgpt
from .langchaintools import get_Response


def run_chatgpt(prompt, messages):
    return run_prompt_chatgpt(prompt, messages)

def run_langchain(prompt, messages):
    return get_Response(prompt, messages)

def run_prompt(prompt, messages=[], llmtype=LLM_TYPE):
    if llmtype == 'chatgpt':
        return run_chatgpt(prompt, messages)
    if llmtype == 'langchain':
        return run_langchain(prompt, messages)
    