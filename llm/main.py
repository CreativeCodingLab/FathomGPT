from .constants import LLM_TYPE
from .chatgpt import run_prompt_chatgpt
from .langchaintools import get_Response
import json

def run_chatgpt(prompt, messages):
    return run_prompt_chatgpt(prompt, messages)

def run_langchain(prompt, messages, isEventStream):
    return get_Response(prompt, messages, isEventStream)

def run_prompt(prompt, messages=[], llmtype=LLM_TYPE, isEventStream=False):
    if llmtype == 'chatgpt':
        return run_chatgpt(prompt, messages, isEventStream)
    if llmtype == 'langchain':
        response = yield from run_langchain(prompt, messages, isEventStream)
        return response
    