from .constants import LLM_TYPE
from .chatgpt import run_prompt_chatgpt
from .langchaintools import get_Response
import json

def run_chatgpt(prompt, messages):
    return run_prompt_chatgpt(prompt, messages)

def run_langchain(prompt, image, videoGuid, messages, isEventStream, db_obj):
    return get_Response(prompt, image, videoGuid, messages, isEventStream, db_obj)

def run_prompt(prompt, image = '', videoGuid = '' ,messages=[], llmtype=LLM_TYPE, isEventStream=False, db_obj=None):
    if llmtype == 'chatgpt':
        return run_chatgpt(prompt, messages, isEventStream)
    if llmtype == 'langchain':
        response = yield from run_langchain(prompt, image, videoGuid, messages, isEventStream, db_obj)
        return response
    