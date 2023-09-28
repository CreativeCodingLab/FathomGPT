from constants import LLM_TYPE
from chatgpt import run_prompt_chatgpt
from langchaintools import initLangchain, get_Response


def run_chatgpt(prompt, messages):
    return run_prompt_chatgpt(prompt, messages)

def run_langchain(prompt, messages):
    agent_chain = initLangchain()
    return get_Response(prompt, agent_chain)

def run_prompt(prompt, messages=[]):
    if LLM_TYPE == 'chatgpt':
        return run_chatgpt(prompt, messages)
    if LLM_TYPE == 'langchain':
        return run_langchain(prompt, messages)
    