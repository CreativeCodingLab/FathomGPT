from functions import *
from constants import *

import openai
import json

openai.api_key = KEYS['openai']

def call_function(function_call):
    function_name = function_call.get('name')
    args = json.loads(function_call.get('arguments'))

    for key, value in args.items():
        if isinstance(value, str):
            if value == "True":
                args[key] = True
            elif value == "False":
                args[key] = False
            else:
                try:
                    args[key] = float(value) if '.' in value else int(value)
                except ValueError:
                    pass


    function_to_call = globals().get(function_name)
    if function_to_call:
        return function_to_call(**args)
    else:
        return f"No function named '{function_name}' in the global scope"

def get_response(messages):
    return openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k-0613",
        temperature=0,
        messages=messages,
        functions=AVAILABLE_FUNCTIONS,
        function_call="auto",
    )

def run_prompt(prompt, messages=[]):
    messages.append({"role": "user", "content": prompt})
    chatResponse = get_response(messages)
    if DEBUG_LEVEL >= 1:
        print(chatResponse)
    messages.append(
        {"role": "assistant", "content": json.dumps(chatResponse.choices[0])},
    )
    if hasattr(chatResponse.choices[0].message, 'function_call'):
        function_response = call_function(chatResponse.choices[0].message.function_call)
        messages.append({"role": "function", "name": chatResponse.choices[0].message.function_call.name, "content": json.dumps(function_response)})
        messages.append({"role": "user", "content": "Summarize the last function content in a human readable format"})
        summaryResponse = get_response(messages)
        if(chatResponse.choices[0].message.function_call.name=='findImages'):
            return {'images': function_response, 'text': summaryResponse.choices[0].message.content}
        else:
            return {'text': summaryResponse.choices[0].message.content}
    else:
        return {'text': chatResponse.choices[0].message.content}
