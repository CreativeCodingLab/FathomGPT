### Install
```
python -m pip install fathomnet
pip install openai
```

### Test
```
python server/test.py
```

### Documentation
The server-side code lives in the `server/` folder
- test.py - For debugging. Not used in production
- llm.py - Main function `run_prompt` is called by the client-side. Processes the input prompt with LLM
- functions.py - Functions called by the LLM to fetch data from FathomNet
- utils.py - Util functions mostly used by `functions.py` to pre-process the arguments or post-process the fetched data
- constants.py - Constants such as default values, function arguments used by the LLM, and functions available to the LLM. Also fetches API keys from the secret `.env` file
