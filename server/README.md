### Install
Tested with Python3
```
pip install fathomnet
pip install openai
```

### Test
Obtain the `.env` file containing the secret API keys and save it in the same directory

Run: 
```
python test.py
```
To change the amount of debugging messages, change the `DEBUG_LEVEL` in `constants.py`.

### Documentation
- `test.py` - For debugging. Not used in production
- `llm.py` - Main function `run_prompt` is called by the client-side. Processes the input prompt with LLM
- `functions.py` - Functions called by the LLM to fetch data from FathomNet
- `utils.py` - Util functions used by `functions.py` to pre-process the arguments or post-process the fetched data
- `constants.py` - Default values, function arguments used by the LLM, and functions available to the LLM. Also fetches API keys from the secret `.env` file
