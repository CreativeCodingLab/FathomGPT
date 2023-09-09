### Install
Tested with Python3
```
pip install fathomnet
pip install openai
```

### Running
Create a `.env` file in this folder containing:
```
OPENAI_KEY=<your openai key>
```

Run: 
```
python debug.py
```
To change the amount of debugging messages, change the `DEBUG_LEVEL` in `constants.py`.

### Unit testing
```
python test.py <test_name1> <test_name2> ...
```
If you run without args, it will run all unit tests. That could take a while and could be costly to run too often.

### Documentation
- `debug.py` - For debugging. Not used in production
- `test.py` - Unit tests. Not used in production
- `llm.py` - Main function `run_prompt` is called by the client-side. Processes the input prompt with LLM
- `functions.py` - Functions called by the LLM to fetch data from FathomNet
- `utils.py` - Util functions used by `functions.py` to pre-process the arguments or post-process the fetched data
- `constants.py` - Default values, function arguments used by the LLM, and functions available to the LLM. Also fetches API keys from the secret `.env` file
