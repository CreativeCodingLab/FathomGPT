from llm import run_prompt

import json

#TEST_PROMPT = "Find me images of species 'Aurelia aurita'"
#TEST_PROMPT = "Find me 3 newest images of species 'Aurelia aurita' within 100km from San Francisco and depth less than 5000m"
TEST_PROMPT = "Find me image of species 'Aurelia aurita' in the Pacific or Atlantic oceans and depth less than 5000m"
#TEST_PROMPT = "Find me images of species 'Aurelia aurita' in Monterey Bay and depth less than 5000m"
#TEST_PROMPT = "Find me 3 images of 'Aurelia aurita' in Monterey Bay"
#TEST_PROMPT = "Find me images of the closest relatives of 'Aegina rosea' in Monterey Bay"
#TEST_PROMPT = "Find me images of species commonly found with 'Sebastolobus'"
#TEST_PROMPT = "Find me images of 'Sebastolobus' by themselves"
#TEST_PROMPT = "Find me images only containing 'Sebastolobus'"
#TEST_PROMPT = "Find me good images of species 'Aurelia aurita'"
#TEST_PROMPT = "Find me images of Moon Jelly"

print(json.dumps(run_prompt(TEST_PROMPT)))
