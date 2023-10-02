from main import run_prompt

import json

#TEST_PROMPT = "Find me images of species 'Aurelia aurita'"
#TEST_PROMPT = "Find me 3 newest images of species 'Aurelia aurita' within 100km from San Francisco and depth less than 5000m"
#TEST_PROMPT = "Find me image of species 'Aurelia aurita' in the Pacific or Atlantic oceans and depth less than 5000m"
#TEST_PROMPT = "Find me images of species 'Aurelia aurita' in Monterey Bay and depth less than 5000m"
#TEST_PROMPT = "Find me 3 images of 'Aurelia aurita' in Monterey Bay"
#TEST_PROMPT = "Find me 3 images of the closest relatives of moon jellyfish in Monterey Bay"
#TEST_PROMPT = "Find me images of species commonly found with 'Sebastolobus'"
#TEST_PROMPT = "Find me images of 'Sebastolobus' by themselves"
#TEST_PROMPT = "Find me images only containing 'Sebastolobus'"
#TEST_PROMPT = "Find me good images of species 'Aurelia aurita'"
#TEST_PROMPT = "Find me images of Moon Jelly fish"
#TEST_PROMPT = "Find me images of jellyfish"
#TEST_PROMPT = "Find me the best images containing only octopus"
#TEST_PROMPT = "Find me the worst images containing only octopus"
#TEST_PROMPT = "Find me images of species Strongylocentrotus fragilis"
#TEST_PROMPT = "Find me images of creatures with tentacles in Monterey Bay"
#TEST_PROMPT = "Provide the data that correlates depth with the distribution of Moon jellyfish in Monterey Bay"
#TEST_PROMPT = "Provide the data for images of Moon jellyfish in Monterey Bay" # badly phrased prompts cause langchain to fail
#TEST_PROMPT = "Find me 3 images of creatures in Monterey Bay"
#TEST_PROMPT = "Find me 3 images of creatures with tentacles in Monterey Bay"
#TEST_PROMPT = "Find me 3 images of moon jellyfish in Monterey Bay sorted by depth"
#TEST_PROMPT = "Find me a list of creatures frequently found near Aurelia aurita"
TEST_PROMPT = "Find me the best images of Aurelia aurita"

print(run_prompt(TEST_PROMPT))
