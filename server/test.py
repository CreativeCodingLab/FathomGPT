from llm import run_prompt
import sys


TEST_CASES = {
    "basic": {
      "input": "Find me images of species 'Aurelia aurita'",
      "expected_uuids": [],
    },
    "distanceFrom": {
      "input": "Find me 3 newest images of species 'Aurelia aurita' within 100km from San Francisco and depth less than 5000m",
      "expected_uuids": [],
    },
    "bodyOfWater": {
      "input": "Find me images of species 'Aurelia aurita' in Monterey Bay and depth less than 5000m",
      "expected_uuids": [],
    },
    "bodiesOfWater": {
      "input": "Find me image of species 'Aurelia aurita' in the Pacific or Atlantic oceans and depth less than 5000m",
      "expected_uuids": [],
    },
    "goodImages": {
      "input": "Find me good images of species 'Aurelia aurita'",
      "expected_uuids": [],
    },
    "closestRelatives": {
      "input": "Find me images of the closest relatives of 'Aegina rosea' in Monterey Bay",
      "expected_uuids": [],
    },
    "commonlyFoundWith": {
      "input": "Find me images of species commonly found with 'Sebastolobus'",
      "expected_uuids": [],
    },
    "excludeOthers": {
      "input": "Find me images of 'Sebastolobus' by themselves",
      "expected_uuids": [],
    },
    "commonName": {
      "input": "Find me images of Moon Jellyfish",
      "expected_uuids": [],
    },
    "commonNameCategory": {
      "input": "Find me images of jellyfish",
      "expected_uuids": [],
    },
    "bestImages": {
      "input": "Find me the best images containing only octopus",
      "expected_uuids": [],
    },
    "worstImages": {
      "input": "Find me the worst images containing only octopus",
      "expected_uuids": [],
    },
}

if len(sys.argv) > 0:
    test_names = sys.argv
else:
    test_names = TEST_CASES

for test in test_names:
    t = TEST_CASES[test]
    print("Running test: "+test)
    output = run_prompt(t['input'])
    # todo: implement unit tests
