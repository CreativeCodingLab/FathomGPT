from llm import run_prompt


TEST_CASES = [
    {
      "name": "basic",
      "input": "Find me images of species 'Aurelia aurita'",
      "expected_uuids": [],
    },
    {
      "name": "distanceFrom",
      "input": "Find me 3 newest images of species 'Aurelia aurita' within 100km from San Francisco and depth less than 5000m",
      "expected_uuids": [],
    },
    {
      "name": "bodyOfWater",
      "input": "Find me images of species 'Aurelia aurita' in Monterey Bay and depth less than 5000m",
      "expected_uuids": [],
    },
    {
      "name": "bodiesOfWater",
      "input": "Find me image of species 'Aurelia aurita' in the Pacific or Atlantic oceans and depth less than 5000m",
      "expected_uuids": [],
    },
    {
      "name": "goodImages",
      "input": "Find me good images of species 'Aurelia aurita'",
      "expected_uuids": [],
    },
    {
      "name": "closestRelatives",
      "input": "Find me images of the closest relatives of 'Aegina rosea' in Monterey Bay",
      "expected_uuids": [],
    },
    {
      "name": "commonlyFoundWith",
      "input": "Find me images of species commonly found with 'Sebastolobus'",
      "expected_uuids": [],
    },
    {
      "name": "excludeOthers",
      "input": "Find me images of 'Sebastolobus' by themselves",
      "expected_uuids": [],
    },
    {
      "name": "commonName",
      "input": "Find me images of Moon Jellyfish",
      "expected_uuids": [],
    },
    {
      "name": "commonNameCategory",
      "input": "Find me images of jellyfish",
      "expected_uuids": [],
    },
    {
      "name": "bestImages",
      "input": "Find me the best images containing only octopus",
      "expected_uuids": [],
    },
    {
      "name": "worstImages",
      "input": "Find me the worst images containing only octopus",
      "expected_uuids": [],
    },
]

for test in TEST_CASES:
    output = run_prompt(test['input'])
    # todo: implement unit tests
