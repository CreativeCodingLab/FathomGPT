from llm import run_prompt
import sys


TEST_CASES = {
    "basic": {
      "input": "Find me images of species 'Aurelia aurita'",
      "expected": {
        'concept': ['Aurelia aurita']
      },
      "has_only": True,
      "num_imgs": 5,
    },
    "distanceFrom": {
      "input": "Find me 3 newest images of species 'Aurelia aurita' within 100km from San Francisco and depth less than 5000m",
      "expected": {},
    },
    "bodyOfWater": {
      "input": "Find me images of species 'Aurelia aurita' in Monterey Bay and depth less than 5000m",
      "expected": {},
    },
    "bodiesOfWater": {
      "input": "Find me image of species 'Aurelia aurita' in the Pacific or Atlantic oceans and depth less than 5000m",
      "expected": {},
    },
    "goodImages": {
      "input": "Find me good images of species 'Aurelia aurita'",
      "expected": {},
    },
    "closestRelatives": {
      "input": "Find me images of the closest relatives of 'Aegina rosea' in Monterey Bay",
      "expected": {},
    },
    "commonlyFoundWith": {
      "input": "Find me images of species commonly found with 'Sebastolobus'",
      "expected": {},
    },
    "excludeOthers": {
      "input": "Find me images of 'Sebastolobus' by themselves",
      "expected": {},
    },
    "commonName": {
      "input": "Find me images of Moon Jellyfish",
      "expected": {},
    },
    "commonNameCategory": {
      "input": "Find me images of jellyfish",
      "expected": {},
    },
    "bestImages": {
      "input": "Find me the best images containing only octopus",
      "expected": {},
    },
    "worstImages": {
      "input": "Find me the worst images containing only octopus",
      "expected": {},
    },
    "stressTest": {
      "input": "Find me 5000 images of species Strongylocentrotus fragilis",
      "expected": {},
    }
}

if len(sys.argv) > 0:
    test_names = sys.argv
else:
    test_names = TEST_CASES

def get_concepts(images):
    concepts = []
    for img in images:
        concepts.extend([box['concept'] for box in img['boundingBoxes']])
    return set(concepts)

def run_tests()
    for test in test_names[:1]:
        t = TEST_CASES[test]
        print("Running test: "+test)
        output = run_prompt(t['input'])
        print(output)
        
        vals = {
            'uuid': [img['uuid'] for img in output['images']]
            'concepts' list(get_concepts(img))
        }
        
        for param in t['expected']:
            for expected in t['expected'][param]:
                if expected not in vals['param']:
                    print('Test output:')
                    print(output)
                    print('Expected '+param+':')
                    print(t['expected'])
                    print('Test'+param+':')
                    print(vals['param'])
                    print('Missing: '+expected)
                    return test
    return None

failed_test = run_tests()
if failed_test is None:
    print('All tests PASSED')
else:
    print('Test FAILED: '+failed_test)
