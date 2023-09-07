This folder contains scripts used for data pipelining, model training, generating files, etc.

These scripts should not run in production

- `pull_fathomnet_data.py`: Pull data for each concept from FathomNet (metadata only, not image files) and saves them locally as json (1 json file per concept). The retrieved data is available in `data/fathomnet.zip`
- `pull_wikipedia_data.py`: Pull data from Wikipedia for each concept. If the Wikipedia page doesn't exist, use the ancestor's page as the concept's page
- `extract_wikipedia_features.py`: Extract common names and categories from the 1st paragraph of each concept's Wikipedia page and save as a JSON mapping: `"common name"/"category": concept names list`
