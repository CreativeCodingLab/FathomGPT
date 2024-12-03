# FathomGPT: A Natural Language Interface for Interactively Exploring Ocean Science Data

| [Demo](https://chew-z.ecn.purdue.edu/) | [Paper](https://dl.acm.org/doi/10.1145/3654777.3676462) |

FathomGPT is an open source system for the interactive investigation of ocean science images and data via a natural language web interface. It was designed in close collaboration with marine scientists to enable researchers and ocean enthusiasts to explore and analyze the FathomNet database. 

FathomGPT introduces a custom information retrieval pipeline that leverages OpenAI’s GPT technologies to enable: the creation of complex database queries to retrieve images, taxonomic information, and scientific measurements; mapping common names and morphological features to scientific names; generating interactive charts on demand; and searching by image or specified patterns within an image. In designing FathomGPT, particular emphasis was placed on the user experience, facilitating free-form exploration and optimizing response times.

https://github.com/user-attachments/assets/72f6ba39-d185-4715-b9f9-8a30cb3aaae6

## Usage Scenarios

In this scenario, the scientist wants to compare information about deep sea creatures. FathomGPT would first use name resolution to find creatures that match the description in the prompt. It would then generate a scatterplot of the data to help the user understand the relationships between the temperature and pressure levels.

<img width="471" alt="usecase-deepsea (1)" src="https://github.com/user-attachments/assets/5ee714c1-a099-4a6a-bb26-2ec888f882a8">
<img width="489" alt="usecase-scatterplot (1)" src="https://github.com/user-attachments/assets/f6a2d92f-70a4-4aba-8e6f-c253daf15be9">

FathomGPT is also able to search for images similar to an image that the user uploaded. This would help the user identify species and allow them to ask followup questions to learn more.

<img width="471" alt="usecase-similar-jellyfish (1)" src="https://github.com/user-attachments/assets/6f67a5b1-187a-4402-a976-ac07910e52b0">

## Architecture

Here we show a high-level pipeline of how an input prompt is processed by the system to produce a JSON output response, which is sent to the frontend webpage to be rendered.

<img width="654" alt="Architecture (1)" src="https://github.com/user-attachments/assets/488b2d53-4339-4db0-b098-20c4fc2b2e4c">

We use an LLM (GPT-3.5) to understand the user prompt and determine which functions to call (eg. name resolution, sql generation, fetch taxonomy tree, etc). 

Name resolution converts common names or descriptions into scientific names that could be used to fetch data from the FathomNet database. We use knowledge graph alignment between the prompt and the species data to resolve descriptions (eg. habitat, morphology, predator/prey relations).

We use an LLM to convert natural language prompts into SQL queries. We fine-tuned it to work specifically for the FathomNet database.

