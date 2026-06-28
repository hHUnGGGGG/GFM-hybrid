This guide explains how to create a knowledge graph index and process QA data that can be used by GFM-RAG for training and testing.

## Data Preparation
Please follow the instructions in the [Data Preparation](data_preparation.md) to prepare your dataset in the following structure:

```
data_name/
├── raw/
│   ├── dataset_corpus.json
│   ├── train.json # (optional)
│   └── test.json # (optional)
└── processed/ # Output directory
```

## Config
You need to create a KG-index configuration file.

??? example "gfmrag/workflow/config/stage1_index_dataset.yaml"

    ```yaml title="gfmrag/workflow/config/stage1_index_dataset.yaml"
    --8<-- "gfmrag/workflow/config/stage1_index_dataset.yaml"
    ```

Details of the configuration parameters are explained in the [KG-index Configuration][kg-index-configuration] page.

## Index dataset

This method performs two main tasks:

1. Creates and saves knowledge graph related files (`kg.txt` and `document2entities.json`) from the `dataset_corpus.json` file
2. Identify the query entities and supporting entities in training and testing data if available in the raw data directory

Files created:

- `kg.txt`: Contains knowledge graph triples
- `document2entities.json`: Maps documents to their entities
- `train.json`: Processed training data (if raw exists)
- `test.json`: Processed test data (if raw exists)

Directory structure:
```
root/
└── data_name/
	├── raw/
	│   ├── dataset_corpus.json
	│   ├── train.json (optional)
	│   └── test.json (optional)
	└── processed/
		└── stage1/
			├── kg.txt
			├── document2entities.json
			├── train.json
			└── test.json
```

To index the data, run the following command:

??? example "gfmrag/workflow/stage1_index_dataset.py"

	<!-- blacken-docs:off -->
    ```python title="gfmrag/workflow/stage1_index_dataset.py"
    --8<-- "gfmrag/workflow/stage1_index_dataset.py"
    ```
	<!-- blacken-docs:on -->

```bash
python -m gfmrag.workflow.stage1_index_dataset
```

You can overwrite the configuration like this:

```bash
python -m gfmrag.workflow.stage1_index_dataset kg_constructor.num_processes=5
```

## Output Files

### `kg.txt`
The `kg.txt` file contains the knowledge graph triples in the following format:

```
subject,relation,object
```

Example:
```
fred gehrke,was,american football player
fred gehrke,was,executive
fred gehrke,played for,cleveland   los angeles rams
```

### `document2entities.json`
The `document2entities.json` file contains the mapping of documents to their entities in the following format:

- `key`: The title or unique id of the document.
- `value`: A list of entities in the document.

Example:
```json
{
    "Fred Gehrke": [
		"1977",
		"1981",
		"american football player",
		"chicago cardinals",
		"cleveland   los angeles rams",
		"executive",
		"first painted logo on helmets",
		"fred gehrke",
		"gehrke",
		"general manager of denver broncos",
		"los angeles rams",
		"los angeles rams logo",
		"miami marlin christian yelich",
		"san francisco 49ers"
	],
	"Manny Machado": [
		"2010 major league baseball draft",
		"american",
		"baltimore orioles",
		"brito high school",
		"july 6  1992",
		"major league baseball",
		"manny machado",
		"right handed"
	],
}
```

### `train.json` and `test.json`
The `train.json` and `test.json` files contain the processed training and test data in the following format:

- `id`: A unique identifier for the example.
- `question`: The question or query.
- `supporting_facts`: A list of supporting facts for the question. Each supporting fact is a list containing the title of the document that can be found in the `dataset_corpus.json` file.
- `question_entities`: A list of entities in the question.
- `supporting_entities`: A list of entities in the supporting facts.
- Additional fields copied from the raw data

Examples
```json
[
	{
		"id": "5abc553a554299700f9d7871",
		"question": "Kyle Ezell is a professor at what School of Architecture building at Ohio State?",
		"answer": "Knowlton Hall",
		"supporting_facts": [
			"Knowlton Hall",
			"Kyle Ezell"
		],
		"question_entities": [
			"kyle ezell",
			"architectural association school of architecture",
			"ohio state"
		],
		"supporting_entities": [
			"10 million donation",
			"2004",
			"architecture",
			"austin e  knowlton",
			"austin e  knowlton school of architecture",
			"bachelor s in architectural engineering",
			"city and regional planning",
			"columbus  ohio  united states",
			"ives hall",
			"july 2002",
			"knowlton hall",
			"ksa",
			"landscape architecture",
			"ohio",
			"replacement for ives hall",
			"the ohio state university",
			"the ohio state university in 1931",
			"american urban planning practitioner",
			"expressing local culture",
			"knowlton school",
			"kyle ezell",
			"lawrenceburg  tennessee",
			"professor",
			"the ohio state university",
			"theorist",
			"undergraduate planning program",
			"vibrant downtowns",
			"writer"
		]
	},
    ...
]
```
