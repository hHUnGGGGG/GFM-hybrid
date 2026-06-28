# Build Your Own Dataset

This guide explains how to prepare your dataset for use with GFM-RAG.

## Directory Structure

You need to prepare the following files:

- `dataset_corpus.json`: A JSON file containing the entire document corpus.
- `train.json` (optional): A JSON file containing the training data.
- `test.json` (optional): A JSON file containing the test data.

Place your files in the following structure:
```
data_name/
├── raw/
│   ├── dataset_corpus.json
│   ├── train.json # (optional)
│   └── test.json # (optional)
└── processed/ # Output directory
```

## Data Format

### `dataset_corpus.json`

The `dataset_corpus.json` is a dictionary where each key is the title or unique id of a document and the value is the text of the document. Each document should be structured as follows:

- `key`: The title or unique id of the document.
- `value`: The text of the document.

Example:
```json
{
    "Fred Gehrke":
        "Clarence Fred Gehrke (April 24, 1918 – February 9, 2002) was an American football player and executive.  He played in the National Football League (NFL) for the Cleveland / Los Angeles Rams, San Francisco 49ers and Chicago Cardinals from 1940 through 1950.  To boost team morale, Gehrke designed and painted the Los Angeles Rams logo in 1948, which was the first painted on the helmets of an NFL team.  He later served as the general manager of the Denver Broncos from 1977 through 1981.  He is the great-grandfather of Miami Marlin Christian Yelich"
    ,
    "Manny Machado":
        "Manuel Arturo Machado (] ; born July 6, 1992) is an American professional baseball third baseman and shortstop for the Baltimore Orioles of Major League Baseball (MLB).  He attended Brito High School in Miami and was drafted by the Orioles with the third overall pick in the 2010 Major League Baseball draft.  He bats and throws right-handed."
    ,
    ...
 }
```

### `train.json` and `test.json`
If you want to train and evaluate the model, you need to provide training and test data in the form of a JSON file. Each entry in the JSON file should contain the following fields:

- `id`: A unique identifier for the example.
- `question`: The question or query.
- `supporting_facts`: A list of supporting facts for the question. Each supporting fact is a list containing the title of the document that can be found in the `dataset_corpus.json` file.

Each entry can also contain additional fields depending on the task. For example:

- `answer`: The answer to the question.

The additional fields will be copied during the following steps of the pipeline.

Example:
```json
[
	{
		"id": "5adf5e285542992d7e9f9323",
		"question": "When was the judge born who made notable contributions to the trial of the man who tortured, raped, and murdered eight student nurses from South Chicago Community Hospital on the night of July 13-14, 1966?",
		"answer": "June 4, 1931",
		"supporting_facts": [
			"Louis B. Garippo",
			"Richard Speck"
		]
	},
	{
		"id": "5a7f7b365542992097ad2f80",
		"question": "Did the Beaulieu Mine or the McIntyre Mines yield gold and copper?",
		"answer": "The McIntyre also yielded a considerable amount of copper",
		"supporting_facts": [
			"Beaulieu Mine",
			"McIntyre Mines"
		]
	}
    ...
]
```
