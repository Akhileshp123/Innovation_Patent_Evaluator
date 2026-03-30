# Innovation Patent Evaluator

A lightweight Python prototype for generating patent novelty and similarity assessments using a mock patent corpus and semantic embeddings.

## Features

- Generates a mock patent corpus (`mock_patent_corpus.csv`) for evaluation
- Loads an input patent from `input_patent.txt` (or path argument)
- Computes a simulated novelty score with heuristics
- Computes semantic similarity with `sentence-transformers` and cosine similarity
- Produces publication-ready visualization (`patent_trends_visualization.png`)
- Writes a markdown report (`patent_evaluation_report.md`)

## Prerequisites

- Python 3.9+ (tested on Python 3.13)
- Internet access for the first model download

## Install dependencies

```powershell
python -m pip install --upgrade pip
pip install pandas numpy matplotlib scikit-learn sentence-transformers
```

## Run

From the project directory:

```powershell
python app.py
```

Optional, custom input file path:

```powershell
python app.py path\to\my_input_patent.txt
```

## Input file format

`input_patent.txt` is auto-generated if missing. Use this structure:

```
Title: <Your Patent Title>

Abstract: <Your Patent Abstract text>
```

## Output

- `patent_evaluation_report.md` (detailed analysis)
- `patent_trends_visualization.png` (trend charts)
- `mock_patent_corpus.csv` (generated corpus)

## Notes

- First run downloads the embedding model (~91MB) into cache.
- If you want to iterate quickly, edit `input_patent.txt` and rerun.
