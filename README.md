# TextRank Summarizer

A text summarizer application with a GUI built using Python, Tkinter, and spaCy, implementing the TextRank algorithm.

## Features

- Clean, user-friendly interface
- Adjustable summary length (10%-50% of original text)
- Graph-based TextRank summarization using spaCy NLP
- Status bar to track summarization progress

## Installation

1. Clone this repository:
```
git clone https://github.com/yourusername/text-summarizer.git
cd text-summarizer
```

2. Create a virtual environment (optional but recommended):
```
python -m venv venv
```

3. Activate the virtual environment:
   - Windows:
   ```
   venv\Scripts\activate
   ```
   - macOS/Linux:
   ```
   source venv/bin/activate
   ```

4. Install the required packages:
```
pip install -r requirements.txt
```

5. Download the spaCy language model:
```
python -m spacy download en_core_web_sm
```

## Usage

1. Run the application:
```
python text_summarizer.py
```

2. Enter or paste the text you want to summarize in the left panel.
3. Select the desired summary length using the dropdown menu.
4. Click the "Summarize" button to generate the summary.
5. The summary will appear in the right panel.
6. Use the "Clear" button to reset both text areas.

## How It Works

The summarizer uses the TextRank algorithm, a graph-based ranking model for text processing:

1. The text is processed using spaCy to tokenize and analyze the content.
2. Each sentence is represented as a node in a graph.
3. Edges between sentences are weighted based on their similarity (cosine similarity of sentence vectors).
4. The PageRank algorithm is applied to score sentences based on their importance in the graph.
5. The highest-scoring sentences are selected for the summary, preserving their original order.

This approach is more sophisticated than simple word frequency models as it captures the relationships between sentences and identifies the most central ideas in the text.

## API Usage

You can also use the summarizer programmatically:

```python
from summarizer_api import TextSummarizer

# Create a summarizer
summarizer = TextSummarizer()

# Generate summary (30% of original length)
summary = summarizer.summarize(your_text, 0.3)
```

## Requirements

- Python 3.6+
- spaCy
- networkx
- numpy
- tkinter (usually comes with Python installation)

## License

MIT 
=======
# Text-Summarizer
A robust text summarization tool leveraging neural networks to extract key information from PDF and Word documents. Built with Python, spaCy, and Hugging Face Transformers, it supports user-friendly file uploads and delivers concise summaries efficiently.

