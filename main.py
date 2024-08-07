import argparse
import json
import random
from typing import Optional

import attrs

from gpt2 import get_classification


@attrs.define(auto_attribs=True)
class Paper:
    title: str
    abstract: str
    id: str
    is_ai: Optional[bool] = None

def split_text(text: str) -> (str, str):
    # The format is 'title. abstract'
    try:
        title, abstract = text.split('. ', 1)
        return title, abstract
    except ValueError:
        raise ValueError(f'Error splitting text: {text}')

def just_ask_llm(paper: Paper) -> bool:
    prompt = f"Paper Details\n\nTitle (str): {paper.title}\nAbstract (str): {paper.abstract}\nAI relevance (True/False):"
    classification_result = get_classification(prompt, [" True", " False"], print_all_probs=True)
    print(f"Classification Result: {classification_result}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict AI relevance for papers.')
    parser.add_argument('--train_file', type=str, help='Train file.', default='data/dev.jsonl')  # Change this to 'data/train.jsonl' for the final run
    parser.add_argument('--test_file', type=str, help='Test file.', default='data/test_no_labels.jsonl')
    parser.add_argument("--num_process", type=int, help="Number of papers to process.", default=1)
    args = parser.parse_args()

    train_set = []
    with open(args.train_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            title, abstract = split_text(data['text'])
            train_set.append(Paper(
                title=title,
                abstract=abstract,
                id=data['meta']['id'],
                is_ai=data.get('label', None),
            ))

    # Shuffle the papers, for variety during development
    random.shuffle(train_set)

    # Part 1, Naively asking GPT2 to predict AI relevance
    for paper in train_set[:args.num_process]:
        print(f"{paper.title} --- {paper.abstract}"[:100])
        just_ask_llm(paper)

