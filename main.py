import argparse
import json
import random
from typing import Optional

import attrs


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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict AI relevance for papers.')
    parser.add_argument('--train_file', type=str, help='Train file.', default='data/dev.jsonl')  # Change this to 'data/train.jsonl' for the final run
    parser.add_argument('--test_file', type=str, help='Test file.', default='data/test_no_labels.jsonl')
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

    # Print a random paper to show the structure
    print(train_set[random.randint(0, len(train_set))])
