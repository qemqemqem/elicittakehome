import argparse
import json
import random
from typing import Optional
from sklearn.metrics import confusion_matrix

import attrs

from gpt2 import get_classification


@attrs.define(auto_attribs=True)
class Paper:
    title: str
    abstract: str
    id: str
    is_ai: Optional[bool] = None

    def __attrs_post_init__(self):
        if isinstance(self.is_ai, str):
            self.is_ai = self.is_ai.lower().strip() == 'true'

def split_text(text: str) -> (str, str):
    # The format is 'title. abstract'
    try:
        title, abstract = text.split('. ', 1)
        return title, abstract
    except ValueError:
        raise ValueError(f'Error splitting text: {text}')

def just_ask_llm(paper: Paper) -> float:
    examples = """Title (str): qualitative and quantitative estimates for minimal hypersurfaces with bounded index and area
Abstract (str): we prove qualitative estimates on the total curvature of closed minimal hypersurfaces in closed riemannian manifolds in terms of their index and area restricting to the case where the hypersurface has dimension less than seven.
AI relevance (True/False): False

Title (str): updating the transformer for alien languages
Abstract (str): Transformers are a powerful tool for natural language processing, but they are not always effective for alien languages. This paper proposes a new method for updating the transformer to work with alien languages.
AI relevance (True/False): True"""

    guidance = "In this dataset, AI relevance is a strict category. If the paper is narrowly about artificial intelligence, this field will be true. Papers in other fields, such as math, cognitive science, or linguistics, will be marked as false."

    prompt = f"{guidance}\n\n{examples}\n\nTitle (str): {paper.title}\nAbstract (str): {paper.abstract}\nAI relevance (True/False):"
    classification_result = get_classification(prompt, [" True", " False"], print_all_probs=False)
    print(f"Classification Result: {classification_result}")

    true_prob = classification_result[" True"]
    false_prob = classification_result[" False"]
    return true_prob / (true_prob + false_prob)

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
    num_correct = 0
    correctnesses = []
    true_labels = []
    predicted_labels_binary = []
    predicted_labels_prob = []

    for paper in train_set[:args.num_process]:
        print(f"{paper.title} --- {paper.abstract}"[:100])
        true_prob = just_ask_llm(paper)
        true_or_false = true_prob > 0.5
        correct_label = paper.is_ai
        true_labels.append(correct_label)
        predicted_labels_binary.append(true_or_false)
        predicted_labels_prob.append(true_prob)
        print(f"True Prob: {true_prob}, True or False: {true_or_false}, Correct Label: {correct_label}")
        if true_or_false == correct_label:
            num_correct += 1
        correctnesses.append(true_prob if correct_label else 1 - true_prob)

    print(f"Accuracy: {num_correct} / {args.num_process} == {num_correct / args.num_process}")
    print(f"Average correctness: {sum(correctnesses) / len(correctnesses)}")

    # Compute confusion matrix for binary classification
    cm_binary = confusion_matrix(true_labels, predicted_labels_binary)
    print("Confusion Matrix (Binary):")
    print(cm_binary)

    # Compute confusion matrix for probabilistic classification
    thresholded_probs = [prob > 0.5 for prob in predicted_labels_prob]
    cm_prob = confusion_matrix(true_labels, thresholded_probs)
    print("Confusion Matrix (Probabilistic):")
    print(cm_prob)

