import argparse
import json
import random
from typing import Optional, Dict
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
import numpy as np

import attrs

from gpt2 import get_classification


@attrs.define(auto_attribs=True)
class Paper:
    title: str
    abstract: str
    id: str
    is_ai: Optional[bool] = None

    features: Dict[str, float] = attrs.field(factory=dict)

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


def naive_asking(num_examples: int, train_set: list[Paper]):
    if num_examples == 0:
        return  # Avoid division by zero

    num_correct = 0
    correctnesses = []
    true_labels = []
    predicted_labels_binary = []
    predicted_labels_prob = []
    for paper in train_set[:num_examples]:
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
    print(f"Accuracy: {num_correct} / {num_examples} == {num_correct / num_examples}")
    print(f"Average correctness: {sum(correctnesses) / len(correctnesses)}")
    # Compute confusion matrix
    cm_binary = confusion_matrix(true_labels, predicted_labels_binary)
    print("Confusion Matrix:")
    print(cm_binary)
    tn, fp, fn, tp = cm_binary.ravel()
    print(f"Paper is AI, Model predicts AI: {tp}")
    print(f"Paper is AI, Model predicts not AI: {fn}")
    print(f"Paper is not AI, Model predicts AI: {fp}")
    print(f"Paper is not AI, Model predicts not AI: {tn}")


def load_train_test(args) -> (list[Paper], list[Paper]):
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
    random.shuffle(train_set)
    train_set = train_set[:args.num_train]

    test_set = []
    with open(args.test_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            title, abstract = split_text(data['text'])
            test_set.append(Paper(
                title=title,
                abstract=abstract,
                id=data['meta']['id'],
            ))
    random.shuffle(test_set)
    test_set = test_set[:args.num_test]

    return train_set, test_set

def train_logistic_regression(train_data: list[Paper]) -> LogisticRegression:
    # Extract features and labels from the training data
    X_train = np.array([list(paper.features.values()) for paper in train_data])
    y_train = np.array([paper.is_ai for paper in train_data])

    # Print the shape of the training data
    print(f"Training data: {X_train.shape[0]} samples, {X_train.shape[1]} features")

    # Create and train the logistic regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Print model coefficients and intercept
    print("Model coefficients:", model.coef_)
    print("Model intercept:", model.intercept_)

    return model


def test_logistic_regression(model: LogisticRegression, test_data: list[Paper]) -> float:
    # Extract features and labels from the test data
    X_test = np.array([list(paper.features.values()) for paper in test_data])
    y_test = np.array([paper.is_ai for paper in test_data])

    # Print the shape of the test data
    print(f"Test data: {X_test.shape[0]} samples, {X_test.shape[1]} features")

    # Predict and evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    # Print classification report and confusion matrix
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return accuracy

def main():
    parser = argparse.ArgumentParser(description='Predict AI relevance for papers.')
    parser.add_argument('--train_file', type=str, help='Train file.', default='data/dev.jsonl')  # Change this to 'data/train.jsonl' for the final run
    parser.add_argument('--test_file', type=str, help='Test file.', default='data/test_no_labels.jsonl')
    parser.add_argument("--num_train", type=int, help="Number of papers to process.", default=20)
    parser.add_argument("--num_naive_ask", type=int, help="Number of papers to naively ask.", default=0)
    parser.add_argument("--num_test", type=int, help="Number of papers to test.", default=10)
    args = parser.parse_args()

    train_set, test_set = load_train_test(args)

    # Part 1, Naively asking GPT2 to predict AI relevance, to establish a baseline
    naive_asking(num_examples=args.num_naive_ask, train_set=train_set)

    # Part 2, Annotate the training and test data
    for paper in train_set:
        paper.features['just_ask_llm'] = just_ask_llm(paper)
    for paper in test_set:
        paper.features['just_ask_llm'] = just_ask_llm(paper)

    # Part 3, Train a model to predict AI relevance
    train_data, test_data = load_train_test(args)

    # Train the model
    model = train_logistic_regression(train_data)

    # Test the model
    accuracy = test_logistic_regression(model, test_data)

    print(f"Accuracy: {accuracy:.2f}")

if __name__ == '__main__':
    main()