import argparse
import json
import os
import random
from collections import defaultdict
from typing import Optional, Dict

from sklearn.ensemble import RandomForestClassifier
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

def just_ask_llm(paper: Paper, question_loc: str) -> float:
    file_loc = f"questions/{question_loc}.txt"
    with open(file_loc, 'r') as f:
        guidance = f.read()

    prompt = f"{guidance}\n\nTitle (str): {paper.title}\nAbstract (str): {paper.abstract}\nAI relevance (True/False):"
    classification_result = get_classification(prompt, [" True", " False"], print_all_probs=False)
    print(f"Classification Result: {paper.id}.{question_loc} -> {classification_result}")

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

def balance_train_set(train_set: list[Paper], num_of_each: int) -> list[Paper]:
    # Introducing this function to balance the training set to a 50/50 split. This is to avoid the model learning to predict the majority class (False) all the time. This is something I'll revisit if I have time.
    class_papers = defaultdict(list)
    for paper in train_set:
        class_papers[paper.is_ai].append(paper)

    class_papers[True] = class_papers[True][:num_of_each]
    class_papers[False] = class_papers[False][:num_of_each]
    balanced_set = (
        class_papers[True] +
        class_papers[False]
    )

    print(f"Balanced training set to {num_of_each} samples per class.")
    return balanced_set

def load_train_test(args, balance_classes: bool = True) -> (list[Paper], list[Paper]):
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
    if balance_classes:
        train_set = balance_train_set(train_set, args.num_train // 2)
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
                is_ai=data.get('label', None),
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
    # print(X_train)
    print(f"Training labels: {y_train.shape[0]} samples")
    # print(y_train)

    # Create and train the logistic regression model
    model = RandomForestClassifier(class_weight='balanced')
    model.fit(X_train, y_train)

    # Print model coefficients and intercept
    try:
        print("Model coefficients:", model.coef_)
        print("Model intercept:", model.intercept_)
    except AttributeError:
        print("Model does not have coefficients and intercept, skipping...")

    return model


def test_logistic_regression(model: LogisticRegression, test_data: list[Paper]) -> float:
    # Extract features and labels from the test data
    X_test = np.array([list(paper.features.values()) for paper in test_data])
    y_test = np.array([paper.is_ai for paper in test_data])

    # Print the shape of the test data
    print(f"Test data: {X_test.shape[0]} samples, {X_test.shape[1]} features")
    # print(X_test)
    print(f"Test labels: {y_test.shape[0]} samples")
    # print(y_test)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    # Print classification report and confusion matrix
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print(f"Paper is AI, Model predicts AI: {tp}")
    print(f"Paper is AI, Model predicts not AI: {fn}")
    print(f"Paper is not AI, Model predicts AI: {fp}")
    print(f"Paper is not AI, Model predicts not AI: {tn}")

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
    questions: list[str] = [os.path.splitext(f)[0] for f in os.listdir('questions') if f.endswith('.txt') and os.path.isfile(os.path.join('questions', f))]
    print("Classifying Train Data Papers")
    for paper in train_set:
        for question in questions:
            paper.features[question] = just_ask_llm(paper, question)
    print("Classifying Test Data Papers")
    for paper in test_set:
        for question in questions:
            paper.features[question] = just_ask_llm(paper, question)

    # Part 3, Train a model to predict AI relevance
    # Train the model
    model = train_logistic_regression(train_set)

    # Test the model
    accuracy = test_logistic_regression(model, test_set)

    print(f"Accuracy: {accuracy:.2f}")

if __name__ == '__main__':
    main()