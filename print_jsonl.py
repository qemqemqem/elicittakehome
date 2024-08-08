import argparse
import json
import random


def main():
    parser = argparse.ArgumentParser(description='Print JSONL file.')
    parser.add_argument('--file', type=str, help='File to print.', default='data/dev.jsonl')
    parser.add_argument("--shuffle", action="store_true", help="Shuffle the data.")
    parser.add_argument("-n", "--num", type=int, help="Number of samples to print.", default=10)
    args = parser.parse_args()

    objs = []

    with open(args.file, 'r') as f:
        for line in f:
            obj = json.loads(line)
            objs.append(obj)
    if args.shuffle:
        random.shuffle(objs)
    if args.num > 0:
        objs = objs[:args.num]

    for i, obj in enumerate(objs):
        text = obj['text']
        label = obj.get('label', None)
        id = obj['meta']['id']
        title, abstract = text.split('. ', 1)

        print(f"Paper {i + 1}\n")
        print(title)
        print()
        print(abstract)
        print()
        print(f"Is AI Relevant: {label}")
        print(f"ID: {id}")
        print()


if __name__ == "__main__":
    main()