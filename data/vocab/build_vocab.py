import json
import re
from collections import Counter
from pathlib import Path

QUESTION_DIR = Path('C:/Users/Baldu/Desktop/Temp/VQA/data/questions')
ANNOTATION_DIR = Path('C:/Users/Baldu/Desktop/Temp/VQA/data/annotations')

QUESTION_FILES = ['train_questions.json', 'val_questions.json']
ANNOTATION_FILES = ['train_annotations.json', 'val_annotations.json']

OUT_QUESTION_VOCAB = 'vocab_questions.txt'
OUT_ANSWER_VOCAB = 'vocab_answers.txt'

MIN_WORD_FREQ = 5
TOP_K_ANSWERS = 3000


def tokenize_question(q):
    return re.findall(r'\w+', q.lower())

def normalize_answer(ans):
    return ans.lower().strip()

def load_questions(paths):
    counter = Counter()
    for file in paths:
        with open(QUESTION_DIR / file, 'r') as f:
            data = json.load(f)
            for q in data['questions']:
                tokens = tokenize_question(q['question'])
                counter.update(tokens)
    return counter

def load_answers(paths):
    counter = Counter()
    for file in paths:
        with open(ANNOTATION_DIR / file, 'r') as f:
            data = json.load(f)
            for ann in data['annotations']:
                for ans in ann['answers']:
                    norm = normalize_answer(ans['answer'])
                    counter[norm] += 1
    return counter

def save_vocab(vocab_list, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        for token in vocab_list:
            f.write(token + '\n')


if __name__ == '__main__':
    print("Building vocabularies...")

    print(" Tokenizing questions...")
    question_counter = load_questions(QUESTION_FILES)

    print("Normalizing answers...")
    answer_counter = load_answers(ANNOTATION_FILES)

    question_vocab = ['<pad>', '<unk>']
    question_vocab += [word for word, freq in question_counter.items() if freq >= MIN_WORD_FREQ]

    answer_vocab = ['<unk>']
    answer_vocab += [ans for ans, _ in answer_counter.most_common(TOP_K_ANSWERS)]

    print(f"Saving question vocab ({len(question_vocab)} tokens) to {OUT_QUESTION_VOCAB}")
    save_vocab(question_vocab, OUT_QUESTION_VOCAB)

    print(f"Saving answer vocab ({len(answer_vocab)} tokens) to {OUT_ANSWER_VOCAB}")
    save_vocab(answer_vocab, OUT_ANSWER_VOCAB)

    print("Done!")
