import os
import json
import tqdm
import numpy as np
from preprocess import get_cleaned
import argparse

import spacy
nlp = spacy.load("en_core_web_md")

parser = argparse.ArgumentParser(description='evaluation script')

parser.add_argument('--prediction-file', type=str, help='Prediction file path')
parser.add_argument('--question-file', type=str, help='Question file path')

args = parser.parse_args()

prediction_file = args.prediction_file
question_file = args.question_file


def check_similar_in_meaning(word1, word2, similarity_threshold=0.7):
    vec1 = nlp(word1).vector
    vec2 = nlp(word2).vector

    # Calculate cosine similarity between the vectors
    similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    # Return True if similarity is above the threshold
    return similarity >= similarity_threshold


with open(question_file) as file:
    annotations = json.load(file)


with open(prediction_file) as f:

    data = json.load(f)
    new_list = []
    correct_count = 0
    total_question_count = 0
    not_mention_count = 0
    data = tqdm.tqdm(data)
    not_count = 0

    structural_types = {"verify": 0,
            "query": 0,
            "choose": 0,
            "logical": 0,
            "compare": 0
            }

    semantic_types = {"obj": 0,
            "attr": 0,
            "cat": 0,
            "rel": 0,
            "global": 0
            }

    for datum in data:
        question_id = datum['question_id']
        img_id = datum['img_id']
        question = datum['question']
        label = datum['answer'].lower()
        prediction = datum['prediction'].lower()
        # text = datum["text"]
        prediction = get_cleaned(prediction)

        meta = annotations[question_id]

        label = meta["answer"].lower()
        structural_type = meta["types"]["structural"]
        semantic_type = meta["types"]["semantic"]

        if check_similar_in_meaning(prediction, label):
            correct_count += 1

            if structural_type in structural_types:
                structural_types[structural_type] += 1

            if semantic_type in semantic_types:
                semantic_types[semantic_type] += 1

        total_question_count += 1


print(f"Total answers: {total_question_count}, matched answers: {correct_count}. "
      f"Accuracy: {100*correct_count/total_question_count}")


print("structural results")
for key, val in structural_types.items():
    print(f"{key}: {val}")


print("semantic results")
for key, val in semantic_types.items():
    print(f"{key}: {val}")
