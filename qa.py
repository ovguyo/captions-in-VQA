from openai import OpenAI
import json
from preprocess import remove_s, remove_newline
from tqdm import tqdm
import argparse

# Modify the api key with yours
client = OpenAI(api_key="OPENAI_API_KEY")

parser = argparse.ArgumentParser(description='question answering script')

parser.add_argument('--caption-file', type=str, help='Caption file path')
parser.add_argument('--question-file', type=str, help='Question file path')

args = parser.parse_args()

caption_file = args.caption_file
question_file = args.question_file


def get_answer(messages):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        max_tokens=5,
        temperature=0.2,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        messages=messages
    )

    return response


with open(caption_file) as file:
    descriptions = json.load(file)


with open(question_file) as f:
    data = json.load(f)
    new_data = []

    for key, datum in tqdm(data.items()):
        img_id = datum["imageId"]
        question = datum["question"]
        answer = datum["answer"]

        for i in range(len(descriptions)):
            desc_datum = descriptions[i]

            if img_id == desc_datum["img_id"]:
                text = desc_datum["description"]
                text = remove_s(text)
                text = remove_newline(text)

                prompt = (f"Answer the question in maximum two words based on the text."
                          f"Consider the type of question in your answer. For example, if it is a yes/no question, answer should be yes or no."
                          f"Text: {text} "
                          f"Question: {question}")

                messages = [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]

                response = get_answer(messages)

                new_datum = {
                    'question_id': key,
                    'img_id': img_id,
                    'question': question,
                    'answer': answer
                }

                prediction = response.choices[0].message.content
                new_datum["prediction"] = prediction
                new_datum["text"] = text

                new_data.append(new_datum)


json.dump(new_data, open("predictions/predicted_answers.json", 'w'), indent=4, sort_keys=True)