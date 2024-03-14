import os
from PIL import Image
import json
import torch
import argparse
from transformers import AutoModelForCausalLM, LlamaTokenizer
from tqdm import tqdm
from keybert import KeyBERT

kw_model = KeyBERT()

parser = argparse.ArgumentParser()
parser.add_argument("--quant", choices=[4], type=int, default=None, help='quantization bits')
parser.add_argument("--from_pretrained", type=str, default="THUDM/cogagent-chat-hf", help='pretrained ckpt')
parser.add_argument("--local_tokenizer", type=str, default="lmsys/vicuna-7b-v1.5", help='tokenizer path')
parser.add_argument("--fp16", action="store_true")
parser.add_argument("--bf16", action="store_true")
parser.add_argument("--image-path", type=str)
parser.add_argument("--question-path", type=str)


args = parser.parse_args()
MODEL_PATH = args.from_pretrained
TOKENIZER_PATH = args.local_tokenizer
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
image_path = args.image_path
question_path = args.question_path


tokenizer = LlamaTokenizer.from_pretrained(TOKENIZER_PATH)
if args.bf16:
    torch_type = torch.bfloat16
else:
    torch_type = torch.float16

print("========Use torch type as:{} with device:{}========\n\n".format(torch_type, DEVICE))

if args.quant:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch_type,
        low_cpu_mem_usage=True,
        load_in_4bit=True,
        trust_remote_code=True
    ).eval()
else:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch_type,
        low_cpu_mem_usage=True,
        load_in_4bit=args.quant is not None,
        trust_remote_code=True
    ).to(DEVICE).eval()

text_only_template = ("A chat between a curious user and an artificial intelligence assistant. "
                      "The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {} ASSISTANT:")

# query = "Discuss the elements of the picture."
# query = "Describe the scene in this image:"


with open(question_path) as f:
    data = json.load(f)
    pair_list = []
    answer_list = []
    count = 0
    new_data = []
    img_ids_in_testdev = []

    for key, datum in tqdm(data.items()):
        new_datum = {
            'img_id': datum['imageId'],
            'question': datum['question'],
            'answer': datum['answer'],
            'question_id': key
        }

        question = datum['question']
        keywords = kw_model.extract_keywords(question)
        extracted_keys = [word[0] for word in keywords]

        query = (f"Describe the scene in this image considering the keywords: {extracted_keys}")

        image = Image.open(image_path + '/' + datum["imageId"] + '.jpg').convert('RGB')

        inputs = model.build_conversation_input_ids(tokenizer, query=query, images=[image])
        inputs = {
            'input_ids': inputs['input_ids'].unsqueeze(0).to('cuda'),
            'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to('cuda'),
            'attention_mask': inputs['attention_mask'].unsqueeze(0).to('cuda'),
            'images': [[inputs['images'][0].to(DEVICE).to(torch_type)]],
        }
        gen_kwargs = {"max_length": 2048, "do_sample": False}

        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
            response = tokenizer.decode(outputs[0])

        new_datum["description"] = response
        new_data.append(new_datum)


json.dump(new_data, open("qd_captions.json", 'w'), indent=4, sort_keys=True)




