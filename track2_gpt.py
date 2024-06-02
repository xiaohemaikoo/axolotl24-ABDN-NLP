import argparse
import pathlib
import pandas as pd
import torch
import tqdm

from prompt_gpt import assessment_def_block
import warnings
warnings.filterwarnings("ignore")
# --pretrain_data ../../data/russian/axolotl.train.ru.tsv --train_data ../../data/russian/axolotl.train.ru.tsv --test_data ../../data/russian/axolotl.dev.ru.tsv --predictions_file ./pred_dev_ru.track2.tsv

parser = argparse.ArgumentParser()

parser.add_argument(
    "--test_data",
    type=pathlib.Path,
    help="path to test data.",
    required=True,
)
parser.add_argument(
    "--predictions_file",
    type=pathlib.Path,
    help="path to prediction output file.",
    default="preds.tsv",
)

parser.add_argument("--lang", help="Langauge", type=str, default='ru')

args = parser.parse_args()

test_dataset = pd.read_csv(args.test_data, sep="\t")

# helper function to remove unusable datapoints
def remove_unusable(dataset):
    pre_drop = len(dataset)
    #dataset = dataset.dropna(subset=["gloss", "example", "word"])
    dataset = dataset.dropna(subset=["example", "word"])
    post_drop = len(dataset)
    if pre_drop != post_drop:
        n_dropped = pre_drop - post_drop
        print(
            f"[Warn] some datapoints were dropped {n_dropped} / {pre_drop}, i.e. "
            f"{(n_dropped * 100) / pre_drop:.2f}%"
        )
    return dataset

def do_prompt_to_def(dataset, language, u_prompt, a_prompt, lm):
    #u_prompt = "### Human:"
    #a_prompt = "### Assistant:"

    prompt_placeholder = u_prompt
    response_placeholder = a_prompt

    dataset = remove_unusable(dataset)
    print(len(dataset.sense_id.unique()))
    old_senses = set(dataset[dataset.period == "old"].sense_id.unique())
    dataset = dataset[~dataset.sense_id.apply(old_senses.__contains__)]

    print('len:',len(dataset.sense_id.unique()))
    dataset["predicted gloss"] = ''
    print("sense_ids: ", dataset.sense_id.unique())
    for sense_id in tqdm.tqdm(dataset.sense_id.unique()):
        #if sense_id != 'pitaeae__LFRpHQ2zJM':
        #    continue
        sense_data = dataset[dataset.sense_id == sense_id]
        if language == 'Finnish':
            headwords = [example[int(pos.split(';')[0].split(':')[0]):int(pos.split(';')[0].split(':')[1])] for pos, example in zip(sense_data.indices_target_token.tolist(), sense_data.example.tolist())]
        else:
            headwords = None
        #print(sense_id, len(sense_data), headwords)
        sentences = sense_data.example.tolist()
        #print("sentences", sentences)
        target_word = sense_data.word.tolist()[0]
        #print("target_word", target_word)

        gloss = assessment_def_block(
            target_word,
            sentences,
            language,
            headwords,
            lm,
            response_placeholder=response_placeholder,
            prompt_placeholder=prompt_placeholder)

        print('definition', gloss)

        # save predict gloss
        row_number = sense_data.index
        dataset.loc[row_number, "predicted gloss"] = gloss

        # tmp test 1 case
        # break
        # print("in loop")

    # save predictions
    dataset[["sense_id", "word", "predicted gloss"]].rename(
        columns={"predicted gloss": "gloss"}
    ).to_csv(args.predictions_file, sep="\t")

if args.lang=='ru':
  language = 'Russian'
if args.lang == 'fi':
  language = 'Finnish'
if args.lang == 'de':
  language = 'German'
#from model_dict import load_from_catalogue
#modelname = "NousResearch/Nous-Hermes-13b"
#modelname = "TheBloke/guanaco-65B-GPTQ"
#modelname = "TheBloke/WizardLM-13B-V1.1-GPTQ"

#model, tokenizer, u_prompt, a_prompt = load_from_catalogue(modelname)

#llm = guidance.llms.Transformers(model, tokenizer=tokenizer, trust_remote_code=True)
u_prompt = 'Instructions'
a_prompt = 'Requirements'

import os
os.environ["OPENAI_API_KEY"]= 'key-for-openai'
from guidance import models, system, user, assistant, gen

gpt35 = models.OpenAI('gpt-3.5-turbo')
#gpt35 = models.OpenAI('gpt-4-turbo-2024-04-09')
lm = gpt35

do_prompt_to_def(test_dataset, language, u_prompt, a_prompt, lm)
