import argparse
from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaModel, LlamaConfig, Trainer, TrainingArguments
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from transformers import BioGptTokenizer, BioGptForCausalLM
from tqdm import tqdm
import os
import torch
import math
import pandas as pd
from rdkit import Chem
import numpy as np



def collect_txt(idx, txt):
    tmp = []
    for i in idx:
        tmp.append(txt[i])
    return tmp


def process_text(text):
    refined_txt_arr = []
    for txt in text:
        refined_txt = txt.split('\n')[1]
        assert refined_txt[:10] == 'Abstract: '
        refined_txt_arr.append(refined_txt[10:])
    return refined_txt_arr


def save_hidden_states(text, name, path, max_length=512, llm_model='llama', llm_model_path='the path of LLM_models'):
    assert llm_model in ['llama', 'bert', 'baichuan', 'vicuna', 'qwen', 'biobert', 'biogpt', 'pubmedbert']

    if llm_model == 'baichuan':
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(llm_model_path, 'Baichuan2-7B-Base'), use_fast=False,
                                                  trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(os.path.join(llm_model_path, 'Baichuan2-7B-Base'),
                                                     device_map='auto', torch_dtype=torch.float16,
                                                     trust_remote_code=True, output_hidden_states=True,
                                                     return_dict=True)
        model = model.model  # only use encoder
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

        hidden_layers = len(model.layers)
    elif llm_model == 'vicuna':
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(llm_model_path, 'vicuna-7b-v1.5'), use_fast=False,
                                                  trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(os.path.join(llm_model_path, 'vicuna-7b-v1.5'), device_map='auto',
                                                     torch_dtype=torch.float16, trust_remote_code=True,
                                                     output_hidden_states=True, return_dict=True)
        model = model.model  # only use encoder
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

        hidden_layers = len(model.layers)

    elif llm_model == 'llama':
        # Llama
        token_id = os.path.join(llm_model_path, 'llama-2-7b-hf')
        model_id = os.path.join(llm_model_path, 'llama-2-7b-hf')

        tokenizer = LlamaTokenizer.from_pretrained(token_id)
        tokenizer.pad_token = tokenizer.eos_token

        model = LlamaModel.from_pretrained(model_id, load_in_8bit=True, device_map='auto', torch_dtype=torch.float16,
                                           output_hidden_states=True, return_dict=True)
        model.config.pad_token_id = model.config.eos_token_id

        hidden_layers = len(model.layers)
    # ***************************************************************

    elif llm_model == 'bert':
        # Sentence BERT
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(llm_model_path, 'bert-large-uncased'))
        model = AutoModel.from_pretrained(os.path.join(llm_model_path, 'bert-large-uncased'),
                                          output_hidden_states=True, return_dict=True).cuda()
        hidden_layers = 12
    
    elif llm_model == 'qwen':

        tokenizer = AutoTokenizer.from_pretrained(os.path.join(llm_model_path, 'Qwen2.5-7B-Instruct'), use_fast=False,
                                                  trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(os.path.join(llm_model_path, 'Qwen2.5-7B-Instruct'), device_map='auto',
                                                     torch_dtype=torch.float16, trust_remote_code=True,
                                                     output_hidden_states=True, return_dict=True)

        model = model.model  # only use encoder
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

        hidden_layers = len(model.layers)

    elif llm_model == 'biobert':
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(llm_model_path, 'biobert-large-cased-v1.1'))
        model = AutoModel.from_pretrained(os.path.join(llm_model_path, 'biobert-large-cased-v1.1'),
                                          output_hidden_states=True, return_dict=True).cuda()
        hidden_layers = 12

    elif llm_model == 'biogpt':

        tokenizer = BioGptTokenizer.from_pretrained(os.path.join(llm_model_path, 'biogpt'))
        model = BioGptForCausalLM.from_pretrained(os.path.join(llm_model_path, 'biogpt')).cuda()

        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id


    elif llm_model == 'pubmedbert':
        # Sentence BERT
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(llm_model_path, 'pubmedbert-base-embeddings'))
        model = AutoModel.from_pretrained(os.path.join(llm_model_path, 'pubmedbert-base-embeddings'),
                                          output_hidden_states=True, return_dict=True).cuda()
        hidden_layers = 12

    # Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    # **************************************************************

    batch_size = 8
    model.eval()
    embs_hid = []
    for i in tqdm(range(math.ceil(len(text) / batch_size))):
        if (i + 1) * batch_size <= len(text):
            txt = text[(i) * batch_size: (i + 1) * batch_size]
        else:
            txt = text[(i) * batch_size:]
        # txt = process_text(txt)

        model_input = tokenizer(txt, truncation=True, padding=True, return_tensors="pt", max_length=max_length).to(
            device)
        with torch.no_grad():
            if llm_model not in ['biogpt', 'qwen']:
                out = model(**model_input)
            else:
                out = model(**model_input, output_hidden_states=True)

        batch_size = model_input['input_ids'].shape[0]
        sequence_lengths = (torch.eq(model_input['input_ids'], model.config.pad_token_id).long().argmax(-1) - 1)
        hidden_states = out['hidden_states']

        emb_hid = hidden_states[-1]
        emb_hid = emb_hid.cpu()
        emb_node_hid = mean_pooling(emb_hid, model_input['attention_mask'].cpu())
        embs_hid.append(emb_node_hid.cpu())
    embs_hid = torch.cat(embs_hid).float()

    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(embs_hid, f=os.path.join(path, name + '_' + llm_model + '_embedding.pt'))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--path', default='down_task')
    parser.add_argument('--task', default='bace', type=str,
                        choices=['bace', 'bbbp', 'clintox', 'sider', 'tox21', 'hiv', 'muv', 'toxcast'])
    parser.add_argument('--llm_model', nargs='?', default='llama',
                        help='Choose a LLM. [llama/bert/qwen/biobert/biogpt/pubmedbert]')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    task = args.task
    csv_file = 'dataset/' + task + '/raw/smiles.csv'
    data = pd.read_csv(csv_file, header=None)

    smiles_all = data.iloc[:, 0].values
    properties_all = data.iloc[:, 1:].values

    smiles_all = [str(smile) for smile in smiles_all]

    smiles = []
    properties = []

    for i, smile in enumerate(smiles_all):
        mol = Chem.MolFromSmiles(smile)

        if mol is not None:
            # smiles.append(smile)
            smiles.append(Chem.MolToSmiles(mol, isomericSmiles=True, kekuleSmiles=False))
            properties.append(properties_all[i])

    smiles = np.array(smiles)
    properties = np.array(properties)


    device = torch.device('cuda:%d' % args.gpu if torch.cuda.is_available() else 'cpu')
    path = os.path.join(args.path, 'processed')

    print('generating llm embedding......')
    embs_hid = save_hidden_states(smiles.tolist(), task, path, 512, args.llm_model, llm_model_path='/media/yjz/LLM_models')
