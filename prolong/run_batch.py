import torch
import time
from tqdm import tqdm
import json
import os
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt

from transformers import LlamaTokenizer, LlamaForCausalLM, AutoTokenizer, AutoModelForCausalLM
from torch.nn import CrossEntropyLoss


IGNORE_INDEX = -100


def parse_config():
    parser = argparse.ArgumentParser()

    # data parameter
    parser.add_argument('--data_file', type=str)

    # lds parameter 
    parser.add_argument('--chunk_size', type=int, default=128)
    parser.add_argument('--dlt_ppl_threshold', type=float, default=0.1)
    parser.add_argument('--window_size', type=int, default=32768)

    # save parameter
    parser.add_argument('--save_file', type=str)
    parser.add_argument('--score_file', type=str)
    parser.add_argument('--pic_dir', type=str)

    # model configuration
    parser.add_argument('--model_name', type=str, default='facebook/opt-350m')
    parser.add_argument('--use_flash_attention_2', action='store_true')

    # other 
    parser.add_argument('--seed', type=int, default=11)

    parser.add_argument('--single_ppl_batch_size', type=int, default=256)
    parser.add_argument('--pair_ppl_batch_size', type=int, default=256)
    parser.add_argument('--sample_size', type=int, default=500)
    parser.add_argument('--need_draw', type=bool, default=False)

    return parser.parse_args()


def set_seed(seed):
    """ fix random seed """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def sample_preserve_order(array, sample_size):
    indices = list(range(len(array)))
    assert sample_size <= len(indices)
    sampled_indices = sorted(random.sample(indices, sample_size))
    return [array[i] for i in sampled_indices]

def construct_data(data, tokenizer, chunk_size, chunk_num):
    tokenized_data = tokenizer(data)['input_ids']
    data_list = [tokenized_data[i:i + chunk_size] for i in range(0, len(tokenized_data), chunk_size)]

    if len(data_list[-1]) < chunk_size:
        data_list = data_list[:-1]
    if len(data_list) > chunk_num:
        data_list = sample_preserve_order(array=data_list, sample_size=chunk_num)
    return data_list

def compute_ppl(logits, labels, nums):
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # Flatten the tokens
    loss_fct = CrossEntropyLoss(reduction='none')
    shift_logits = shift_logits.view(-1, model.config.vocab_size)
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)
    loss = loss_fct(shift_logits, shift_labels)
    loss = loss.view(labels.size(0), -1)  # reshape loss back to sequence length

    batch_ppl = []
    for i, num in enumerate(nums):
        avg_loss = loss[i, -num:].mean()
        batch_ppl.append(torch.exp(avg_loss).float().cpu().item())
    return batch_ppl


def compute_single_ppl(data_list, batch_size):
    single_ppl = [0 for _ in range(len(data_list))]
    with torch.no_grad():
        model.eval()
        for i in range(0, len(data_list), batch_size):
            batch = data_list[i:i+batch_size]
            nums = [len(b) - 1 for b in batch]
            inputs = torch.tensor(batch).to(device)
            labels = inputs.clone()
            logits = model(input_ids=inputs)[0]
            batch_ppl = compute_ppl(logits, labels, nums)
            single_ppl[i:i+batch_size] = batch_ppl
    return single_ppl

    
def compute_pair_ppl(data_list, batch_size, sample_size=-1):
    pair_ppl = [[float('inf') for _ in range(len(data_list))] for _ in range(len(data_list))]
    with torch.no_grad():
        model.eval()
        pairs = [(i, j) for i in range(len(data_list)) for j in range(i)]
        if sample_size > 0:
            if len(pairs) < sample_size:
                return pair_ppl
            pairs = random.sample(pairs, sample_size)
        for batch_start in range(0, len(pairs), batch_size):
            batch_pairs = pairs[batch_start:batch_start+batch_size]
            nums = [len(data_list[i]) - 1 for i, _ in batch_pairs]
            inputs = [data_list[j] + data_list[i] for i, j in batch_pairs]
            inputs = torch.tensor(inputs).to(device)
            labels = torch.tensor([[IGNORE_INDEX] * (len(data_list[j]) + 1) + data_list[i][1:] for i, j in batch_pairs]).to(device)
            logits = model(input_ids=inputs)[0]
            batch_ppl = compute_ppl(logits, labels, nums)
            for k, (i, j) in enumerate(batch_pairs):
                pair_ppl[i][j] = batch_ppl[k]
    return pair_ppl

def compute_de(logits):
    
    def _softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def _compute_entropy(x):
        if 0 in x:
            x += 1e-12
            x /= np.sum(x)
        entropy = -np.sum(x * np.log(x))
        return entropy

    if len(logits) == 1:
        return 1
    
    max_entropy = np.log(len(logits))
    entropy = _compute_entropy(_softmax(logits))
    return np.clip((max_entropy - entropy) / max_entropy, 0, 1)

def compute_lds(single_ppl, pair_ppl):
    dlt_ppl = [[0 for _ in range(len(single_ppl))] for _ in range(len(single_ppl))]
    dependency_entropy = [0 for _ in range(len(single_ppl))]
    dis_scale = 1 / (args.chunk_num - 1)
    lds = 0

    for i in range(len(single_ppl)):
        row_logits = []
        for j in range(i):
            dlt_ppl[i][j] = single_ppl[i] - pair_ppl[i][j]
            if pair_ppl[i][j] != float('inf'):
                row_logits.append(dlt_ppl[i][j])
        if len(row_logits) > 0:
            dependency_entropy[i] = compute_de(logits=row_logits)

    for i in range(len(single_ppl)):
        for j in range(i):
            dlt_ppl[i][j] /= single_ppl[i]
            if dlt_ppl[i][j] > args.dlt_ppl_threshold:
                distance_gain = np.clip((i - j) * dis_scale, 0, 1)
                lds += (dlt_ppl[i][j] + distance_gain) * dependency_entropy[i]

    if np.isnan(lds):
        lds = 0.
    return lds, dlt_ppl, dependency_entropy


def draw(save_path, matrix_data, lds):
    matrix_array = np.array(matrix_data)

    plt.imshow(matrix_array, cmap='viridis', interpolation='nearest')
    plt.colorbar()
    plt.title(f'LDS = {lds}')
    plt.show()
    plt.savefig(save_path)
    plt.clf()


if __name__ == "__main__":
    # cuda 
    cuda_available = torch.cuda.is_available()

    if cuda_available:
        print('Cuda is available.')
        device = torch.device('cuda')
    else:
        print('Cuda is not available')
        device = torch.device('cpu')

    # args
    args = parse_config()
    args.chunk_num = args.window_size // args.chunk_size

    # set seed
    set_seed(seed=args.seed)

    # model
    print ('Start loading tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        use_fast=False
    )
    if hasattr(tokenizer, "add_bos_token"): 
        setattr(tokenizer, "add_bos_token", False)
    if hasattr(tokenizer, "add_eos_token"):
        setattr(tokenizer, "add_eos_token", False)

    # model
    print ('Start loading model...')
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, 
        torch_dtype=torch.float16,
        use_flash_attention_2=args.use_flash_attention_2,
        trust_remote_code=True,
    )
    model.to(device)
    print ('Model loaded')

    with open(args.data_file, 'r', encoding='utf-8') as i:
        all_data = i.readlines()
    for idx, single_data in tqdm(enumerate(all_data), total=len(all_data)):
        # Loading Data
        if args.data_file.endswith('.jsonl'):
            json_data = json.loads(single_data)
            data = json_data['text']
        else:
            data = single_data
        # Construct Data
        data_list = construct_data(data=data, tokenizer=tokenizer, chunk_size=args.chunk_size, chunk_num=args.chunk_num)

        start = time.time()
        try:
            single_ppl = compute_single_ppl(data_list, args.single_ppl_batch_size)
            pair_ppl = compute_pair_ppl(data_list, args.pair_ppl_batch_size, args.sample_size)
            long_dependency_score, dlt_ppl, dependency_entropy = compute_lds(single_ppl=single_ppl, pair_ppl=pair_ppl)
        except Exception as e:
            print (f'Error: {e}, set LDS to 0.')
            long_dependency_score = 0.

        print(f'long_dependency_score: {long_dependency_score}')
        end = time.time()
        print(f'cost time: {end - start} seconds')

        # draw
        if args.need_draw:
            if idx % 1000 == 0:
                save_path = os.path.join(args.pic_dir, str(idx) + '.png')
                draw(save_path, dlt_ppl, long_dependency_score)

        # wirte score to file
        with open(args.score_file, 'a', encoding='utf-8') as o:
            o.write(str(long_dependency_score) + '\n')

        # store data with score
        new_data = {'text': data, 'lds': long_dependency_score}
        with open(args.save_file, 'a', encoding='utf-8') as o:
            o.write(json.dumps(new_data, ensure_ascii=False) + '\n')
        
