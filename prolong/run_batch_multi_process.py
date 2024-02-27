import torch
import time
from tqdm import tqdm
import json
import os
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
from transformers import LlamaTokenizer, LlamaForCausalLM, AutoTokenizer, AutoModelForCausalLM
from torch.nn import CrossEntropyLoss

multiprocessing.set_start_method('spawn', force=True)

IGNORE_INDEX = -100

def set_seed(seed):
    """ fix random seed """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

class DataProcessor:
    def __init__(self, model_name, use_flash_attention_2, data_file, input_chunk_path, output_chunk_path, score_chunk_path, pic_dir, chunk_size, dlt_ppl_threshold, window_size, single_ppl_batch_size, pair_ppl_batch_size, sample_size, need_draw, seed, device):
        self.model_name = model_name
        self.data_file = data_file
        self.input_chunk_path = input_chunk_path
        self.output_chunk_path = output_chunk_path
        self.score_chunk_path = score_chunk_path
        self.pic_dir = pic_dir
        self.chunk_size = chunk_size
        self.dlt_ppl_threshold = dlt_ppl_threshold
        self.window_size = window_size
        self.chunk_num = window_size // chunk_size
        self.single_ppl_batch_size = single_ppl_batch_size
        self.pair_ppl_batch_size = pair_ppl_batch_size
        self.sample_size = sample_size
        self.need_draw = need_draw
        self.seed = seed
        self.device = device

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            use_flash_attention_2=use_flash_attention_2,
            trust_remote_code=True,
            device_map="auto"
        )
        # self.model.to(device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=False,
            trust_remote_code=True
        )
        if hasattr(self.tokenizer, "add_bos_token"): 
            setattr(self.tokenizer, "add_bos_token", False)
        if hasattr(self.tokenizer, "add_eos_token"):
            setattr(self.tokenizer, "add_eos_token", False)

    def sample_preserve_order(self, array, sample_size):
        indices = list(range(len(array)))
        assert sample_size <= len(indices)
        sampled_indices = sorted(random.sample(indices, sample_size))
        return [array[i] for i in sampled_indices]


    def construct_data(self, data, tokenizer, chunk_size, chunk_num):
        tokenized_data = tokenizer(data)['input_ids']
        data_list = [tokenized_data[i:i + chunk_size] for i in range(0, len(tokenized_data), chunk_size)]

        if len(data_list[-1]) < chunk_size:
            data_list = data_list[:-1]
        if len(data_list) > chunk_num:
            data_list = self.sample_preserve_order(array=data_list, sample_size=chunk_num)
        return data_list
    

    def compute_de(self, logits):
    
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


    def compute_lds(self, single_ppl, pair_ppl):
        dlt_ppl = [[0 for _ in range(len(single_ppl))] for _ in range(len(single_ppl))]
        dependency_entropy = [0 for _ in range(len(single_ppl))]
        dis_scale = 1 / (self.chunk_num - 1)
        lds = 0

        for i in range(len(single_ppl)):
            row_logits = []
            for j in range(i):
                dlt_ppl[i][j] = single_ppl[i] - pair_ppl[i][j]
                if pair_ppl[i][j] != float('inf'):
                    row_logits.append(dlt_ppl[i][j])
            if len(row_logits) > 0:
                dependency_entropy[i] = self.compute_de(logits=row_logits)

        for i in range(len(single_ppl)):
            for j in range(i):
                dlt_ppl[i][j] /= single_ppl[i]
                if dlt_ppl[i][j] > self.dlt_ppl_threshold:
                    distance_gain = np.clip((i - j) * dis_scale, 0, 1)
                    lds += (dlt_ppl[i][j] + distance_gain) * dependency_entropy[i]

        return lds, dlt_ppl, dependency_entropy

    def compute_single_ppl(self, data_list, batch_size):
        single_ppl = [0 for _ in range(len(data_list))]
        with torch.no_grad():
            self.model.eval()
            for i in range(0, len(data_list), batch_size):
                batch = data_list[i:i+batch_size]
                nums = [len(b) - 1 for b in batch]
                inputs = torch.tensor(batch).to(self.device)
                labels = inputs.clone()
                logits = self.model(input_ids=inputs)[0]
                batch_ppl = self.compute_ppl(logits, labels, nums)
                single_ppl[i:i+batch_size] = batch_ppl
        return single_ppl
    

    def compute_pair_ppl(self, data_list, batch_size, sample_size=-1):
        pair_ppl = [[float('inf') for _ in range(len(data_list))] for _ in range(len(data_list))]
        with torch.no_grad():
            self.model.eval()
            pairs = [(i, j) for i in range(len(data_list)) for j in range(i)]
            if sample_size > 0:
                if len(pairs) < sample_size:
                    return pair_ppl
                pairs = random.sample(pairs, sample_size)
            for batch_start in range(0, len(pairs), batch_size):
                batch_pairs = pairs[batch_start:batch_start+batch_size]
                nums = [len(data_list[i]) - 1 for i, _ in batch_pairs]
                inputs = [data_list[j] + data_list[i] for i, j in batch_pairs]
                inputs = torch.tensor(inputs).to(self.device)
                labels = torch.tensor([[IGNORE_INDEX] * (len(data_list[j]) + 1) + data_list[i][1:] for i, j in batch_pairs]).to(self.device)
                logits = self.model(input_ids=inputs)[0]
                batch_ppl = self.compute_ppl(logits, labels, nums)
                for k, (i, j) in enumerate(batch_pairs):
                    pair_ppl[i][j] = batch_ppl[k]
        return pair_ppl


    def compute_ppl(self, logits, labels, nums):
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss(reduction='none')
        shift_logits = shift_logits.view(-1, self.model.config.vocab_size)
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

    def draw(self, save_path, matrix_data, lds):
        matrix_array = np.array(matrix_data)

        plt.imshow(matrix_array, cmap='viridis', interpolation='nearest')
        plt.colorbar()
        plt.title(f'LDS = {lds}')
        plt.show()
        plt.savefig(save_path)
        plt.clf()

    def process_data(self):
        base_name = self.data_file.split('.')[0]
        input_file = os.path.join(self.input_chunk_path, self.data_file)
        output_file = os.path.join(self.output_chunk_path, base_name+'.jsonl')
        score_chunk_path = os.path.join(self.score_chunk_path, base_name+'.txt')
        pic_dir = os.path.join(self.pic_dir, base_name)
        if not os.path.exists(score_chunk_path):
            # create file
            with open(score_chunk_path, 'w', encoding='utf-8') as f:
                pass
        chunk_num = self.window_size // self.chunk_size
        if not os.path.exists(pic_dir):
            os.makedirs(pic_dir)
        exist_score_lines = sum(1 for _ in open(score_chunk_path, 'r', encoding='utf-8'))
        print(f'[INFO] file {score_chunk_path} exist_score_lines: {exist_score_lines}')
        with open(input_file, 'r', encoding='utf-8') as fin, \
            open(score_chunk_path, 'a', encoding='utf-8') as fscore, \
            open(output_file, 'a', encoding='utf-8') as fout:
            total_lines = sum(1 for _ in fin)
            fin.seek(0)
            idx = 0
            for single_data in tqdm(fin, total=total_lines):
                # Loading Data
                if idx < exist_score_lines:
                    idx += 1
                    continue
                if input_file.endswith('.jsonl'):
                    json_data = json.loads(single_data)
                    if 'text' in json_data:
                        data = json_data['text']
                else:
                    data = single_data
                try:
                    data_list = self.construct_data(data, self.tokenizer, self.chunk_size, chunk_num)
                    start = time.time()

                    single_ppl = self.compute_single_ppl(data_list, self.single_ppl_batch_size)

                    pair_ppl = self.compute_pair_ppl(data_list, self.pair_ppl_batch_size, self.sample_size)
                    
                    long_dependency_score, dlt_ppl, dependency_entropy = self.compute_lds(single_ppl=single_ppl, pair_ppl=pair_ppl)
                except Exception as e:
                    print (f'[Error]: {e}, [file_name]: {input_file}, [idx]: {idx}, set LDS to 0.')
                    long_dependency_score = 0
                end = time.time()

                # draw
                if self.need_draw:
                    if idx % 5000 == 0:
                        print(f'file: {base_name} idx: {idx}')
                        save_pic_path = os.path.join(pic_dir, str(idx) + '.png')
                        self.draw(save_pic_path, dlt_ppl, long_dependency_score)

                # wirte score to file
                fscore.write(str(long_dependency_score) + '\n')

                # store data with score
                new_data = {'text': data, 'lds': long_dependency_score}
                fout.write(json.dumps(new_data, ensure_ascii=False) + '\n')
                idx += 1
        os.remove(input_file)

def split_jsonl(input_file, output_path, lines_per_file, total_lines):
    with open(input_file, 'r', encoding='utf-8') as f:
        file_count, line_count = 0, 0
        current_out = open(f"{output_path}/chunk_{file_count}.jsonl", 'w', encoding='utf-8')
        for line in tqdm(f, total=total_lines):
            if line_count < lines_per_file:
                current_out.write(line)
                line_count += 1
            else:
                current_out.close()
                file_count += 1
                current_out = open(f"{output_path}/chunk_{file_count}.jsonl", 'w', encoding='utf-8')
                current_out.write(line)
                line_count = 1
        current_out.close()

def merge(file_list, output_file, output_chunk_path):
    count = 0
    with open(output_file, 'w', encoding='utf-8') as fout:
        for file_name in file_list:
            with open(os.path.join(output_chunk_path, file_name), 'r', encoding='utf-8') as fin:
                for line in fin:
                    print(line, end='', file=fout)
                    count += 1
    return count


def process_single_chunk(gpu_id, file_name, input_chunk_path, output_chunk_path, score_chunk_path, pic_dir, chunk_size, window_size, sample_size, single_ppl_batch_size, pair_ppl_batch_size, dlt_ppl_threshold, model_name, use_flash_attention_2, seed, need_draw):
    pseed = int(file_name.split('.jsonl')[0][6:])
    random.seed(seed + pseed)

    process_id = os.getpid()
    print(f'[PID-{process_id}] {file_name} start!')

    # cuda
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        print('Cuda is available.')
        device = torch.device(f'cuda:{gpu_id}')
    else:
        print('Cuda is not available')
        device = torch.device('cpu')

    data_processor = DataProcessor( model_name=model_name, 
                                    use_flash_attention_2=use_flash_attention_2,
                                    data_file=file_name, 
                                    input_chunk_path=input_chunk_path, 
                                    output_chunk_path=output_chunk_path, 
                                    score_chunk_path=score_chunk_path, 
                                    pic_dir=pic_dir, 
                                    chunk_size=chunk_size, 
                                    dlt_ppl_threshold=dlt_ppl_threshold, 
                                    window_size=window_size, 
                                    single_ppl_batch_size=single_ppl_batch_size, 
                                    pair_ppl_batch_size=pair_ppl_batch_size, 
                                    sample_size=sample_size, 
                                    need_draw=need_draw, 
                                    seed=seed,
                                    device=device)
    data_processor.process_data()
    print(f'[PID-{process_id}] {file_name} end!')
    return process_id

def parse_config():
    parser = argparse.ArgumentParser()

    # data parameter
    parser.add_argument('--data_file', type=str)
    parser.add_argument('--root_path', type=str)
    # lds parameter 
    parser.add_argument('--chunk_size', type=int, default=128)
    parser.add_argument('--dlt_ppl_threshold', type=float, default=0.1)
    parser.add_argument('--window_size', type=int, default=32768)

    # model configuration
    parser.add_argument('--model_name', type=str, default='facebook/opt-350m')
    parser.add_argument('--use_flash_attention_2', action='store_true')

    # other 
    parser.add_argument('--seed', type=int, default=11)

    parser.add_argument('--single_ppl_batch_size', type=int, default=256)
    parser.add_argument('--pair_ppl_batch_size', type=int, default=256)
    parser.add_argument('--sample_size', type=int, default=5000)
    parser.add_argument('--need_draw', action='store_true')

    parser.add_argument('--gpu_ids', nargs='+', type=int, default=[0,1,2,3,4,5,6,7])

    return parser.parse_args()


if __name__ == "__main__":
    # args
    args = parse_config()
    seed = args.seed
    # set seed
    set_seed(seed=seed)

    # model
    model_name = args.model_name
    use_flash_attention_2 = args.use_flash_attention_2
    
    # data path
    data_file = args.data_file
    root_path = args.root_path

    # output path
    input_chunk_path = f'{root_path}/raw'
    output_chunk_path = f'{root_path}/processed'
    save_file = f'{root_path}/merged'
    score_chunk_path = f'{root_path}/scored'
    pic_dir = f'{root_path}/pic'

    # create dir
    if not os.path.exists(input_chunk_path):
        os.makedirs(input_chunk_path)
    if not os.path.exists(output_chunk_path):
        os.makedirs(output_chunk_path)
    if not os.path.exists(save_file):
        os.makedirs(save_file)
    if not os.path.exists(score_chunk_path):
        os.makedirs(score_chunk_path)
    if not os.path.exists(pic_dir):
        os.makedirs(pic_dir)

    # hyper-parameters
    chunk_size = args.chunk_size
    dlt_ppl_threshold = args.dlt_ppl_threshold
    window_size = args.window_size
    single_ppl_batch_size = args.single_ppl_batch_size
    pair_ppl_batch_size = args.pair_ppl_batch_size
    sample_size = args.sample_size
    need_draw = args.need_draw
    gpu_ids = args.gpu_ids
    num_process = len(gpu_ids)

    with open(data_file, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
        f.seek(0)
    if total_lines != -1:
        print(f"The file {data_file} has {total_lines} lines.")
        lines_per_file = total_lines // num_process if total_lines % num_process == 0 else total_lines // num_process + 1

        # split data into chunks
        split_jsonl(data_file, input_chunk_path, lines_per_file, total_lines)

        assert num_process < multiprocessing.cpu_count()
        pool = multiprocessing.Pool(processes=num_process)


        file_list = [file_name for file_name in os.listdir(input_chunk_path) if file_name.endswith('.jsonl')]
        # sort file_list
        file_list.sort(key=lambda x: int(x.split('.jsonl')[0][6:]))
        print(f'[INFO] {len(file_list)} files found in {input_chunk_path}')
        assert len(file_list) == num_process


        results = []
        for idx, file_name in enumerate(file_list):
            gpu_id = gpu_ids[idx]
            results.append(
                pool.apply_async(
                    process_single_chunk, 
                    (gpu_id, file_name, input_chunk_path, output_chunk_path, score_chunk_path, pic_dir, chunk_size, window_size, sample_size, single_ppl_batch_size, pair_ppl_batch_size, dlt_ppl_threshold, model_name, use_flash_attention_2, seed, need_draw)))

        pool.close()
        pool.join()

        results = [result.get() for result in results]

        score_output_file = os.path.join(save_file, 'scores.txt')
        score_file_list = [file_name for file_name in os.listdir(score_chunk_path) if file_name.endswith('.txt')]
        score_file_list.sort(key=lambda x: int(x.split('.txt')[0][6:]))
        count = merge(file_list=score_file_list, output_file=score_output_file, output_chunk_path=score_chunk_path)
        print(f'[INFO] {score_output_file}: {count} lines in total.')

        output_file = os.path.join(save_file, 'merged.jsonl')
        count = merge(file_list=file_list, output_file=output_file, output_chunk_path=output_chunk_path)
        print(f'[INFO] {output_file}: {count} lines in total.')
