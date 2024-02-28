import json
from datasets import load_dataset

def sort_scores(in_filename, out_filename):
    with open(in_filename, 'r', encoding='utf-8') as f:
        scores = [(float(line.strip()), i) for i, line in enumerate(f)]
    scores.sort(reverse=True)

    scores_json = [{'Index': i, 'Score': score} for score, i in scores]

    with open(out_filename, 'w', encoding='utf-8') as f:
        for score in scores_json:
            f.write(json.dumps(score) + '\n')

    print('Done!')


def get_dataset(dataset_path, index_path, output_path):
    dataset = load_dataset('text', data_files=dataset_path)
    with open(index_path, 'r', encoding='utf-8') as i:
        idx_data = i.readlines()

    idx_data = idx_data[:len(idx_data)//2] # choose top 50% TODO: use parameter to control the percentage
    index_list = [int(json.loads(idx)['Index']) for idx in idx_data]
    new_dataset = dataset['train'].select(index_list)

    with open(output_path, 'w', encoding='utf-8') as f:
        for row in new_dataset:
            text = row['text']
            f.write(text + '\n')

get_dataset('data.jsonl', 'score.jsonl', 'new_data.jsonl')