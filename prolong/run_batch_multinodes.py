import os
import argparse
import subprocess

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=str)
    parser.add_argument('--output-dir', type=str)
    parser.add_argument('--ip-hostfile', type=str)
    parser.add_argument('--file-list', type=str)

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
    parser.add_argument('--sample_size', type=int, default=500)
    parser.add_argument('--need_draw', action='store_true')

    parser.add_argument('--gpu_ids', nargs='+', type=int, default=[0,1,2,3,4,5,6,7])

    args = parser.parse_args()
    return args


def get_ips(hostfile_path):
    """Read ips from hostfiles"""
    with open(hostfile_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    ip_list = []
    for line in lines:
        ip = line.strip().split(' ')[0]
        ip_list.append(ip)
    return ip_list

def get_files(file_list):
    """Read files from file lists"""
    with open(file_list, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    lines = [line.strip() for line in lines]
    return lines

def get_cmd(args, filename, output_dir):
    path = os.path.abspath("run_batch_multi_process.py")
    cmd = "python {}".format(path)
    cmd += " --data_file {}".format(filename)
    cmd += " --root_path {}".format(output_dir)
    cmd += " --chunk_size {}".format(args.chunk_size)
    cmd += " --dlt_ppl_threshold {}".format(args.dlt_ppl_threshold)
    cmd += " --window_size {}".format(args.window_size)
    cmd += " --model_name {}".format(args.model_name)
    if args.use_flash_attention_2:
        cmd += " --use_flash_attention_2"
    cmd += " --seed {}".format(args.seed)
    cmd += " --single_ppl_batch_size {}".format(args.single_ppl_batch_size)
    cmd += " --pair_ppl_batch_size {}".format(args.pair_ppl_batch_size)
    cmd += " --sample_size {}".format(args.sample_size)
    if args.need_draw:
        cmd += " --need_draw"
    gpu_ids = ' '.join([str(gpu_id) for gpu_id in args.gpu_ids])
    cmd += " --gpu_ids {}".format(gpu_ids)

    return cmd

def main():
    args = get_arguments()
    ip_list = get_ips(args.ip_hostfile)
    # filenames = os.listdir(args.input_dir)
    filenames = get_files(args.file_list)

    # get basename of filenames
    basename_list = []
    for filename in filenames:
        basename = os.path.splitext(filename)[0]
        basename_list.append(basename)
    
    # output dir
    output_path_list = []
    for basename in basename_list:
        output_path = os.path.join(args.output_dir, basename)
        output_path_list.append(output_path)
        os.makedirs(output_path, exist_ok=True)
    
    for idx, filename in enumerate(filenames):
        ip = ip_list[idx]
        input_path = os.path.join(args.input_dir, filename)
        cmd = get_cmd(args, input_path, output_path_list[idx])
        
        meta_cmd = 'pdsh -R ssh -w {} {} &'.format(ip, cmd)
        
        print(f"ip: {ip}, cmd: {cmd}")
        os.system(meta_cmd)
        

if __name__ == '__main__':
    main()