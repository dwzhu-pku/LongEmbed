import os
import json
import argparse
import logging

from tabulate import tabulate
from mteb import MTEB

from utils import logger, get_detailed_instruct, get_task_def_by_task_name_and_type, get_args
from encoder_model import RetrievalModel

logging.getLogger().setLevel(logging.INFO)

def main():

    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    model = RetrievalModel(args)
    retrieval_task_list = []
    needle_passkey_task_list = []
    output_dict = dict()
    retrieval_task_results = list()
    needle_passkey_score_list = list()

    for task in ["LEMBSummScreenFDRetrieval", "LEMBQMSumRetrieval","LEMBWikimQARetrieval","LEMBNarrativeQARetrieval"]:
        if task in args.task_list:
            retrieval_task_list.append(task)
    
    for task in ["LEMBNeedleRetrieval", "LEMBPasskeyRetrieval"]:
        if task in args.task_list:
            needle_passkey_task_list.append(task)

    # evaluating needle and passkey retrieval tasks
    if needle_passkey_task_list != []:

        context_length_list = list(args.window_length_list)
        context_length_list.sort()

        for ctx_len in context_length_list:
            print(f"Running task: NeedlesRetrieval, PasskeyRetrieval, context length: {ctx_len}")
            evaluation = MTEB(tasks=["LEMBNeedleRetrieval", "LEMBPasskeyRetrieval"])
            results = evaluation.run(model, context_length=ctx_len,overwrite_results=True,batch_size=args.batch_size)
            needle_passkey_score_list.append([ctx_len, results["LEMBNeedleRetrieval"]["test"]["ndcg_at_1"], results["LEMBPasskeyRetrieval"]["test"]["ndcg_at_1"]])

        needle_passkey_score_list.append(["avg", sum([x[1] for x in needle_passkey_score_list])/len(context_length_list), sum([x[2] for x in needle_passkey_score_list])/len(context_length_list)])

        output_dict["needle"] = {item[0]: item[1] for item in needle_passkey_score_list}
        output_dict["passkey"] = {item[0]: item[2] for item in needle_passkey_score_list}

        print(tabulate(needle_passkey_score_list, headers=["Context Length", "Needle-ACC", "Passkey-ACC"]))

    # evaluating retrieval tasks
    if retrieval_task_list != []:

        evaluation = MTEB(tasks=retrieval_task_list)
        results = evaluation.run(model,output_folder=args.output_dir, overwrite_results=True, batch_size=args.batch_size,verbosity=0)

        for key, value in results.items():
            split = "test" if "test" in value else "validation"
            retrieval_task_results.append([key, value[split]["ndcg_at_1"], value[split]["ndcg_at_10"]])
            output_dict[key] = {"ndcg@1": value[split]["ndcg_at_1"], "ndcg@10": value[split]["ndcg_at_10"]}
        
        print(tabulate(retrieval_task_results, headers=["Task", "NDCG@1", "NDCG@10"]))
        
        if needle_passkey_score_list != []:
            print(tabulate(needle_passkey_score_list, headers=["Context Length", "Needle-ACC", "Passkey-ACC"]))

    # set output file name
    output_file_name: str = os.path.basename(os.path.normpath(args.model_name_or_path))
    
    chunking_mode: str = os.getenv('CHUNKING_MODE')
    if chunking_mode:
        if chunking_mode == "no_chunk":
            chunk_max_len = args.encode_max_length
        else:
            chunk_max_len = os.getenv('MAX_TOKEN_NUM', "0")
        output_file_name += f"_{chunking_mode}-{chunk_max_len}"
    if args.pos_mode != "original":
        output_file_name += f'_{args.pos_mode}'
    if args.use_self_extend == True:
        output_file_name += "_se"
    if args.rope_theta != 10000:
        output_file_name += f"_theta{args.rope_theta}"
    if args.rearrange_pids:
        output_file_name += "_rearrange"
    
    output_file_name += '.json'
    if len(args.task_list) >= 6:
        with open(os.path.join(args.output_dir, output_file_name), 'w') as f:
            json.dump(output_dict, f, indent=4)

    
        

if __name__ == "__main__":
    main()