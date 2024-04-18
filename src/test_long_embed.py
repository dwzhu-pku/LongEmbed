import os
import json
import logging

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

        evaluation = MTEB(tasks=needle_passkey_task_list)
        results = evaluation.run(model, output_folder=args.output_dir, overwrite_results=True,batch_size=args.batch_size,verbosity=0)
        for key, value in results.items():
            needle_passkey_score_list = []
            for ctx_len in context_length_list:
                needle_passkey_score_list.append([ctx_len, value[f"test_{ctx_len}"]["ndcg_at_1"]])
            needle_passkey_score_list.append(["avg", sum([x[1] for x in needle_passkey_score_list])/len(context_length_list)])
            output_dict[key] = {item[0]: item[1] for item in needle_passkey_score_list}

    # evaluating retrieval tasks
    if retrieval_task_list != []:

        evaluation = MTEB(tasks=retrieval_task_list)
        results = evaluation.run(model,output_folder=args.output_dir, overwrite_results=True, batch_size=args.batch_size,verbosity=0)

        for key, value in results.items():
            split = "test" if "test" in value else "validation"
            output_dict[key] = {"ndcg@1": value[split]["ndcg_at_1"], "ndcg@10": value[split]["ndcg_at_10"]}
        
    print(output_dict)
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
    
    output_file_name += '.json'
    if len(args.task_list) >= 6:
        with open(os.path.join(args.output_dir, output_file_name), 'w') as f:
            json.dump(output_dict, f, indent=4)

    
        

if __name__ == "__main__":
    main()