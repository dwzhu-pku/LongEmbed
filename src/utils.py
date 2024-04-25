import os
import json
import torch
import logging
import argparse

from torch import Tensor
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast, BatchEncoding, AutoTokenizer
from typing import Mapping, Dict, List, Tuple, Optional, Sequence

from model_config import MODEL_NAME_TO_POOL_TYPE, MODEL_NAME_TO_PREFIX_TYPE

token_per_word: float = 4/3 # using empirical evidence that 1 token is about 3/4 of a word

def _setup_logger():
    log_format = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]

    return logger


logger = _setup_logger()


def move_to_cuda(sample):
    if len(sample) == 0:
        return {}

    def _move_to_cuda(maybe_tensor):
        if torch.is_tensor(maybe_tensor):
            return maybe_tensor.cuda(non_blocking=True)
        elif isinstance(maybe_tensor, dict):
            return {key: _move_to_cuda(value) for key, value in maybe_tensor.items()}
        elif isinstance(maybe_tensor, list):
            return [_move_to_cuda(x) for x in maybe_tensor]
        elif isinstance(maybe_tensor, tuple):
            return tuple([_move_to_cuda(x) for x in maybe_tensor])
        elif isinstance(maybe_tensor, Mapping):
            return type(maybe_tensor)({k: _move_to_cuda(v) for k, v in maybe_tensor.items()})
        else:
            return maybe_tensor

    return _move_to_cuda(sample)


def pool(last_hidden_states: Tensor,
         attention_mask: Tensor,
         pool_type: str) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)

    if pool_type == "avg":
        emb = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    elif pool_type == "weightedavg":  # position-weighted mean pooling from SGPT (https://arxiv.org/abs/2202.08904)
        attention_mask *= attention_mask.cumsum(dim=1)  # [0,1,1,1,0,0] -> [0,1,2,3,0,0]
        s = torch.sum(last_hidden * attention_mask.unsqueeze(-1).float(), dim=1)
        d = attention_mask.sum(dim=1, keepdim=True).float()
        emb = s / d
    elif pool_type == "cls":
        emb = last_hidden[:, 0]
    elif pool_type == "last":
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            emb = last_hidden[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden.shape[0]
            emb = last_hidden[torch.arange(batch_size, device=last_hidden.device), sequence_lengths]
    else:
        raise ValueError(f"pool_type {pool_type} not supported")

    return emb


def create_batch_dict(tokenizer: PreTrainedTokenizerFast, input_texts: List[str], always_add_eos: bool, max_length: int = 512) -> BatchEncoding:
    
    if not always_add_eos:
        return tokenizer(
            input_texts,
            max_length=max_length,
            padding=True,
            pad_to_multiple_of=8,
            return_token_type_ids=False,
            add_special_tokens=True,
            truncation=True,
            return_tensors='pt'
        )
    else:
        batch_dict = tokenizer(
            input_texts,
            max_length=max_length - 1,
            return_token_type_ids=False,
            return_attention_mask=False,
            padding=False,
            truncation=True
        )

        # append eos_token_id to every input_ids
        batch_dict['input_ids'] = [input_ids + [tokenizer.eos_token_id] for input_ids in batch_dict['input_ids']]

        return tokenizer.pad(
            batch_dict,
            padding=True,
            pad_to_multiple_of=8,
            return_attention_mask=True,
            return_tensors="pt",
        )


def get_task_def_by_task_name_and_type(task_name: str, task_type: str) -> str:
    if task_type in ['STS']:
        return "Retrieve semantically similar text."

    if task_type in ['Summarization']:
        return "Given a news summary, retrieve other semantically similar summaries"

    if task_type in ['BitextMining']:
        return "Retrieve parallel sentences."

    if task_type in ['Classification']:
        task_name_to_instruct: Dict[str, str] = {
            'AmazonCounterfactualClassification': 'Classify a given Amazon customer review text as either counterfactual or not-counterfactual',
            'AmazonPolarityClassification': 'Classify Amazon reviews into positive or negative sentiment',
            'AmazonReviewsClassification': 'Classify the given Amazon review into its appropriate rating category',
            'Banking77Classification': 'Given a online banking query, find the corresponding intents',
            'EmotionClassification': 'Classify the emotion expressed in the given Twitter message into one of the six emotions: anger, fear, joy, love, sadness, and surprise',
            'ImdbClassification': 'Classify the sentiment expressed in the given movie review text from the IMDB dataset',
            'MassiveIntentClassification': 'Given a user utterance as query, find the user intents',
            'MassiveScenarioClassification': 'Given a user utterance as query, find the user scenarios',
            'MTOPDomainClassification': 'Classify the intent domain of the given utterance in task-oriented conversation',
            'MTOPIntentClassification': 'Classify the intent of the given utterance in task-oriented conversation',
            'ToxicConversationsClassification': 'Classify the given comments as either toxic or not toxic',
            'TweetSentimentExtractionClassification': 'Classify the sentiment of a given tweet as either positive, negative, or neutral',
            # C-MTEB eval instructions
            'TNews': 'Classify the fine-grained category of the given news title',
            'IFlyTek': 'Given an App description text, find the appropriate fine-grained category',
            'MultilingualSentiment': 'Classify sentiment of the customer review into positive, neutral, or negative',
            'JDReview': 'Classify the customer review for iPhone on e-commerce platform into positive or negative',
            'OnlineShopping': 'Classify the customer review for online shopping into positive or negative',
            'Waimai': 'Classify the customer review from a food takeaway platform into positive or negative',
        }
        return task_name_to_instruct[task_name]

    if task_type in ['Clustering']:
        task_name_to_instruct: Dict[str, str] = {
            'ArxivClusteringP2P': 'Identify the main and secondary category of Arxiv papers based on the titles and abstracts',
            'ArxivClusteringS2S': 'Identify the main and secondary category of Arxiv papers based on the titles',
            'BiorxivClusteringP2P': 'Identify the main category of Biorxiv papers based on the titles and abstracts',
            'BiorxivClusteringS2S': 'Identify the main category of Biorxiv papers based on the titles',
            'MedrxivClusteringP2P': 'Identify the main category of Medrxiv papers based on the titles and abstracts',
            'MedrxivClusteringS2S': 'Identify the main category of Medrxiv papers based on the titles',
            'RedditClustering': 'Identify the topic or theme of Reddit posts based on the titles',
            'RedditClusteringP2P': 'Identify the topic or theme of Reddit posts based on the titles and posts',
            'StackExchangeClustering': 'Identify the topic or theme of StackExchange posts based on the titles',
            'StackExchangeClusteringP2P': 'Identify the topic or theme of StackExchange posts based on the given paragraphs',
            'TwentyNewsgroupsClustering': 'Identify the topic or theme of the given news articles',
            # C-MTEB eval instructions
            'CLSClusteringS2S': 'Identify the main category of scholar papers based on the titles',
            'CLSClusteringP2P': 'Identify the main category of scholar papers based on the titles and abstracts',
            'ThuNewsClusteringS2S': 'Identify the topic or theme of the given news articles based on the titles',
            'ThuNewsClusteringP2P': 'Identify the topic or theme of the given news articles based on the titles and contents',
        }
        return task_name_to_instruct[task_name]

    if task_type in ['Reranking', 'PairClassification']:
        task_name_to_instruct: Dict[str, str] = {
            'AskUbuntuDupQuestions': 'Retrieve duplicate questions from AskUbuntu forum',
            'MindSmallReranking': 'Retrieve relevant news articles based on user browsing history',
            'SciDocsRR': 'Given a title of a scientific paper, retrieve the titles of other relevant papers',
            'StackOverflowDupQuestions': 'Retrieve duplicate questions from StackOverflow forum',
            'SprintDuplicateQuestions': 'Retrieve duplicate questions from Sprint forum',
            'TwitterSemEval2015': 'Retrieve tweets that are semantically similar to the given tweet',
            'TwitterURLCorpus': 'Retrieve tweets that are semantically similar to the given tweet',
            # C-MTEB eval instructions
            'T2Reranking': 'Given a Chinese search query, retrieve web passages that answer the question',
            'MMarcoReranking': 'Given a Chinese search query, retrieve web passages that answer the question',
            'CMedQAv1': 'Given a Chinese community medical question, retrieve replies that best answer the question',
            'CMedQAv2': 'Given a Chinese community medical question, retrieve replies that best answer the question',
            'Ocnli': 'Retrieve semantically similar text.',
            'Cmnli': 'Retrieve semantically similar text.',
        }
        return task_name_to_instruct[task_name]

    if task_type in ['Retrieval']:
        if task_name.lower().startswith('cqadupstack'):
            return 'Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question'

        task_name_to_instruct: Dict[str, str] = {
            'ArguAna': 'Given a claim, find documents that refute the claim',
            'ClimateFEVER': 'Given a claim about climate change, retrieve documents that support or refute the claim',
            'DBPedia': 'Given a query, retrieve relevant entity descriptions from DBPedia',
            'FEVER': 'Given a claim, retrieve documents that support or refute the claim',
            'FiQA2018': 'Given a financial question, retrieve user replies that best answer the question',
            'HotpotQA': 'Given a multi-hop question, retrieve documents that can help answer the question',
            'MSMARCO': 'Given a web search query, retrieve relevant passages that answer the query',
            'MSMARCODOC': 'Given a web search query, retrieve relevant passages that answer the query',
            'Needles': 'Given a web search query, retrieve relevant passages that answer the query',
            'NFCorpus': 'Given a question, retrieve relevant documents that best answer the question',
            'NQ': 'Given a question, retrieve Wikipedia passages that answer the question',
            'QuoraRetrieval': 'Given a question, retrieve questions that are semantically equivalent to the given question',
            'SCIDOCS': 'Given a scientific paper title, retrieve paper abstracts that are cited by the given paper',
            'SciFact': 'Given a scientific claim, retrieve documents that support or refute the claim',
            'Touche2020': 'Given a question, retrieve detailed and persuasive arguments that answer the question',
            'TRECCOVID': 'Given a query on COVID-19, retrieve documents that answer the query',
            # C-MTEB eval instructions
            'T2Retrieval': 'Given a Chinese search query, retrieve web passages that answer the question',
            'MMarcoRetrieval': 'Given a web search query, retrieve relevant passages that answer the query',
            'DuRetrieval': 'Given a Chinese search query, retrieve web passages that answer the question',
            'CovidRetrieval': 'Given a question on COVID-19, retrieve news articles that answer the question',
            'CmedqaRetrieval': 'Given a Chinese community medical question, retrieve replies that best answer the question',
            'EcomRetrieval': 'Given a user query from an e-commerce website, retrieve description sentences of relevant products',
            'MedicalRetrieval': 'Given a medical question, retrieve user replies that best answer the question',
            'VideoRetrieval': 'Given a video search query, retrieve the titles of relevant videos',
        }

        # add lower case keys to match some beir names
        task_name_to_instruct.update({k.lower(): v for k, v in task_name_to_instruct.items()})
        # other cases where lower case match still doesn't work
        task_name_to_instruct['trec-covid'] = task_name_to_instruct['TRECCOVID']
        task_name_to_instruct['climate-fever'] = task_name_to_instruct['ClimateFEVER']
        task_name_to_instruct['dbpedia-entity'] = task_name_to_instruct['DBPedia']
        task_name_to_instruct['webis-touche2020'] = task_name_to_instruct['Touche2020']
        task_name_to_instruct['fiqa'] = task_name_to_instruct['FiQA2018']
        task_name_to_instruct['quora'] = task_name_to_instruct['QuoraRetrieval']

        # for miracl evaluation
        task_name_to_instruct['miracl'] = 'Given a question, retrieve Wikipedia passages that answer the question'

        return task_name_to_instruct[task_name]

    raise ValueError(f"No instruction config for task {task_name} with type {task_type}")


def get_detailed_instruct(task_description: str) -> str:
    if not task_description:
        return ''

    return 'Instruct: {}\nQuery: '.format(task_description)

def get_chunked_docs(args: argparse.Namespace, doc_list: List[str]) -> Tuple[List[str],List[Tuple[int, int]]]:

    chunk_size = int(os.getenv("CHUNK_SIZE", 512))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", 0))
    max_token_num = int(os.getenv("MAX_TOKEN_NUM", 0))
    
    chunked_doc_list: List[str] = []
    chunked_index_list: List[Tuple[int, int]] = []

    logger.info(f'start chunking corpus, chunk_size: {chunk_size}, chunk_overlap: {chunk_overlap}, max_token_num: {max_token_num}')

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    chunk_size = chunk_size - 2 # for [CLS] and [SEP]
    truncate_length = max_token_num if max_token_num > 0 else None
    truncate_length = truncate_length - 2 if truncate_length is not None else None

    doc_token_list = tokenizer(
        doc_list, 
        add_special_tokens=False, 
        truncation=True if truncate_length is not None else False, 
        max_length = truncate_length
    )['input_ids']

    for token_list in tqdm(doc_token_list, desc='chunking', mininterval=10):
        
        st = len(chunked_doc_list)
        for idx in range(0, len(token_list), chunk_size):
            if idx + chunk_size > len(token_list):
                chunk = token_list[-chunk_size:]
                chunked_doc_list.append(tokenizer.decode(chunk, skip_special_tokens=True))
                break
            else:
                chunk = token_list[idx:idx + chunk_size]
                chunked_doc_list.append(tokenizer.decode(chunk, skip_special_tokens=True))
        ed = len(chunked_doc_list)
        chunked_index_list.append((st, ed))


    return chunked_doc_list, chunked_index_list


def get_args(input_args: Optional[Sequence[str]] = None) -> argparse.Namespace:
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--pos_mode", type=str, default="original", help="original / recurrent / group")
    parser.add_argument('--pool_type', type=str, default='avg', help='pool type')
    parser.add_argument('--prefix_type', type=str, default='query_or_passage', help='prefix type')
    parser.add_argument('--dataset_name_or_path', type=str, default=None)
    parser.add_argument("--chunk_mode", type=str, default="token", help="word / token")
    parser.add_argument('--task_list', type=str, nargs='+', default=["LEMBSummScreenFDRetrieval", "LEMBQMSumRetrieval","LEMBWikimQARetrieval","LEMBNarrativeQARetrieval","LEMBNeedleRetrieval", "LEMBPasskeyRetrieval"])

    parser.add_argument("--encode_max_length", type=int, default=None)
    parser.add_argument("--group_size_1", type=int, default=8)
    parser.add_argument("--group_size_2", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument('--window_length_list', type=int, nargs='+', default=[])
    parser.add_argument("--rope_theta", type=int, default=10000)
    parser.add_argument("--rotary_scaling_factor", type=int, default=None)

    parser.add_argument("--use_fp16", action="store_true", default=False)
    parser.add_argument("--no_l2_norm", action="store_true", default=False)
    parser.add_argument("--use_xformers", action="store_true", default=False)
    parser.add_argument("--use_self_extend", action="store_true", default=False)

    args = parser.parse_args(input_args)

    base_name: str = os.path.basename(os.path.normpath(args.model_name_or_path))
    logger.info(f'base_name: {base_name}')
    args.pool_type = MODEL_NAME_TO_POOL_TYPE.get(base_name, args.pool_type)
    args.prefix_type = MODEL_NAME_TO_PREFIX_TYPE.get(base_name, args.prefix_type)

    if args.prefix_type == 'bge':
        args.prompt: str = """Represent this sentence for searching relevant passages: """
    elif args.prefix_type == 'instruction':
        task_def: str = get_task_def_by_task_name_and_type(task_name="Needles", task_type="Retrieval")
        args.prompt: str = get_detailed_instruct(task_def)
    elif args.prefix_type == 'nomic':
        args.prompt: str = 'search_query: '
    else:
        args.prompt: str = ''

    logger.info('Args: {}'.format(json.dumps(args.__dict__, ensure_ascii=False, indent=4)))
    assert args.pool_type in ['cls', 'avg', 'last', 'weightedavg'], 'pool_type should be cls / avg / last'
    assert args.prefix_type in ['query_or_passage', 'instruction', 'bge', 'none', "nomic"], 'prefix_type should be query_or_passage / instruction'

    return args