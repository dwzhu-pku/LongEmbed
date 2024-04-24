import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from typing import Dict, List, Tuple
from tqdm import tqdm
from mteb.evaluation.evaluators import DRESModel
from torch import Tensor
from transformers import AutoModel, AutoTokenizer, AutoConfig

from utils import move_to_cuda, create_batch_dict, pool, logger, get_chunked_docs
from model_utils import replace_with_xformers, use_self_extend


def get_position_ids(input_ids: Tensor, mode: str = "original", max_original_positions: int=512, encode_max_length: int=512) -> Tensor:

    position_ids = list(range(input_ids.size(1)))
    if mode == "recurrent":
        position_ids = [(pid) % max_original_positions for pid in position_ids]
    elif mode == "group":
        factor = math.ceil(input_ids.size(1) / max_original_positions)
        position_ids = [(pid // factor) for pid in position_ids]
    elif mode == "interpolate":
        factor = max(encode_max_length // max_original_positions, 1)
        if input_ids.size(1) <= max_original_positions:
            position_ids = [(pid * factor) for pid in position_ids]
        
    position_ids = torch.tensor(position_ids, dtype=torch.long)
    position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
    
    return position_ids

class RetrievalModel(DRESModel):
    def __init__(self, args):
        self.args = args
        self.model_name_or_path = args.model_name_or_path

        config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=True)
        logger.info(f"Loading model: {args.model_name_or_path}, model max_position_embeddings: {config.max_position_embeddings}")

        if config.model_type in ["mistral", "e5rope"]:
            if config.rope_theta != args.rope_theta:
                logger.info(f"Adjusting rope_theta of {config.model_type} from {config.rope_theta} to {args.rope_theta}")
                
            config.rope_theta = args.rope_theta
            if config.model_type == "mistral":
                config._attn_implementation = "flash_attention_2"
                config.use_cache = False
                config.sliding_window=None

        model_kwargs = {}
        # needed for nomic dynamic ntk rope scaling
        if args.rotary_scaling_factor:
            model_kwargs["rotary_scaling_factor"] = args.rotary_scaling_factor
        
        self.encoder = AutoModel.from_pretrained(args.model_name_or_path, torch_dtype=torch.float16 if args.use_fp16 else torch.float32, config=config, ignore_mismatched_sizes=False, trust_remote_code=True, **model_kwargs)
            
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        
        if args.encode_max_length is not None:
            self.encode_max_length = args.encode_max_length
        else:
            self.encode_max_length = self.encoder.config.max_position_embeddings
        
        self.pos_mode = args.pos_mode
        self.prompt = args.prompt
        self.prefix_type = args.prefix_type
        self.pool_type = args.pool_type
        self.gpu_count = torch.cuda.device_count()
        self.l2_norm = True if args.no_l2_norm == False else False

        if self.gpu_count > 1:
            self.encoder = torch.nn.DataParallel(self.encoder)

        # monkey patch
        if args.use_xformers:
            replace_with_xformers()
        if args.use_self_extend:
            use_self_extend(args, self.encoder)

        # a hack to use grouped / recurrent pids for bert models
        if hasattr(self.encoder, "embeddings") and hasattr(self.encoder.embeddings, "token_type_ids"):
            self.encoder.embeddings.register_buffer(
                "token_type_ids", torch.zeros([1,self.encode_max_length], dtype=torch.long), persistent=False
            )
        
        self.encoder.cuda()
        self.encoder.eval()

        if self.tokenizer.padding_side == 'left':
            logger.warning('tokenizer.padding_side == left, change it to right')
            self.tokenizer.padding_side = 'right'


    def encode_queries(self, queries: List[str], batch_size: int = 64, **kwargs) -> np.ndarray:

        # in retrieval settings, queries are usually short, so we don't need to chunk them
        batch_size = max(batch_size, 64)
        if self.prefix_type == 'query_or_passage':
            input_texts = [f'query: {q}' for q in queries]
        else:
            input_texts = [self.prompt + q for q in queries]

        encoded_embeds: np.ndarray = self._do_encode(input_texts, batch_size)
        # normalize each row of encoded_embeds, using numpy
        if self.l2_norm:
            encoded_embeds = encoded_embeds / np.linalg.norm(encoded_embeds, axis=1, keepdims=True)

        return encoded_embeds

    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int, **kwargs) -> np.ndarray:

        chunking_mode: str = os.getenv('CHUNKING_MODE')
        chunked_corpus: List[Dict[str, str]] = []
        chunked_index_list: List[Tuple[int, int]] = []

        if chunking_mode == 'chunk':
            doc_list = [doc['text'] for doc in corpus]
            chunked_doc_list, chunked_index_list = get_chunked_docs(self.args, doc_list)
            chunked_corpus = [{'text': doc} for doc in chunked_doc_list]
        else:
            chunked_corpus = corpus

        input_texts = ['{} {}'.format(doc.get('title', ''), doc['text']).strip() for doc in chunked_corpus]
        # no need to add prefix for instruct models
        if self.prefix_type == 'query_or_passage':
            input_texts = ['passage: {}'.format(t) for t in input_texts]
        elif self.prefix_type == 'nomic':
            input_texts = ['search_document: {}'.format(t) for t in input_texts]
        # doing nothing for bge, none, instruction models

        encoded_embeds: np.ndarray = self._do_encode(input_texts, batch_size)
        # restore using chunked_index_list
        if chunking_mode == 'chunk':
            restored_embeds = []
            for st, ed in chunked_index_list:
                avg_embed = encoded_embeds[st:ed].mean(axis=0)
                if self.l2_norm:
                    avg_embed = avg_embed / np.linalg.norm(avg_embed)
                restored_embeds.append(avg_embed)
            encoded_embeds = np.array(restored_embeds)
        assert len(encoded_embeds) == len(corpus)

        return encoded_embeds

    @torch.no_grad()
    def _do_encode(self, input_texts: List[str], batch_size: int) -> np.ndarray:
        encoded_embeds = []
        batch_size = batch_size * self.gpu_count
        for start_idx in tqdm(range(0, len(input_texts), batch_size), desc='encoding', mininterval=10):
            batch_input_texts: List[str] = input_texts[start_idx: start_idx + batch_size]

            batch_dict = create_batch_dict(self.tokenizer, batch_input_texts, always_add_eos=(self.pool_type == 'last'), max_length=self.encode_max_length)
            if self.pos_mode != 'original':
                max_positions = 4096 if 'mistral' in self.encoder.config.model_type else 512
                batch_dict['position_ids'] = get_position_ids(batch_dict['input_ids'], mode=self.pos_mode, max_original_positions=max_positions, encode_max_length=self.encode_max_length)
            batch_dict = move_to_cuda(batch_dict)

            with torch.cuda.amp.autocast():
                outputs = self.encoder(**batch_dict)
                embeds = pool(outputs.last_hidden_state, batch_dict['attention_mask'], self.pool_type)
                chunking_mode: str = os.getenv('CHUNKING_MODE')
                if self.l2_norm and chunking_mode != 'chunk':
                    embeds = F.normalize(embeds, p=2, dim=-1)
                encoded_embeds.append(embeds.cpu().numpy())

        encoded_embeds = np.concatenate(encoded_embeds, axis=0)
        # check nan
        if np.isnan(encoded_embeds).any():
            logger.info('nan detected in encoded_embeds')
            exit(1)

        return encoded_embeds

    def set_prompt(self, prompt: str):
        self.prompt = prompt