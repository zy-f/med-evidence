import os
import sys
import json
import time
import logging
import threading
import concurrent.futures
from functools import partial

from easydict import EasyDict as edict
from tqdm import tqdm

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(FILE_DIR, 'src'))
import rag

# Set up logging configuration
def make_logger(name=None):
    """
    Sets up a logger to capture and display log messages for QueryLLM.

    Returns:
        logging.Logger: Configured logger object for QueryLLM operations.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    if logger.hasHandlers():
        logger.handlers.clear()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    s_handler = logging.StreamHandler()
    s_handler.setFormatter(formatter)
    s_handler.setLevel(logging.DEBUG)
    logger.addHandler(s_handler)
    if name is not None:
        f_handler = logging.FileHandler(f"{FILE_DIR}/___res/{name}.log", mode='a', encoding='utf-8')
        f_handler.setFormatter(formatter)
        f_handler.setLevel(logging.DEBUG)
        logger.addHandler(f_handler)
    return logger


def make_rag_system(cfg, logger):
    if cfg.evidence.retrieve:
        raise NotImplementedError
    else:
        if cfg.evidence.summarize:
            cfg.rag_setup.filter_evidence = cfg.evidence.llm_filter
            RAGClass = rag.SummarizingClosedRAG
        else:
            RAGClass = rag.SimpleClosedRAG
    rag_sys = RAGClass(cfg.prompt_set, cfg.model_type, logger=logger, **cfg.rag_setup)
    return rag_sys


class FlatDataset:
    def __init__(self, dataset_name, skip_ids=[]):
        with open(f'datasets/{dataset_name}.jsonl', 'r') as fh:
            self.systematic_reviews = [json.loads(line) for line in fh]
        self.dataset_len = sum([len(qa['relevant_sources'])
                                for review in self.systematic_reviews \
                                for qa in review['question_data']])
        self.skip_ids = set(skip_ids)
    
    def __len__(self):
        return self.dataset_len - len(self.skip_ids)
    
    def _iterate_questions(self):
        for review in self.systematic_reviews:
            qa_data = review['question_data']
            sources = review['sources']
            source_map = {src['article_id']: src for src in sources}
            for qa in qa_data:
                for pmid in qa['relevant_sources']:
                    sub_qa = {**qa}
                    sub_qa['question_id'] = f"{sub_qa['question_id']}-{pmid}"
                    if sub_qa['question_id'] in self.skip_ids:
                        continue
                    question_sources = [source_map[pmid]]
                    yield review['original_review'], sub_qa, question_sources
    
    def __iter__(self):
        return self._iterate_questions()

thread_local_storage = threading.local()

def answer_question(question_data, rag_system=None, logger=None):
    # unpack and setup
    review_pmid, qa, question_sources = question_data
    is_parallel = rag_system is None
    if is_parallel:
        rag_system = thread_local_storage.rag_system
    # run
    q_id = qa.pop('question_id')
    if not is_parallel:
        logger.info(rag.emph_message(f"EXAMPLE {q_id}", rep=20))
    start = time.time()
    output = rag_system.invoke(qa['question'], question_sources)
    end = time.time()
    used_articles = [a['article_id'] for a in output["articles"]]
    if not is_parallel:
        logger.info(output["answer"])
        logger.info(f"GT ANSWER: {qa['answer']}")
        logger.info(f"SOURCES USED: {used_articles}")
        logger.info(f"RUNTIME: {end-start:.2f}s")
    output['question_id'] = q_id
    for k,v in qa.items():
        output[f'gt_{k}'] = v
    output['runtime'] = round(end - start,2)
    output['original_review'] = review_pmid
    if not is_parallel:
        return output
    else:
        str_out = f"""{rag.emph_message(f"EXAMPLE {q_id}", rep=20)}\n""" \
                + f"""QUESTION: {qa['question']}\n{rag_system.out_str.strip()}\n""" \
                + f"""{output["answer"]}\nGT ANSWER: {qa['answer']}\n""" \
                + f"""SOURCES USED: {used_articles}\nRUNTIME: {end-start:.2f}s"""
        return output, str_out


def launch(mode='overwrite', parallel_size=0):
    config = {
        "shorthand_name": "deepseekV3_SLA",
        "dataset_name": "med-evidence-fulltext",
        "model_type": "deepseekV3",
        "prompt_set": "source_level_agreement",
        "evidence": {
            "summarize": False,
            "llm_filter": False,
        },
        "rag_setup": {
            "prompt_buffer_size": 1000
        }
    }
    cfg = edict(config)
    is_parallel = parallel_size > 1
    logger = make_logger(name=cfg.shorthand_name)
    logger.info(f"Running config for {cfg.shorthand_name}")
    OUTFILE = f'{FILE_DIR}/___res/{cfg.shorthand_name}.jsonl'
    skip_ids = []
    if os.path.exists(OUTFILE):
        if mode == 'overwrite':
            logger.warning(f"\n{'!'*20}\nOVERWRITING EXISTING VERSION\n{'!'*20}")
            os.remove(OUTFILE)
        elif mode == 'append':
            with open(OUTFILE, 'r') as fh:
                existing_output = [json.loads(line) for line in fh]
            logger.warning(f"EXISTING OUTPUT THUS FAR: {len(existing_output)}")
            skip_ids += [o['question_id'] for o in existing_output]
        else:
            logger.error("OUTPUT FILE ALREADY EXISTS -- EXITING")
            return 1
    
    if len(skip_ids) == 0:
        logger.info(f"FULL CONFIG:\n{json.dumps(cfg, indent=2)}")
    dataset = FlatDataset(cfg.dataset_name, skip_ids=skip_ids)
    rag_setup = (cfg, None if is_parallel else logger)
    if is_parallel:
        def init_thread():
            thread_local_storage.rag_system = make_rag_system(*rag_setup)
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=parallel_size, initializer=init_thread
        ) as executor:
            futures = [executor.submit(answer_question, q) for q in dataset]
            with tqdm(total=len(dataset), desc="answering questions") as pbar:
                for future in concurrent.futures.as_completed(futures):
                    (output, str_out) = future.result()
                    logger.info(str_out)
                    with open(OUTFILE, 'a') as fh:
                        json.dump(output, fh)
                        fh.write('\n')
                    pbar.set_description(output['question_id'])
                    pbar.update()
    else:
        rag_system = make_rag_system(*rag_setup)
        for question_data in tqdm(dataset, desc="answering questions"):
            output = answer_question(question_data, rag_system=rag_system, logger=logger)
            with open(OUTFILE, 'a') as fh:
                json.dump(output, fh)
                fh.write('\n')
    return 0


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='append', choices=['new', 'append', 'overwrite'])
    parser.add_argument('--parallel-size', type=int, default=1, help="Number of multiprocessing processes to use")
    parser.add_argument('--server', required=False)
    args = parser.parse_args()
    if args.server:
        os.environ["VLLM_SERVER"] = args.server
    launch(mode=args.mode, parallel_size=args.parallel_size)