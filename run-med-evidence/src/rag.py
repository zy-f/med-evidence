import os
import json
import re
import logging

from tqdm import tqdm
from langchain_core.globals import set_verbose, set_debug
set_verbose(False) # Disable langchain verbose logging
set_debug(False)   # Disable  langchain_core debug logging

from queryllms import QueryLLM, QuerySummaryChainLLM
from utils.text_utils import TextSplitter, Page
from utils.prompts import load_prompts
from utils.CONSTS import LLM_MODEL_MAP, KNOWN_MAX_TOKENS

def emph_message(message:str,rep:int):
    return f"""\n{"*"*rep}\n{message}\n{"*"*rep}\n"""

def sep(rep:int=2):
    return f"""\n{"*"*rep}\n{"-"*50}"""


class SimpleClosedRAG:
    def __init__(self, 
        prompt_set,
        model_type,
        prompt_buffer_size=1000,
        max_input_token=4096,
        logger:logging.Logger=logging.getLogger(),
        **extra_llm_kwargs
    ):
        self.prompts = load_prompts(prompt_set, check_relevance=False)
        provider, model_name = LLM_MODEL_MAP[model_type] if isinstance(model_type, str) else model_type
        llm_config = dict(
            provider=provider,
            api_key=None,
            model=model_name,
            parameters={
                "temperature":0,
                "timeout":None,
                "max_retries":10,
                "max_tokens":None
            },
            host_vllm_manually=True
        )
        if model_type == 'gpto1':
            llm_config['parameters'].pop('temperature')
        llm_config.update(**extra_llm_kwargs)
        self.llm = QueryLLM(**llm_config)
        
        self.compile_source_chain = QuerySummaryChainLLM(
            llm_client=self.llm.client,
            instruction_prompt=self.prompts['final_answer'])

        self.max_input_token = KNOWN_MAX_TOKENS.get(self.llm.model, max_input_token)
        self.splitter = TextSplitter(max_num_tokens=self.max_input_token - prompt_buffer_size)
        self.logger = logger
        self.out_str = ""


    def _merge_texts(self,
        header:str,
        texts:list,
    ) -> list[Page]:
        max_len = self.splitter.max_num_tokens
        init_token_count = self.splitter.get_num_tokens(header)
        cur_page = header
        running_token_count = init_token_count
        chunks = []
        def add_page(page): # process a chunk of summaries as a Page and reset for the new article
            chunked_page = self.splitter.split(page)
            if len(chunked_page) > 1:
                if self.logger is None:
                    self.out_str += "WARNING: Text for a single element exceeds token limit, ignoring past first chunk\n"
                else:
                    self.logger.warning("Text for a single element exceeds token limit, ignoring past first chunk")
            chunks.append(chunked_page[0])

        for text in texts:
            text_num_tokens = self.splitter.get_num_tokens(text)
            if running_token_count + text_num_tokens > max_len:
                add_page(cur_page)
                # reset the count and string
                running_token_count = init_token_count
                cur_page = header
            running_token_count += text_num_tokens
            cur_page += text
        # make the last page
        if running_token_count == init_token_count: # EDGE CASE: no additional text
            cur_page += "No text available"
        add_page(cur_page)
        return chunks
    

    def _format_article_summaries(self,
        summaries:list,
        metadata:list
    ) -> list[str]:
        formatted_summaries:list[str] = []
        for (meta, summary) in zip(metadata, summaries):
            title, article_id = meta.get('title'), meta.get('article_id')
            date  = meta.get('date')
            if not date:
                date = "No date provided"
            article = f"----\n\nArticle ID {article_id}:\nTITLE: {title}\nPublished Date: {date}\n\nRelevant Information:\n{summary}\n\n----"
            formatted_summaries.append(article)
        return formatted_summaries


    def parse_articles(self, articles):
        article_metadata = []
        article_summaries = []
        for i, article_data in enumerate(articles):
            title, content = article_data.get('title'), article_data.get('content')
            if not title:
                title = "No title provided"
            if not content:
                content = "No article content provided"
            a_id = article_data.get('article_id', i)
            date = article_data.get('date')
            metadata = {'title': title, 'article_id': a_id, 'date': date}
            article_metadata.append(metadata)
            article_summaries.append(content)
        return article_summaries, article_metadata


    def _format_final_answer(self,STORE):
        # Define a pattern to extract the key-value pairs
        pattern = r"(-\s*)?\*\*(.*?)\*\*:\s*((.|\n)*?)(?=\n+(-\s*)?\*\*|\Z)"
        matches = re.findall(pattern, STORE["answer"] , re.DOTALL) # Extract matches
        if matches:
            # Convert to a dictionary
            parsed_data = {}
            for (_,key,value,_,_) in matches:
                formatted_key = key.strip().lower().replace(" ","_")
                parsed_data[formatted_key] = value.strip()
            STORE["parsed_answer"] = parsed_data
        else:
            STORE["parsed_answer"] = "Could not parse"


    def answer_question(self,
            question:str,
            article_summaries:list,
            article_metadata:list
    ) -> dict:
        log_msg = "Answer with RAG"
        formatted_summaries:list[str] = self._format_article_summaries(article_summaries, article_metadata)

        summaries_header = self.prompts['final_answer_values'].replace('{question}', question).replace('{context}',"")
        prompt_chunks:list[Page] = self._merge_texts(summaries_header, formatted_summaries)
        use_chain:bool = len(prompt_chunks) > 1
        if use_chain:
            log_msg += f" - long overall summary split into {len(prompt_chunks)} chunk and using llmchain"
            answer_log:str = self.compile_source_chain.querychain(documents=prompt_chunks)
            final_answer:str = answer_log["output_clean"]
        else:
            # add the header prompt
            summaries_text = prompt_chunks[0].page_content
            prompt_chunks[0].update_content(self.prompts['final_answer'].replace("{text}",summaries_text))
            final_answer:str = self.llm.simple_query("follow instruction",prompt_chunks[0].page_content)

        STORE = {}
        STORE["n_articles"] = len(formatted_summaries)
        STORE["question"]   = question
        STORE["articles"]   = article_metadata
        STORE["summaries"]  = article_summaries
        STORE["answer"]     = final_answer
        self._format_final_answer(STORE)

        if self.logger is None:
            self.out_str += f"{emph_message(message='',rep=20)}\n{emph_message(message=log_msg,rep=20)}\n{sep(rep=2)}\n"
        else:
            self.logger.info(emph_message(message="",rep=20))
            self.logger.info(emph_message(message=log_msg,rep=20))
            self.logger.info(sep(rep=2))
        return STORE


    def invoke(self,
            question:str,
            articles:list,
    ) -> dict:
        if self.logger is None:
            self.out_str = ""
        article_summaries, article_metadata = self.parse_articles(articles)
        answer:dict = self.answer_question(question, article_summaries, article_metadata)
        return answer


class LongSummaryClosedRAG(SimpleClosedRAG):
    def _format_article_summaries(self, # update: let article summaries run across multiple page chunks
        summaries:list,
        metadata:list
    ) -> list[str]:
        formatted_summaries:list[str] = []
        for (meta, summary) in zip(metadata, summaries):
            title, article_id = meta.get('title'), meta.get('article_id')
            date = meta.get('date')
            if not date:
                date = "No date provided"
            article_header = f"----\n\nArticle ID {article_id}:\nTITLE: {title}\nPublished Date: {date}\n\n"
            chunked_summary = self.splitter.split(summary)
            if len(chunked_summary) > 1:
                self.out_str += f"Relevant article {article_id} split into {len(chunked_summary)} separate summaries\n"
            for page in chunked_summary:
                formatted_summary = article_header+f"Relevant Information:\n{page.page_content}\n\n----"
                formatted_summaries.append(formatted_summary)
        return formatted_summaries


class SummarizingClosedRAG(SimpleClosedRAG):
    def __init__(self, 
        prompt_set,
        model_type,
        prompt_buffer_size=1000,
        filter_evidence=False,
        **kwargs
    ):
        super().__init__(prompt_set, model_type, 
            prompt_buffer_size=prompt_buffer_size, **kwargs)
        self.prompts = load_prompts(prompt_set, check_relevance=filter_evidence)
        self.summarize_source_chain = QuerySummaryChainLLM(
            llm_client=self.llm.client,
            instruction_prompt=self.prompts['get_info'])
        self.filter_evidence = filter_evidence


    def summarize_evidence_sequential(self,
        question:str,
        articles:list,
    ):
        n_articles = len(articles)

        if self.logger is None:
            self.out_str += emph_message(message="Reading Articles",rep=18) + '\n'
        else:
            self.logger.info(emph_message(message="Reading Articles",rep=18))

        article_metadata = []
        article_summaries = []
        for i, article_data in enumerate(articles):
            title, content = article_data.get('title'), article_data.get('content')
            if not title:
                title = "No title provided"
            if not content:
                content = "No article content provided"
            a_id = article_data.get('article_id', i)
            date = article_data.get('date')
            metadata = {'title': title, 'article_id': a_id, 'date': date}

            query_msg:str = f"[{i+1}/{n_articles}] Getting summary for: \"{title}\""
            chunks = self.splitter.split(content)
            use_chain:bool = len(chunks) > 1
            if use_chain == False:
                prompt = self.prompts['get_info'].replace("{text}","") + self.prompts['get_info_values'].replace("{question}",question).replace("{title}",title).replace("{context}",chunks[0].page_content)
                summary:str = self.llm.simple_query("follow instruction", prompt)
            else:
                query_msg += f" -> Article was split into {len(chunks)} chunks, thus use_llm_chain={use_chain}"
                [c.update_content(self.prompts['get_info_values'].replace("{question}",question).replace("{title}",title).replace("{context}",c.page_content)) for c in chunks]
                summary_log:str = self.summarize_source_chain.querychain(documents=chunks)
                summary:str = summary_log["output_clean"]
            
            if self.logger is None:
                self.out_str += query_msg + '\n'
            else:
                self.logger.info(query_msg)
            article_metadata.append(metadata)
            article_summaries.append(summary)
        return article_summaries, article_metadata


    def _filter_summaries(self,
        summaries:list,
        metadata:list
    ):
        filtered_summaries, filtered_metadata = [], []
        for s,m in zip(summaries, metadata):
            if s.upper().strip() != 'N/A':
                filtered_summaries.append(s)
                filtered_metadata.append(m)
        return filtered_summaries, filtered_metadata


    def invoke(self,
            question:str,
            articles:list,
    ) -> dict:
        if self.logger is None:
            self.out_str = ""
        article_summaries, article_metadata = self.summarize_evidence_sequential(question, articles)
        if self.filter_evidence:
            article_summaries, article_metadata = self._filter_summaries(article_summaries, article_metadata)
        answer:dict = self.answer_question(question, article_summaries, article_metadata)
        return answer