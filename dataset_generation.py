import os
import re
import pickle
import json
import time

import pandas as pd
import numpy as np
from tqdm import tqdm

import download_pubmed_data as dpd

ROOT = "dset_gen"

def load_metadata():
    import ast
    setupdf = pd.read_csv('run-med-evidence/___res/dataset_stats.csv')
    setupdf.loc[:,'relevant_sources'] = setupdf['relevant_sources'].apply(ast.literal_eval)
    setupdf['review_pmid'] = setupdf['review_pmid'].astype(str)
    setupdf.loc[:, 'evidence_quality'] = setupdf['evidence_quality'].fillna('N/A')
    meta = {}
    for _, row in setupdf.iterrows():
        meta[row.question_id] = {'source_concordance': row.sla, 'medical_specialty': row.med_specialty}
    return meta

def make_manual_dataset(dataset_name, use_fulltext=False):
    with open(f"{ROOT}/{dataset_name}-setup.json", 'r') as fh:
        setup_data = json.load(fh)
    rct_text_data = dpd.get_rct_pmid_data()
    fulltext_used = 0
    content_type = 'fulltext' if use_fulltext else 'abstract'
    if use_fulltext:
        rct_fulltext_data = dpd.get_rct_fulltext()
    review_to_rct_pmids = dpd.get_review_to_references_map(id_type="pmid")
    qa_dataset = []
    for review_pmid, review_qs in tqdm(setup_data.items()):
        if len(review_qs) < 1:
            continue
        all_review_rcts = review_to_rct_pmids[review_pmid]
        for q in review_qs:
            assert all([pmid in all_review_rcts for pmid in q['relevant_sources']]),\
                f"relevant={q['relevant_sources']}, all={all_review_rcts}"
        article_data = []
        for rct_pmid in all_review_rcts:
            rct_data = {'article_id': rct_pmid, **rct_text_data[rct_pmid]}
            if use_fulltext and (rct_pmid in rct_fulltext_data):
                rct_data['content'] = rct_fulltext_data[rct_pmid]
                fulltext_used += 1
            article_data.append(rct_data)
        qa_dataset.append({
            "original_review": review_pmid,
            "question_data": review_qs,
            "sources": article_data
        })
    if use_fulltext:
        print("fulltexts used:", fulltext_used)
    dset_path = f'run-med-evidence/datasets/{dataset_name}-{content_type}.jsonl'
    with open(dset_path, 'w') as fh:
        for review_data in qa_dataset:
            json.dump(review_data, fh)
            fh.write('\n')
    print("dataset made and written to", dset_path)

def make_manual_dataset_v2(dataset_name, use_fulltext=True):
    with open(f"{ROOT}/{dataset_name}-setup.json", 'r') as fh:
        setup_data = json.load(fh)
    rct_text_data = dpd.get_rct_pmid_data()
    review_pmid_data = dpd.get_review_pmid_data()
    metadata = load_metadata()
    fulltext_used = 0
    content_type = 'fulltext' if use_fulltext else 'abstract'
    if use_fulltext:
        rct_fulltext_data = dpd.get_rct_fulltext()
    review_to_rct_pmids = dpd.get_review_to_references_map(id_type="pmid")
    qa_dataset = []
    for review_pmid, review_qs in tqdm(setup_data.items()):
        if len(review_qs) < 1:
            continue
        all_review_rcts = review_to_rct_pmids[review_pmid]
        review_rct_data = {}
        for rct_pmid in all_review_rcts:
            rct_data = {'article_id': rct_pmid, **rct_text_data[rct_pmid]}
            if use_fulltext and (rct_pmid in rct_fulltext_data):
                rct_data['content'] = rct_fulltext_data[rct_pmid]
                fulltext_used += 1
            review_rct_data[rct_pmid] = rct_data
        for q in review_qs:
            _date = review_pmid_data[review_pmid].get('date')
            year = int(_date[:4]) if isinstance(_date, str) else None
            qid = q['question_id']
            q['original_review'] = review_pmid
            q['review_year'] = year
            q['sources'] = {_pmid: review_rct_data[_pmid] for _pmid in q['relevant_sources']}
            q.update(metadata[qid])
            qa_dataset.append(q)
    if use_fulltext:
        print("fulltexts used:", fulltext_used)
    dset_path = f'run-med-evidence/datasets/{dataset_name}-{content_type}.jsonl'
    with open(dset_path, 'w') as fh:
        for review_data in qa_dataset:
            json.dump(review_data, fh)
            fh.write('\n')
    print("dataset made and written to", dset_path)
    csv_path = f'run-med-evidence/datasets/{dataset_name}-{content_type}.csv'
    csv_data = pd.DataFrame(qa_dataset)
    csv_data.to_csv(csv_path, index=False)
    print("csv dataset made and written to", csv_path)

def convert_manual_spreadsheet(dataset_name):
    csv_path = f'{ROOT}/{dataset_name}-data_spreadsheet.csv'
    df = pd.read_csv(csv_path).fillna("")
    df['relevant_sources'] = [
        [elem.strip() for elem in text_list.split(',')]
        for text_list in df["Relevant Cited PubMed IDs"].values
    ]
    print(df.head())
    rows = df.drop(columns=['Relevant Cited PubMed IDs', "Comment"]).rename(columns={
        "Cochrane PubMed ID": "original_review",
        "Question": "question",
        "Answer": "answer",
        "Evidence Quality": "evidence_certainty",
        "Open-Access Full-text Needed": "fulltext_required",
    }).reset_index(names="question_id").to_dict(orient='records')
    proto_dataset = {}
    for row in rows:
        review_pmid = row.pop('original_review')
        proto_dataset.setdefault(review_pmid, []).append(row)
    out_path = f"{ROOT}/{dataset_name}-setup.json"
    with open(out_path, 'w') as fh:
        json.dump(proto_dataset, fh, indent=2)
    print('dataset setup written to', out_path)

if __name__ == '__main__':
    name = 'med-evidence'
    # convert_manual_spreadsheet(name)
    # make_manual_dataset(name, use_fulltext=False)
    make_manual_dataset_v2(name, use_fulltext=True)