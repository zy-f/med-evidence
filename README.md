![Core skills evaluated by MedEvidence, including: medical domain expertise across 10 different specialties, synthesizing conflicting evidence, and applying scientific skepticism when studies exhibit a high risk of bias (e.g. due to small sample sizes or insufficient supporting evidence)](assets/medEv-pullFigure.svg)
<h1 align="center">MedEvidence</h1>
<p align="center">
  <strong><a href="https://arxiv.org/pdf/2505.22787" target="_blank">Arxiv</a></strong>
  &nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;
  <strong><a href="https://zy-f.github.io/website-med-evidence/" target="_blank">Website</a></strong>
  &nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;
  <strong><a href="https://huggingface.co/datasets/clcp/med-evidence" target="_blank">HuggingFace Dataset</a></strong>
</p>

This repository contains a streamlined and simple version of the minimal code needed to replicate the results of the MedEvidence paper.

# Setup
Create a virtual environment and install necessary requirements using `pip install -r requirements.txt` using the requirements file in the project root directory.

# Using the dataset
If you are just interested in using the dataset for evaluation of your own LLMs, it is available in an easier-to-use jsonl format at `run-med-evidence/datasets/med-evidence-fulltext.jsonl`.

If you want to make use of our infrastructure to test your own models, we currently support API endpoints hosted via OpenAI, TogetherAI, and most flexibly via vLLM (i.e. using `vllm serve`).
To evaluate a custom model, create a custom json config file like:
```json
{
    "dataset_name": "med-evidence-fulltext",
    "model_type": [
        "vllm OR together OR openai",
        "HOSTED/PATH (see run-med-evidence/src/utils/CONSTS.py for examples)"
    ],
    "prompt_set": "basic",
    "evidence": {
        "summarize": false,
        "llm_filter": false
    },
    "rag_setup": {
        "prompt_buffer_size": 1000
        "max_input_token": [YOUR MODEL'S CONTEXT LIMIT HERE (int)]
    }
}
```
Then, switch to the `run-med-evidence` directory and run `python run.py path/to/your/config.json`. Your outputs will appear in `run-med-evidence/___res/` as `[config_name].log` (for printed logged output) and `[config_name].jsonl` (for full outputs that can be evaluated and analyzed).

Note that if you are using vLLM, you will need to specify `--server [vllm_host_address:port]` when executing `run.py`.

As an example, a simple evaluation of accuracy can be accomplished as follows:
```python
with open("run-med-evidence/___res/[config_name].jsonl", 'r') as fh:
    outputs = [json.loads(line) for line in fh]
scores = []
for o in outputs:
    model_ans = out.get('parsed_answer')
    ans = model_ans.get('answer', "").lower().strip() if isinstance(model_ans, dict) else ""
    scores.append(out['gt_answer'].lower() == ans)
accuracy = np.array(scores).mean()
print("Accuracy:", accuracy)
```
For more advanced analysis, please see `inspect_outputs.ipynb > final figure plots > define funcs`

# Replicating our results
## Replicating LLM evaluation
LLM evaluations can be replicated using the `run-med-evidence` subdirectory. The fulltext version of the dataset has been pre-provided in the expected jsonl format in `run-med-evidence/datasets`.

To run experiments, specify configs in the `run-med-evidence/configs` directory (an example is provided, and large batches of configs can be made programmatically via `run-med-evidence/make_configs.py`), switch to the `run-med-evidence` directory, then execute using the run file, as follows: 
```
python run.py <config folder and/or file paths>
```

Multithreading is implemented via the `--parallel-size [num threads]` option to allow multiple questions to be evaluated at once. For more details on using the run file, run `python run-med-evidence/run.py --help`.

Please note that you will have to provide your own API keys for TogetherAI (`TOGETHER_API_KEY`) and OpenAI (`OPENAI_API_KEY`), and that locally-run models (such as the medically-finetuned models) will need to be separately hosted as API endpoints using `vllm serve`.

## Replicating analysis
The generation of source-level concordance was performed via `run-med-evidence/souce_level_agreement.py`. Otherwise, analysis and generation of all figures can be inspected (and if all experiments are re-run, replicated) in the `inspect_outputs.ipynb` file.

## Replicating dataset creation
To replicate the extraction of Cochrane reviews and relevant studies, please run `download_pubmed_data.py`. Note that you will need to provide your own email address to use PubMed's Entrez tools to extract the data.

Once that data is available and written to the `cochrane_data` directory, it is then possible to run `dataset_generation.py` to convert the raw, human-generated spreadsheet in `dset_gen/med-evidence-data_spreadsheet.csv` into the final jsonl data format expected by the run file.
