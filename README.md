## Quick Start

### Installation

> aigco requires **Python 3.12** or higher

#### From PyPI

```bash
pip install aigco[flash_attn]
```

or

with uv:

```bash
uv pip install aigco[flash_attn]

or add to deependencies
"flash-attn @ https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.8.3%2Bcu130torch2.10-cp312-cp312-linux_x86_64.whl"
```

#### From Source

```bash
# pull the source code from Github
git clone --depth 1 https://github.com/TorrentBrave/aigco.git

# Install the package in editable mode
cd aigco

pip install -e .
# or with uv
# uv pip install -e .
```

#### As a submodule in Src

```bash
mkdir src && git -C src clone https://github.com/TorrentBrave/aigco.git

git submodule add --force https://github.com/TorrentBrave/aigco.git src/aigco

uv add --editable ./src/aigco/ <!-- will update aigco.egg.info -->

cd src/aigco

uv lock

uv sync
```

## Example

### Inference Qwen3-0.6B like vllm

```python
import aigco
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download
from dotenv import load_dotenv

load_dotenv()
REPO_ID = "Qwen/Qwen3-0.6B"

logger = aigco.logger(name="qwen3_inference")

def main():
    try:
        model_path = snapshot_download(repo_id=REPO_ID, local_files_only=True)
        logger.info(f"📍 Find model path: {model_path}")
    except Exception as e:
        logger.error(f"❌ Can't find model in cache {REPO_ID}, Reason: {e}")
        return

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    logger.info("Starting init LLM Engine...")
    llm = aigco.inference.LLM(model_path, enforce_eager=True, tensor_parallel_size=1)

    sampling_params = aigco.inference.SamplingParams(temperature=0.6, max_tokens=256)
    prompts_text = ["introduce yourself", "list all prime numbers within 100"]

    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts_text
    ]

    logger.info(f"Generating reponse, The number of samples: {len(prompts)}...")
    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        log_message = f"\nPrompt: {prompt!r}\nCompletion: {output['text']!r}"
        logger.info(log_message)

    logger.info("Finished Inference Task")

if __name__ == "__main__":
    main()
```

