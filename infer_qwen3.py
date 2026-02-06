import os
import aigco
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download  # <--- 新增导入
from dotenv import load_dotenv

load_dotenv()

# 模型在 HF 上的 ID
REPO_ID = "Qwen/Qwen3-0.6B"

logger = aigco.logger(name="qwen3_inference")


def main():
    # 自动获取缓存中的真实绝对路径
    try:
        # local_files_only=True 确保它只从本地找，不会去联网下载
        model_path = snapshot_download(repo_id=REPO_ID, local_files_only=True)
        print(f"📍 找到模型路径: {model_path}")
    except Exception as e:
        print(f"❌ 无法在缓存中找到模型 {REPO_ID}，请确认是否已下载。")
        return

    # 使用自动获取的路径
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    llm = aigco.inference.LLM(model_path, enforce_eager=True, tensor_parallel_size=1)

    sampling_params = aigco.inference.SamplingParams(temperature=0.6, max_tokens=256)
    prompts = ["introduce yourself", "list all prime numbers within 100"]

    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]

    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print(f"\nPrompt: {prompt!r}\nCompletion: {output['text']!r}")


if __name__ == "__main__":
    main()
