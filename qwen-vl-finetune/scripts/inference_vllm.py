# -*- coding: utf-8 -*-
import json
from tqdm import tqdm
import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

import os
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

def prepare_inputs_for_vllm(messages, processor):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # qwen_vl_utils 0.0.14+ reqired
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True,
        return_video_metadata=True
    )
    print(f"video_kwargs: {video_kwargs}")

    mm_data = {}
    if image_inputs is not None:
        mm_data['image'] = image_inputs
    if video_inputs is not None:
        mm_data['video'] = video_inputs

    return {
        'prompt': text,
        'multi_modal_data': mm_data,
        'mm_processor_kwargs': video_kwargs
    }


if __name__ == '__main__':
    test_path = "/input/generate_image/test/labels.jsonl"
    with open(test_path) as f:
        lines = f.readlines()
    test_data = [json.loads(l) for l in lines]



    # TODO: change to your own checkpoint path
    checkpoint_path = "/workspace/qwen-vl-finetune/scripts/output/exp001/merged_model"
    processor = AutoProcessor.from_pretrained(checkpoint_path)

    # Ensure sensible parallel settings for this environment.
    tp_size = max(1, torch.cuda.device_count())
    # Some model builds do not include experts; disable expert parallelism by default to avoid
    # vllm configuration validation errors. If you have a model with experts, set this to True
    # and provide the appropriate configuration.
    try:
        # Reduce max_model_len to fit available GPU KV cache memory.
        # Adjust gpu_memory_utilization (0..1) as needed for your environment.
        llm = LLM(
            model=checkpoint_path,
            mm_encoder_tp_mode="data",
            enable_expert_parallel=False,
            tensor_parallel_size=tp_size,
            max_model_len=4096,
            gpu_memory_utilization=0.05,
            kv_cache_memory_bytes=1024**3,
            seed=0,
            max_num_batched_tokens=16,
            max_num_seqs=4,
            trust_remote_code=True
        )
    except Exception as e:
        print(f"Failed to initialize LLM: {e}")
        raise

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=1024,
        top_k=-1,
        stop_token_ids=[],
    )

    batch_size = 16
    inputs = []

    for data in tqdm(test_data[:80]):


        message = [
            {"role": "system", "content": "You are a helpful assistant that helps people find information and answer questions. You are good at reading text in images and videos."},
            {
                "role": "user",
                "content": [
                {
                    "type": "image",
                    "image": data["image"].replace("/kaggle", ""),
                },
                {"type": "text", "text": data["conversations"][0]["value"]}
                ],
            }
        ]

        inputs.append(prepare_inputs_for_vllm(message, processor))

        if (len(inputs) % batch_size) == 0:
            # prepare a batch (vLLM expects an iterable of input dicts)
            batch_inputs = inputs if isinstance(inputs, list) else [inputs]
            # for i, input_ in enumerate(batch_inputs):
            #     print()
            #     print('=' * 40)
            #     print(f"Inputs[{i}]: prompt={input_.get('prompt')!r}")
            # print('\n' + '>' * 40)

            outputs = llm.generate(batch_inputs, sampling_params=sampling_params, use_tqdm=False)
            for i, output in enumerate(outputs):
                generated_text = output.outputs[0].text
                print()
                print('=' * 40)
                print(f"Generated text: {generated_text!r}")

            inputs = []