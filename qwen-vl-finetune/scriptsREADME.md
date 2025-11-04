# vLLMをVision Language Modelで使ってみよう

# vLLMとは

PythonベースのLLM推論エンジンで、LLMの推論を高速かつ簡単に提供することを目的としています。

vLLMの特長は以下の通りです。

- **高速なサービングスループット**
    - PagedAttentionによるAttentionのKVキャッシュ管理、継続的なバッチ処理、CUDA/HIPグラフを用いた高速なモデル実行などにより、高いスループットを実現
- **豊富な最適化機能**
    - GPTQ・AWQなどの量子化、INT4/INT8/FP8 などの低精度演算、FlashAttention/FlashInferと統合された最適化カーネル等の機能が備わっている
- **柔軟性と使いやすさ**
    - Hugging Faceのモデルとのシームレスな連携、様々なデコードアルゴリズム(ビームサーチや並列サンプリングなど)への対応が可能
- **多様なハードウェア対応**
    - NVIDIA/AMD/IntelのGPUやCPU、PowerPC、TPUに加え、Intel Gaudi・IBM Spyre・Huawei Ascendなどの専用ハードウェアプラグインにも対応
- **OpenAI互換APIやストリーミング出力**
    - OpenAI互換のREST APIサーバやツール呼び出し付きチャット補完など、アプリケーション開発に便利な機能が標準で用意されている

# Quick Start(Qwen3-VL)

## Qwen3-VLとは

Alibaba Cloudが開発する最新のVision Language Model(VLM)です。純粋なテキスト生成能力と、視覚情報の理解に長けており、オープンソースなVLMの中ではかなり高精度のモデルです。

インプット：テキスト、画像、動画

アウトプット：テキスト

Qwen3-VLをvLLMで動かすサンプルコードはQwenのGithubにあるので、こちらを利用します。

https://github.com/QwenLM/Qwen3-VL

## インストール

vLLMでQwen3-VLを動作させるには0.11.0以降のバージョンをインストールする必要があります。

また、オフライン推論を行う場合はqwen-vl-utilsという画像前処理用のライブラリもインストールします。

```jsx
pip install accelerate
pip install vllm==0.11.0
pip install qwen-vl-utils==0.0.14
```

## サンプルコード

```jsx
import torch
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor

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
    
llm = LLM(
    model="Qwen/Qwen3-VL-8B-Instruct",
    tensor_parallel_size=torch.cuda.device_count(),
    max_model_len=4096,
    gpu_memory_utilization=0.5,
    seed=0,
    max_num_batched_tokens=4*4096,
    max_num_seqs=4,
    trust_remote_code=True
)

processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")

message = [
    {"role": "system", "content": システムプロンプト},
    {
        "role": "user",
        "content": [
        {
            "type": "image",
            "image": 画像パス,
        },
        {"type": "text", "text": ユーザプロンプト}
        ],
    }
]

inputs = [prepare_inputs_for_vllm(message, processor)]

outputs = llm.generate(batch_inputs, sampling_params=sampling_params, use_tqdm=False)
for i, output in enumerate(outputs):
		# generated_text: Qwen3-VLの推論結果
    generated_text = output.outputs[0].text
    print()
    print('=' * 40)
    print(f"Generated text: {generated_text!r}")
```

LLM関数の各引数の説明は以下です

| model | モデルパス、チェックポイントパス | チェックポイントパスを指定する場合、対象フォルダにprocessorの配置する必要がある(余談に記載) |
| --- | --- | --- |
| tensor_parallel_size | 使用するGPUの枚数 | 複数指定するとテンソル並列化が可能 |
| max_model_len | 最大コンテキスト長 | コンテキストが長い場合や出力トークンが多い場合はこちらを増やしていく |
| gpu_memory_utilization | GPUメモリ使用量の制限 | 0~1を指定。値が少ないとメモリが足りないとエラーが出る。デフォルト0.9(90%使用する) |
| kv_cache_memory_bytes | KVキャッシュで確保するメモリ量 | バイト単位で指定。今回は1024**3=1GiB。この値はgpu_memory_utilizationの設定より優先される |
| seed | ランダムシード固定 |  |
| max_num_batched_tokens | 一度にバッチ処理するトークン数の上限 | バッチ推論する際には気を付けたい引数。大きくするとメモリ使用量は増える |
| max_num_seqs | 一度に処理する最大バッチサイズ | max_num_batched_tokensと一緒に調整したい引数。これも大きくするとメモリ使用量は増える |

# メモリ節約術

## kv_cache_memory_bytes

| kv_cache_memory_bytes | KVキャッシュで確保するメモリ量 | バイト単位で指定。この値はgpu_memory_utilizationの設定より優先される |
| --- | --- | --- |

LLM関数にこの引数を指定すると、gpu_memory_utilizationよりも精緻にKVキャッシュの確保量を指定できます。

```jsx
llm = LLM(
    model="Qwen/Qwen3-VL-8B-Instruct",
    tensor_parallel_size=torch.cuda.device_count(),
    max_model_len=4096,
    # gpu_memory_utilization=0.5,
    kv_cache_memory_bytes=1024**3, # 1024**3 = 1GiB確保
    seed=0,
    max_num_batched_tokens=4*4096,
    max_num_seqs=4,
    trust_remote_code=True
)
```

## limit_mm_per_prompt

| limit_mm_per_prompt | 推論時にマルチモーダルの数を制限 | 画像や動画の数を予め制限することでメモリ使用量を少なくできる |
| --- | --- | --- |

```jsx
llm = LLM(
    model="Qwen/Qwen3-VL-8B-Instruct",
    tensor_parallel_size=torch.cuda.device_count(),
    max_model_len=4096,
    # gpu_memory_utilization=0.5,
    kv_cache_memory_bytes=1024**3, # 1024**3 = 1GiB確保
    seed=0,
    max_num_batched_tokens=4*4096,
    max_num_seqs=4,
    trust_remote_code=True,
    limit_mm_per_prompt={"image": 1, "video": 0} # 画像は1枚だけ、動画は使わない
)
```

# バッチ推論

vLLMの真髄です。

1件ずつ推論より複数件同時に推論した時の効率がとても良いです。

```jsx
batch_size = 8

llm = LLM(
    model="Qwen/Qwen3-VL-8B-Instruct",
    tensor_parallel_size=torch.cuda.device_count(),
    max_model_len=4096,
    kv_cache_memory_bytes=1024**3, # 1024**3 = 1GiB確保
    seed=0,
    max_num_batched_tokens=batch_size*4096,
    max_num_seqs=batch_size,
    trust_remote_code=True,
    limit_mm_per_prompt={"image": 1, "video": 0} # 画像は1枚だけ、動画は使わない
)

message = [
    {"role": "system", "content": システムプロンプト},
    {
        "role": "user",
        "content": [
        {
            "type": "image",
            "image": 画像パス,
        },
        {"type": "text", "text": ユーザプロンプト}
        ],
    }
]

# バッチ推論
inputs = [prepare_inputs_for_vllm(message, processor) for i in range(batch_size)]

outputs = llm.generate(batch_inputs, sampling_params=sampling_params, use_tqdm=False)
for i, output in enumerate(outputs):
		# generated_text: Qwen3-VLの推論結果
    generated_text = output.outputs[0].text
    print()
    print('=' * 40)
    print(f"Generated text: {generated_text!r}")
```

### 速度比較

|  | 1件当たりの推論速度 |
| --- | --- |
| vLLM, バッチサイズ1 | 約2.8秒 |
| vLLM, バッチサイズ8 | 約0.9秒 |
| transformers, バッチサイズ1 | 約10秒 |

# おわりに

vLLMでVision Language Model(Qwen3-VL)の使い方を紹介しました。特に行数も少なく高速に推論できてとても便利です。

vLLMを使う際は用途に応じてメモリ節約術やバッチ推論を検討していただけたらと思います。
