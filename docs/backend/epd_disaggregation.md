# EPD Disaggregation

## Why and What is EPD Disaggregation?

Multimodal Large Language Model (MLLM) inference comprises two distinct phases: **Prefill** and **Decode**. The Prefill phase is computation-intensive, processing the entire input sequence, while the Decode phase is memory-intensive, managing the Key-Value (KV) cache for token generation. Traditionally, these phases are handled within a unified engine, where combined scheduling of prefill and decode batches introduces inefficiencies. To address these challenges, we introduce **Prefill and Decoding (PD) Disaggregation** in SGLang.

### Issues with Unified Scheduling

The conventional unified engine, which processes prefill and decode batches together, results in two significant problems:

1. **Prefill Interruption**: Incoming prefill batches frequently interrupt ongoing decode batches, causing substantial delays in token generation.
2. **DP Attention Imbalance**: In data-parallel (DP) attention, one DP worker may process a prefill batch while another handles a decode batch simultaneously, leading to increased decode latency.

PD Disaggregation resolves these by separating the two stages, enabling tailored optimizations for each.

For the design details, please refer to [link](https://docs.google.com/document/d/1rQXJwKd5b9b1aOzLh98mnyMhBMhlxXA5ATZTHoQrwvc/edit?tab=t.0).

Currently, we support Mooncake and NIXL as the transfer engine.

### Usage

### QwenVL, PD not disaggregated

```bash
# start an encode server
$ python -m sglang.launch_server --model-path Qwen/Qwen2-VL-7B-Instruct --host 0.0.0.0  --disaggregation-mode encode --port 60001
# start a text server
$ python -m sglang.launch_server --model-path Qwen/Qwen2-VL-7B-Instruct --host 0.0.0.0  --disaggregation-mode text --port 60002
# mini_lb
$ python -m sglang.srt.disaggregation.mini_lb --encode http://127.0.0.1:60001 --text http://127.0.0.1:60002 --host 0.0.0.0 --port 9080
```


### QwenVL, EPD all disaggregated

```bash
# start an encode server
$ python -m sglang.launch_server --model-path Qwen/Qwen2-VL-7B-Instruct --host 0.0.0.0  --disaggregation-mode encode --port 60001
# start a prefill server, make sure to set --encoder-disaggregated
$ python -m sglang.launch_server --model-path Qwen/Qwen2-VL-7B-Instruct --host 0.0.0.0  --disaggregation-mode prefill --port 60002 --encoder-disaggregated
# start a decode server
$ python -m sglang.launch_server --model-path Qwen/Qwen2-VL-7B-Instruct --host 0.0.0.0  --disaggregation-mode decode --port 60003
# mini_lb
$ python -m sglang.srt.disaggregation.mini_lb --encode http://127.0.0.1:60001 --prefill http://127.0.0.1:60002 --decode http://127.0.0.1:60003 --host 0.0.0.0 --port 9080
```
