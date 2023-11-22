Going through Chris Hayes' "[Python with Stanford Alpaca and Vicuna 13B AI models - A llama-cpp-python Tutorial!](https://www.youtube.com/watch?v=-BidzsQYZM4)"

- Note: the url where Chris got his model says "obsolete", so I searched HuggingFace and found (as of 2023-Nov-21) [a different file named `ggml-vicuna-13b-4bit-rev1.bin`](https://huggingface.co/Bleak/ggml-vicuna-13b-4bit-rev1/tree/main)

- at 4:11

- Getting an error just trying to load the model:

```
(venv_ml_llama) birkin@birkinbox-2021 ml_llama_python_code % 
(venv_ml_llama) birkin@birkinbox-2021 ml_llama_python_code % python ./main.py
[22/Nov/2023 12:10:14] DEBUG [main-<module>()::11] loading model
gguf_init_from_file: invalid magic characters tjgg??k.
error loading model: llama_model_loader: failed to load model from ../models/ggml-vicuna-13b-4bit-rev1.bin

llama_load_model_from_file: failed to load model
AVX = 0 | AVX2 = 0 | AVX512 = 0 | AVX512_VBMI = 0 | AVX512_VNNI = 0 | FMA = 0 | NEON = 1 | ARM_FMA = 1 | F16C = 0 | FP16_VA = 1 | WASM_SIMD = 0 | BLAS = 1 | SSE3 = 0 | SSSE3 = 0 | VSX = 0 | 
Traceback (most recent call last):
  File "./main.py", line 12, in <module>
    llm = Llama( model_path='../models/ggml-vicuna-13b-4bit-rev1.bin' )
...etc...
```

Some info on that `invalid magic characters tjgg`...
<https://github.com/ggerganov/llama.cpp/issues/3708>

Trying a suggestion from the youtube video...
```
spheres5531
5 months ago
hey i found this and it worked for me: pip freeze | grep llama
pip cache purge
pip install llama-cpp-python==0.1.39
```

---
