(- at 4:11)

# Goal

To go through Chris Hayes' "[Python with Stanford Alpaca and Vicuna 13B AI models - A llama-cpp-python Tutorial!](https://www.youtube.com/watch?v=-BidzsQYZM4)"

---


# Install notes

- `makedir ./ml_llama_python_stuff`

- `cd ./ml_llama_python_stuff`

- `python3.8 -m venv ./venv_ml_llama`

    - Note that I'm using an older version of python, 3.8x, because that's the version of python for a server I regularly deploy to.

- `ln -s ./venv_ml_llama ./env`

- `mkdir ./ml_llama_python_code`

- `cd ./ml_llama_python_code`

- `source ../env/bin/activate`

- Re `requirements.in` -- For reproducibility, I put the main requirements into `requirements.in`, then `pip install pip-tools` into the venv, then `pip-compile ./requirements.in` (pip-compile is part of pip-tools) to produce the `requirements.txt`, then `pip-sync ./requirements.txt` (pip-sync is part of pip-tools).

- The url where Chris got his model says "obsolete", so I searched HuggingFace and found (as of 2023-Nov-21) [a different file named `ggml-vicuna-13b-4bit-rev1.bin`](https://huggingface.co/Bleak/ggml-vicuna-13b-4bit-rev1/tree/main)

- Getting an error just trying to load the model -- see video at 5:40:

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

    Ok, that worked! Thanks `spheres5531`!

- TODO: the `llama.cpp/issues/3708` thread and other stuff indicates that modern versions of `llama-ccp-python`... maybe don't work on `.bin` files? But instead expect `.gguf` files? (New to all this.) So after I get this working, perhaps explore newer versions of `llama-ccp-python` (this is the nice benefit of being able to have multiple venvs and just point the `env` simlink to the active one) with newer models. For reference, [a gpt4 overview of the gguf format](https://chat.openai.com/share/6826ff67-432e-4e04-99b4-6e9be08242bd).

---


# Usage

## main07.py

### usage
`time python3 ./main07.py --url "https://url/to/storage/test:2etz4x4h/EXTRACTED_TEXT/"`

### output

[26/Nov/2023 18:10:56] DEBUG [main07-summarize_text()::65] cleaned_summary, ``The text describes an event where a group of lecturers, including the speaker, were briefed by government spokesmen about the world view of the United States government. One of the speakers was the Under Secretary of State, Dean Gooderham Acheson, who described the Soviet leaders as "like little boys who enjoy throwing brickbats at other people's greenhouses." The speaker, a naturalized citizen, disagreed with this view and argued that the difference...``

---

## main04.py output

Goal: experiment with summarization.

```
[22/Nov/2023 15:11:06] DEBUG [main04-<module>()::56] output, ``{'choices': [{'finish_reason': 'length',
              'index': 0,
              'message': {'content': ' The text is a speech given by President '
                                     'Barack Obama after his re-election in '
                                     '2012. In the speech, Obama thanks his '
                                     'supporters and expresses his desire to '
                                     'work with Mitt Romney and their teams to '
                                     'move forward as one nation. He '
                                     'acknowledges that while they may have '
                                     'had differences, they both share a '
                                     'commitment to certain hopes for '
                                     "America's future, such as education, "
                                     'innovation, and a strong military. Obama '
                                     'believes that',
                          'role': 'assistant'}}],
 'created': 1700683764,
 'id': 'chatcmpl-898cf652-ce58-4f9f-aaa7-7cd1445b787b',
 'model': '../models/ggml-vicuna-13b-4bit-rev1.bin',
 'object': 'chat.completion',
 'usage': {'completion_tokens': 100,
           'prompt_tokens': 1951,
           'total_tokens': 2051}}``
```

## main03.py output

Goal: experiment with regular output text.

```
[22/Nov/2023 14:14:50] DEBUG [main03-<module>()::46] output, ``{'choices': [{'finish_reason': 'length',
              'index': 0,
              'message': {'content': ' Ada Lovelace was a British '
                                     'mathematician and writer, born in 1815. '
                                     'She is best known for her work on '
                                     "Charles Babbage's early mechanical "
                                     'general-purpose computer, the Analytical '
                                     'Engine. Lovelace is credited with '
                                     'writing the first computer program, '
                                     'which was designed to calculate the '
                                     'Bernoulli numbers. She was also a '
                                     'prolific writer and published several '
                                     'books, including a translation of the '
                                     'Italian novel "The Betrothed"',
                          'role': 'assistant'}}],
 'created': 1700680479,
 'id': 'chatcmpl-389e00aa-52c0-4de0-8522-460f076dd81a',
 'model': '../models/ggml-vicuna-13b-4bit-rev1.bin',
 'object': 'chat.completion',
 'usage': {'completion_tokens': 100, 'prompt_tokens': 20, 'total_tokens': 120}}``
```


## main02.py output

Goal: just following video tutorial.

```
[22/Nov/2023 13:45:48] DEBUG [main02-<module>()::34] text, `` Ada``
[22/Nov/2023 13:45:48] DEBUG [main02-<module>()::34] text, `` Lov``
[22/Nov/2023 13:45:49] DEBUG [main02-<module>()::34] text, ``el``
[22/Nov/2023 13:45:49] DEBUG [main02-<module>()::34] text, ``ace``
[22/Nov/2023 13:45:49] DEBUG [main02-<module>()::34] text, `` was``
[22/Nov/2023 13:45:49] DEBUG [main02-<module>()::34] text, `` an``
[22/Nov/2023 13:45:49] DEBUG [main02-<module>()::34] text, `` English``
[22/Nov/2023 13:45:49] DEBUG [main02-<module>()::34] text, `` math``
[22/Nov/2023 13:45:49] DEBUG [main02-<module>()::34] text, ``ematic``
[22/Nov/2023 13:45:49] DEBUG [main02-<module>()::34] text, ``ian``
[22/Nov/2023 13:45:49] DEBUG [main02-<module>()::34] text, `` and``
[22/Nov/2023 13:45:49] DEBUG [main02-<module>()::34] text, `` writer``
...etc...
```


## main01.py output

Goal: just followign video tutorial.

```
[22/Nov/2023 12:55:35] DEBUG [main-<module>()::28] output: {
  "id": "cmpl-03b92ca8-2d67-4acc-b6f4-2964398e2bd2",
  "object": "text_completion",
  "created": 1700675728,
  "model": "../models/ggml-vicuna-13b-4bit-rev1.bin",
  "choices": [
    {
      "text": "Question: Who is Ada Lovelace? Answer: Ada Lovelace was an English mathematician and writer, born in 1815. She is known for her work on Charles Babbage's early mechanical general-purpose computer, the Analytical Engine. She is considered to be the world's first computer programmer.",
      "index": 0,
      "logprobs": null,
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 12,
    "completion_tokens": 62,
    "total_tokens": 74
  }
}
python ./main.py  33.89s user 0.72s system 486% cpu 7.117 total
```