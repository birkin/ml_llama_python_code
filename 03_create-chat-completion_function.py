"""
Explores [create_chat_completion](https://abetlen.github.io/llama-cpp-python/#llama_cpp.llama.Llama.create_chat_completion) function
"""

import copy, json, logging, pprint
from llama_cpp import Llama

logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s] %(levelname)s [%(module)s-%(funcName)s()::%(lineno)d] %(message)s',
    datefmt='%d/%b/%Y %H:%M:%S' )
log = logging.getLogger( '__name__' )

## load model -------------------------------------------------------
log.debug( 'loading model' )
llm = Llama( model_path='../models/ggml-vicuna-13b-4bit-rev1.bin' )
log.debug( 'model loaded' )


## run model --------------------------------------------------------
log.debug( 'running model' )
messages = [ {'role': 'user', 'content': 'Question: Who is Ada Lovelace? Answer:'} ]
output = llm.create_chat_completion(
    messages, 
    temperature=0.2, 
    top_p=0.95, 
    top_k=40, 
    stream=False, 
    stop=[], 
    max_tokens=100,
    repeat_penalty=1.1
    )


## show output ------------------------------------------------------
log.debug( f'output, ``{pprint.pformat(output)}``' )