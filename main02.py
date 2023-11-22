"""
Code from about 9:40 through 11:30 of video.
"""

import copy, json, logging
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
stream = llm( 
    'Question: Who is Ada Lovelace? Answer:',
    max_tokens=100,
    temperature=0.8,
    stop=['\n', 'Question:', 'Q:'],
    stream=True,
)

## show output ------------------------------------------------------
for output in stream:
    completion_fragment = copy.deepcopy( output )
    txt = completion_fragment['choices'][0]['text']
    log.debug( f'text, ``{txt}``' )
