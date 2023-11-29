"""
Code through about 9:40 of the video.
"""

import json, logging
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
output = llm( 
    'Question: Who is Ada Lovelace? Answer:',
    max_tokens=100,
    temperature=0.8,
    stop=['\n', 'Question:', 'Q:'],
    echo=True,
)

## show output ------------------------------------------------------
jsn = json.dumps( output, indent=2)
log.debug( f'output: {jsn}' )
