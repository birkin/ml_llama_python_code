import logging
from llama_cpp import Llama

logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s] %(levelname)s [%(module)s-%(funcName)s()::%(lineno)d] %(message)s',
    datefmt='%d/%b/%Y %H:%M:%S' )
log = logging.getLogger( '__name__' )

# load the model
log.debug( 'loading model' )
llm = Llama( model_path='../models/ggml-vicuna-13b-4bit-rev1.bin' )
log.debug( 'model loaded' )
