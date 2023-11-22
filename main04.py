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
llm = Llama( 
    model_path='../models/ggml-vicuna-13b-4bit-rev1.bin',
    n_ctx=4000 )
log.debug( 'model loaded' )


## run model --------------------------------------------------------
log.debug( 'running model' )
obama_speech = '''
Thank you so much. Tonight, more than 200 years after a former colony won the right to determine its own destiny, the task of perfecting our union moves forward. It moves forward because of you. It moves forward because you reaffirmed the spirit that has triumphed over war and depression, the spirit that has lifted this country from the depths of despair to the great heights of hope, the belief that while each of us will pursue our own individual dreams, we are an American family and we rise or fall together as one nation and as one people. Tonight, in this election, you, the American people, reminded us that while our road has been hard, while our journey has been long, we have picked ourselves up, we have fought our way back, and we know in our hearts that for the United States of America the best is yet to come. I want to thank every American who participated in this election, whether you voted for the very first time or waited in line for a very long time. By the way, we have to fix that.
'''
# obama_speech = '''
# Thank you so much. Tonight, more than 200 years after a former colony won the right to determine its own destiny, the task of perfecting our union moves forward. It moves forward because of you. It moves forward because you reaffirmed the spirit that has triumphed over war and depression, the spirit that has lifted this country from the depths of despair to the great heights of hope, the belief that while each of us will pursue our own individual dreams, we are an American family and we rise or fall together as one nation and as one people. Tonight, in this election, you, the American people, reminded us that while our road has been hard, while our journey has been long, we have picked ourselves up, we have fought our way back, and we know in our hearts that for the United States of America the best is yet to come. I want to thank every American who participated in this election, whether you voted for the very first time or waited in line for a very long time. By the way, we have to fix that.
# '''
message = f'Summarize the following text using a neutral tone, describing main themes and topics: {obama_speech}'
log.debug( f'message, ``{message}``' )
messages = [ {'role': 'user', 'content': message} ]
output = llm.create_chat_completion(
    messages, 
    temperature=0.2, 
    top_p=0.95, 
    top_k=40, 
    stream=False, 
    stop=[], 
    max_tokens=75,
    repeat_penalty=1.1
    )


## show output ------------------------------------------------------
log.debug( f'output, ``{pprint.pformat(output)}``' )