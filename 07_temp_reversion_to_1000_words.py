# -*- coding: utf-8 -*-

"""
Attempts to summarize url to text.

Only uses the first 1000 words of the text, for now.

TODO: apply the logic in '06_...' to compare summary-of-summaries method. Or just use a newer model that can handle more tokens.
"""

import argparse, copy, json, logging, pprint
from urllib.parse import urlparse

## imported packages ------------------------------------------------
# import nltk
import requests
# nltk.download( 'punkt' )
# from nltk.tokenize import sent_tokenize
from llama_cpp import Llama


logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s] %(levelname)s [%(module)s-%(funcName)s()::%(lineno)d] %(message)s',
    datefmt='%d/%b/%Y %H:%M:%S' )
log = logging.getLogger( '__name__' )


## manager function -------------------------------------------------
def summarize_text( url: str ) -> str:
    """ Manages summarization.
        Called by dundermain. """
    ## validate input -----------------------------------------------
    text_to_summarize: str = validate_input( url )
    ## load model ---------------------------------------------------
    LLM = load_model()
    ## get the first 1000 words -------------------------------------
    word_count = len( text_to_summarize.split() )
    log.debug( f'word_count, ``{word_count}``' )
    text_to_summarize = ' '.join( text_to_summarize.split()[0:1000] )  # I want to keep the max-tokens under 2,000
    ## summarize ----------------------------------------------------
    max_tokens_for_summarization = 100
    if word_count < 500:
        max_tokens_for_summarization = 75
    summarization_dict = summarize( text_to_summarize, LLM, max_tokens_for_summarization )
    ## extract summary-text -----------------------------------------
    summary: str = summarization_dict['choices'][0]['message']['content']
    cleaned_summary = f'{summary.strip()}... (auto-summarization of extracted-text)'
    cleaned_summary = cleaned_summary.replace( '\n', ' ' )
    cleaned_summary = cleaned_summary.replace( '  ', ' ' )
    log.debug( f'cleaned_summary, ``{cleaned_summary}``' )
    return cleaned_summary


## helper functions -------------------------------------------------


def validate_input( url: str ) -> str:
    """ Validates input.
        Accesses url and returns text.
        Called by summarize_text() """
    log.debug( 'validating input' )
    ## check url ----------------------------------------------------
    parsed_url = urlparse(url)
    if not parsed_url.scheme:
        raise Exception( 'url does not appear to have a scheme' )
    if parsed_url.scheme not in ['https']:
        raise Exception( 'url scheme must be https' )
    if not parsed_url.netloc:
        raise Exception( 'url does not appear to have a domain' )
    ## get text -----------------------------------------------------
    r = requests.get( url )
    if r.status_code != 200:
        raise Exception( 'url did not return a 200 status code; instead returned ``{r.status_code}``' )
    byte_response = r.content
    try:
        text_response = byte_response.decode( 'utf-8' )
    except Exception as e:
        raise Exception( 'url did not return utf-8' )
    if not text_response:
        raise Exception( 'url did not return any text' )
    log.debug( f'text_response, ``{text_response}``' )
    return text_response


def load_model() -> Llama:
    """ Loads model.
        Called by summarize_text() """
    log.debug( 'loading model' )
    LLM = Llama( 
        model_path='../models/ggml-vicuna-13b-4bit-rev1.bin',
        n_ctx=2000 )  # meximum number of tokens
    assert type(LLM) == Llama, type(LLM)
    log.debug( 'model loaded' )
    return LLM


def summarize( text_to_summarize: str, LLM, max_tokens_for_summarization=100 ) -> dict:
    """ Summarizes text.
        Explanation of model parameters: <https://chat.openai.com/share/e5923e46-4236-4034-9d99-829ab18dbfdc> (chatgpt4, 2023-Nov-25)
        Called by summarize_text() """
    log.debug( 'starting summarize()' )
    log.debug( f'text_to_summarize, \n\n``{text_to_summarize}``\n\n' )
    
    ## define prompt ------------------------------------------------
    # (message experimentation)
    # message = f'Summarize the following text ({word_text}-word-maximum) using a neutral tone, describing main themes and topics. The text: {f"{text_to_summarize} (end-of-text)"}'
    # message = f'Summarize, in one short sentence, using a neutral tone -- the following text: {text_to_summarize}'
    # message = f'Give a three-sentence summary of the following text using a neutral tone, describing main themes and topics. The text: {text_to_summarize}'
    # (title experimentation)
    # message = f'Give a 7-word-max summary (in title-case) of the following text, using a neutral tone: {text_to_summarize}'
    message = f'What would a good short tile be for following text?: {text_to_summarize}'
    log.debug( f'message, ``{message}``' )
    
    ## complete prompt-data -----------------------------------------
    messages = [ {'role': 'user', 'content': message} ]
    ## run model ----------------------------------------------------
    output: dict = LLM.create_chat_completion(
        messages, 
        temperature=0.2,  # lower is more conservative, higher is more creative
        top_p=0.95, 
        top_k=40, 
        stream=False, 
        stop=[], 
        max_tokens=max_tokens_for_summarization,
        repeat_penalty=1.1
        )
    log.debug( f'output, ``{pprint.pformat(output)}``' )
    ## return -------------------------------------------------------
    return output


## dundermain -------------------------------------------------------


if __name__ == '__main__':
    ## set up argparser ---------------------------------------------
    parser = argparse.ArgumentParser( description='provide url to text-to-summarize' )
    parser.add_argument('--url', type=str, help='url to extracted-text datastream' )
    args = parser.parse_args()
    log.debug( f'args: {args}' )
    ## get text -----------------------------------------------------
    url = args.url
    ## get to work ---------------------------------------------------
    summarize_text ( url )
    log.debug( 'done' )
