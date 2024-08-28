from openai import OpenAI
import string, os
import numpy as np
from enum import Enum

import pickle
#from TypeChat.typechat.typechat import TypeChat

class Ranking(Enum):
    GREATER = 0
    LESSER = 1
    EQUAL = 2

#from keys import API_KEY, ORG_KEY


def decode_obs(obs, new_obs, step_idx):
    old_height = obs[1]
    new_height = new_obs[1]

    old_x = obs[0]
    new_x = new_obs[0]

    q = "Q:\n"
    q += "State[" + str(step_idx) + "]:\n"
    q += "The hopper is " + str(old_x) + " m to the right from the starting point, and this number will become larger in the next state if and only if the hopper is moving rightwards.\n"
    q += "The hopper is now " + str(old_height) + "m in height.\n\n"
    
    q += "State[" + str(step_idx+1) + "]:\n"
    q += "The hopper is " + str(new_x) + " m to the right from the starting point.\n"
    q += "The hopper is now " + str(new_height) + "m in height.\n\n"
    
    q+= "Is the transtion from State[" + str(step_idx) + "] to State[" + str(step_idx+1) + "] a good transition? (Yes/No)\n\n"
    q+= "A:\nLet's think step by step.\n\n"

    return q

def decode_llm_msg(msg):
    #print(msg)
    lines = msg.split('\n')
    punctuation = string.punctuation
    lines = [line.strip(punctuation + string.whitespace) for line in lines]
    #lines[1] = lines[1].strip(punctuation + ' ')

    rank = False
    for line in lines:
        # if line[-3:].lower() == 'yes':
        #     rank = Ranking.GREATER
        # elif line[-2:].lower() == 'no':
        #     rank = Ranking.LESSER
        if "yes" in line.lower():
            rank = Ranking.GREATER
        elif "no" in line.lower():
            rank = Ranking.LESSER
    return rank


def value_est(q, type_check = False, prompts = None, anss = None):
    request = [
        {"role": "user", "content": q}
    ]

    client = OpenAI(
        base_url = 'http://localhost:23100/v1',
        api_key='ollama', # required, but unused
    )
    #client = OpenAI(api_key=API_KEY, organization=ORG_KEY, base_url="http://localhost:23002/v1")

    # create a chat completion
    print('prepared to ask')
    completion = client.chat.completions.create(
        model= 'llama3:70b', #'llama3:70b',#"Llama-2-13b-chat-hf", #'mixtral',#"Starling-LM-7B-alpha",#"gpt-4-1106-preview",
        messages=request,
        temperature=0.1,
        max_tokens=400
    )
    print(completion.choices[0].message.content)
    return completion.choices[0].message.content
