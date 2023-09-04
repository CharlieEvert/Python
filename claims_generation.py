#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 09:38:19 2023

@author: cevert
"""

import pandas as pd 
import math  
import openai

# Set up OpenAI API  
openai.api_key = '34c44fc2f72b45eab5cd1c5325cc8bc1'  
  
def split_string(string, tokens, tokens_per_split):  
    num_splits = math.ceil(tokens / tokens_per_split) - 1  
    if num_splits == 0:  
        return [string]  
      
    split_length = math.ceil(len(string) / (num_splits + 1))  
    splits = [string[i:i + split_length] for i in range(0, len(string), split_length)]  
    return splits  

tokens_per_split = 4_000 

df = pd.read_csv('/Users/cevert/Desktop/ai_projects/aprs/aprs.csv')  

# Ask the user to input the document name  
document_name = '0901c66d800c817d.pdf'#input("Enter the document name: ")  
  
# Filter the DataFrame to keep only the rows with the specified document name  
filtered_df = df[df['name'] == document_name]  
  
if not filtered_df.empty:  
    # Convert all values in the 'text' column to strings and concatenate them into a single string  
    full_text = ' '.join(f'Page {page+1}: {text}' for text, page in zip(filtered_df['text'].astype(str), filtered_df['page']))  
  
    # Get the total number of tokens  
    tokens = sum(filtered_df['tokens'])  
    
    split_text = split_string(full_text, tokens, tokens_per_split) 
    
    list_of_claims = []
    for text in split_text:
    
        claims = openai.ChatCompletion.create(  
            engine="gpt-35-turbo-16k",  
            messages=[  
                {"role": "system", "content": "You turn the text I provide into a list of up to 50 claims that could be made about a product that would help to sell the product and are not simply general in nature. Do not duplicate the same claims. Next to each claim, list the page number they were derived from. The text I provide is an excerpt from a larger document. Do not make anything up. Only respond with a bulleted list of numbered claims and nothing else."},  
                {"role": "user", "content": text}  
            ]  
        )['choices'][0]['message']['content'] 
        
        list_of_claims.append(claims)
    
else:  
    print("No document found with the specified name.")  
    
string_of_claims = ' '.join(list_of_claims) 
summary_of_claims = openai.ChatCompletion.create(  
    engine="chat-large",  
    messages=[  
        {"role": "system", "content": "You turn the list of claims I provide you with into one cohesive sequential list. Do not make anything up. Only keep the claims that would help to sell a product and are relevant. Group claims as needed. Respond in a bulleted list and do not provide anything else in your response. Maintin the page number citations."},  
        {"role": "user", "content": string_of_claims}  
    ]  
)['choices'][0]['message']['content'] 

pitch = openai.ChatCompletion.create(  
    engine="chat",  
    messages=[  
        {"role": "system", "content": "Using the product claims I provide, write me an advertisement for a product."},  
        {"role": "user", "content": summary_of_claims}  
    ]  
)['choices'][0]['message']['content'] 

