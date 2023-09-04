#!/usr/bin/env python3  
# -*- coding: utf-8 -*-  
"""  
Created on Tue Aug 29 12:43:54 2023  
  
@author: cevert  
"""  
  
import tkinter as tk  
import pandasql as psql  
import pandas as pd  
import openai  
import time  
import json  
  
def create_query_string(query):  
    sql_query = ""  
  
    # Add SELECT statement  
    if 'SELECT' in query:  
        sql_query += f"SELECT {query['SELECT']}"  
  
    # Add FROM statement  
    if 'FROM' in query:  
        sql_query += f" FROM {query['FROM']}"  
  
    # Add WHERE statement  
    if 'WHERE' in query:  
        sql_query += f" WHERE {query['WHERE']}"  
  
    # Add GROUP BY statement  
    if 'GROUP BY' in query:  
        sql_query += f" GROUP BY {query['GROUP BY']}"  
  
    # Add ORDER BY statement  
    if 'ORDER BY' in query:  
        sql_query += f" ORDER BY {query['ORDER BY']}"  
  
    # Add LIMIT statement  
    if 'LIMIT' in query:  
        sql_query += f" LIMIT {query['LIMIT']}"  
  
    return sql_query  
  
# Set up OpenAI API  
openai.api_key = '34c44fc2f72b45eab5cd1c5325cc8bc1'  
  
# Load CSV file into a DataFrame  
df = pd.read_csv('/Users/cevert/Desktop/ai_projects/aprs/aprs.csv')  
  
initial_context = "You help me generate SQL Queries. If there is an error, I am using the error as well as the response as context to further guide your response. If context indicates an error, then modify the query to avoid this error. If there is no error, then nothing else is needed. Make sure that the response is properly formatted as a JSON object with double quotes enclosing the property names."  
  
# Define available functions for GPT-3.5  
functions = [  
    {  
        "name": "aprs",  
        "description": "Generates SQL Queries based on terms and fields provided to get information from documents. Each record has a doctype (str), name (str), page (int), tokens (int), text (str). Text is the text content of each page and name is the name of the document. If a question asks if a document contains something, the field that would indicate this would be text. All WHERE clauses should contain LIKE to avoid case sensitivity. If asking about how/if something appears in documents, then select distinct documents so that the same document is not listed duplicatively.",  
        "parameters": {  
            "type": "object",  
            "properties": {  
                "SELECT": {  
                    "type": "string",  
                    "description": "Choose from the following fields and separate by a comma to fill in the select part of the query doctype, name, page, tokens, text."  
                },  
                "FROM": {  
                    "type": "string",  
                    "description": "Choose from the following tables: df"  
                },  
                "WHERE": {  
                    "type": "string",  
                    "description": "Create a filtering statement (WHERE clause) that will work with the following fields doctype, name, page, tokens, text."  
                },  
                "GROUP BY": {  
                    "type": "string",  
                    "description": "Create a SQL group by statement if needed with the following fields doctype, name, page, tokens, text."  
                },  
                "ORDER BY": {  
                    "type": "string",  
                    "description": "Create an ORDER BY SQL statement."  
                },  
                "LIMIT": {  
                    "type": "string",  
                    "description": "Create an LIMIT SQL statement if needed."  
                }  
            },  
            "required": ["SELECT", "FROM"],  
        },  
    }  
]  
  
# Retry logic constants  
max_retries = 10  
retry_delay = .1  # in seconds  
  
def get_response_with_retry(messages, functions, max_retries, retry_delay, user_question):      
    retries = 0      
    temp_messages = []  # Store temporary error messages and responses      
    while retries < max_retries:      
        try:      
            response = openai.ChatCompletion.create(      
                engine="gpt-35-turbo",      
                messages=messages,      
                functions=functions,      
                function_call="auto",      
            )['choices'][0]['message']['function_call']['arguments']      
                
            print("response: ", response)    
      
            # Convert response to dictionary    
            response = response.replace("%", "%%")  
            response_json = json.loads(response)      
            query = create_query_string(response_json)      
            print("query:", query)      
      
            # Query the db for the result      
            result = psql.sqldf(query)    
            result_table = result.to_string(index=True, index_names=['index'])   
    
            if len(result) > 100:    
                result_head = result.head(50)    
                result_tail = result.tail(50)    
                result_string = f"Length: {len(result)}\n\nHead:\n{result_head.to_string(index=True, index_names=['index'])}...\n\nTail:\n{result_tail.to_string(index=True, index_names=['index'])}"    
            else:    
                result_string = result.to_string(index=True, index_names=['index'])    
                
            print(result_string)      
      
            # synthesize the response into natural language for the user      
            synthesizing_prompt = "Interpret the given data to answer the user's question accurately. Keep the answer as concise as possible. For example, if the answer lists multiple documents to answer the question 'how many documents contain the word Listerine', and the data is a list of documents, the answer would be based on this list. If you do not see the exact answer, approximate it or say that 'The dataset is too large to provide a natural language answer on, please refer to the table below for your answer.' Do not say that the data is insufficient or ask follow up questions, simply use the data, the question and the context of the query that led to the data to answer the question."  
            synthesizing_context = f"User Question: '{user_question}'. Data: '{result_string}'. Query to get Data: '{query}'.\n\n\n"  
              
            result_in_words = openai.ChatCompletion.create(      
                engine="chat",      #gpt-35-turbo-16k
                messages=[      
                    {"role": "system", "content": synthesizing_prompt},      
                    {"role": "user", "content": synthesizing_context}      
                ]      
            )['choices'][0]['message']['content']      
      
            print(result_in_words)      
            response_dict = {      
                "query": query,      
                "result": result_table,      
                "result_in_words": result_in_words      
            }      
            return response_dict      
      
            # If no exceptions are raised, break out of the loop      
            break 
      
        except Exception as e:  
            retries += 1  
            messages.append({"role": "assistant", "content": response})  
            error_message = f"""Error: {str(e)}. Retrying {retries}/{max_retries}. 
            Code: 
                response = openai.ChatCompletion.create(  
                    engine="gpt-35-turbo",  
                    messages=messages,  
                    functions=functions,  
                    function_call="auto",  
                )['choices'][0]['message']['function_call']['arguments']  
      
                # Convert response to dictionary  
                response_json = ast.literal_eval(response)  
                query = create_query_string(response_json)  
                print("query:", query)  
                
                # Query the db for the result
                result = psql.sqldf(query)  
                result_string = result.to_string()  
                print(result_string)
                
                # synthesize the response into natural language for the user
                synthesizing_prompt = f"Answer this question using the data I provide: {user_question}. If the data is blank, then the answer is 'There are no search results. Please rephrase your question.' If the data is not blank, this is the answer."  
      
                result_in_words = openai.ChatCompletion.create(  
                    engine="chat",  
                    messages=[  
                        {"role": "system", "content": synthesizing_prompt},  
                        {"role": "user", "content": result_string}  
                    ]  
                )['choices'][0]['message']['content']  
      
                print(result_in_words)  
                response_dict = {  
                    "query": query,  
                    "result": result,  
                    "result_in_words": result_in_words  
                }  
                return response_dict  
            Values:
                response: {response}
                response_json: {response_json}
                query: {query}
                result: {result}
                result_string: {result_string}
                result_in_words: {result_in_words}
                response_dict: {response_dict}
                """
            messages.append({"role": "user", "content": error_message})  
            print(error_message)  
            time.sleep(retry_delay)  
    else:  
        raise Exception("Max retries exceeded. Please check the input or try again later.")  
  
def process_user_input(user_input):  
    messages = [  
        {"role": "system", "content": initial_context},
        {"role": "user", "content": user_input}
    ]  
    interim_response = get_response_with_retry(messages, functions, max_retries, retry_delay, user_input)  
    response = str(interim_response.get('result_in_words')) + "\n\nData:\n" + str(interim_response.get('result')) + "\n\nQuery:\n" + str(interim_response.get('query'))  
    return response  
  
def on_send(event=None):  
    user_input = user_entry.get()  
    if user_input:  
        chatbox.insert(tk.END, f"You: {user_input}\n")  
        user_entry.delete(0, tk.END)  
        response = process_user_input(user_input)  
        chatbox.insert(tk.END, f"APR Assistant: {response}\n")  
  
def on_close():  
    root.destroy()  
  
root = tk.Tk()  
root.title("Chat with APR Structured Data")  
  
frame = tk.Frame(root)  
scrollbar = tk.Scrollbar(frame)  
chatbox = tk.Text(frame, wrap=tk.WORD, yscrollcommand=scrollbar.set)  
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)  
chatbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)  
frame.pack(fill=tk.BOTH, expand=True)  
  
user_entry = tk.Entry(root, width=50)  
user_entry.bind("<Return>", on_send)  
user_entry.pack(fill=tk.X, padx=5, pady=5)  
  
send_button = tk.Button(root, text="Send", command=on_send)  
send_button.pack(side=tk.RIGHT, padx=5, pady=5)  
  
root.protocol("WM_DELETE_WINDOW", on_close)  
root.mainloop()  
