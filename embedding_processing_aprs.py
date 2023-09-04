import pandas as pd  
import openai  
import re  
import requests  
import time  
from openai.embeddings_utils import get_embedding, cosine_similarity  
import uuid 
import time 

get_embedding("test", engine = 'text-embedding-ada-002')
  
def normalize_text(s, sep_token=" \n "):  
    if not isinstance(s, str):  
        s = str(s)  
    s = re.sub(r'\s+', ' ', s).strip()  
    s = re.sub(r". ,", "", s)  
    s = s.replace("..", ".")  
    s = s.replace(". .", ".")  
    s = s.replace("\n", "")  
    s = s.strip()  
    return s  
  
openai.api_key = API_KEY  
  
df = pd.read_csv('/Users/cevert/Desktop/ai_projects/aprs/aprs.csv')  
df = df.dropna(subset=['text'])  
df['text'] = df["text"].apply(lambda x: normalize_text(x))  
df = df[df['tokens'] != 0]  
df = df[df.tokens < 8192]  
df['uuid'] = [uuid.uuid4() for _ in range(len(df))]  
  
embeddings = []  
counter = 0  
total_iterations = len(df)  
start_time = time.time()  
elapsed_times = []  
  
for index, row in df.iterrows():  
    iteration_start_time = time.time()  
     
    embedding_dict = {  
        "id": row.uuid,  
        "doctype": row.doctype,  
        "name": row.name,  
        "page": row.page,  
        "tokens" : row.tokens,  
        "text" : row.text,  
        "embedding": get_embedding(str(row['text']), engine = 'text-embedding-ada-002'),  
    }  
    embeddings.append(embedding_dict)  
    counter += 1  
  
    iteration_end_time = time.time()  
    elapsed_time = iteration_end_time - iteration_start_time  
    elapsed_times.append(elapsed_time)  
  
    average_time_per_iteration = sum(elapsed_times) / counter  
    remaining_iterations = total_iterations - counter  
    estimated_time_to_completion = average_time_per_iteration * remaining_iterations  
    estimated_time_to_completion_hours = estimated_time_to_completion / 3600  
  
    print(f"Done with iteration {counter}. Estimated time to completion: {estimated_time_to_completion_hours:.2f} hours")  
  
total_time = time.time() - start_time  
total_time_hours = total_time / 3600  
print(f"Total time taken: {total_time_hours:.2f} hours") 
    