#!/usr/bin/env python3  
# -*- coding: utf-8 -*-  
import time  
import os  
import re  
import csv  
import PyPDF2  
import tiktoken  
from concurrent.futures import ThreadPoolExecutor  
  
num_cpu_cores = os.cpu_count()  
folder_path = "/Volumes/My Passport for Mac/aprs/First_Load"  
  
def num_tokens_from_string(string: str, encoding_name: str) -> int:  
    encoding = tiktoken.get_encoding(encoding_name)  
    num_tokens = len(encoding.encode(string))  
    return num_tokens  
  
def save_to_csv(embeddings, output_file):  
    with open(output_file, 'a', newline='', encoding="utf-8") as f:  
        writer = csv.DictWriter(f, fieldnames=["doctype", "name", "page", "tokens", "text"])  
        writer.writerows(embeddings)  
        
    with open(output_file, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        row_count = sum(1 for row in reader) - 1  
        print(f"Current length of CSV: {row_count} rows")

  
def process_pdf(file_path, counter, start_time, output_file):  
    local_embeddings = []  
    name = os.path.basename(file_path)
    try:  
        if name.endswith(".pdf"):  
            with open(file_path, "rb") as file:  
                pdf_reader = PyPDF2.PdfReader(file)  
                for page_num in range(len(pdf_reader.pages)):  
                    text = pdf_reader.pages[page_num].extract_text()  
                    text = re.sub(r"^\s+|\s+?$", "", text)  
                    text = re.sub(" +", " ", text).strip()  
                    tokens = num_tokens_from_string(text, "cl100k_base")  
                    embedding = {  
                        "doctype": "apr",  
                        "name": name,  
                        "page": page_num,  
                        "tokens": tokens,  
                        "text": text  
                    }  
                    local_embeddings.append(embedding)  
  
                    if len(local_embeddings) >= 1000:  
                        save_to_csv(local_embeddings, output_file)  
                        local_embeddings = []  
        
        print(f"Successfully processed file: {counter}, Filename: {name}")  
    except Exception as e:  
        print(f"Failed to process file: {counter}, Filename: {name}, Error: {str(e)}")  
    finally:  
        elapsed_time = (time.time() - start_time) / 3600  
        print(f"Processed file: {counter}, Elapsed time: {elapsed_time:.2f} hours")  
  
csv_file = "/Users/cevert/Desktop/ai_projects/aprs/aprs.csv"  # Full path to aprs.csv

# Always create a new aprs.csv file, overwriting the old one.
with open(csv_file, "w", newline="", encoding="utf-8") as f:  
    writer = csv.DictWriter(f, fieldnames=["doctype", "name", "page", "tokens", "text"])  
    writer.writeheader()  

pdf_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".pdf")]  
start_time = time.time()  
  
with ThreadPoolExecutor(max_workers=int(num_cpu_cores)) as executor:  
    executor.map(process_pdf, pdf_files, range(len(pdf_files)), [start_time] * len(pdf_files), [csv_file] * len(pdf_files))  
  
print("Processing completed.")
