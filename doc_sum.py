#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 10:49:38 2023

@author: cevert
"""

import openai  
import PyPDF2  
    
openai.api_key = key
  
pdf_filename = "/Users/cevert/Downloads/OpenAI Document Q&A.pdf"  
  
system_prompt = "Your job is to summarize an entire document 5 pages at a time into a single paragraph (based on the previous summary). Do not make anything up, only use the context in the summary and the documents. Summaries should be returned as one paragraph with the main idea as the first sentence."  
base_prompt = "Using the summary below, update it to include relevant information from a page. Make sure the central point of the summary is not lost and only keep the most pertinent details. Do not show the original summary or provide any input besides for the summary."  
  
current_summary = ""  
summaries = []  
  
# Read the PDF  
with open(pdf_filename, 'rb') as pdf_file:  
    # Create a PDF reader object  
    pdf_reader = PyPDF2.PdfReader(pdf_file)  
  
    # Get the total number of pages in the PDF  
    num_pages = len(pdf_reader.pages)  
  
    # Loop through each 5 pages and summarize them  
    for start_page in range(0, num_pages, 5):  
        # Initialize the page text for the current batch  
        batch_page_text = ""  
  
        # Extract text from the 5 pages in the current batch  
        for page_num in range(start_page, min(start_page + 5, num_pages)):  
            # Get the page and extract the text  
            page = pdf_reader.pages[page_num]  
            page_text = page.extract_text()  
  
            # Add the page text to the batch_page_text  
            batch_page_text += f"Page {page_num + 1}:\n {page_text}\n\n"  
  
        # Prepare the prompt for the API call  
        prompt = f"""{base_prompt}  
  
        Current Summary:  
        {current_summary}  
  
        New Information:  
        {batch_page_text}  
        """  
  
        # Summarize the current batch using OpenAI API  
        current_summary = openai.ChatCompletion.create(  
            engine="chat",  
            messages=[  
                {"role": "system", "content": system_prompt},  
                {"role": "user", "content": prompt}  
            ]  
        )['choices'][0]['message']['content']  
  
        # Print the current summary  
        print(current_summary)  
  
        # Add the current summary to the list of summaries  
        summaries.append(current_summary)  

        
        
        
        
        
        