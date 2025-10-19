import pandas as pd
import json
import os
import re
from bs4 import BeautifulSoup
import nltk
from openai import OpenAI
import pickle


# read text
def read_news_list_from_excel(excel_path, column_name):
    df = pd.read_excel(excel_path)
    return df[column_name].astype(str).tolist()

# text cleaning
def clean_text(text):
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9,.，。！？:：\\-— ]', '', text)
    text = text.replace('\r', ' ').replace('\n', ' ')
    return text.strip()

print("cleaning text...")
print("cleaning text finished")

# text chunking
def chunk_text(cleaned_news_text, max_chars=3000, overlap_chars=100):
    chunks = []
    text_length = len(cleaned_news_text)
    start = 0

    while start < text_length:
        end = min(start + max_chars, text_length)
        if end < text_length:
            last_sentence_end = cleaned_news_text.rfind('.', start, end)
            last_question_end = cleaned_news_text.rfind('?', start, end)
            last_exclamation_end = cleaned_news_text.rfind('!', start, end)
            last_boundary = max(last_sentence_end, last_question_end, last_exclamation_end)
            if last_boundary > start:
                end = last_boundary + 1
        chunk = cleaned_news_text[start:end]
        chunks.append(chunk.strip())
        start = end - overlap_chars if end < text_length else end
    return chunks

print("chunking text")
chunks = chunk_text(cleaned_news_text, max_chars=30000, overlap_chars=1000)
print(f"chuncking text finished，{len(chunks)}")
for i, chunk in enumerate(chunks):
    print(f' {i+1} : {len(chunk)}')

# llm API
def call_llm_with_openai_sdk(text, prompt, api_key, base_url, model, max_retries=3):
    key = api_key or os.getenv('OPENAI_API_KEY')

    for attempt in range(max_retries):
        try:
            client = OpenAI(
                api_key=key,
                base_url=base_url,
            )
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": text},
            ]
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=2000,  # add token restrictions
                timeout=60  # add timeout
            )

prompt = """
your prompt
"""

api_key = "your api"
base_url = "your url"
model = "your model"
input_excel_path = 'your path'

def test_api_connection():
    try:
        test_client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        response = test_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=10
        )
        print("API connection ")
        return True
    except Exception as e:
        print(f"API error: {e}")
        print("please check")
        print(f"  - API Key: {api_key[:10]}...")
        print(f"  - Base URL: {base_url}")
        print(f"  - Model: {model}")
        return False

# test API connection
print("try API connection...")
if not test_api_connection():
    print("API error")
    exit()

print("reading Excel file")
news_list = read_news_list_from_excel(input_excel_path, 'text')
print(f"Excel read finished，all{len(news_list)}news。")


# save
def save_results_to_excel(results, output_path, input_excel_path, input_column_name='text'):
    input_df = pd.read_excel(input_excel_path)
    data = []
    for idx, result in enumerate(results):
        row_data = list(input_df.iloc[idx])
        if result == "No results":
            data.append(row_data + ["No results", "-", "-", "-", "-", "-"])
        else:
            groups = re.findall(r'\[(.*?)\]', result)
            if not groups:
                groups = [result.strip('[]')]
            for group in groups:
                items = [item.strip() if item.strip() else '-' for item in group.split(';')]
                while len(items) < 6:
                    items.append('-')
                if len(items) > 6:
                    items = items[:6]
                data.append(row_data + items)
    output_columns = list(input_df.columns) + [
        "Supply Chain Stage",
        "Disruption Event",
        "Risk Category",
        "Location/Country",
        "Involved Parties",
        "Event Description"
    ]
    df = pd.DataFrame(data, columns=output_columns)
    df.to_excel(output_path, index=False)
    print(f"save reuslts to {output_path}")