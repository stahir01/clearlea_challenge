import os
import re 
import spacy 
import langchain
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import openai
import groq
import string
import re
import json

from dotenv import load_dotenv, dotenv_values 
from groq import Groq

#TODO: Data Cleaning using OpenAI
#TODO: Perform Word Embedding (Not sure if it's mandatory or not)
#TODO: Create Model Pipiline using Gemini
#TODO: Create Model Pipiline using LLAMA2.0
#TODO: Model Evaluation using OPENAI + Human Evaluation


def read_documents(file_path):
    """
    Reads all .txt files in the specified directory and stores their content in a list of JSON objects.
    :param file_path: Path to the directory containing the .txt files.
    :return: A list of JSON objects containing file names and their content.
    """
    # List to store the document contents
    documents = []

    # Get all .txt files in the directory
    file_dir = os.listdir(file_path)
    document_list = [file for file in file_dir if file.endswith('.txt')]
    i = 0
    # Read each file and store its content in the list
    for file in document_list:
        with open(os.path.join(file_path, file), 'r', encoding='latin-1') as f:
            content = f.read()
            document = {
                "file_name_{i}": file,
                "content": content
            }
            documents.append(document)
        i += 1

    return documents








