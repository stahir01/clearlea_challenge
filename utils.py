import os
import re 
# import spacy 
# import langchain
import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import openai
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


class MissingEnvironmentVariable(Exception):
    pass

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
    i = 1
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

def load_questions(file_path):
    """
    Reads a JSON file and returns its contents.
    :param file_path: Path to the JSON file.
    :return: The contents of the JSON file.
    """
    with open(file_path, 'r') as f:
        data = f.read()
    questions = json.loads(data)["questions"]
    return questions


def model_result(model_name, api_var, documents, questions):
    """
    Creates a model pipeline to read documents and answer a list of questions.

    :param model_name: The name of the model to be used.
    :param api_var: The environment variable for the API key.
    :param documents: A list of dictionaries containing document file names and their content.
    :param questions: A list of questions to be answered by the model.
    :return: The model's responses to the questions.
    """

    # Load environment variables
    if not load_dotenv():
        raise MissingEnvironmentVariable(f"The environment variable file does not exist")

    api_key = os.environ.get(api_var)
    if not api_key:
        raise MissingEnvironmentVariable(f"API key for {api_var} not found in environment variables")

    # Create a Groq client
    client = Groq(api_key=api_key)
    MODEL = model_name

    # Combine document contents
    document_content = documents['content']
    #print(f"Test document output: {combined_content}")

    responses = []
    for question in questions:
        # Create chat completion request
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant.",
                },
                {
                    "role": "user",
                    "content": f"Document: {document_content}\n\nQuestion: {question}",
                }
            ],
            model=MODEL,
        )

        # Get the model's response
        response_content = chat_completion.choices[0].message.content
        actual_question = question.replace("Please extract the following data from the document. If you can’t find the answer please answer ‘not available’. ", "")
        #print(f"Question: {question}\nAnswer: {response_content}")
        responses.append({
            "question": actual_question,
            "answer": response_content
        })

    return responses


def store_model_results(model_name, document_name, model_responses):
    """
    Stores the model's responses to a list of questions in a JSON file.

    :param model_name: The name of the model used to answer the questions.
    :param document_name: The name of the document used to answer the questions.
    :param model_responses: A list of dictionaries containing the questions and the model's responses.
    """

    if model_name == 'mixtral-8x7b-32768':
        output_dir = './MixtralModelOutput'

    elif model_name == 'llama3-70b-8192':
        output_dir = './LLamaModelOutput'
        
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    # Ensure the directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract the document's name key
    document_name = next(value for value in document_name.values() if value.startswith('rental_contract'))

    # Remove the .txt extension if present
    if document_name.endswith('.txt'):
        document_name = document_name[:-4]

    # Define the output file path
    output_file = os.path.join(output_dir, f'{document_name}.json')

    # Store the model's responses in a JSON file
    with open(output_file, 'w') as f:
        json.dump(model_responses, f, indent=4)







