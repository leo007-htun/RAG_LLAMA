import ollama
import ast
import chromadb
import psycopg
from colorama import Fore
from tqdm import tqdm
from psycopg.rows import dict_row
from melo.api import TTS
from playsound import playsound 
import base64
import requests
import json
import speech_recognition as sr
import warnings

# Suppress all warnings
warnings.simplefilter('ignore')


# Initialize the Chromadb client
client = chromadb.Client()

# System prompt for the assistant
system_prompt = (
    'YOU ARE A PERSONAL ASSISTANT NAMED JARVIS.'
)

# Conversation history
convo = [{'role': 'system', 'content': system_prompt}]

# Database connection parameters
DB_PARAMS = {
    'dbname': 'memory_agent',
    'user': 'leo',
    'password': 'table[838]',
    'host': 'localhost',
    'port': '5432'
}

# Initialize recognizer
recognizer = sr.Recognizer()
#recognizer.energy_threshold = 4000
recognizer.dynamic_energy_threshold = True

device = 'auto'  # Automatically use GPU if available, otherwise use CPU
speed = 0.9 # Speech speed adjustment
tts_model = TTS(language='EN', device=device)  # Load model once
speaker_ids = tts_model.hps.data.spk2id  # Get speaker IDs once


# Function to connect to the database
def connect_db():
    """Connects to the PostgreSQL database."""
    return psycopg.connect(**DB_PARAMS)

# Function to fetch conversations from the database
def fetch_conversations():
    """Fetches all conversations from the database."""
    conn = connect_db()
    with conn.cursor(row_factory=dict_row) as cursor:
        cursor.execute('SELECT * FROM conversations')
        conversations = cursor.fetchall()
    conn.close()
    return conversations

# Function to store conversations in the database
def store_conversations(prompt, response):
    """Stores a conversation prompt and response in the database."""
    conn = connect_db()
    with conn.cursor() as cursor:
        cursor.execute(
            'INSERT INTO conversations (timestamp, prompt, response) VALUES (CURRENT_TIMESTAMP, %s, %s)',
            (prompt, response)
        )
        conn.commit()
    conn.close()

# Function to remove the last conversation from the database
def remove_last_conversation():
    """Removes the last conversation from the database."""
    conn = connect_db()
    with conn.cursor() as cursor:
        cursor.execute('DELETE FROM conversations WHERE id = (SELECT MAX(id) FROM conversations)')
        conn.commit()
    conn.close()

# Function to stream the assistant's response
def stream_response(prompt):
    """Streams the assistant's response to a prompt."""
    response = ''
    stream = ollama.chat(model='llama3', messages=convo, stream=True)
    print(Fore.LIGHTGREEN_EX + '\nASSISTANT:')

    for chunk in stream:
        content = chunk['message']['content']
        response += content
        print(content, end='', flush=True)

    print('\n')
    store_conversations(prompt=prompt, response=response)
    convo.append({'role': 'assistant', 'content': response})
    speak_response(response)

# Function to create a vector database from conversations
def create_vector_db(conversations):
    """Creates a vector database from existing conversations."""
    vector_db_name = 'conversations'

    # Attempt to delete the existing collection if it exists
    try:
        client.delete_collection(name=vector_db_name)
    except ValueError:
        pass

    vector_db = client.create_collection(name=vector_db_name)

    for c in conversations:
        serialized_convo = f"prompt: {c['prompt']} response: {c['response']}"
        response = ollama.embeddings(model='nomic-embed-text', prompt=serialized_convo)
        embedding = response['embedding']

        vector_db.add(
            ids=[str(c['id'])],
            embeddings=[embedding],
            documents=[serialized_convo]
        )

# Function to retrieve relevant embeddings based on queries
def retrieve_embeddings(queries, results_per_query=2):
    """Retrieves embeddings from the vector database based on the queries."""
    embeddings = set()

    for query in tqdm(queries, desc='Processing queries to vector database'):
        response = ollama.embeddings(model='nomic-embed-text', prompt=query)
        query_embedding = response['embedding']
        vector_db = client.get_collection(name='conversations')
        results = vector_db.query(query_embeddings=[query_embedding], n_results=results_per_query)
        best_embeddings = results['documents'][0]

        for best in best_embeddings:
            if best not in embeddings:
                if 'yes' in classify_embedding(query=query, context=best):
                    embeddings.add(best)

    return embeddings

# Function to create queries for the embedding database
def create_queries(prompt):
    """Creates a list of search queries to retrieve relevant conversations."""
    query_msg = (
        'You are a first-principles reasoning search query AI agent. '
        'Your list of search queries will be run on an embedding database of all conversations you have had with the user. '
        'With first principles, create a Python list of queries to search the embeddings database for any data that would be necessary to correctly respond to the prompt. '
        'Your response must be a Python list with no syntax errors. '
        'Do not explain anything and do not generate anything but a perfect syntax Python list.'
    )

    query_convo = [
        {'role': 'system', 'content': query_msg},
        {'role': 'user', 'content': 'What is machine learning?'},
        {'role': 'assistant', 'content': 'Machine learning is a field of AI focused on enabling systems to learn from data and improve over time without being explicitly programmed.'},
        {'role': 'user', 'content': 'Can you give an example of machine learning?'},
        {'role': 'assistant', 'content': 'Sure! A common example is spam filtering in emails. The system learns to classify emails as spam or not spam based on patterns in previous emails.'},
        {'role': 'user', 'content': prompt}
    ]

    response = ollama.chat(model='llama3', messages=query_convo)
    print(Fore.YELLOW + f'\nVector database queries: {response["message"]["content"]}\n')

    try:
        return ast.literal_eval(response['message']['content'])
    except Exception as e:
        print(f"Error parsing response: {e}")
        return [prompt]

# Function to classify embeddings based on query relevance
def classify_embedding(query, context):
    """Classifies whether the context is relevant to the search query."""
    classify_msg = (
        'You are an embedding classification AI agent. Your input will be a prompt and one embedded chunk of text. '
        'You will not respond as an AI assistant. You only respond "yes" or "no". '
        'Determine whether the context contains data that directly relates to the search query. '
        'If the context is exactly what the search query needs, respond "yes"; otherwise respond "no".'
    )

    classify_convo = [
        {'role': 'system', 'content': classify_msg},
        {'role': 'user', 'content': f'SEARCH QUERY: {query}\nEMBEDDED CONTEXT: {context}'}
    ]

    response = ollama.chat(model='llama3', messages=classify_convo)
    return response['message']['content'].strip().lower()

# Function to recall memories based on the user prompt
def recall(prompt):
    """Recalls memories related to the user prompt."""
    queries = create_queries(prompt=prompt)
    embeddings = retrieve_embeddings(queries=queries)
    convo.append({'role': 'user', 'content': f'MEMORIES: {embeddings}\n\nUSER PROMPT: {prompt}'})
    print(f'\n{len(embeddings)} message:response embeddings added for context')

def speak_response(response):
    output_path = 'response.wav'
    # Use a specific English speaker ID, for example, 'EN-US'
    if 'EN-BR' in speaker_ids:
        tts_model.tts_to_file(response, speaker_ids['EN-BR'], output_path, speed=speed)
        playsound(output_path)
    else:
        print("Error: English speaker ID not found.")

# Capture audio from the microphone and convert to text
def listen_for_prompt():
    with sr.Microphone() as source:
        print("Listening for your prompt...")
        audio = recognizer.listen(source)

    try:
        prompt = recognizer.recognize_google(audio)
        print(f"You said: {prompt}")
        return prompt
    except sr.UnknownValueError:
        print("Sorry, I could not understand the audio.")
        return ""
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
        return ""


# Main execution flow
if __name__ == "__main__":
    # Fetch conversations from the database and create a vector database
    conversations = fetch_conversations()
    create_vector_db(conversations=conversations)

    while True:
        # Prompt user for input
        #prompt = input(Fore.WHITE + 'YOU:\n')
        prompt = listen_for_prompt()  # Listen for speech input

        if prompt == "":
            continue  # If no valid prompt, skip the iteration

        # Check if the input is a recall command
        if prompt.lower().startswith('/recall'):
            prompt = prompt[8:]  # Remove the command prefix
            recall(prompt=prompt)  # Recall memories related to the prompt

        elif prompt.lower().startswith('/forget'):
            remove_last_conversation()
            convo.pop()  # Remove the last assistant response
            convo.pop()  # Remove the last user input
            print('\n')

        elif prompt.lower().startswith('/memorize'):
            prompt = prompt[10:]  # Remove the command prefix
            store_conversations(prompt=prompt, response='Memory Stored!!!')
            print('\n')

        else:
            convo.append({'role': 'user', 'content': prompt})  # Add user input to the conversation

        # Generate and stream the response
        stream_response(prompt=prompt)
