# RAG_LLAMA Integration

This project integrates the [MeloTTS](https://github.com/myshell-ai/MeloTTS) speech synthesis engine. Follow the instructions below to set up and run the project successfully.

## Prerequisites

- Python 3.10
- PostgreSQL


## Installation Steps

### 1. git clone the repo

     git clone https://github.com/leo007-htun/RAG_LLAMA.git

     cd RAG_LLAMA

### 2. Clone the MeloTTS Repository

To get started, you need to install **MeloTTS** from its source, as `pip` will not install all the required libraries.


    git clone https://github.com/myshell-ai/MeloTTS.git
    cd MeloTTS
    pip install -e .
    python -m unidic download

### 4. Install Requirements

    cd ..
    pip install -r requirements.txt

### 5. Create POSGRESQL

     sudo -u postgres psql #login to postgres


     CREATE USER leo WITH PASSWORD '123456' SUPERUSER;
     CREATE DATABASE memory_agent;
     GRANT ALL PRIVILEGES ON SCHEMA public TO leo;
     GRANT ALL PRIVILEGES ON DATABASE memory_agent TO leo;
     
     
     \l # list of tables
     
     \c memory_agent #connect db
     
     CREATE TABLE conversations (
     id SERIAL PRIMARY KEY,
     timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
     prompt TEXT NOT NULL,
     response TEXT NOT NULL
     );
     
     
     INSERT INTO conversations (timestamp, prompt, response) VALUES (CURRENT_TIMESTAMP, 'What is my name?', 'Your name is LEO');
     
     INSERT INTO conversations (timestamp, prompt, response) VALUES (CURRENT_TIMESTAMP, 'What is your purpose?', 'Your purpose is to serve humans');
     
     SELECT * FROM conversations;
     
     \dt # display table contents
     \d conversations



### 6. RUN RUN RUN >>>>

     python3 assistant5.py

     
