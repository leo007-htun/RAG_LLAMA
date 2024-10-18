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

### 3. Install Requirements

    cd ..
    pip install -r requirements.txt

### 3. RUN RUN RUN >>>>

     python3 assistant5.py
