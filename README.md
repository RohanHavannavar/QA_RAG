# Chatbot for Answering Questions in Documents using RAG

This project implements a chatbot designed to answer questions from documents using the Retrieval-Augmented Generation (RAG) model. The chatbot leverages state-of-the-art NLP techniques to retrieve relevant information from a corpus of documents and generate accurate and context-aware responses.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Introduction

In the age of information, efficiently extracting relevant information from a vast collection of documents is crucial. This project aims to provide a sophisticated solution by integrating retrieval and generation capabilities, allowing the chatbot to understand and answer queries based on the content of provided documents.

## Features

- **Document Ingestion**: Load and process multiple documents in various formats (e.g., PDF, DOCX, TXT).
- **Retrieval-Augmented Generation (RAG)**: Combines the strengths of retrieval-based and generation-based models for accurate responses.
- **Natural Language Understanding**: Advanced NLP techniques for understanding complex queries.
- **Scalability**: Designed to handle large volumes of documents and queries.

## Architecture

The chatbot is built using the RAG architecture, which consists of:

1. **Retriever**: Identifies relevant passages from the document corpus.
2. **Generator**: Generates coherent and contextually accurate responses based on the retrieved passages.

![Architecture Diagram](path/to/architecture-diagram.png)

## Installation

To set up the project locally, follow these steps:

1. **Clone the repository**:
    ```sh
    git clone https://github.com/your-username/chatbot-rag.git
    cd chatbot-rag
    ```

2. **Install dependencies**:
    ```sh
    poetry install
    ```

3. **Download pre-trained models**:
    Follow the instructions to download necessary pre-trained models from the respective repositories (e.g., Hugging Face).

## Usage

To start the chatbot, run:

```sh
poetry run python main.py
