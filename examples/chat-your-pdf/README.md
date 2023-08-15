# Chat Your PDF!

This example app shows how to build a simple chatbot powered by Redis and Google's Vertex AI. It contains the following elements:

1) a Jupyter notebook that demonstrates setting up Redis as a vector database to store and retrieve document vectors. The notebook also shows how to use LangChain to perform semantic search for context within a pdf.
2) a Streamlit app that showcases the chat completion functionality of Vertex AI Palm 2 Chat Completion model, powered by Redis for RAG (retrieval augmented generation), Semantic Caching for the LLM, and chat history persistence.

https://user-images.githubusercontent.com/13009163/237002719-26e3118d-77ee-4ded-96f5-6ba801cae66c.mov


## Setting up the Environment

The tutorial can run in multiple ways. The first step is to configure the ``.env`` file in this repository. This file contains the following variables:

```bash
OPENAI_TEMPERATURE=0.7
OPENAI_MAX_TOKENS=50
CHUNK_SIZE=500
CHUNK_OVERLAP=100

REDIS_ADDRESS=localhost
REDIS_PORT=6379
REDIS_PASSWORD=
```

### Redis Cloud on GCP Marketplace
Once you set up your Redis Enterprise database on GCP, please provide the following credentials to your `.env` file:

```bash
REDIS_PASSWORD=<your password here>
REDIS_HOST=<your redis address here>
REDIS_PORT=<your redis port here>
```

#### Manual Setup (GCP Console)
For each, the following options are required.

#### Terraform Setup



There are some ``docker-compose.yml`` files in the ``docker`` directory that will help spin up
redis-stack locally and redisinsight in the case where a remote Redis is being used (like ACRE).

### Run


To run the script, follow these steps:

1. Clone this repository to your local machine.
2. copy the ``.env.template`` to ``.env`` and configure the values as outlined above.
3. If using a local Python environment, just run the notebook
3. Otherwise, select a docker-compose file to run and execute ``docker-compose up`` to start the environment.
