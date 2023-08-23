# LLM Reference Architecture using Redis & Google Cloud Platform

<a href="https://colab.research.google.com/github/RedisVentures/redis-google-llms/blob/main/BigQuery_Palm_Redis.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

☁️ Google's Vertex AI has expanded its capabilities by introducing [Generative AI](https://cloud.google.com/vertex-ai/docs/generative-ai/learn/overview). This advanced technology comes with a specialized [in-console studio experience](https://cloud.google.com/vertex-ai/docs/generative-ai/start/quickstarts/quickstart), a [dedicated API](https://cloud.google.com/vertex-ai/docs/generative-ai/start/quickstarts/api-quickstart) and [Python SDK](https://cloud.google.com/vertex-ai/docs/python-sdk/use-vertex-ai-python-sdk) designed for deploying and managing instances of Google's powerful PaLM language models. With a distinct focus on text generation, summarization, chat completion, and embedding creation, PaLM models are reshaping the boundaries of natural language processing and machine learning.

⚡ Redis Enterprise offers robust [vector database](https://redis.com/solutions/use-cases/vector-database) features, with an API for index creation, management, distance metric selection, similarity search, and hybrid filtering. When coupled with its [versatile data structures](https://redis.io/docs/data-types/) - including lists, hashes, JSON, and sets - Redis Enterprise shines as the optimal solution for crafting high-quality Large Language Model (LLM)-based applications. It embodies a streamlined, ["shared-nothing" architecture](https://redis.com/redis-enterprise/technology/redis-enterprise-cluster-architecture/) and exceptional [SLAs](https://redis.com/legal/redis-enterprise-cloud-service-level-agreement/), making it an instrumental tool for production environments.

>This repo serves as a foundational architecture for building LLM applications with Redis and GCP services.

## Reference Architecture

![](assets/GCP_RE_GenAI.drawio.png)

### Core Components
1. Primary Storage >>> **GCP BigQuery**
2. Foundation Models >>> **GCP Vertex AI**
    - PaLM API for text embedding creation
    - PaLM API for text generation
    - PaLM API for chat completion
3. High-Performance Data Layer >>> **Redis Enterprise**
    - Vector database for semantic search + context retrieval
    - LLM Cache
    - LLM Memory for application chat history and session metadata

### Setup Workflow
1. **Load Libraries and Tools**: Before building language modeling applications, we install the right Python libraries, connect to the proper datastores, and authenticate with GCP.
2. **Create BigQuery Table**: Drawing in data from one or more sources, we populate an enriched table in BigQuery that holds the primary data for building language model applications. This could be a custom knowledge base, domain-specific proprietary data, customer records (typically any kind of data that has text fields).
3. **Generate Embeddings**: Leveraging Google’s Vertex PaLM API, we generate semantic text embeddings that characterize & represent “chunks” of underlying text. These embeddings are lists of numbers that capture the meaning and context and can be used for similarity search between a user input question or prompt and source text.
4. **Load Embeddings**: We store the rich embeddings in Redis Enterprise as an additional low-latency data layer on top of BigQuery.
4. **Create Vector Index**: We create a search index in Redis Enterprise that enables real-time semantic search. While BigQuery holds the primary data, Redis holds the embeddings.

## Potential Use Cases
This architecture contains many essential elements required to build real-world LLM applications that can enhance your business. A few examples include:

- [Customer Support Bot](examples/chat-your-pdf/)
- [Virtual Shopping Assistant](https://github.com/RedisVentures/redis-langchain-chatbot)
- [Document Retrieval Engine](https://github.com/RedisVentures/redis-arXiv-search)

## Tutorial
<a href="https://colab.research.google.com/github/RedisVentures/redis-google-llms/blob/main/BigQuery_Palm_Redis.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

Open the code tutorial in a Colab notebook (recommended) to get your hands dirty with LLMs on GCP. This guide is a step-by-step walkthrough of setting up the required data, services, and databases in order to build LLM applications, and then highlights a few Redis & LLM design principles including: **Semantic Search**, **Retrieval**, **Caching**, and **Memory** storage.


## Additional Resources
- [DRAFT Blog Post](https://docs.google.com/document/d/1nGelpYQaFcTd1lqLOC3W0ZoXVDNMiI3W3pG7LY4U3n4/edit?usp=sharing)
- [LangChain Example](https://github.com/antonum/Redis-Workshops/blob/main/05-LangChain_Redis/05.3_VertexAI_LangChain_Redis.ipynb)
- [Redis Vector Search Documentation](https://redis.io/docs/interact/search-and-query/search/vectors/)
- [More Redis AI Resources](https://github.com/RedisVentures)
- [Google VertexAI Resources](https://cloud.google.com/vertex-ai)
- [Google BigQuery Resources](https://cloud.google.com/bigquery)
