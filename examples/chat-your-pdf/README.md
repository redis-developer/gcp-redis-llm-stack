# 📃 Chat Your PDF!

This example Streamlit app demonstrates how to build a simple chatbot powered by Redis, LangChain, and Google's Vertex AI. It contains the following elements:

- ⚙️ [LangChain](https://python.langchain.com/docs/get_started/introduction.html) for app orchestration, agent construction, and tools
- 🖥️ [Streamlit](https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps) for the front end and conversational interface
- ☁️ [GCP Vertex AI Palm 2](https://cloud.google.com/vertex-ai/docs/generative-ai/start/quickstarts/api-quickstart) models for embedding creation and chat completion
- 💥 Redis and [RedisVL](https://redisvl.com) for Retrieval-Augmented Generation (RAG), LLM Semantic Caching, and chat history persistence

{ IMAGE FORTHCOMING }


## About
Redis is well-versed to power chatbots thanks to its flexible data models, query engine, and high performance. This enables users to leverage redis for a variety of gen AI needs:
- **RAG** -- ensures that relevant context is retrieved from Redis as a [Vector Database](https://redis.com/solutions/use-cases/vector-database), given a users question
- **Semantic Caching** -- ensures that duplicate requests for identical or very *similar* information are not exhuastive. Ex:
    ```bash
    streamlit    | Full Response Time (secs) 1.6435627937316895
    streamlit    | Cache Response Time (secs) 0.11130380630493164
    ```
- **Chat History** -- ensures distributed & low latency access to conversation history in Redis [Lists](https://redis.io/docs/data-types/lists/)

## Setting up the Environment

### Obtain Google Cloud Credentials
You need a valid GCP project, service account, an an application JSON key file in order to auth with GCP. The credentials file will be mounted to the docker container of the chatbot app and exposed through the `GOOGLE_APPLICATION_CREDENTIALS` environment variable. For more information [check out this link](https://cloud.google.com/docs/authentication/application-default-credentials#GAC).


1) **Download a GCP credentials JSON file**:
    - Go to "IAM & Admin" panel in the GCP console.
    - On the left navbar select "Service Accounts".
    - Select the name of the service account.
    - On the top bar, select the "Keys" tab.
    - Above the list of active keys, select "ADD KEY" to create a new JSON key file.
2) **Move the credentials file into the root level of this folder here as `gcp_credentials.json`**
    ```bash
    mv ~/Downloads/<your-gcp-secret-key-file>.json  app/gcp_credentials.json
    ```

### Update environment configuration
The project comes with a template `.env.template` file with the following values. Make a coy of this as `.env`. Update the values below accordingly.

```bash
CHUNK_SIZE=500
CHUNK_OVERLAP=100
DOCS_FOLDER="pdfs/"
REDIS_URL="redis://localhost:6379"
GCP_PROJECT_ID="YOUR_GCP_PROJECT_NAME" #nifty-456098
GCP_LOCATION="YOUR_VERTEXAI_REGION"    #us-central1
```

- Update the `GCP_PROJECT_ID` and `GCP_LOCATION` variables based on your GCP project and vertex AI configuration.
- Update the `REDIS_URL` based on your Redis Enterprise database deployed in GCP.


## Run

To run the app, follow these steps:

1. Clone this repository to your local machine.
2. Set up your GCP credentials as outlined above.
2. Copy the `.env.template` to `.env` and configure the values as outlined above.
3. Run the app with Docker compose: `docker-compose up`.
