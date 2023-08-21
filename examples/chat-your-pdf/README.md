# Chat Your PDF!

This example app shows how to build a simple chatbot powered by Redis and Google's Vertex AI. It contains the following elements:

1) a Jupyter notebook that demonstrates setting up Redis as a vector database to store and retrieve document vectors. The notebook also shows how to use LangChain to perform semantic search for context within a pdf.
2) a Streamlit app that showcases the chat completion functionality of Vertex AI Palm 2 Chat Completion model, powered by Redis for RAG (retrieval augmented generation), Semantic Caching for the LLM, and chat history persistence.

{ IMAGE FORTHCOMING }


## Setting up the Environment

### Obtain Google Cloud Credentials
You will need a GCP project, service account, an an application JSON key file in order to auth with GCP. The credentials file will be mounted to the docker container of the chatbot app and exposed with the `GOOGLE_APPLICATION_CREDENTIALS` environment variable. For more information [check out this link](https://cloud.google.com/docs/authentication/application-default-credentials#GAC).


1) Download a GCP credentials JSON file:
    - Go to "IAM & Admin" panel in the GCP console.
    - On the left navbar select "Service Accounts".
    - Select the name of the service account.
    - On the top bar, select the "Keys" tab.
    - Above the list of active keys, select "ADD KEY" to create a new JSON key file.
2) **Move the credentials file into the root level of this folder here as `gcp_credentials.json`**
    ```bash
    mv ~/Downloads/.......json  gcp_credentials.json
    ```

### Update environment file
The project comes with a template `.env.template` file with the following values. Make a coy of this as `.env`. Update the values below accordingly.

```bash
CHUNK_SIZE=500
CHUNK_OVERLAP=100
DOCS_FOLDER="pdfs/"
REDIS_URL="redis://localhost:6379"
PROJECT_ID="YOUR_GCP_PROJECT_NAME" #nifty-456098
LOCATION="YOUR_VERTEXAI_REGION"    #us-central1
```

Update the `PROJECT_ID` and `LOCATION` based on your GCP project and vertex AI configuration. Update the `REDIS_URL` based on your Redis Enterprise database deployed in GCP.


## Run

To run the app, follow these steps:

1. Clone this repository to your local machine.
2. Set up your GCP credentials as outlined above.
2. Copy the `.env.template` to `.env` and configure the values as outlined above.
3. Run the app with Docker compose: `docker-compose up`.
