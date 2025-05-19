import os
import pandas as pd
from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# Explicitly specify the path to the .env file
dotenv_path = r"C:\Users\Lenovo\OneDrive\Desktop\langchain\.env"
load_dotenv(dotenv_path=dotenv_path)

# Check if the .env file is being loaded correctly
if not os.path.exists(dotenv_path):
    print(f"Error: The .env file does not exist at the specified path: {dotenv_path}")
else:
    print(f".env file found at {dotenv_path}")

# Fetch the Google API Key from environment
google_api_key = os.getenv("GOOGLE_API_KEY")

# Check if the API key is loaded correctly
if google_api_key is None:
    print("Error: GOOGLE_API_KEY not found in the .env file!")
else:
    print(f"Google API Key loaded: {google_api_key[:10]}...")  # Only show first 10 characters for security

# If no API key, raise an error
if not google_api_key:
    raise ValueError("Missing Google API Key in .env")

# Path to your CSV file
csv_path = r"c:\Users\Lenovo\OneDrive\Desktop\langchain\merged_mobiles_data.csv"

# Read the CSV file into a DataFrame
df = pd.read_csv(csv_path, encoding='utf-8')

# Initialize the LLM with the correct API key
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=google_api_key)

def fill_missing_with_llm(row):
    row_dict = row.to_dict()
    prompt = "Fill in the missing values based on the available data and general knowledge.\n\n"
    for key, value in row_dict.items():
        prompt += f"{key}: {value if pd.notna(value) else '[MISSING]'}\n"
    prompt += "\nReturn only the missing values in the format:\nColumn: Value"

    # Get the response from the LLM
    response = llm.invoke(prompt)
    print("\n🔍 Gemini Response:")
    print(response.content)

    # Process the response to fill missing values
    for line in response.content.splitlines():
        if ':' in line:
            col, val = line.split(':', 1)
            col = col.strip()
            val = val.strip()
            if col in row.index and pd.isna(row[col]):
                row[col] = val
    return row

# Create a copy of the dataframe to fill missing values
filled_df = df.copy()

# Identify rows with missing values
missing_rows = filled_df[filled_df.isnull().any(axis=1)]

# Fill missing values row by row
for idx, row in missing_rows.iterrows():
    filled_row = fill_missing_with_llm(row)
    filled_df.loc[idx] = filled_row

# Prepare documents for vectorization
docs = []
for _, row in filled_df.iterrows():
    content = "\n".join([f"{col}: {row[col]}" for col in filled_df.columns])
    docs.append(Document(page_content=content))

# Split documents into smaller chunks
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = splitter.split_documents(docs)

# Initialize embedding model
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)

# Create FAISS vector store
vectorstore = FAISS.from_documents(split_docs, embedding_model)

# Create a retriever from the vector store
retriever = vectorstore.as_retriever()

# Set up the RetrievalQA chain
qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Run the chatbot
print("\nRAG chatbot ready. Ask about the CSV data.")
while True:
    query = input("\nYou: ")
    if query.lower() in ["exit", "quit"]:
        print("Exiting. Have a great day!")
        break
    answer = qa.run(query)
    print(f"Bot: {answer}")
