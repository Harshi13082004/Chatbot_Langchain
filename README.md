# LangChain Chatbot

This is a chatbot built using LangChain and OpenAI that answers user questions based on a structured CSV dataset.

## Features
- Answers questions using data from a CSV file
- Uses FAISS for vector search
- Generates responses with OpenAI through LangChain

## How to Run

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Create a `.env` file and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_openai_api_key
   ```

3. Run the chatbot:
   ```
   python src/main.py
   ```

## Files
- `src/main.py` – Chatbot code
- `data/sample_mobiles_data.csv` – Sample data file
- `.env.example` – Environment variable template
- `requirements.txt` – List of dependencies

## Notes
- Replace the sample data with your own CSV if needed.
- Do not upload real or sensitive data to public repositories.
