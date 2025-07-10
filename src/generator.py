import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from retriever import retriever
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from logger import setup_logger

load_dotenv()
logger = setup_logger(name="generator", log_file="logs/generator.log")

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))


def create_prompt(query, context_docs):
    """
    Create a well-structured prompt for the LLM using retrieved context
    """
    documents = context_docs.get("documents", [[]])[0]
    formatted_context = "\n\n---\n\n".join([doc for doc in documents])
    
    prompt = f"""
    You are an AI assistant that provides accurate, helpful, and concise responses based on the provided context.
    
    # Context Information
    {formatted_context}
    
    # Instructions
    1. Answer the user's question using ONLY the information from the provided context.
    2. If the context doesn't contain enough information to fully answer the question, acknowledge this limitation.
    3. Do not make up information or use knowledge outside of the provided context.
    4. Provide direct quotes from the context when appropriate to support your answer.
    5. Format your response in a clear, structured way using markdown formatting when helpful.
    6. Be concise but comprehensive in your response.
    7. If the question is completely unrelated to the context, politely state that you don't have information on that topic.
    
    # User Question
    {query}
    
    # Response
    """
    
    logger.info(f"Created prompt for query: {query[:50]}...")
    return prompt


def generate_response(query):
    """
    Generate a response to the user's query using RAG.
    """
    try:

        logger.info(f"Retrieving documents for query: {query}")
        retrieved_docs = retriever(query)
        
        if not retrieved_docs or not retrieved_docs.get("documents", [[]])[0]:
            logger.info("No documents retrieved from the database")
            return "I couldn't find any relevant information to answer your question. Please try rephrasing or ask something else."
        
        prompt = create_prompt(query, retrieved_docs)
        
        response = llm.invoke(prompt).content
        
        logger.info(f"Generated response for query.")
        return response,retrieved_docs
        
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        return f"An error occurred while generating a response: {str(e)}"


if __name__ == "__main__":
    # Example usage
    user_query = "What is the maximum number of items that can be listed on eBay?"
    response,retrieved_docs = generate_response(user_query)
    print(f"\nQuery: {user_query}\n")
    print(f"Response:\n{response}")
    print(f"Retrieved Documents:\n{retrieved_docs}")