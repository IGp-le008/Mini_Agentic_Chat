import os
import sys
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import HumanMessage

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

chat_model = ChatHuggingFace(llm=llm)

print("--- AI Chat Initialized. Type 'exit' or 'quit' to end. ---", flush=True)

while True:
    try:
        user_input = input("\nYou: ")
        
        if user_input.strip().lower() in ['exit', 'quit']:
            print("Goodbye!")
            break
            
        if not user_input.strip():
            continue

        messages = [HumanMessage(content=user_input)]
        response = chat_model.invoke(messages)
        print(f"AI: {response.content}")
        
    except KeyboardInterrupt:
        print("\nGoodbye!")
        break
