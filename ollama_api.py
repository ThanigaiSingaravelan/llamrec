import requests

def check_ollama_connection():
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            print("Ollama API connected successfully!")
            return True
        else:
            print("Ollama API connection failed!")
            return False
    except:
        print("Cannot connect to Ollama API. Make sure Ollama is running!")
        return False

def ai_generate(prompt):
    response = requests.post("http://localhost:11434/api/generate",
                             json={"model": "llama3:8b", "prompt": prompt, "stream": False})
    return response.json()["response"]

if not check_ollama_connection():
    exit()


print("Start chatting! (type 'quit' to exit)")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'quit':
        break

    ai_response = ai_generate(user_input)
    print("AI:", ai_response)