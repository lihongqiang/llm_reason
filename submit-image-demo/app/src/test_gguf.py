import ollama
response = ollama.chat(model='llama3.1-70b', messages=[
  {
    'role': 'user',
    'content': 'Why is the sky blue?',
  },
])
print(response)