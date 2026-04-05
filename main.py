from openai import OpenAI

# Load the API key from the .env file
load_dotenv(dotenv_path="../.env")

client = OpenAI()

# This is the "personality" of the AI — try changing it!
system_prompt = "You are a helpful assistant."

# This is what you're asking — change this to whatever you want
user_prompt = "Explain what a large language model is in 3 sentences."

print(f"You: {user_prompt}\n")
print("AI: ", end="", flush=True)

# Call the model and stream the response (so it prints word by word)
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ],
    temperature=0.7,
    stream=True,
)

for chunk in response:
    content = chunk.choices[0].delta.content
    if content:
        print(content, end="", flush=True)

print()  # newline at the end
