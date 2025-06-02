from openai import OpenAI
client = OpenAI()

response = client.responses.create(
    model="gpt-4",
    input = """Write a One-sentence bedtime story about a unicorn"""
)
print(response.output_text)