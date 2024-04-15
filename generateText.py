# import the OpenAI Python library for calling the OpenAI API
from openai import OpenAI
import os

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "<your OpenAI API key if not set as env var>"))

# Example OpenAI Python library request
MODEL = "gpt-3.5-turbo"
# Note you can change the model to your prefered model
# response = client.chat.completions.create(
#     model=MODEL,
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": "Knock knock."},
#         {"role": "assistant", "content": "Who's there?"},
#         {"role": "user", "content": "Orange."},
#     ],
#     temperature=0,
# )

# response.choices[0].message.content

# example with a system message
response = client.chat.completions.create(
    model=MODEL,
    # Use messages as prompt
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain asynchronous programming in the style of the pirate Blackbeard."},
    ],
    temperature=0,
)

print(response.choices[0].message.content)

