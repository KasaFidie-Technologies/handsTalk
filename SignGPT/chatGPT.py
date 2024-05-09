
# import the OpenAI Python library for calling the OpenAI API
from openai import OpenAI
import os


def respond(word):
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "your api"))

    # Example OpenAI Python library request
    MODEL = "gpt-3.5-turbo"

    user_message = word
    joinSentence = " ".join(user_message)


    # example with a system message
    response = client.chat.completions.create(
        model=MODEL,
        # Use messages as prompt
        messages=[
            # {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": joinSentence},
        ],
        temperature=0,
    )

    res = response.choices[0].message.content

    return res
