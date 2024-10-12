# %%
from dotenv import dotenv_values
from openai import OpenAI
from io import BytesIO
import requests

config = dotenv_values('.env')
api_key = config["OPENAI_API_KEY"]

'''
Given a section of text, ask GPT to generate a DALLE-3 prompt that would be used to generate an image that best capture the visual elements of that section.
'''

def generate_image_prompt(text: str) -> str:
    system_prompt = "We are making an app where users can input the text of a book and get visual depictions, real-time narration, and live music throughout each chapter of the book to give a new visual experience to readers."
    user_prompt = f"Given a section of the text from the book, generate an effective prompt that would be used to generate an image that best capture the visual elements of that section. \n\n"
    user_prompt += text

    client = OpenAI(api_key=api_key)
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": user_prompt
            }
        ]
    )

    return completion.choices[0].message.content

def generate_image(text: str, model="dall-e-3", size="1024x1024") -> str:
    prompt = generate_image_prompt(text)
    client = OpenAI(api_key=api_key)
    response = client.images.generate(
        model=model,
        prompt=prompt,
        size=size,
        quality="standard",
        n=1,
    )
    url = response.data[0].url
    image = requests.get(url)
    img = BytesIO(image.content)

    return img


section = "Six days into what should be the greatest two months of my life, and it’s turned into a nightmare. I don’t even know who’ll read this. I guess someone will find it eventually. Maybe a hundred years from now. For the record…I didn’t die on Sol 6. Certainly the rest of the crew thought I did, and I can’t blame them. Maybe there’ll be a day of national mourning for me, and my Wikipedia page will say, “Mark Watney is the only human being to have died on Mars.”"

img = generate_image(section)

# %%
from PIL import Image
Image.open(img)


# %%
