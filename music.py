from transformers import MusicgenForConditionalGeneration
import torch
import scipy
from dotenv import dotenv_values
from openai import OpenAI
from transformers import AutoProcessor

config = dotenv_values('.env')
api_key = config["OPENAI_API_KEY"]

'''
Given a section of text, ask GPT to generate a Musicgen prompt that would be used to generate a music that best capture the auditory elements of that section.
'''

def generate_music_prompt(text: str) -> str:
    system_prompt = "We are making an app where users can input the text of a book and get visual depictions, real-time narration, and live music throughout each chapter of the book to give a new visual experience to readers."
    user_prompt = f"Given a section of the text from the book, generate an effective SINGLE-LINE prompt that would be used as input to the MusicGen model. This should generate music that's good to play in the background as the images display. \n\n"
    user_prompt += text

    client = OpenAI(api_key = api_key)
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
    print(completion.choices[0].message.content)

    return completion.choices[0].message.content

def generate_music(text: str, seconds) -> str:
    prompt = generate_music_prompt(text)

    model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.to(device)

    processor = AutoProcessor.from_pretrained("facebook/musicgen-small")

    inputs = processor(
        text=[text],
        padding=True,
        return_tensors="pt",
    )

    audio_values = model.generate(**inputs.to(device), do_sample=True, guidance_scale=3, max_new_tokens=(256 // 5) * seconds)

    sampling_rate = model.config.audio_encoder.sampling_rate
    
    scipy.io.wavfile.write("musicgen_out.wav", rate=sampling_rate, data=audio_values[0, 0].cpu().numpy())

    return "musicgen_out.wav"

section = "Six days into what should be the greatest two months of my life, and it’s turned into a nightmare. I don’t even know who’ll read this. I guess someone will find it eventually. Maybe a hundred years from now. For the record…I didn’t die on Sol 6. Certainly the rest of the crew thought I did, and I can’t blame them. Maybe there’ll be a day of national mourning for me, and my Wikipedia page will say, “Mark Watney is the only human being to have died on Mars.”"

music = generate_music(section, 10)