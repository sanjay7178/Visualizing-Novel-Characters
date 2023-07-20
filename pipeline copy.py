import os
import json
import nltk
from tqdm import tqdm
from multiprocessing import Pool
from diffusers.models import AutoencoderKL
from diffusers import StableDiffusionPipeline
from ner_test import *
from coref_dependency import *
from summarize import *

# Set the number of processes for multiprocessing
num_processes = 16

# Define the extraction function
def extract_descriptions(sentence):
    character_desc = []
    normal_desc = []
    entities = ner_test(sentence, model_checkpoint_ner)
    try:
        if entities['PER'] == character or entities['MISC'] == character:
            descriptive_sentence = descriptive_filter(sentence)
            if descriptive_sentence:
                character_desc.append(descriptive_sentence)
            else:
                normal_desc.append(sentence)
    except:
        pass
    return character_desc, normal_desc

# Set the paths
path_to_chars = "Frankenstein_contents.json"
path_to_book = "new_books/Frankenstein.txt"
path_to_book_desc= "book_descs"
path_to_book_summary = "book_summaries"

# Read the book
with open(path_to_book, 'r', encoding='utf-8') as file:
    contents = file.read()

# Read the characters of the book
with open(path_to_chars, 'r') as j:
    character_contents = json.loads(j.read())

character_list = character_contents["Frankenstein"]
coref = coref_resolution(contents)
sentences = nltk.sent_tokenize(coref)

# Initialize the multiprocessing pool
pool = Pool(processes=num_processes)

for character in character_list:
    character_desc = []
    normal_desc = []
    print(f"Extracting descriptions for {character}")

    # Create a list of sentences for multiprocessing
    sentence_list = [(sentence,) for sentence in sentences]

    # Extract descriptions in parallel with tqdm progress bar
    results = []
    with tqdm(total=len(sentence_list), ncols=80) as pbar:
        for res in pool.starmap(extract_descriptions, sentence_list):
            results.append(res)
            pbar.update(1)

    # Collect the results
    for character_desc_batch, normal_desc_batch in results:
        character_desc.extend(character_desc_batch)
        normal_desc.extend(normal_desc_batch)

    full_desc = " ".join(character_desc)
    full_desc_normal = " ".join(normal_desc)

    with open(os.path.join(path_to_book_desc, str(character) + "_desc_.json"), 'w') as f:
        f.write(json.dumps(full_desc) + "\n")

    with open(os.path.join(path_to_book_desc, str(character) + "_normal_.json"), 'w') as f:
        f.write(json.dumps(full_desc_normal) + "\n")

print("All Descriptions Extracted!!")

# Close the multiprocessing pool
pool.close()
pool.join()


# Summarizer
print("Begin Abstractive Summarizing")
for char_desc in os.listdir(path_to_book_desc):
    char_path = os.path.join(path_to_book_desc, char_desc)
    description = get_character_summary(char_path)

    ARTICLE_TO_SUMMARIZE = description
    inputs = tokenizer.encode(ARTICLE_TO_SUMMARIZE, return_tensors="pt")

    # Global attention on the first token (cf. Beltagy et al. 2020)
    global_attention_mask = torch.zeros_like(inputs)
    global_attention_mask[:, 0] = 1

    # Generate Summary
    summary_ids = model.generate(inputs, global_attention_mask=global_attention_mask, num_beams=3, max_length=80)
    abstractive_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)

    # Dump the summary
    with open(os.path.join(path_to_book_summary, str(character)+"_normal_.json"), 'w') as f:
        f.write(json.dumps(full_desc_normal) + "\n")

# Text-to-Image Generation
print("Begin Text-to-Image Generation")

from diffusers.models import AutoencoderKL
from diffusers import StableDiffusionPipeline

model = "CompVis/stable-diffusion-v1-4"
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
pipe = StableDiffusionPipeline.from_pretrained(model, vae=vae)
pipe =  pipe.to("cuda")

for summary in os.listdir(path_to_book_summary):
    with open(os.path.join(path_to_book_summary, summary), 'r', encoding='utf-8') as file:
        # Read the contents of the file
        prompt = file.read()

    # Generate 5 images from runwayml
    # for i in range(5):
    #     runway_pipe(prompt).images[0].save(f"results/runawayml{i}.png")

    # # Generate 5 images from stability ai
    # for i in range(5):
    #     stability_pipe(prompt).images[0].save(f"results/stabilityAI{i}.png")

    for i in range(5):
        pipe(prompt).images[0].save(f"results/stabilityAI-vae{i}.png")
