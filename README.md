# Visualizing-Novel-Characters
An end to end NLP and CV pipeline to generate textually accurate depictions (images) of characters from Novels, Story books etc.

Install packages
1. neuralcoref
2. transformers
3. spacy
4. spacy (en)
5. stanza

1. Clone this repo.
2. If you want to use a new book
    1. Upload .txt file to "new_books" directory.
    2. Run `character_list.py` with the appropriate paths set.
    3. Run `pipeline.py` with the appropriate paths set.
    4. Check results in "results" dir.
3. If you want to experiment with existing books
    1. Run `pipeline.py` and sit back.
    2. Check results in "results" dir.

NOTE: This experiment will take a while to be sure to grab popcorn and your fav soda ;).

## Each File Description
1. All .json files - These all represent character list, descriptive filter text, abstractive summaries. They are spread out in "book_descs", "book_summaries" folders respectively.
2. All .txt files in "new_books" directory - these are the books (data) for the model to parse and generate images.
3. `character_list.py` - this file extracts all the characters from a given book and saves them as .json.
4. `checks.py` - has all the helper functions to check and read the .json files for each specific tasks.
5. `coref_dependency.py` - has the methods defined for coreference resolution, descriptive filtering, character_extraction and all descriptions extraction.
6. `ner_test.py` - has the method for NER tagging
7. `stable_diffusion.py` - defines the stable diffusion models.
8. `summarize.py` - defines the models (longformer) for abstractive summarization.
7. `pipeline.py` - this is the main file which combines all of the processes together.

### Fine Tuning files
1. `gutenbergData.py` - fine tunes the summarizer on LISCU dataset.
2. `token_classification.py` - fine tunes the NER and POS tagger models on CONLL datasets.

### Datasets
1. Refer all files named - "lisu_train", "liscu_val", "liscu_test" .jsonl files.

Please refer to the comments in the files to know exactly how the pipeline works. Specifically check the files `pipeline.py` for the entire process.

GITHUB LINK: https://github.com/Sudhendra/Visualizing-Novel-Characters
