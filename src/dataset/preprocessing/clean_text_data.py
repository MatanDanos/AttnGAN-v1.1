import os
import re
from string import printable

from dataset.birds_dataset import BirdsDataset

from nltk.tokenize import sent_tokenize, word_tokenize


def replace_unknown_characters(bad_string, replace_char=" "):
    """ A function that will replace the bad characters that appear in some of the text files
    with a single whitespace (by default)
    If this doesn't work, the loop below will, but it's slower..
    
    ! ------> slow loop that will surely work:
    unknown_chars_indices = [i for i, char in enumerate(bad_string) if char not in printable]
    new_string_chars = []
    for i, char in enumerate(bad_string):
        if i in unknown_chars_indices:
            new_string_chars.append(" ")
    else:
        new_string_chars.append(bad_string[i])
    s = "".join(new_string_chars)
    collapse_whitespaces_regex = re.compile(r'\W+')
    result = collapse_whitespaces_regex.sub(' ', s) """
    
    collapse_unknown_chars_regex = re.compile(r'[ן¿½ן¿]+')
    return collapse_unknown_chars_regex.sub(replace_char, bad_string)


def fix_invalid_chars(base_dataset_path, badfiles_filepath, replace_char=" "):
    """ This will fix the files that contain text with invalid characeters
    The files that contain the bad characters should be listed in the `badfiles_filepath` parameter
    Parameters:
    @basedataset_path (string): path to the base of the preprocessed data
    @badfiles_filepath (string): path to a file that each line in the file is a path in 
    dataset/text/* folder to a file with invalid characeters.
    @repalce_char (string): a single char that will repalce all bad characters. 
    """
    with open(badfiles_filepath, "r") as f:
        for line in f:
            line = line.strip("\n")
            filepath = os.path.join(base_dataset_path, "text", line + ".txt")
            new_lines = []
            with open(filepath, "r") as f2:
                bad_lines = f2.readlines()
                for bad_line in bad_lines:
                    new_lines.append(replace_unknown_characters(bad_line, replace_char))
            os.remove(filepath)
            with open(filepath, "w") as f2:
                for good_line in new_lines:
                    f2.write(good_line)
 

def save_bird_vocabulary(birds_dataset_path, vocab_filename):
    dataset = BirdsDataset(birds_dataset_path)
    with open(vocab_filename, "w") as f:
        for word in dataset.vocabulary.counts.keys():
            f.write(word + "\n")
    