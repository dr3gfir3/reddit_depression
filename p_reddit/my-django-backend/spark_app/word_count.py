def get_word_count(text):
    words = text.split()
    word_count = len(words)
    return word_count

def get_word_count_from_file(file_path):
    with open(file_path, 'r') as file:
        text = file.read()
    return get_word_count(text)