import pandas as pd
import wordfreq

# We create a dictionary containing the feature name and the special characters we want to check.
dict = {
    'vow': 'aeiou',
    'acc': 'àèìòùáéíóú',
    'accl': 'àèìòù',
    'accr': 'áéíóú',
    'die': 'äëïöü',
    'cir': 'âêîôû',
    'ñ': 'ñ',
    'ç': 'ç',
    'ale': 'ß',
    'rus': 'бвгджзийклмнпрстфцчшщъыьэюя',
    'pol': 'ąćęłńóśźż',
    'por': 'ãõ',
    'sue': 'åäö',
    'esp': 'áéíóúü',
    'ita': 'àèéìíîòóùú',
    'fra': 'àâæçéèêëîïôœùûüÿ',
    'ger': 'äöüß',
    'cat': 'àèéíïòóúüç',
    'num_words': ' ',
    "apos": "'",
    "hyph": "-",
    "rares": "kqwxyz"
}

common_prefixes = {
    'pre_eng': ["anti", "be", "de", "dis", "en", "ex", "im", "in", "non", "pre", "re", "un"],
    'pre_esp': ["anti", "auto", "contra", "des", "en", "ex", "in", "inter", "pre", "re", "sub", "trans"],
    'pre_cat': ["anti", "ab", "avant", "arxi", "dia", "hemi", "auto", "contra", "des", "en", "ex", "in", "inter", "pre", "re", "sub", "trans"],
    'pre_ita': ["auto", "dis", "en", "ex", "im", "in", "ir", "mal", "per", "pre", "pro", "re", "sott", "sotto", "tran", "ab"],
    'pre_fra': ["anti", "auto", "co", "con", "contre", "de", "des", "en", "ex", "in", "inter", "mal", "pre", "pro", "re", "sub", "sur"],
    'pre_por': ["auto", "co", "contra", "des", "em", "en", "ex", "in", "inter", "pre", "pro", "re", "sub"],
    'pre_ale': ["be", "ein", "ent", "er", "ge", "hin", "ver", "zer"],
    'pre_sue': ["be", "för", "in", "om", "över", "under"],
    'pre_pol': ["przed", "nad", "na", "pod", "z", "w"],
    'pre_rus': ["анти", "без", "в", "во", "до", "за", "из", "над", "пере", "под", "по", "пре", "раз"]
}

common_suffixes = {
    'suf_eng': ["able", "al", "ation", "er", "est", "ful", "ing", "ion", "ive", "less", "ly", "ness", "ous", "s", "y"],
    'suf_esp': ["ado", "ador", "aje", "anza", "ar", "ario", "ero", "iente", "illa", "ina", "izar", "oso", "ón", "udo", "er", "ir"],
    'suf_cat': ["ana", "aca", "ada", "al", "am", "ador", "tge", "isme", "nça", "ar", "ista", "istic", "mente", "ment", "ina", "tzar", "nça" "on", "um", "ut", "uda", "er", "ir", "re"],
    'suf_ita': ["abile", "are", "ario", "atore", "azione", "ente", "evole", "ificare", "ivo", "izzare", "ore", "orente", "orevole", "oso", "ura"],
    'suf_fra': ["age", "aille", "ance", "eau", "eux", "eur", "eurse", "ie", "iment", "ion", "ique", "isme", "iste", "ition", "ive", "oire", "ure", "y"],
    'suf_por': ["al", "ão", "ar", "ês", "ência", "eza", "ia", "ício", "imento", "ir", "or", "oso", "ura"],
    'suf_ale': ["bar", "e", "ei", "er", "heit", "ich", "ig", "in", "keit", "lich", "ling", "sam", "schaft", "ung"],
    'suf_sue': ["ande", "are", "bar", "dom", "else", "en", "eri", "het", "ing", "isk", "itet", "lig", "lighet", "ning", "ningen", "ningar", "ningen"],
    'suf_pol': ["acja", "ać", "anie", "eć", "enie", "enie", "enie", "enie", "enie", "enie", "enie", "enie", "enie", "enie", "enie", "enie"],
    'suf_rus': ["больше", "енький", "ик", "ичка", "ок", "онок", "ушко", "ца", "чек", "шка", "шко", "ящик", "ец", "ин", "ист", "ник", "овец", "щик", "ёнок", "ь"]
}

language_codes = ['en', 'de', 'ca', 'es', 'fr', 'it', 'pl', 'pt', 'ru', 'sv']


def count_special_characters(word: str, string: str) -> int:
    num_special_characters = 0
    for char in word:
        if char in string:
            num_special_characters += 1
    return num_special_characters

# Returns if word starts with a prefix from prefixes list.
def has_prefix(word: str, prefixes: list) -> int:
    for prefix in prefixes:
        if word.startswith(prefix):
            return 1
    return 0

# Returns if word ends with a prefix from suffixes list.
def has_suffix(word: str, suffixes: list) -> int:
    for suffix in suffixes:
        if word.endswith(suffix):
            return 1
    return 0

def process_phrase(phrase):
    phrase = phrase.split()
    df_phrase = pd.DataFrame(phrase, columns=["word"])
    df_phrase["lang"] = "cat"

    # A
    df_phrase['len'] = df_phrase['word'].str.len()
    # For every entry in the dictionary we create that column using the function above.
    for column in dict:
        df_phrase[column] = df_phrase['word'].apply(
            lambda row: count_special_characters(row, dict[column]))

    # C
    for column in common_prefixes:
        df_phrase[column] = df_phrase['word'].apply(
            lambda row: has_prefix(row, common_prefixes[column]))
    for column in common_suffixes:
        df_phrase[column] = df_phrase['word'].apply(
            lambda row: has_suffix(row, common_suffixes[column]))

    # D
    for code in language_codes:
        df_phrase[code+'_zipf'] = df_phrase['word'].apply(
            lambda row: wordfreq.zipf_frequency(row, code))

    df_phrase['lang'] = df_phrase['lang'].astype('category').cat.codes
    return df_phrase
