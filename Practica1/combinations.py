#!/usr/bin/env python
# coding: utf-8

# In[21]:


import pandas as pd
import wordfreq


# Returns the number of appearances of any char from string in word.
def count_special_characters(word: str, string: str) -> int:
    num_special_characters=0
    for char in word:
        if char in string:
            num_special_characters+=1
    return num_special_characters

# Function that creates all the pairs of 2 vowels.
def diptongos() -> list:
    list=[]
    vowels = "aeiouàèìòùáéíóúäëïöüâêîôû"
    for i in vowels:
        for j in vowels:
            list.append(i+j)
    return list

# Length
dict = {
        'vow':'aeiou', 
        'acc':'àèìòùáéíóú', 
        'accl':'àèìòù', 
        'accr':'áéíóú', 
        'die':'äëïöü',
        'cir':'âêîôû', 
        'ñ':'ñ', 
        'ç':'ç', 
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

# Dictionary containing feature name, and list of the leetter groups in that language.
groups = {
        "pairs_eng": ["sh", "th", "ch", "ck", "ph", "ng", "qu", "tr", "st", "wh", "tr"],
        "pairs_cat": ["ny", "tx", "sc", "nc", "rc", "ll", "nc", "pc", "pr", "br", "fr", "ts", "ix",  "nd", "pr", "bl"],
        "pairs_esp": ["nd", "nt", "ch", "rr", "ll", "qu", "gu", "nc", "mb", "pr"],
        "pairs_ger": ["tch", "ck", "ng", "qu", "tz", "ss", "st", "sp", "str", "sch"],
        "pairs_por": ["tch", "lh", "nh", "qu", "sc", "rr", "nc", "gu", "lm", "rm"],
        "pairs_pol": ["ch", "dz", "dł", "di", "rz", "sz", " sc", "ed", "id"],
        "pairs_ita": ["ch", "gl", "gn", "sc", "qu", "scl", "ch", "ci", "gli", "gn", "io", "la", "leu", "ii", "io", "ne"],
        "pairs_swe": ["ch", "ck", "cid", "dt", "gg", "ll", "ng", "sk", "st", "tt"],
        "pairs_fre": ["ch", "che", "eau", "ent", "es", "ette", "eur", "iau", "ie", "in"],
        "pairs_rus": ["бл", "вл", "гл", "дл", "жл", "зл", "кл", "лл", "мл", "нл", "пл", "рл", "сл", "тл", "фл", "хл", "цл", "чл", "шл", "щл"],
        "diptongos": diptongos()
}

# Dictionary containing feature name, and list of the common prefixes in that language.
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

# Dictionary containing feature name, and list of the common suffixes in that language.
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

language_codes = ['en', 'es', 'ca', 'it', 'fr', 'pt', 'de', 'sv', 'pl', 'ru']


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


def count_group(word: str, groups: list) -> int:
    num_groups=0
    for group in groups:
        if group in word:
            num_groups+=1
    return num_groups


def A(df_AUX):
    df_AUX['len'] = df_AUX['word'].str.len()
    for column in dict:
        df_AUX[column] = df_AUX['word'].apply(lambda row: count_special_characters(row, dict[column]))
    return df_AUX

def B(df_AUX):
    # For every entry in the dictionary we create that column using the function above.
    for column in groups:
            df_AUX[column] = df_AUX['word'].apply(lambda row: count_group(row, groups[column]))
    return df_AUX

def C(df_AUX):
    # For every entry in the dictionary we create that column using the function above.
    for column in common_prefixes:
            df_AUX[column] = df_AUX['word'].apply(lambda row: has_prefix(row, common_prefixes[column]))
    # For every entry in the dictionary we create that column using the function above.
    for column in common_suffixes:
        df_AUX[column] = df_AUX['word'].apply(lambda row: has_suffix(row, common_suffixes[column]))

    return df_AUX

def D(df_AUX):

    for code in language_codes:
            df_AUX[code+'_zipf'] = df_AUX['word'].apply(lambda row: wordfreq.zipf_frequency(row, code))
    return df_AUX


# Read Data and convert to lowercase strings
df = pd.read_csv('data/data3.csv', sep=',')
df = df.apply(lambda x: x.astype(str).str.lower())

# concat all languages into one Dataframe
dfs = list()
for lang in df.columns:
    df_lang = pd.DataFrame(df[lang])
    df_lang['lang'] = lang[:3]
    df_lang = df_lang.rename(columns={lang: 'word'})
    dfs.append(df_lang)
df = pd.concat(dfs, ignore_index=True)
df.head()

df['lang'] = df['lang'].astype('category').cat.codes


combinations = [("A", A), ("B", B), ("C", C), ("D", D), 
                ("A", A, "B", B), ("A", A, "C", C), ("A", A, "D", D), 
                ("B", B, "C", C), ("B", B, "D", D), ("C", C, "D", D), 
                ("A", A, "B", B, "C", C), ("A", A, "B", B, "D", D), 
                ("A", A, "C", C, "D", D), ("B", B, "C", C, "D", D), 
                ("A", A, "B", B, "C", C, "D", D)]

# Call the functions for each combination
for combination in combinations:
    df2 = df.copy()
    str = ""
    for i in range(0, len(combination), 2):        
        df2 = combination[i+1](df2)
        str += combination[i]
    print(str)
    df2.to_csv('data/comb/'+str+'.csv')
