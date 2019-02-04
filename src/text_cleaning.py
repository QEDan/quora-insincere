import re
from nltk.corpus import stopwords
import string

def remove_stops(sentence):
    """ Should apply only after tokenization """
    stop = set(stopwords.words('english'))
    filtered = list()
    for w in sentence.split(" "):
        if w not in stop:
            filtered.append(w)
    return " ".join(filtered)


def clean_apos(text):
    apos = ["’", "‘", "´", "`"]
    for s in apos:
        text = text.replace(s, "'")
    quotes = ['"', '"', '“', '”', '’']
    for q in quotes:
        text = text.replace(q, '"')
    return text


def clean_specials(text):
    # punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'
    # puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=',
    #           '#', '*', '+', '\\', '•', '~', '@', '£',
    #           '·', '_', '{', '}', '©', '^', '®', '`', '<', '→', '°', '€', '™', '›', '♥', '←', '×', '§', '″', '′',
    #           'Â', '█', '½', 'à', '…',
    #           '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║',
    #           '―', '¥', '▓', '—', '‹', '─',
    #           '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è',
    #           '¸', '¾', 'Ã', '⋅', '‘', '∞',
    #           '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï',
    #           'Ø', '¹', '≤', '‡', '√']
    punct_mapping = {"‘": "'", "´": "'", "′": "'", '″': '"', '¨': '"', 'ʻ': "'", '،': ',',
                     "—": "-", "–": "-", '−': '-', "’": "'", "`": "'", '“': '"', '”': '"',
                     '？': '?',
                     '∞': ' infinity ', 'θ': ' theta ', '÷': ' divide ', '℃': ' temperature ',
                     'ū': 'u', 'ú': 'u',
                     'á': 'a', 'ã': 'a', 'à': 'a', 'а': 'a', 'α': 'a', 'â': 'a', 'å': 'a', 'ā': 'a', 'ạ': 'a', 'ä': 'a',
                     'ă': 'a',
                     'ь': 'b', 'ß': 'b', 'в': 'b', 'β': 'b',
                     'ς': 'c', 'ç': 'c', 'ć': 'c', 'с': 'c',
                     'ë': 'e', 'е': 'e', 'é': 'e', 'ê': 'e', 'è': 'e',
                     'ğ': 'g',
                     'н': 'h', 'ḥ': 'h',
                     'í': 'i', 'î': 'i', 'ì': 'i', 'i̇': 'i', '¡': 'i',
                     'η': 'n',
                     'о': 'o', 'ô': 'o', 'ó': 'o', 'ο': 'o', 'ō': 'o',
                     'ρ': 'p',
                     'ș': 's', 'ѕ': 's', 'ş': 's',
                     'т': 't',
                     'ü': 'u', 'û': 'u',
                     'υ': 'v', 'ν': 'v',
                     'ω': 'w',
                     '×': 'x',
                     'ý': 'y', 'у': 'y',
                     'ž': 'z',
                     '¿': '?',
                     '½': ' 0.5 ', '¼': ' 0.25 ', 'π': ' pi ', '卐': ' nazi symbol '}
    for p in punct_mapping.keys():
        text = text.replace(p, punct_mapping[p])
    # for p in set(list(punct) + puncts) - set(punct_mapping.keys()):
    #     text = text.replace(p, f' {p} ')

    # specials = {'\u200b': ' ', '…': ' ... ', '\ufeff': '', 'करना': '',
    #             'है': ''}
    # for s in specials:
    #     text = text.replace(s, specials[s])
    return text


def clean_spelling(text, case_sensitive=False):
    misspell_dict = {'colour': 'color', 'centre': 'center', 'favourite': 'favorite', 'travelling': 'traveling',
                     'counselling': 'counseling', 'theatre': 'theater', 'cancelled': 'canceled', 'labour': 'labor',
                     'organisation': 'organization', 'wwii': 'world war 2', 'citicise': 'criticize',
                     'youtu ': 'youtube ', 'Qoura': 'Quora', 'sallary': 'salary', 'Whta': 'What',
                     'narcisist': 'narcissist', 'howdo': 'how do', 'whatare': 'what are', 'howcan': 'how can',
                     'howmuch': 'how much', 'howmany': 'how many', 'whydo': 'why do', 'doI': 'do I',
                     'theBest': 'the best', 'howdoes': 'how does', 'mastrubation': 'masturbation',
                     'mastrubate': 'masturbate', "mastrubating": 'masturbating', 'pennis': 'penis',
                     'Etherium': 'Ethereum', 'narcissit': 'narcissist', 'bigdata': 'big data', '2k17': '2017',
                     '2k18': '2018', 'qouta': 'quota', 'exboyfriend': 'ex boyfriend', 'airhostess': 'air hostess',
                     "whst": 'what', 'watsapp': 'whatsapp', 'demonitisation': 'demonetization',
                     'demonitization': 'demonetization', 'demonetisation': 'demonetization'}
    for word in misspell_dict.keys():
        if case_sensitive:
            text = text.replace(word, misspell_dict[word])
        else:
            re_insensitive = re.compile(re.escape(word), re.IGNORECASE)
            text = re_insensitive.sub(misspell_dict[word], text)
    return text


def clean_acronyms(text, case_sensitive=False):
    acronym_dict = {'upsc': 'union public service commission',
                    'aiims': 'all india institute of medical sciences',
                    'cgl': 'graduate level examination',
                    'icse': 'indian school certificate exam',
                    'iiit': 'indian institute of information technology',
                    'cgpa': 'cumulative grade point average',
                    'ielts': 'international english language training system',
                    'ncert': 'national council of education research training',
                    'isro': 'indian space research organization',
                    'clat': 'common law admission test',
                    'ibps': 'institute of banking personnel selection',
                    'iiser': 'indian institute of science education and research',
                    'iisc': 'indian institute of science',
                    'iims': 'indian institutes of management',
                    'cpec': 'china pakistan economic corridor'

                    }
    for word in acronym_dict.keys():
        if case_sensitive:
            text = text.replace(word, acronym_dict[word])
        else:
            re_insensitive = re.compile(re.escape(word), re.IGNORECASE)
            text = re_insensitive.sub(acronym_dict[word], text)
    return text


def clean_non_dictionary(text, case_sensitive=False):
    replace_dict = {'quorans': 'users',
                    'quoran': 'user',
                    'jio': 'phone manufacturer',
                    'manipal': 'city',
                    'bitsat': 'exam',
                    'mtech': 'technical university',
                    'pilani': 'town',
                    'bhu': 'university',
                    'h1b': 'visa',
                    'redmi': 'phone manufacturer',
                    'nift': 'university',
                    'kvpy': 'exam',
                    'thanos': 'comic villain',
                    'paytm': 'payment system',
                    'comedk': 'medical consortium',
                    'accenture': 'management consulting company',
                    'llb': 'bachelor of laws',
                    'ignou': 'university',
                    'dtu': 'university',
                    'aadhar': 'social number',
                    'lenovo': 'computer manufacturer',
                    'gmat': 'exam',
                    'kiit': 'institute of technology',
                    'shopify': 'music streaming',
                    'fitjee': 'exam',
                    'kejriwal': 'politician',
                    'wbjee': 'exam',
                    'pgdm': 'master of business administration',
                    'trudeau': 'politician',
                    'nri': 'research institute',
                    'deloitte': 'accounting company',
                    'jinping': 'politician',
                    'bcom': 'bachelor of commerce',
                    'mcom': 'masters of commerce',
                    'virat': 'athlete',
                    'kcet': 'television network',
                    'wipro': 'information technology company',
                    'articleship': 'internship',
                    'comey': 'law enforcement director',
                    'jnu': 'university',
                    'acca': 'chartered accountants',
                    'aakash': 'phone manufacturer',
                    'brexit': 'british succession',
                    'crypto': 'digital currency',
                    'cryptocurrency': 'digital currency',
                    'cryptocurrencies': 'digital currencies',
                    'etherium': 'digital currency',
                    'bitcoin': 'digital currency',
                    'viteee': 'exam',
                    'iocl': 'indian oil company',
                    'nmims': 'management school',
                    'rohingya': 'myanmar people',
                    'fortnite': 'videogame',
                    'upes': 'university',
                    'nsit': 'university',
                    'coinbase': 'digital currency exchange'
                    }
    for word in replace_dict.keys():
        if case_sensitive:
            text = text.replace(word, replace_dict[word])
        else:
            re_insensitive = re.compile(re.escape(word), re.IGNORECASE)
            text = re_insensitive.sub(replace_dict[word], text)
    return text


def clean_numbers(text, min_magnitude=2, max_magnitude=10):
    for n in range(min_magnitude, max_magnitude):
        text = re.sub('[0-9]{' + str(n) + '}', '#' * n, text)
    return text
