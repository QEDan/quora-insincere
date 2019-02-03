import re
from nltk.corpus import stopwords


def remove_stops(sentence):
    """ Should apply only after tokenization """
    stop = set(stopwords.words('english'))
    filtered = list()
    for w in sentence.split(" "):
        if w not in stop:
            filtered.append(w)
    return " ".join(filtered)


def clean_contractions(text):
    contraction_mapping = {"ain't": "is not", "aren't": "are not", "can't": "cannot", "'cause": "because",
                           "could've": "could have", "couldn't": "could not", "didn't": "did not",
                           "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not",
                           "haven't": "have not", "he'd": "he would", "he'll": "he will", "he's": "he is",
                           "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",
                           "I'd": "i would", "I'd've": "I would have", "I'll": "i will", "I'll've": "i will have",
                           "I'm": "i am", "I've": "i have", "i'd": "i would", "i'd've": "i would have",
                           "i'll": "i will", "i'll've": "i will have", "i'm": "i am", "i've": "i have",
                           "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will",
                           "it'll've": "it will have", "it's": "it is", "let's": "let us", "ma'am": "madam",
                           "mayn't": "may not", "might've": "might have", "mightn't": "might not",
                           "mightn't've": "might not have", "must've": "must have", "mustn't": "must not",
                           "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have",
                           "o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have",
                           "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",
                           "she'd": "she would", "she'd've": "she would have", "she'll": "she will",
                           "she'll've": "she will have", "she's": "she is", "should've": "should have",
                           "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have",
                           "so's": "so as", "this's": "this is", "that'd": "that would",
                           "that'd've": "that would have", "that's": "that is", "there'd": "there would",
                           "there'd've": "there would have", "there's": "there is", "here's": "here is",
                           "they'd": "they would", "they'd've": "they would have", "they'll": "they will",
                           "they'll've": "they will have", "they're": "they are", "they've": "they have",
                           "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have",
                           "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have",
                           "weren't": "were not", "what'll": "what will", "what'll've": "what will have",
                           "what're": "what are", "what's": "what is", "what've": "what have", "when's": "when is",
                           "when've": "when have", "where'd": "where did", "where's": "where is",
                           "where've": "where have", "who'll": "who will", "who'll've": "who will have",
                           "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have",
                           "will've": "will have", "won't": "will not", "won't've": "will not have",
                           "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have",
                           "y'all": "you all", "y'all'd": "you all would", "y'all'd've": "you all would have",
                           "y'all're": "you all are", "y'all've": "you all have", "you'd": "you would",
                           "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",
                           "you're": "you are", "you've": "you have"}
    specials = ["’", "‘", "´", "`"]
    for s in specials:
        text = text.replace(s, "'")
    text = ' '.join([contraction_mapping[t] if t in contraction_mapping else t for t in text.split(" ")])
    return text


def clean_specials(text):
    punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'
    puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=',
              '#', '*', '+', '\\', '•', '~', '@', '£',
              '·', '_', '{', '}', '©', '^', '®', '`', '<', '→', '°', '€', '™', '›', '♥', '←', '×', '§', '″', '′',
              'Â', '█', '½', 'à', '…',
              '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║',
              '―', '¥', '▓', '—', '‹', '─',
              '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è',
              '¸', '¾', 'Ã', '⋅', '‘', '∞',
              '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï',
              'Ø', '¹', '≤', '‡', '√', ]
    punct_mapping = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ", "×": "x", "²": "2",
                     "—": "-", "–": "-", "’": "'", "_": "-", "`": "'", '“': '"', '”': '"', '“': '"', "£": "e",
                     '∞': 'infinity', 'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-',
                     'β': 'beta', '∅': '', '³': '3', 'π': 'pi', }
    for p in punct_mapping:
        text = text.replace(p, punct_mapping[p])
    for p in set(list(punct) + puncts) - set(punct_mapping.keys()):
        text = text.replace(p, f' {p} ')

    specials = {'\u200b': ' ', '…': ' ... ', '\ufeff': '', 'करना': '',
                'है': ''}
    for s in specials:
        text = text.replace(s, specials[s])
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
