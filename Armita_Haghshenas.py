# گام ۱: تنظیمات اولیه و کلمات ایست (Stopwords)
import re
import json

ENGLISH_STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd",
    'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers',
    'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
    'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been',
    'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if',
    'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
    'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out',
    'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
    'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
    'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should',
    "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't",
    'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't",
    'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't",
    'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"
}

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    try:
        nltk.download('stopwords', quiet=True)
        STOP_WORDS = set(stopwords.words('english'))
    except Exception:
        STOP_WORDS = ENGLISH_STOPWORDS
except ImportError:
    from nltk.stem import PorterStemmer
    STOP_WORDS = ENGLISH_STOPWORDS

print('✓ Stop words and the PorterStemmer algorithm have been loaded.')
# گام ۲: کلاس پیش‌پردازش عمومی متون بازیابی اطلاعات
class GeneralIRPreprocessor:
    def __init__(self, stop_words_set=None):
        self.stop_words = stop_words_set if stop_words_set is not None else STOP_WORDS
        self.stemmer = PorterStemmer()

    def tokenize(self, text: str) -> list:
        # توکن‌سازی عمومی استاندارد بدون قاعده خاص
        return re.findall(r'\b\w+\b', text)

    def normalize(self, tokens: list) -> list:
        # یکسان‌سازی حروف کوچک
        return [tok.lower() for tok in tokens]

    def remove_stopwords(self, tokens: list) -> list:
        # حذف کلمات ایست
        return [tok for tok in tokens if tok not in self.stop_words]

    def stem(self, tokens: list) -> list:
        # ریشه‌یابی عمومی پورتر
        return [self.stemmer.stem(tok) for tok in tokens]

    def process_with_details(self, text: str) -> dict:
        tokens = self.tokenize(text)
        normalized = self.normalize(tokens)
        no_stopwords = self.remove_stopwords(normalized)
        stemmed_terms = self.stem(no_stopwords)
        return {
            'raw_text': text,
            '1_tokens': tokens,
            '2_normalized': normalized,
            '3_no_stopwords': no_stopwords,
            '4_stemmed_terms': stemmed_terms,
            'unique_terms': sorted(list(set(stemmed_terms)))
        }

    def process(self, text: str) -> list:
        return self.process_with_details(text)['4_stemmed_terms']

print('✓ The GeneralIRPreprocessor  class is ready for use.')

# گام ۳: ماژول ارزیابی Precision و Recall
def evaluate_output(program_output: list, gold_standard_input) -> dict:
    if isinstance(gold_standard_input, str):
        g_list = [w.strip().lower() for w in re.split(r'[,\s]+', gold_standard_input) if w.strip()]
        G = set(g_list)
    elif isinstance(gold_standard_input, (list, tuple, set)):
        G = set([str(x).lower().strip() for x in gold_standard_input])
    else:
        G = set()

    O = set(program_output)
    intersection = O.intersection(G)

    precision = len(intersection) / len(O) if len(O) > 0 else 0.0
    recall = len(intersection) / len(G) if len(G) > 0 else 0.0

    return {
        'O': sorted(list(O)),
        'G': sorted(list(G)),
        'intersection': sorted(list(intersection)),
        'precision': precision,
        'recall': recall,
        'extra_terms': sorted(list(O - G)),
        'missed_terms': sorted(list(G - O))
    }

print('✓ The evaluation function evaluate_output is ready.')

SAMPLE_TEXTS = {
    'text1': 'Friends, Romans, countrymen, lend me your ears;',
    'text2': "John's new U.S.A. laptop is a state-of-the-art device. The price is high, but the quality is amazing!",
    'text3': 'The authorization for this authorized project is pending. He authorizes the payments regularly.'
}

GOLD_STANDARDS = {
    'text1': {'friend', 'roman', 'countrymen', 'lend', 'ear'},
    'text2': {'john', 'new', 'usa', 'laptop', 'state-of-the-art', 'devic', 'price', 'high', 'qualiti', 'amaz'},
    'text3': {'author', 'project', 'pend', 'payment', 'regularli'}
}

preprocessor = GeneralIRPreprocessor()
final_outputs = {}

print('==================================================================')
print('       General pipeline test without overfitting on project texts       ')
print('==================================================================\n')

for text_id, raw_text in SAMPLE_TEXTS.items():
    terms = preprocessor.process(raw_text)
    final_outputs[text_id] = terms
    res = evaluate_output(terms, GOLD_STANDARDS[text_id])

    print(f'--- [ {text_id} ] ---')
    print(f'Input text: {raw_text}')
    print(f'Output sequence: {terms}')
    print(f'Program Collection (O): {res["O"]}')
    print(f'Golden Collection  (G): {res["G"]}')
    print(f'intersection (O ∩ G):    {res["intersection"]}')
    print(f'Precision:          {res["precision"]:.4f}')
    print(f'Recall:             {res["recall"]:.4f}')
    print(f'extra_terms  (O - G): {res["extra_terms"]}')
    print(f'missed_terms (G - O): {res["missed_terms"]}')
    print('-' * 66 + '\n')

with open('outputs.json', 'w', encoding='utf-8') as f:
    json.dump(final_outputs, f, ensure_ascii=False, indent=2)
print('✓ The output sequence of terms has been saved in outputs.json.')


user_text = input('Please enter the new English text:\n> ').strip()
if user_text:
    details = preprocessor.process_with_details(user_text)
    print('\n--- Preprocessing steps ---')
    print(f'Initial Token: {details["1_tokens"]}')
    print(f'Normalization: {details["2_normalized"]}')
    print(f'Removal of stop words: {details["3_no_stopwords"]}')
    print(f'Root Cause Analysis: {details["4_stemmed_terms"]}')
    print(f'Program Output Set (O): {details["unique_terms"]}\n')

    gold_str = input('Enter the expected Gold Standard words (separated by spaces/commas) :\n> ').strip()
    if not gold_str:
        gold_str = details['unique_terms']

    eval_res = evaluate_output(details['4_stemmed_terms'], gold_str)

    print('\n' + '=' * 50)
    print('        Precision and Recall Evaluation Report        ')
    print('=' * 50)
    print(f'Program Output Set (O):    {eval_res["O"]}')
    print(f'Golden Output Collection  (G):    {eval_res["G"]}')
    print(f'Intersection of Sets (O ∩ G):    {eval_res["intersection"]}')
    print(f'Precision Metric (Precision):      {eval_res["precision"]:.4f}')
    print(f'Recall Metric    (Recall):  {eval_res["recall"]:.4f}')
    print(f'extra terms  (O - G):   {eval_res["extra_terms"]}')
    print(f'missed terms (G - O):   {eval_res["missed_terms"]}')
    print('=' * 50)