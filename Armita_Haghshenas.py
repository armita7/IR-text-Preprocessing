import re
import json

# Standard English Stop Word List (with Offline Support)
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
    from nltk.stem import PorterStemmer, SnowballStemmer, WordNetLemmatizer
    try:
        nltk.download('stopwords', quiet=True)
        STOP_WORDS = set(stopwords.words('english'))
    except Exception:
        STOP_WORDS = ENGLISH_STOPWORDS
except ImportError:
    STOP_WORDS = ENGLISH_STOPWORDS

print("Library and stopword settings have been loaded.")

class TextPreprocessor:
    def __init__(self, mode="porter"):
        self.stop_words = STOP_WORDS
        self.mode = mode.lower()
        self.porter = PorterStemmer()
        try:
            self.snowball = SnowballStemmer('english')
        except Exception:
            self.snowball = self.porter
        try:
            self.lemmatizer = WordNetLemmatizer()
        except Exception:
            self.lemmatizer = None

    def tokenize(self, text: str) -> list:
        text = re.sub(r"['’]s\b", "", text, flags=re.IGNORECASE)
        pattern = r"(?:[A-Za-z]\.){2,}|[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*"
        return re.findall(pattern, text)

    def normalize(self, tokens: list) -> list:
        normalized = []
        for token in tokens:
            tok = token.lower()
            if re.match(r"^(?:[a-z]\.){2,}$", tok):
                tok = tok.replace(".", "")
            normalized.append(tok)
        return normalized

    def remove_stopwords(self, tokens: list) -> list:
        return [tok for tok in tokens if tok not in self.stop_words]

    def stem(self, tokens: list) -> list:
        stemmed = []
        for tok in tokens:
            if self.mode == "snowball":
                stemmed.append(self.snowball.stem(tok))
            elif self.mode == "lemmatizer" and self.lemmatizer:
                try:
                    stemmed.append(self.lemmatizer.lemmatize(tok))
                except Exception:
                    stemmed.append(self.porter.stem(tok))
            else:
                stemmed.append(self.porter.stem(tok))
        return stemmed

    def process_with_details(self, text: str) -> dict:
        tokens = self.tokenize(text)
        normalized = self.normalize(tokens)
        no_stopwords = self.remove_stopwords(normalized)
        stemmed_terms = self.stem(no_stopwords)
        return {
            "raw_text": text,
            "1_tokens": tokens,
            "2_normalized": normalized,
            "3_no_stopwords": no_stopwords,
            "4_stemmed_terms": stemmed_terms,
            "unique_terms": sorted(list(set(stemmed_terms)))
        }

    def process(self, text: str) -> list:
        return self.process_with_details(text)["4_stemmed_terms"]

print("The TextPreprocessor class is ready.")

def evaluate_output(program_output: list, gold_standard_input) -> dict:
    if isinstance(gold_standard_input, str):
        g_list = [w.strip().lower() for w in re.split(r'[,;\s]+', gold_standard_input) if w.strip()]
        G = set(g_list)
    elif isinstance(gold_standard_input, (list, tuple, set)):
        G = set([str(x).lower().strip() for x in gold_standard_input])
    else:
        G = set()

    O = set(program_output)
    intersection = O.intersection(G)

    precision = len(intersection) / len(O) if len(O) > 0 else 0.0
    recall = len(intersection) / len(G) if len(G) > 0 else 0.0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "O": sorted(list(O)),
        "G": sorted(list(G)),
        "intersection": sorted(list(intersection)),
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "extra_terms": sorted(list(O - G)),
        "missed_terms": sorted(list(G - O))
    }

print("The evaluate_output function has been defined.")


SAMPLE_TEXTS = {
    "text1": "Friends, Romans, countrymen, lend me your ears;",
    "text2": "John's new U.S.A. laptop is a state-of-the-art device. The price is high, but the quality is amazing!",
    "text3": "The authorization for this authorized project is pending. He authorizes the payments regularly."
}

GOLD_STANDARDS = {
    "text1": {"friend", "roman", "countrymen", "lend", "ear"},
    "text2": {"john", "new", "usa", "laptop", "state-of-the-art", "devic", "price", "high", "qualiti", "amaz"},
    "text3": {"author", "project", "pend", "payment", "regularli"}
}

preprocessor = TextPreprocessor(mode="porter")
final_outputs = {}

print("=" * 66)
print("         General Preprocessing Pipeline Test on Project Sample Dataset         ")
print("=" * 66 + "\n")

for text_id, raw_text in SAMPLE_TEXTS.items():
    terms_sequence = preprocessor.process(raw_text)
    final_outputs[text_id] = terms_sequence
    eval_res = evaluate_output(terms_sequence, GOLD_STANDARDS[text_id])

    print(f"--- [ {text_id} ] ---")
    print(f"Input text: {raw_text}")
    print(f"Sequence of terms: {terms_sequence}")
    print(f"Program output set (O): {eval_res['O']}")
    print(f"Gold output set (G): {eval_res['G']}")
    print(f"Intersection (O ∩ G):           {eval_res['intersection']}")
    print(f"Precision:          {eval_res['precision']:.4f}")
    print(f"Recall:        {eval_res['recall']:.4f}")
    print(f"F1-Score:                  {eval_res['f1_score']:.4f}")
    print(f"Extra terms (O - G):   {eval_res['extra_terms']}")
    print(f"Missed terms (G - O):   {eval_res['missed_terms']}")
    print("-" * 66 + "\n")

with open("outputs.json", "w", encoding="utf-8") as f:
    json.dump(final_outputs, f, ensure_ascii=False, indent=2)
print("✓ The sequence of output terms has been saved in the outputs.json file.")

# @title Input English Text and Evaluate Precision / Recall
test_text = "Artificial intelligence and information retrieval systems are searching big datasets." # @param {type:"string"}
gold_terms_str = "artifici, intellig, inform, retriev, system, search, big, dataset" # @param {type:"string"}
stemming_mode = "lemmatizer" # @param ["porter", "snowball", "lemmatizer"]

p = TextPreprocessor(mode=stemming_mode)
det = p.process_with_details(test_text)

print("--- Step-by-Step Preprocessing Results ---")
print(f"1. Initial tokens:    {det['1_tokens']}")
print(f"2. Normalization:     {det['2_normalized']}")
print(f"3. Stopword removal:  {det['3_no_stopwords']}")
print(f"4. Stemming/Lemmatization: {det['4_stemmed_terms']}")
print(f"Output set (O):       {det['unique_terms']}")

res = evaluate_output(det['4_stemmed_terms'], gold_terms_str)
print("\n" + "=" * 50)
print("        Precision and Recall Evaluation Report        ")
print("=" * 50)
print(f"Program Output Set (O):    {res['O']}")
print(f"Gold Standard Set (G):     {res['G']}")
print(f"Intersection (O ∩ G):      {res['intersection']}")
print(f"Precision:                 {res['precision']:.4f}")
print(f"Recall:                    {res['recall']:.4f}")
print(f"F1-Score:                  {res['f1_score']:.4f}")
print(f"Extra Terms (O - G):       {res['extra_terms']}")
print(f"Missed Terms (G - O):      {res['missed_terms']}")
print("=" * 50)