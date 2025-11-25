import pandas as pd
import io
import contextlib
import warnings
warnings.filterwarnings('ignore')

try:
    import torch
    import sacrebleu
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    from flask import Flask, render_template, request, jsonify
except ImportError as e:
    print(f"ERROR: Missing dependency - {e}")
    print("Please run: pip install flask transformers torch sacrebleu pandas sentencepiece protobuf")
    exit(1)

import os

print("="*60)
print("Hindi to Punjabi Translation System")
print("="*60)

# --- 1. Data Definitions and Loading (Dictionary) ---
print("\n[1/4] Creating dictionary...")
data_dict = {
    'Hindi': ['नमस्ते', 'तुम', 'कैसे', 'हो', 'मैं', 'ठीक', 'धन्यवाद', 'पानी', 'खाना', 'घर', 'तैयार', 'है'],
    'Punjabi': ['ਸਤ ਸ੍ਰੀ ਅਕਾਲ', 'ਤੁਸੀਂ', 'ਕਿਵੇਂ', 'ਹੋ', 'ਮੈਂ', 'ਠੀਕ', 'ਧੰਨਵਾਦ', 'ਪਾਣੀ', 'ਖਾਣਾ', 'ਘਰ', 'ਤਿਆਰ', 'ਹੈ']
}
df_dict = pd.DataFrame(data_dict)

csv_file_path_dict = 'hindi_punjabi_dictionary.csv'
df_dict.to_csv(csv_file_path_dict, index=False, encoding='utf-8')

# Load the dictionary from the CSV file
dictionary_df = pd.read_csv(csv_file_path_dict, encoding='utf-8')
translation_dict = dict(zip(dictionary_df['Hindi'], dictionary_df['Punjabi']))
print(f"✓ Dictionary loaded: {len(translation_dict)} word pairs")

# --- 2. Data Definitions and Loading (Parallel Corpus) ---
print("\n[2/4] Creating parallel corpus...")
data_corpus = {
    'Hindi_Sentence': [
        'तुम कैसे हो?',
        'मैं ठीक हूँ, धन्यवाद।',
        'क्या तुम घर जा रहे हो?',
        'मुझे पानी चाहिए.',
        'यह मेरा घर है.',
        'आज मौसम अच्छा है.',
        'खाना तैयार है.',
        'नमस्ते, आपका दिन शुभ हो.',
        'अच्छे रिजल्ट लेन के लिए'
    ],
    'Punjabi_Sentence': [
        'ਤੁਸੀਂ ਕਿਵੇਂ ਹੋ?',
        'ਮੈਂ ਠੀਕ ਹਾਂ, ਧੰਨਵਾਦ।',
        'ਕੀ ਤੁਸੀਂ ਘਰ ਜਾ ਰਹੇ ਹੋ?',
        'ਮੈਨੂੰ ਪਾਣੀ ਚਾਹੀਦਾ ਹੈ।',
        'ਇਹ ਮੇਰਾ ਘਰ ਹੈ।',
        'ਅੱਜ ਮੌਸਮ ਵਧੀਆ ਹੈ।',
        'ਖਾਣਾ ਤਿਆਰ ਹੈ।',
        'ਸਤ ਸ੍ਰੀ ਅਕਾਲ, ਤੁਹਾਡਾ ਦਿਨ ਸ਼ੁਭ ਹੋਵੇ।',
        'ਚੰਗੇ ਨਤੀਜੇ ਪ੍ਰਾਪਤ ਕਰਨ ਲਈ'
    ]
}
df_parallel_corpus = pd.DataFrame(data_corpus)

csv_file_path_corpus = 'parallel_corpus.csv'
df_parallel_corpus.to_csv(csv_file_path_corpus, index=False, encoding='utf-8')

df_parallel_corpus_loaded = pd.read_csv(csv_file_path_corpus, encoding='utf-8')
parallel_corpus = list(df_parallel_corpus_loaded.itertuples(index=False, name=None))
print(f"✓ Parallel corpus loaded: {len(parallel_corpus)} sentence pairs")

# --- 3. NMT Model Loading ---
print("\n[3/4] Loading NLLB-200 model...")
print("(This may take 2-5 minutes on first run - downloading ~1.2GB)")

try:
    model_name = "facebook/nllb-200-distilled-600M"

    # Force CPU mode to avoid CUDA errors
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model = model.to(device)

    print("✓ Model loaded successfully!")
except Exception as e:
    print(f"ERROR loading model: {e}")
    print("\nTrying to continue without NMT (Dictionary and EBMT will still work)...")
    model = None
    tokenizer = None

# --- 4. Translation Functions ---
def jaccard_similarity(sentence1, sentence2):
    """Calculate Jaccard similarity between two sentences."""
    words1 = set(sentence1.split())
    words2 = set(sentence2.split())
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    if not union:
        return 0.0
    return len(intersection) / len(union)

def ebmt_translate(hindi_sentence_to_translate, parallel_corpus, similarity_func):
    """Example-Based Machine Translation using parallel corpus."""
    best_match_punjabi = "Translation not found in corpus."
    highest_similarity = -1.0

    for hindi_ref, punjabi_ref in parallel_corpus:
        similarity = similarity_func(hindi_sentence_to_translate, hindi_ref)
        if similarity > highest_similarity:
            highest_similarity = similarity
            best_match_punjabi = punjabi_ref

    if highest_similarity >= 1.0:
        return best_match_punjabi
    else:
        return "No similar sentence found."

def nmt_translate(hindi_sentence, tokenizer, model):
    """Neural Machine Translation using NLLB-200."""
    if model is None or tokenizer is None:
        return "NMT model not available"

    try:
        source_lang = "hin_Deva"
        target_lang = "pan_Guru"

        device = "cuda" if torch.cuda.is_available() else "cpu"

        inputs = tokenizer(hindi_sentence, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            generated_tokens = model.generate(
                **inputs,
                forced_bos_token_id=tokenizer.convert_tokens_to_ids(target_lang),
                max_length=512
            )

        translated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        return translated_text
    except Exception as e:
        return f"NMT Error: {str(e)}"

def translate_hindi_to_punjabi(hindi_sentence, dictionary, parallel_corpus, similarity_func, tokenizer, model):
    """Main translation function with cascade approach."""
    # 1. Dictionary-based translation
    words = hindi_sentence.split()
    translated_words = []
    untranslated_count = 0

    for word in words:
        translated_word = dictionary.get(word, None)
        if translated_word is None:
            untranslated_count += 1
            translated_words.append(word)
        else:
            translated_words.append(translated_word)

    if untranslated_count == 0:
        print("  --> Dictionary translated fully.")
        return ' '.join(translated_words)
    else:
        # 2. EBMT as a fallback
        print(f"  --> Dictionary failed ({untranslated_count} words untranslated). Trying EBMT...")
        ebmt_result = ebmt_translate(hindi_sentence, parallel_corpus, similarity_func)

        if ebmt_result != "No similar sentence found.":
            print("  --> EBMT found a translation.")
            return ebmt_result
        else:
            # 3. NMT as a final fallback
            print("  --> EBMT failed. Trying NMT...")
            nmt_result = nmt_translate(hindi_sentence, tokenizer, model)
            return nmt_result

# --- 5. Metric Calculation Functions ---
def calculate_bleu(candidate, reference):
    """Calculate BLEU score."""
    try:
        return sacrebleu.corpus_bleu([candidate], [[reference]]).score
    except:
        return 0.0

def calculate_chrf(candidate, reference):
    """Calculate chrF score."""
    try:
        return sacrebleu.corpus_chrf([candidate], [[reference]]).score
    except:
        return 0.0

def calculate_exact_match_accuracy(candidate, reference):
    """Calculate exact match accuracy."""
    return 1.0 if candidate.strip() == reference.strip() else 0.0

# --- 6. Flask Application Setup ---
print("\n[4/4] Starting Flask application...")

app = Flask(__name__)

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/translate', methods=['POST'])
def translate_route():
    """Handle translation requests."""
    try:
        data = request.get_json()
        hindi_text = data.get('hindi_text', '').strip()

        if not hindi_text:
            return jsonify({'error': 'No Hindi text provided'}), 400

        # Capture translation process output
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            translated_punjabi_sentence = translate_hindi_to_punjabi(
                hindi_text, translation_dict, parallel_corpus, jaccard_similarity, tokenizer, model
            )
        s = f.getvalue()

        # Determine which method was used
        method_used = "Unknown Method"
        if "Dictionary translated fully." in s:
            method_used = "Dictionary-based Translation"
        elif "EBMT found a translation." in s:
            method_used = "EBMT (Example-Based Machine Translation)"
        elif "Trying NMT..." in s:
            method_used = "NMT (Neural Machine Translation)"

        response_data = {
            'original_hindi': hindi_text,
            'translated_punjabi': translated_punjabi_sentence,
            'method_used': method_used,
            'reference_punjabi': None,
            'bleu_score': None,
            'chrf_score': None,
            'exact_match': None
        }

        # Check if reference translation exists
        reference_punjabi_sentence = None
        for hindi_ref, punjabi_ref in parallel_corpus:
            if hindi_ref.strip() == hindi_text.strip():
                reference_punjabi_sentence = punjabi_ref
                break

        # Calculate metrics if reference exists
        if reference_punjabi_sentence:
            bleu_score = calculate_bleu(translated_punjabi_sentence, reference_punjabi_sentence)
            chrf_score = calculate_chrf(translated_punjabi_sentence, reference_punjabi_sentence)
            exact_match = calculate_exact_match_accuracy(translated_punjabi_sentence, reference_punjabi_sentence)

            response_data['reference_punjabi'] = reference_punjabi_sentence
            response_data['bleu_score'] = round(bleu_score, 2)
            response_data['chrf_score'] = round(chrf_score, 2)
            response_data['exact_match'] = round(exact_match, 2)

        return jsonify(response_data)

    except Exception as e:
        return jsonify({'error': f'Translation error: {str(e)}'}), 500

@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'dictionary_size': len(translation_dict),
        'corpus_size': len(parallel_corpus)
    })

if __name__ == '__main__':
    print("\n" + "="*60)
    print("✅ Server starting...")
    print("="*60)
    print("\n📱 Access the application at: http://localhost:5000")
    print("Press CTRL+C to stop the server\n")

    try:
        app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        print("\nTroubleshooting:")
        print("1. Check if port 5000 is already in use")
        print("2. Try running on different port: app.run(port=8080)")
        print("3. Check if templates/index.html exists")
