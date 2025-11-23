#!/usr/bin/env python3
import csv
import os
import re
import html
import random
import json
from collections import Counter

import requests
from weasyprint import HTML

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer

# -----------------------------------------------------------
# NLTK data
# -----------------------------------------------------------
for pkg in [
    "tokenizers/punkt",
    "corpora/wordnet",
    "corpora/omw-1.4",
    "taggers/averaged_perceptron_tagger",
]:
    try:
        nltk.data.find(pkg)
    except LookupError:
        # download just the last component, e.g. "punkt" from "tokenizers/punkt"
        nltk.download(pkg.split("/", 1)[1] if "/" in pkg else pkg)

# -----------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------
INPUT_TSV = "students.tsv"  # student_id, name, text

PDF_DIR = "PDFs-hybrid-assembler-replacer"

ANSWER_KEY_ASSEMBLER = "answer_key_hybrid_assembler.tsv"
ANSWER_KEY_SYNONYM = "answer_key_hybrid_synonyms.tsv"

FREQ_FILE = "wiki_freq.txt"      # "word count" per line

NUM_SYNONYM_REPLACEMENTS = 5     # how many words you want replaced
NUM_CANDIDATE_OBSCURE = 20       # how many rare words to consider

NUM_WORDS_TO_REPLACE = NUM_SYNONYM_REPLACEMENTS  # alias

AVOID_WORDS = {
    "hufs", "macalister", "minerva", "students", "learners",
    "student", "learner", "hankuk", "university", "foreign", "studies"
}

STOPWORDS = {
    "the","a","an","and","or","but","if","than","then","therefore","so","because",
    "of","to","in","on","for","at","by","from","with","as","about","into","through",
    "after","over","between","out","against","during","without","before","under","around",
    "among","is","am","are","was","were","be","been","being","have","has","had","do","does",
    "did","can","could","will","would","shall","should","may","might","must","i","you",
    "he","she","it","we","they","me","him","her","us","them","my","your","his","their",
    "our","its","this","that","these","those","there","here","up","down","very","also",
    "just","only","not","no","yes","than","such","many","much","few","several","some",
    "any","all","each","every","both","either","neither","one","two","three","four",
    "five","first","second","third"
}

# 🔑 DeepSeek config – put your actual key here
DEEPSEEK_API_KEY = "YOUR-API-KEY-HERE"
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"
DEEPSEEK_MAX_RETRIES = 3

# Synonym cache
SYNONYM_CACHE_FILE = "synonym_cache.json"

lemmatizer = WordNetLemmatizer()

# -----------------------------------------------------------
# Basic helpers
# -----------------------------------------------------------

def sanitize_filename(name):
    return "".join(c for c in name if c not in r'\/:*?"<>|').strip() or "student"

def split_into_sentences(text):
    return [s.strip() for s in sent_tokenize(str(text)) if s.strip()]

def tokenize_preserving_hyphens_apostrophes(text):
    """Hyphenated and apostrophized words count as single words."""
    return re.findall(r"[a-z0-9][a-z0-9'\-]*", str(text).lower())

def tokenize_words_lower(text):
    return re.findall(r"[A-Za-z']+", str(text).lower())

def load_frequency_ranks(freq_file):
    freq_ranks = {}
    try:
        with open(freq_file, encoding="utf-8") as f:
            rank = 1
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                word = parts[0].lower()
                if word not in freq_ranks:
                    freq_ranks[word] = rank
                    rank += 1
    except FileNotFoundError:
        print(f"⚠️ Frequency file {freq_file} not found — all words treated equally.")
    return freq_ranks

# -----------------------------------------------------------
# Synonym cache helpers
# -----------------------------------------------------------

def load_synonym_cache(cache_file=SYNONYM_CACHE_FILE):
    # --- CACHE DISABLED ---
    # Always return an empty dict so nothing is cached or loaded.
    return {}
    try:
        with open(cache_file, encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, dict):
                print(f"⚠️ Cache {cache_file} not a dict, ignoring.")
                return {}
            return data
    except FileNotFoundError:
        print(f"ℹ️ No synonym cache found at {cache_file}, starting fresh.")
        return {}
    except Exception as e:
        print(f"⚠️ Could not load synonym cache {cache_file}: {e}")
        return {}

def save_synonym_cache(cache, cache_file=SYNONYM_CACHE_FILE):
    # --- CACHE DISABLED ---
    return
    try:
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
        print(f"💾 Synonym cache saved to {cache_file}")
    except Exception as e:
        print(f"⚠️ Could not save synonym cache {cache_file}: {e}")

def make_syn_key(surface_word, sentence):
    # key is based on the surface form + sentence context
    return f"{surface_word.strip().lower()}|||{sentence.strip()}"

# -----------------------------------------------------------
# Sentence-assembler: expand block & 20-word window
# -----------------------------------------------------------

def expand_block(sentences, idx_longest):
    """
    Alternating expansion: prev → next → prev2 → next2 → ...
    Returns (block_text, block_tokens, block_indices).
    """
    total_sentences = len(sentences)
    block_indices = [idx_longest]

    left_offset = 1
    right_offset = 1

    def block_tokens_now():
        text = " ".join(sentences[i] for i in block_indices)
        return tokenize_preserving_hyphens_apostrophes(text)

    # Expand until >= 20 tokens or exhausted
    while True:
        tokens_now = block_tokens_now()
        if len(tokens_now) >= 10:
            break

        expanded = False

        if left_offset <= right_offset:
            prev_idx = idx_longest - left_offset
            if prev_idx >= 0:
                block_indices.insert(0, prev_idx)
                left_offset += 1
                expanded = True
            else:
                left_offset += 1
        else:
            next_idx = idx_longest + right_offset
            if next_idx < total_sentences:
                block_indices.append(next_idx)
                right_offset += 1
                expanded = True
            else:
                right_offset += 1

        if not expanded:
            break

    block_text = " ".join(sentences[i] for i in block_indices)
    block_tokens = tokenize_preserving_hyphens_apostrophes(block_text)

    return block_text, block_tokens, block_indices

def select_best_window(tokens, window=20):
    if len(tokens) <= window:
        return list(range(len(tokens)))  # all words

    semantic = [10 if w not in STOPWORDS else 1 for w in tokens]
    length = [len(w) for w in tokens]

    best_score = -1
    best_start = 0

    for i in range(len(tokens) - window + 1):
        score = sum(semantic[i:i+window]) + sum(length[i:i+window])
        if score > best_score:
            best_score = score
            best_start = i

    return list(range(best_start, best_start + window))

def replace_tokens_with_blanks_in_block(block_text, block_tokens, removed_positions):
    pattern = re.compile(r"[A-Za-z0-9][A-Za-z0-9'\-]*|[.,;:!?]")
    parts = pattern.findall(block_text)

    rebuilt = []
    token_i = 0

    for part in parts:
        if re.match(r"[A-Za-z0-9]", part):
            if token_i in removed_positions:
                blank = "_" * (len(part) * 2)
                blank_html = f"<span style='color:white'>{blank}</span>"
                rebuilt.append(blank_html)
            else:
                rebuilt.append(part)
            token_i += 1
        else:
            rebuilt.append(part)

    # --- PATCHED SPACING LOGIC ---
    out = ""
    last = ""
    for piece in rebuilt:
        if last == "":
            out += piece
        else:
            # If last token ends with punctuation → force a space before the next token.
            if last.endswith(('.', ',', ';', ':', '!', '?')):
                out += " " + piece
            # If piece itself IS punctuation → attach directly.
            elif piece in ".,;:!?":
                out += piece
            else:
                out += " " + piece
        last = piece

    return out

def compute_block_info(text):
    """
    Compute the assembler block and window on the ORIGINAL text,
    and also return the set of word types in that block (for excluding synonyms).
    """
    sentences = split_into_sentences(text)
    if not sentences:
        return None

    # Longest sentence by token count
    idx_longest = max(
        range(len(sentences)),
        key=lambda i: len(tokenize_preserving_hyphens_apostrophes(sentences[i]))
    )

    block_text, block_tokens, block_indices = expand_block(sentences, idx_longest)

    removed_positions = select_best_window(block_tokens, window=10)
    removed_words = [block_tokens[i] for i in removed_positions]

    # All word TYPES in the block sentences (for synonym exclusion)
    block_sentence_text = " ".join(sentences[i] for i in block_indices)
    block_types = set(tokenize_words_lower(block_sentence_text))

    return {
        "block_text": block_text,
        "block_tokens": block_tokens,
        "removed_positions": removed_positions,
        "removed_words": removed_words,
        "block_types": block_types,
    }

# -----------------------------------------------------------
# Synonym replacer – helpers
# -----------------------------------------------------------

def levenshtein(a, b):
    """Compute Levenshtein edit distance between strings a and b."""
    m, n = len(a), len(b)
    dp = [[0]*(n+1) for _ in range(m+1)]

    for i in range(m+1):
        dp[i][0] = i
    for j in range(n+1):
        dp[0][j] = j

    for i in range(1, m+1):
        for j in range(1, n+1):
            cost = 0 if a[i-1] == b[j-1] else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,
                dp[i][j-1] + 1,
                dp[i-1][j-1] + cost
            )
    return dp[m][n]

def find_sentence_and_surface_word(text, word_lower):
    """
    Find the first sentence containing word_lower (case-insensitive),
    and return (sentence, surface_form_as_it_appears).
    """
    for sent in split_into_sentences(text):
        m = re.search(r"\b" + re.escape(word_lower) + r"\b", sent, re.IGNORECASE)
        if m:
            return sent, m.group(0)
    return None, None

def find_obscure_words(text, freq_ranks, num_candidates=NUM_CANDIDATE_OBSCURE,
                       forbidden_types=None):
    """
    Return up to num_candidates obscure words (rarest first), EXCLUDING:
      - stopwords
      - AVOID_WORDS
      - forbidden_types (e.g. words used anywhere in the assembler block)
    """
    if forbidden_types is None:
        forbidden_types = set()

    tokens = tokenize_words_lower(text)
    counts = Counter(tokens)
    candidates = []

    for w, c in counts.items():
        if len(w) < 4:
            continue
        if w in STOPWORDS or w in AVOID_WORDS or w in forbidden_types:
            continue
        if "'" in w:
            continue
        rank = freq_ranks.get(w, 10**9)
        candidates.append((rank, w))

    candidates.sort(key=lambda x: x[0], reverse=True)

    result = []
    for _, w in candidates:
        if w not in result:
            result.append(w)
        if len(result) >= num_candidates:
            break

    return result

# -----------------------------------------------------------
# DeepSeek synonym generator with cache (CACHE DISABLED)
# -----------------------------------------------------------

def get_synonym_from_deepseek(surface_word, sentence, all_words_in_text, cache):
    """
    DeepSeek synonym generator with caching.
      - lowercase-only rule
      - Levenshtein > 50% difference from original
      - Levenshtein > 30% difference from any other word in text
      - skip if original contains any capital letters (no proper nouns)
      - retries
      - cache keyed by (surface_word, sentence)
    """
    if not DEEPSEEK_API_KEY:
        return None

    # Skip if original has any capital letters
    if any(c.isupper() for c in surface_word):
        print(f"⚠️ Skipping '{surface_word}' — contains capital letters.")
        return None

    key = make_syn_key(surface_word, sentence)

    # -----------------------------
    # CACHE LOOKUP — DISABLED
    # -----------------------------
    """
    if cache is not None and key in cache:
        cached = cache[key]
        if cached:
            print(f"✓ Using cached synonym for '{surface_word}': {cached}")
            return cached
        else:
            print(f"ℹ️ Cache says 'no synonym' for '{surface_word}', skipping API.")
            return None
    """

    system_prompt = (
        "You are a precise thesaurus assistant. Given an English word as it appears "
        "inside a sentence, produce exactly one lowercase synonym that can replace "
        "the original word WITHOUT ANY CAPITAL LETTERS.\n\n"
        "Requirements:\n"
        "1) Synonym must be lowercase only.\n"
        "2) Must match part of speech and inflection.\n"
        "3) Respond with ONLY the replacement word.\n"
        "4) If no good synonym exists, repeat the original word."
    )

    user_prompt = (
        f"Sentence:\n{sentence}\n\n"
        f"Original word: {surface_word}\n\n"
        "Return a single-word lowercase synonym."
    )

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": 20,
        "temperature": 0.5,
    }

    surface_lower = surface_word.lower()
    attempt = 0

    original_threshold = max(1, len(surface_lower) // 2)

    while attempt < DEEPSEEK_MAX_RETRIES:
        print("\n----------------------------")
        print(f"DeepSeek synonym lookup for '{surface_word}' (attempt {attempt+1})")
        print("Sentence:", sentence)

        try:
            resp = requests.post(DEEPSEEK_URL, headers=headers, json=payload, timeout=20)

            if resp.status_code >= 400:
                print(f"⚠️ HTTP {resp.status_code}: {resp.text}")
                attempt += 1
                continue

            data = resp.json()
            candidate = data["choices"][0]["message"]["content"].strip()

            print("DeepSeek raw content:", candidate)
            print("----------------------------")

            if candidate.startswith(("'", '"')) and candidate.endswith(("'", '"')):
                candidate = candidate[1:-1].strip()

            tokens = re.findall(r"[A-Za-z]+", candidate)
            if not tokens:
                attempt += 1
                continue

            synonym = tokens[0].lower()

            if any(c.isupper() for c in synonym):
                print(f"⚠️ Rejected '{synonym}' — contains capitals.")
                attempt += 1
                continue

            if synonym == surface_lower:
                print(f"⚠️ Rejected '{synonym}' — same as original.")
                attempt += 1
                continue

            dist_orig = levenshtein(surface_lower, synonym)
            if dist_orig <= original_threshold:
                print(f"⚠️ '{synonym}' too similar to '{surface_lower}' "
                      f"(dist={dist_orig}, threshold={original_threshold})")
                attempt += 1
                continue

            conflict = False
            for w in all_words_in_text:
                if w == surface_lower:
                    continue

                threshold_other = max(1, int(len(w) * 0.30))
                dist_other = levenshtein(w, synonym)

                if dist_other <= threshold_other:
                    print(
                        f"⚠️ '{synonym}' rejected — too similar to '{w}' in text "
                        f"(dist={dist_other}, threshold={threshold_other})"
                    )
                    conflict = True
                    break

            if conflict:
                attempt += 1
                continue

            print(f"✓ Accepted synonym for '{surface_word}': {synonym}")

            # -----------------------------
            # CACHE WRITE — DISABLED
            # -----------------------------
            """
            if cache is not None:
                cache[key] = synonym
            """

            return synonym

        except Exception as e:
            print(f"⚠️ DeepSeek error: {e}")
            attempt += 1

    print(f"⚠️ No suitable synonym for '{surface_word}' after retries.")

    # -----------------------------
    # CACHE WRITE (NO-SYNONYM) — DISABLED
    # -----------------------------
    """
    if cache is not None:
        cache[key] = ""  # mark as 'no synonym'
    """

    return None

# -----------------------------------------------------------
# Synonym transform (EXCLUDING assembler block types)
# -----------------------------------------------------------

def transform_text_with_synonyms(text, freq_ranks, forbidden_types=None, cache=None):
    """
    Replace up to NUM_WORDS_TO_REPLACE words with DeepSeek synonyms, but
    never for ANY word type that appears in forbidden_types
    (i.e., anything in the assembler block).
    Returns (modified_text, replacements_list).
    """

    if forbidden_types is None:
        forbidden_types = set()

    all_words = set(tokenize_words_lower(text))

    candidate_words = find_obscure_words(
        text,
        freq_ranks,
        num_candidates=NUM_CANDIDATE_OBSCURE,
        forbidden_types=forbidden_types,
    )

    modified_text = text
    replacements = []

    for w_lower in candidate_words:
        if len(replacements) >= NUM_WORDS_TO_REPLACE:
            break

        sentence, surface_word = find_sentence_and_surface_word(modified_text, w_lower)
        if not sentence or not surface_word:
            continue

        synonym = get_synonym_from_deepseek(surface_word, sentence, all_words, cache)
        if not synonym:
            continue

        pattern = re.compile(r"\b" + re.escape(w_lower) + r"\b", re.IGNORECASE)
        if not pattern.search(modified_text):
            continue

        def repl(m):
            orig = m.group(0)
            if orig.isupper():
                return synonym.upper()
            elif orig[0].isupper():
                return synonym.capitalize()
            else:
                return synonym.lower()

        modified_text = pattern.sub(repl, modified_text)
        replacements.append((surface_word, synonym))

    return modified_text, replacements

# -----------------------------------------------------------
# Word bank grid
# -----------------------------------------------------------

def build_word_grid(words):
    """
    Build a 5×4 grid (20 items) for the word bank.
    Words are lowercased and sorted; empty strings pad to 20 items.
    """
    words = sorted([w.lower() for w in words])
    while len(words) < 10:
        words.append("")
    words = words[:10]
    return [words[i*2:(i+1)*2] for i in range(5)]

# -----------------------------------------------------------
# PDF generation (hybrid layout)
# -----------------------------------------------------------

def generate_hybrid_pdf(student_id, name, final_text, removed_words, replacements,
                        out_dir=PDF_DIR):
    os.makedirs(out_dir, exist_ok=True)
    pdf_path = os.path.join(out_dir, f"{sanitize_filename(name)}.pdf")

    esc_name = html.escape(name)
    esc_id = html.escape(student_id)
    esc_text = final_text  # contains <span> blanks; don't escape

    # -----------------------------
    # Dynamic font size for word bank
    # -----------------------------
    base_font_size = 12  # starting point
    reference_word = "paragraphsparagraphs"
    ref_len = len(reference_word)  # 10
    longest_len = max((len(w) for w in removed_words if w), default=0)

    extra_letters = max(0, longest_len - ref_len)
    reduction_points = extra_letters // 3
    font_size = base_font_size - reduction_points
    if font_size < 6:  # sanity floor
        font_size = 6

    # -----------------------------
    # Word bank grid HTML (left box) – 5 rows × 4 cols
    # -----------------------------
    grid = build_word_grid(removed_words)
    grid_html = (
        f"<table style='width:100%; font-size:{font_size}pt; "
        f"border-collapse:collapse;'>"
    )
    for row in grid:
        grid_html += "<tr>"
        for cell in row:
            grid_html += (
                "<td style='border:1px solid #ccc; padding:4px; text-align:center;'>"
                f"{html.escape(cell)}</td>"
            )
        grid_html += "</tr>"
    grid_html += "</table>"

    # -----------------------------
    # Synonym blanks (right box)
    #   • Exactly 5 lines
    #   • Each line is a single DIV with no wrapping
    # -----------------------------
    total_lines = 5
    blanks_html = []
    for i in range(1, total_lines + 1):
        # slightly shorter underscores to avoid clipping while still long
        blanks_html.append(
            f"<div class='synline'>{i}. _______________ → _______________</div>"
        )
    blanks_block = "\n".join(blanks_html)

    html_doc = f"""
    <html>
    <head>
    <meta charset='utf-8'>
    <style>
        @page {{
            margin: 1.1cm;
            size: A4;
        }}
        body {{
            font-family: Arial, sans-serif;
            font-size: 12pt;
            line-height: 1.4;
            margin: 0;
            padding: 0;
        }}
        .header {{
            font-weight: bold;
            margin-bottom: 0.5em;
        }}
        .subheader {{
            font-weight: bold;
            margin-top: 0.5em;
            margin-bottom: 0.3em;
        }}
        .sentence {{
            margin-top: 1em;
            margin-bottom: 1em;
            white-space: pre-wrap;
        }}
        .instructions {{
            white-space: normal;
            margin: 0;
            text-indent: 0;
            text-align: left;
        }}
        .text {{
            white-space: pre-wrap;
            margin: 0.5em 0;
            text-indent: 2em;
        }}
        /* NEW: for synonym lines – no wrapping, no indent */
        .synline {{
            white-space: nowrap;
            margin: 0.5em 0;
            text-indent: 0;
        }}
        .bottom-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 1em;
        }}
        .bottom-table td {{
            vertical-align: top;
            padding-right: 1em;
            width: 50%;
        }}
    </style>
    </head>
    <body>

        <div class='header'>
            Name: {esc_name}<br>
            Student Number: {esc_id}<br>
            Sentence Assembler &amp; Synonym Replacer
        </div>

        <div class='instructions'>
            <b>20 words have been removed.  
            Use the 20 words in the grid to reconstruct the text.<br>
            Also, 5 words have been replaced.
            Find the 5 new words and provide the 5 original words.</b>
        </div>

        <div class='sentence'>{esc_text}</div>

        <table class="bottom-table">
          <tr>
            <td>
              <div class='subheader'>Word Grid</div>
              {grid_html}
            </td>
            <td>
              <div class='subheader'>Replacement Words → Original Words </div>
              {blanks_block}
            </td>
          </tr>
        </table>

    </body>
    </html>
    """

    HTML(string=html_doc).write_pdf(pdf_path)
    print(f"📝 PDF created: {pdf_path}")

# -----------------------------------------------------------
# Main per-student pipeline
# -----------------------------------------------------------

def process_student(student_id, name, text, freq_ranks,
                    asm_writer, syn_writer, syn_cache):

    # 1) Compute assembler block & window on ORIGINAL text
    block_info = compute_block_info(text)
    if not block_info:
        print(f"⚠️ No sentences for {name}. Skipping.")
        return

    block_text = block_info["block_text"]
    block_tokens = block_info["block_tokens"]
    removed_positions = block_info["removed_positions"]
    removed_words = block_info["removed_words"]
    block_types = block_info["block_types"]

    # 2) Run synonym replacer on words EXCLUSIVELY OUTSIDE that block
    syn_modified_text, replacements = transform_text_with_synonyms(
        text, freq_ranks, forbidden_types=block_types, cache=syn_cache
    )

    # 3) Insert invisible blanks for the 20-word window in the synonym-modified text
    removed_pos_set = set(removed_positions)
    modified_block = replace_tokens_with_blanks_in_block(
        block_text,
        block_tokens,
        removed_pos_set
    )

    final_text = syn_modified_text.replace(block_text, modified_block, 1)

    # 4) Generate PDF
    generate_hybrid_pdf(student_id, name, final_text, removed_words, replacements)

    # 5) Write answer keys
    asm_writer.writerow([student_id, name, ", ".join(removed_words)])
    for orig, syn in replacements:
        syn_writer.writerow([student_id, name, orig, syn])

# -----------------------------------------------------------
# TSV processor
# -----------------------------------------------------------

def process_hybrid(input_tsv, out_asm_tsv, out_syn_tsv, freq_file=FREQ_FILE):
    freq_ranks = load_frequency_ranks(freq_file)
    syn_cache = load_synonym_cache()

    with open(out_asm_tsv, "w", newline="", encoding="utf-8") as f_asm, \
         open(out_syn_tsv, "w", newline="", encoding="utf-8") as f_syn:

        asm_writer = csv.writer(f_asm, delimiter="\t")
        syn_writer = csv.writer(f_syn, delimiter="\t")

        asm_writer.writerow(["student_id", "name", "removed_words"])
        syn_writer.writerow(["student_id", "name", "original", "synonym"])

        with open(input_tsv, newline="", encoding="utf-8") as infile:
            reader = csv.reader(infile, delimiter="\t")
            for row in reader:
                if len(row) < 3:
                    continue
                student_id, name, text = row[0], row[1], row[2]
                print(f"\n=== Processing {student_id} / {name} ===")
                process_student(student_id, name, text, freq_ranks,
                                asm_writer, syn_writer, syn_cache)

    # save_synonym_cache(syn_cache)
    print(f"✅ Done.\n  Assembler key: {out_asm_tsv}\n  Synonym key:   {out_syn_tsv}")

# -----------------------------------------------------------
if __name__ == "__main__":
    process_hybrid(INPUT_TSV, ANSWER_KEY_ASSEMBLER, ANSWER_KEY_SYNONYM)