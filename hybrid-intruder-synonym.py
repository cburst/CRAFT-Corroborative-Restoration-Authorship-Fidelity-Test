#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ====================================================
# IMPORTS & GLOBAL CONFIG
# ====================================================

import csv
import os
import re
import random
import time
import html
import json
import requests
from collections import Counter

from weasyprint import HTML

import nltk
from nltk.tokenize import sent_tokenize

# Ensure NLTK punkt tokenizer
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

# -----------------------------
# FILE PATHS & CONSTANTS
# -----------------------------
INPUT_TSV = "students.tsv"   # student_id, name, text
PDF_DIR = "PDFs-hybrid-synonym-intruders"
ANSWER_KEY = "answer_key_hybrid_synonym_intruders.tsv"
FREQ_FILE = "wiki_freq.txt"  # "word count" per line

NUM_INTRUDERS = 4            # one per quarter
NUM_WORDS_TO_REPLACE = 5     # target number of synonym replacements
NUM_CANDIDATE_OBSCURE = 10   # how many rare words to consider for synonym replacement

# Use environment variable for safety; user can set DEEPSEEK_API_KEY externally
DEEPSEEK_API_KEY = "YOUR-API-KEY-HERE"
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"
DEEPSEEK_MAX_RETRIES = 3     # for synonyms
DEEPSEEK_INTRUDER_MAX_RETRIES = 5  # for intruders

AVOID_WORDS = {
    "hufs", "macalister", "minerva", "students", "learners",
    "student", "learner", "Hankuk", "University", "Foreign", "Studies"
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


# ====================================================
# GENERAL UTILITIES
# ====================================================

def split_into_sentences(text):
    """Split text into sentences using NLTK sent_tokenize and strip whitespace."""
    if not text:
        return []
    return [s.strip() for s in sent_tokenize(str(text)) if s.strip()]


def tokenize_words_lower(text):
    """Return list of lowercased word tokens (A–Z and apostrophe)."""
    return re.findall(r"[A-Za-z']+", str(text).lower())


def sanitize_filename(name):
    """Remove forbidden characters for file names."""
    forbidden = r'\/:*?"<>|'
    safe = "".join(c for c in name if c not in forbidden).strip()
    return safe or "student"


def normalize_sentence(sent):
    """Normalize sentence to a simple lowercased token string."""
    return " ".join(tokenize_words_lower(sent)).strip()


def load_frequency_ranks(freq_file):
    """
    Load frequency ranks from a file with 'word count' per line.
    Lower rank = more frequent. Unseen words get very large rank.
    """
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

def build_unified_paragraph(sentences):
    """
    Convert a list of sentences into one continuous paragraph with
    bracketed 2-digit labels: [ 01 ], [ 02 ], ...
    All sentences appear in one unified paragraph with single spaces between them.
    """
    parts = []
    for i, sent in enumerate(sentences, start=1):
        label = f"[ {i:02d} ]"
        parts.append(f"{label} {sent}")
    return " ".join(parts)
    
# ====================================================
# SYNONYM REPLACEMENT LOGIC
# ====================================================

def find_obscure_words(text, freq_ranks, num_candidates=NUM_CANDIDATE_OBSCURE):
    """
    Return up to num_candidates obscure words (rarest first).
    We'll later attempt to find synonyms for these and replace
    up to NUM_WORDS_TO_REPLACE of them.
    """
    tokens = tokenize_words_lower(text)
    counts = Counter(tokens)
    candidates = []

    for w, c in counts.items():
        if len(w) < 4:
            continue
        if w in STOPWORDS or w in AVOID_WORDS:
            continue
        if "'" in w:
            # skip possessives / contracted forms like "teacher's"
            continue
        rank = freq_ranks.get(w, 10**9)  # unseen = very rare
        candidates.append((rank, w))

    # sort by rarity: highest rank first (rarest)
    candidates.sort(key=lambda x: x[0], reverse=True)

    result = []
    for _, w in candidates:
        if w not in result:
            result.append(w)
        if len(result) >= num_candidates:
            break

    return result


def find_sentence_and_surface_word(text, word_lower):
    """
    Find the first sentence containing word_lower (case-insensitive),
    and return (sentence, surface_form_as_it_appears).
    """
    sentences = split_into_sentences(text)
    pattern = re.compile(r"\b" + re.escape(word_lower) + r"\b", re.IGNORECASE)

    for sent in sentences:
        m = pattern.search(sent)
        if m:
            return sent, m.group(0)  # surface form in that sentence

    return None, None


def get_synonym_from_deepseek(surface_word, sentence, all_words_in_text):
    """
    DeepSeek synonym generator with:
      - lowercase-only rule
      - Levenshtein > 50% difference from original
      - Levenshtein sufficiently different from *every other word* in text
      - retries
    """

    if not DEEPSEEK_API_KEY:
        print("⚠️ DeepSeek API key not set; skipping synonym for", surface_word)
        return None

    # Capitalization rule: skip words with capitals
    if any(c.isupper() for c in surface_word):
        print(f"⚠️ Skipping '{surface_word}' — contains capital letters.")
        return None

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
    original_threshold = max(1, len(surface_lower) // 2)  # too-similar threshold

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

            # unwrap possible quotes
            if candidate.startswith(("'", '"')) and candidate.endswith(("'", '"')):
                candidate = candidate[1:-1].strip()

            tokens = re.findall(r"[A-Za-z]+", candidate)
            if not tokens:
                attempt += 1
                continue

            synonym = tokens[0].lower()

            # Reject uppercase
            if any(c.isupper() for c in synonym):
                print(f"⚠️ Rejected '{synonym}' — contains capitals.")
                attempt += 1
                continue

            # Reject identical
            if synonym == surface_lower:
                print(f"⚠️ Rejected '{synonym}' — same as original.")
                attempt += 1
                continue

            # Reject too similar to original
            dist_orig = levenshtein(surface_lower, synonym)
            if dist_orig <= original_threshold:
                print(f"⚠️ '{synonym}' too similar to '{surface_lower}' "
                      f"(dist={dist_orig}, threshold={original_threshold})")
                attempt += 1
                continue

            # Reject too similar to any other word in text
            conflict = False
            for w in all_words_in_text:
                if w == surface_lower:
                    continue
                threshold_other = max(1, int(len(w) * 0.30))  # 30% threshold
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

            # Accept
            print(f"✓ Accepted synonym for '{surface_word}': {synonym}")
            return synonym

        except Exception as e:
            print(f"⚠️ DeepSeek error: {e}")
            attempt += 1

    print(f"⚠️ No suitable synonym for '{surface_word}' after retries.")
    return None


def apply_synonym_case_preserving(text, original_lower, synonym):
    """
    Replace all occurrences of original_lower in text (case-insensitive),
    preserving case pattern of each occurrence.
    """
    pattern = re.compile(r"\b" + re.escape(original_lower) + r"\b", re.IGNORECASE)
    result_parts = []
    last_end = 0

    for m in pattern.finditer(text):
        result_parts.append(text[last_end:m.start()])
        orig = m.group(0)
        if orig.isupper():
            rep = synonym.upper()
        elif orig[0].isupper():
            rep = synonym.capitalize()
        else:
            rep = synonym.lower()
        result_parts.append(rep)
        last_end = m.end()

    result_parts.append(text[last_end:])
    return "".join(result_parts)


def transform_text_with_synonyms(text, freq_ranks):
    """
    1) Find up to NUM_CANDIDATE_OBSCURE rare words.
    2) For each, find a sentence and the surface form of the word.
    3) Ask DeepSeek for a synonym that matches POS + inflection.
    4) Reject synonyms that:
         - are too similar to the original word
         - OR too similar to ANY other word in the text
         - OR contain capitalization
    5) Replace up to NUM_WORDS_TO_REPLACE words in the text.

    Returns:
      modified_text, replacements_list (list of (original_surface, synonym)).
    """
    all_words = set(tokenize_words_lower(text))

    candidate_words = find_obscure_words(
        text, freq_ranks, num_candidates=NUM_CANDIDATE_OBSCURE
    )

    modified_text = text
    replacements = []

    for w_lower in candidate_words:
        if len(replacements) >= NUM_WORDS_TO_REPLACE:
            break

        sentence, surface_word = find_sentence_and_surface_word(modified_text, w_lower)
        if not sentence or not surface_word:
            continue

        synonym = get_synonym_from_deepseek(surface_word, sentence, all_words)
        if not synonym:
            continue

        pattern = re.compile(r"\b" + re.escape(w_lower) + r"\b", re.IGNORECASE)
        if not pattern.search(modified_text):
            continue

        modified_text = apply_synonym_case_preserving(modified_text, w_lower, synonym)
        replacements.append((surface_word, synonym))

    return modified_text, replacements


# ====================================================
# INTRUDER GENERATION LOGIC
# ====================================================

def generate_intruder_sentence(essay_section_sentences, existing_sentences, intruder_index):
    """
    Use DeepSeek to generate one plausible intruder sentence
    based on a section of the essay.
    """
    if not DEEPSEEK_API_KEY:
        print("⚠️ DeepSeek API key not set; using fallback intruder sentence.")
        return "This sentence relates to the topic but is not from the original essay."

    system_prompt = (
        "You are a careful writing assistant. Given a section of a student's essay, "
        "write one plausible standalone sentence that matches the student's stylistic level "
        "(non-native academic), topic, and tone.\n\n"
        "Requirements:\n"
        "1) It should sound like it could appear anywhere in the essay.\n"
        "2) It should be most influenced by the CHARACTERISTICS of THIS SECTION.\n"
        "3) 10–30 words.\n"
        "4) Must NOT duplicate any existing sentence.\n"
        "5) Output one sentence only, no commentary."
    )

    user_prompt = (
        "Here is one section of the student's essay, representing one part of the essay's topic/style:\n\n"
        f"{' '.join(essay_section_sentences)}\n\n"
        "Write one plausible standalone sentence that matches the STYLE and CONTENT CHARACTERISTICS of THIS SECTION."
    )

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",  "content": user_prompt},
        ],
        "max_tokens": 200,
        "temperature": 0.8,
    }

    existing_norms = {normalize_sentence(s) for s in existing_sentences}
    attempt = 1

    while attempt <= DEEPSEEK_INTRUDER_MAX_RETRIES:
        try:
            print(f"→ Intruder {intruder_index}, attempt {attempt}")

            resp = requests.post(DEEPSEEK_URL, headers=headers, json=payload, timeout=30)
            if resp.status_code >= 400:
                print(f"⚠️ HTTP {resp.status_code}: {resp.text}")
                attempt += 1
                time.sleep(attempt)
                continue

            data = resp.json()
            content = data["choices"][0]["message"]["content"].strip()
            candidate = content.strip("'\"")
            norm = normalize_sentence(candidate)

            if not norm:
                print(f"⚠️ Intruder {intruder_index}: empty result; retrying…")
                attempt += 1
                time.sleep(attempt)
                continue

            if norm in existing_norms:
                print(f"⚠️ Intruder {intruder_index}: duplicate sentence; retrying…")
                attempt += 1
                time.sleep(attempt)
                continue

            print(f"✓ Intruder {intruder_index} accepted after {attempt} attempt(s): {candidate}")
            return candidate

        except Exception as e:
            print(f"⚠️ Intruder {intruder_index}: DeepSeek error: {e}")
            attempt += 1
            time.sleep(attempt)

    return "This sentence relates to the topic but is not from the original essay."


def compute_quarters(n_sentences):
    """
    Split n_sentences into up to 4 quarters as evenly as possible.
    Returns list of (start, end) indices, end-exclusive.
    """
    quarters = []
    base = n_sentences // 4
    remainder = n_sentences % 4
    start = 0

    for i in range(4):
        size = base + (1 if i < remainder else 0)
        end = min(n_sentences, start + size)
        if start < end:
            quarters.append((start, end))
        else:
            quarters.append((start, start))
        start = end

    return quarters


def insert_intruders_into_sentences(sentences):
    """
    Insert NUM_INTRUDERS intruder sentences into the synonym-modified sentences:
    - One intruder per quarter
    - Intruder appears somewhere in the quarter
    - Not as the very first or very last sentence of the overall paragraph.
    Returns:
        augmented_sentences, intruder_positions (0-based), intruder_texts
    """
    original_sentences = list(sentences)
    n = len(original_sentences)

    if n == 0:
        return original_sentences, [], []

    quarters = compute_quarters(n)
    intruder_specs = []
    intruder_index = 1
    existing_sentences_for_intruders = list(original_sentences)

    for q_idx, (start, end) in enumerate(quarters):
        if intruder_index > NUM_INTRUDERS:
            break
        if end <= start:
            continue

        # Determine allowed insertion indices:
        # - indices in [start, end] region
        # - not at index 0 (first sentence of paragraph)
        # - not at index n (would place intruder after last sentence)
        min_i = max(start, 1)       # avoid index 0
        max_i = min(end, n - 1)     # avoid inserting after last sentence

        if min_i > max_i:
            # Fallback: anywhere safe in the middle of the paragraph
            if n > 2:
                min_i = 1
                max_i = n - 1
            else:
                min_i = 1
                max_i = 1

        insert_index = random.randint(min_i, max_i)

        section_sentences = original_sentences[start:end]
        intruder_text = generate_intruder_sentence(
            essay_section_sentences=section_sentences,
            existing_sentences=existing_sentences_for_intruders,
            intruder_index=intruder_index
        )

        existing_sentences_for_intruders.append(intruder_text)
        intruder_specs.append({
            "insert_index": insert_index,
            "text": intruder_text
        })
        intruder_index += 1

    # Now actually insert intruders, from highest index down
    augmented = list(original_sentences)
    for spec in sorted(intruder_specs, key=lambda s: s["insert_index"], reverse=True):
        idx = spec["insert_index"]
        txt = spec["text"]
        if idx < 0:
            idx = 0
        if idx > len(augmented):
            idx = len(augmented)
        augmented.insert(idx, txt)

    intruder_positions = sorted(spec["insert_index"] for spec in intruder_specs)
    intruder_texts = [spec["text"] for spec in intruder_specs]

    return augmented, intruder_positions, intruder_texts


# ====================================================
# PDF GENERATION (COMBINED TEST)
# ====================================================

def generate_pdf(student_id, name, sentences, replacements, pdf_dir=PDF_DIR):
    """
    Generate a single PDF that contains:
      - Instructions for both tests at the top
      - ONE unified paragraph:
            [ 01 ] Sentence. [ 02 ] Sentence. [ 03 ] Sentence. ...
      - A 5-column table at the bottom:
            replacements  (blank cells)
            originals      (first-letter hints)
    """
    os.makedirs(pdf_dir, exist_ok=True)
    safe_name = sanitize_filename(name)
    pdf_path = os.path.join(pdf_dir, f"{safe_name}.pdf")

    esc_name = html.escape(name)
    esc_number = html.escape(student_id)

    # Build unified paragraph
    unified_paragraph = build_unified_paragraph(sentences)
    esc_paragraph = html.escape(unified_paragraph)

    num_replacements = len(replacements)

    html_parts = [
        "<html>",
        "<head>",
        "<meta charset='utf-8'>",
        "<style>",
        "@page { margin: 1.5cm; size: A4; }",
        "body { font-family: Arial, sans-serif; font-size: 13pt; line-height: 1.4; margin: 0; padding: 0; }",
        ".header { font-weight: bold; margin-bottom: 0.5em; }",
        ".instructions { white-space: normal; margin: 0.3em 0; text-indent: 0; }",
        ".paragraph { white-space: pre-wrap; margin: 0.6em 0; text-indent: 2em; }",
        "table.syn-table { border-collapse: collapse; margin-top: 1.2em; }",
        "table.syn-table td { border: 1px solid #000; padding: 0.2em 0.3em; min-width: 2.5cm; }",
        "table.syn-table td.label-cell { font-weight: bold; white-space: nowrap; }",
        "</style>",
        "</head>",
        "<body>",
        f"<div class='header'>Name: {esc_name}<br>Student Number: {esc_number}</div>",
        "<div class='header'>Sentence Intruders & Synonym Replacements</div>",
        "<div class='instructions'>",
        "<b> Extra sentences have been added. Circle the added sentence numbers.<br>",
        f"Five words have been replaced. Find the replacements and provide the originals.</b>",
        "</div>",
        f"<div class='paragraph'>{esc_paragraph}</div>",
        "<table class='syn-table'>",
    ]

    #
    # 🔵 ROW 1 — "replacements" with BLANK CELLS (students must find them)
    #
    html_parts.append("<tr>")
    html_parts.append("<td class='label-cell'>replacements</td>")
    for _ in range(NUM_WORDS_TO_REPLACE):
        html_parts.append("<td>&nbsp;</td>")
    html_parts.append("</tr>")

    #
    # 🔵 ROW 2 — "originals" with FIRST-LETTER HINTS
    #
    html_parts.append("<tr>")
    html_parts.append("<td class='label-cell'>originals</td>")
    for idx in range(NUM_WORDS_TO_REPLACE):
        if idx < len(replacements):
            orig, _ = replacements[idx]
            # first alphabetical character
            first_letter = next((ch.lower() for ch in orig if ch.isalpha()), "")
            cell_text = html.escape(first_letter)
            html_parts.append(f"<td>{cell_text}</td>")
        else:
            html_parts.append("<td>&nbsp;</td>")
    html_parts.append("</tr>")

    html_parts.append("</table>")
    html_parts.append("</body></html>")

    html_doc = "\n".join(html_parts)
    HTML(string=html_doc).write_pdf(pdf_path)
    print(f"📄 PDF created: {pdf_path}")

# ====================================================
# TSV PROCESSOR & MAIN PIPELINE
# ====================================================

def process_tsv(input_tsv, output_tsv):
    """
    Final hybrid processor:
        1. Read (student_id, name, text)
        2. Apply synonym replacements to ORIGINAL sentences only
        3. Insert intruders (4 total) into quarter-based sections
        4. Produce hybrid numbered paragraph PDF
        5. Write simplified answer key with:
               student_id, name, intruder_sentence_numbers,
               replacements (comma-separated),
               originals (comma-separated)
    """

    freq_ranks = load_frequency_ranks(FREQ_FILE)

    with open(output_tsv, "w", newline="", encoding="utf-8") as keyfile:
        writer = csv.writer(keyfile, delimiter="\t")

        # --------------------------
        # HEADER
        # --------------------------
        writer.writerow([
            "student_id",
            "name",
            "intruder_sentence_numbers",
            "replacements",
            "originals"
        ])

        with open(input_tsv, newline="", encoding="utf-8") as infile:
            reader = csv.reader(infile, delimiter="\t")

            for row in reader:
                if len(row) < 3:
                    continue

                student_id, name, text = row[0], row[1], row[2]

                print(f"\n=== Processing {student_id} / {name} ===")

                # ------------------------------------------------------------------
                # 1. SPLIT ORIGINAL SENTENCES
                # ------------------------------------------------------------------
                original_sentences = split_into_sentences(text)
                if not original_sentences:
                    print("⚠️ No sentences found. Skipping student.")
                    continue

                # ------------------------------------------------------------------
                # 2. APPLY SYNONYM TRANSFORMATIONS (to original sentences only)
                # ------------------------------------------------------------------
                synonym_modified_text, replacements = transform_text_with_synonyms(
                    text,
                    freq_ranks
                )

                # remake synonym-modified original sentences
                modified_original_sentences = split_into_sentences(synonym_modified_text)

                # ------------------------------------------------------------------
                # 3. INSERT 4 INTRUDERS anywhere inside each quarter (not edges)
                # ------------------------------------------------------------------
                augmented_sentences, intruder_positions, intruders = \
                    insert_intruders_into_sentences(modified_original_sentences)


                # intruder_positions are 0-based → convert to 1-based numbers
                intruder_numbers_1based = [i + 1 for i in intruder_positions]

                # ------------------------------------------------------------------
                # 4. GENERATE PDF (using unified paragraph + blank replacements box)
                # ------------------------------------------------------------------
                generate_pdf(
                    student_id=student_id,
                    name=name,
                    sentences=augmented_sentences,
                    replacements=replacements,
                    pdf_dir=PDF_DIR
                )

                # ------------------------------------------------------------------
                # 5. WRITE SIMPLIFIED ANSWER KEY ENTRY
                # ------------------------------------------------------------------
                original_words = [orig for (orig, syn) in replacements]
                replacement_words = [syn for (orig, syn) in replacements]

                writer.writerow([
                    student_id,
                    name,
                    ",".join(str(n) for n in intruder_numbers_1based),
                    ",".join(replacement_words),
                    ",".join(original_words)
                ])

    print(f"\n🎯 Done. Answer key saved to: {output_tsv}")


# ====================================================
# MAIN
# ====================================================

if __name__ == "__main__":
    random.seed()
    process_tsv(INPUT_TSV, ANSWER_KEY)