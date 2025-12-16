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
NUM_CANDIDATE_OBSCURE = 20   # how many rare words to consider for synonym replacement

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

def levenshtein_similarity(a, b):
    """
    Normalized Levenshtein similarity in [0,1].
    1.0 = identical, 0.0 = completely different
    """
    if not a or not b:
        return 0.0
    dist = levenshtein(a, b)
    return 1.0 - dist / max(len(a), len(b))


def extract_surface_phrases(sentence):
    """
    Extract surface spans, but return ONLY content-bearing phrases:
      - first word (only if not a stopword)
      - start → first comma
      - internal comma clauses
      - last comma → end
    Any phrase must contain ≥2 NON-stopwords to be kept.
    """
    sent = sentence.strip().lower()
    if not sent:
        return []

    phrases = []

    def content_phrase(text):
        tokens = tokenize_words_lower(text)
        content = [t for t in tokens if t not in STOPWORDS]
        if len(content) >= 2:
            return " ".join(content)
        return None

    # ---- First word (rarely useful, but keep if content-bearing) ----
    m = re.match(r"\s*([a-z']+)", sent)
    if m:
        fw = m.group(1)
        if fw not in STOPWORDS:
            phrases.append(fw)

    # ---- Comma-based spans ----
    if "," in sent:
        parts = [p.strip() for p in sent.split(",")]

        # start → first comma
        p = content_phrase(parts[0])
        if p:
            phrases.append(p)

        # internal clauses
        for mid in parts[1:-1]:
            p = content_phrase(mid)
            if p:
                phrases.append(p)

        # last comma → end
        p = content_phrase(parts[-1])
        if p:
            phrases.append(p)

    return phrases


def intruder_too_similar(candidate, existing_intruders, threshold=0.75):
    """
    Reject candidate ONLY if a content-bearing surface phrase
    is too similar to a previous intruder.
    """

    cand_phrases = extract_surface_phrases(candidate)

    if not cand_phrases:
        return False  # nothing meaningful to compare

    for prev in existing_intruders:
        prev_phrases = extract_surface_phrases(prev)

        for c in cand_phrases:
            for p in prev_phrases:
                sim = levenshtein_similarity(c, p)
                if sim >= threshold:
                    print(
                        f"⚠️ Intruder rejected — content similarity {sim:.2f}\n"
                        f"    '{c}' ~ '{p}'"
                    )
                    return True

    return False

    
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


def get_synonym_from_deepseek(surface_word, sentence, all_words_in_text, freq_ranks):
    """
    DeepSeek synonym generator with FULL diagnostic logging.

    Enforces:
      - lowercase only
      - EXACTLY one unhyphenated alphabetic word
      - POS & inflection matching (prompt-level)
      - sufficiently different from original (Levenshtein)
      - sufficiently different from all other words in text
      - NOT substantially more obscure than original (frequency-rank controlled)

    Prints:
      - every attempt
      - raw DeepSeek output
      - frequency ranks
      - explicit rejection reason
      - explicit acceptance
    """

    if not DEEPSEEK_API_KEY:
        print(f"⚠️ No API key; skipping synonym for '{surface_word}'")
        return None

    # ---- skip capitalized originals ----
    if any(c.isupper() for c in surface_word):
        print(f"⚠️ Skipping '{surface_word}' — contains uppercase letters.")
        return None

    surface_lower = surface_word.lower()
    original_rank = freq_ranks.get(surface_lower, 10**9)

    # ---- obscurity control (loosened but safe) ----
    MAX_MULTIPLIER = 6
    MAX_ABSOLUTE_DELTA = 15000
    max_allowed_rank = max(
        original_rank * MAX_MULTIPLIER,
        original_rank + MAX_ABSOLUTE_DELTA
    )

    system_prompt = (
        "You are a precise thesaurus assistant. Given an English word as it appears "
        "inside a sentence, produce exactly one lowercase synonym that can replace "
        "the original word.\n\n"
        "Requirements:\n"
        "1) lowercase only\n"
        "2) same part of speech and inflection\n"
        "3) similar or slightly higher difficulty, but not much harder\n"
        "4) output ONLY the word\n"
        "5) if unsure, repeat the original word"
    )

    user_prompt = (
        f"Sentence:\n{sentence}\n\n"
        f"Original word: {surface_word}"
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
        "temperature": 0.45,
    }

    original_threshold = max(1, len(surface_lower) // 2)

    for attempt in range(1, DEEPSEEK_MAX_RETRIES + 1):
        print("\n----------------------------")
        print(f"🔍 DeepSeek synonym attempt {attempt} for '{surface_word}'")
        print("Sentence:", sentence)

        try:
            resp = requests.post(DEEPSEEK_URL, headers=headers, json=payload, timeout=20)

            if resp.status_code >= 400:
                print(f"⚠️ HTTP {resp.status_code}: {resp.text}")
                continue

            raw = resp.json()["choices"][0]["message"]["content"].strip()
            print("🧠 DeepSeek raw output:", raw)

            candidate = raw.strip("'\"").strip().lower()

            # ---- must be EXACTLY one alphabetic word ----
            if not re.fullmatch(r"[a-z]+", candidate):
                print("❌ Rejected — synonym must be a single unhyphenated word.")
                continue

            synonym = candidate

            # ---- identical ----
            if synonym == surface_lower:
                print("❌ Rejected — identical to original.")
                continue

            # ---- obscurity check ----
            syn_rank = freq_ranks.get(synonym, 10**9)
            print(
                f"📊 Frequency ranks — original: {original_rank}, "
                f"candidate: {syn_rank}, "
                f"max allowed: {max_allowed_rank}"
            )

            if syn_rank > max_allowed_rank:
                print("❌ Rejected — synonym too obscure relative to original.")
                continue

            # ---- similarity to original ----
            dist_orig = levenshtein(surface_lower, synonym)
            if dist_orig <= original_threshold:
                print(
                    f"❌ Rejected — too similar to original "
                    f"(dist={dist_orig}, threshold={original_threshold})"
                )
                continue

            # ---- similarity to other words in text ----
            conflict = False
            for w in all_words_in_text:
                if w == surface_lower:
                    continue
                threshold_other = max(1, int(len(w) * 0.30))
                dist_other = levenshtein(w, synonym)
                if dist_other <= threshold_other:
                    print(
                        f"❌ Rejected — too similar to existing word '{w}' "
                        f"(dist={dist_other}, threshold={threshold_other})"
                    )
                    conflict = True
                    break

            if conflict:
                continue

            # ---- ACCEPT ----
            print(f"✅ Accepted synonym for '{surface_word}': {synonym}")
            return synonym

        except Exception as e:
            print(f"⚠️ DeepSeek error on attempt {attempt}: {e}")

    print(f"⚠️ No suitable synonym found for '{surface_word}' after retries.")
    return None
    
def get_pos_from_deepseek(surface_word, sentence):
    """
    Ask DeepSeek for the part of speech of a word *as used in the given sentence*.
    Returns a short POS label like: noun, verb, adjective, adverb, preposition, etc.
    """

    if not DEEPSEEK_API_KEY:
        return "?"

    system_prompt = (
        "You are an expert at identifying the part of speech of English words "
        "based on their usage in context. Given a sentence and the target word, "
        "return ONLY the part of speech of the word *as used*, such as "
        "'noun', 'verb', 'adjective', 'adverb', 'preposition', 'conjunction', etc.\n\n"
        "Output only the POS label, no punctuation, no commentary."
    )

    user_prompt = (
        f"Sentence:\n{sentence}\n\n"
        f"Target word: {surface_word}\n\n"
        "What is the part of speech of this word in this sentence?"
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
        "max_tokens": 10,
        "temperature": 0.2,
    }

    try:
        resp = requests.post(DEEPSEEK_URL, headers=headers, json=payload, timeout=15)

        if resp.status_code >= 400:
            print("⚠️ POS HTTP error:", resp.text)
            return "?"

        data = resp.json()
        pos_raw = data["choices"][0]["message"]["content"].strip()
        pos_clean = pos_raw.lower().split()[0]  # take first token, lowercase
        return pos_clean

    except Exception as e:
        print("⚠️ POS DeepSeek error:", e)
        return "?"

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
    4) Ask DeepSeek for POS of the original word.
    5) Reject synonyms that:
         - are too similar to the original word
         - OR too similar to ANY other word in the text
         - OR contain capitalization
    6) Replace up to NUM_WORDS_TO_REPLACE words in the text.

    Returns:
      modified_text,
      replacements_list = list of (original_surface, synonym, pos_label)
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

        # --- (A) SYNONYM GENERATION ---
        synonym = get_synonym_from_deepseek(surface_word, sentence, all_words, freq_ranks)
        if not synonym:
            continue

        # Ensure the word still exists in text
        pattern = re.compile(r"\b" + re.escape(w_lower) + r"\b", re.IGNORECASE)
        if not pattern.search(modified_text):
            continue

        # --- (B) POS LABELING ---
        pos_label = get_pos_from_deepseek(surface_word, sentence)
        if not pos_label:
            pos_label = "?"

        # --- (C) APPLY REPLACEMENT ---
        modified_text = apply_synonym_case_preserving(modified_text, w_lower, synonym)

        # Store (original_surface, synonym, part_of_speech)
        replacements.append((surface_word, synonym, pos_label))

    return modified_text, replacements


# ====================================================
# INTRUDER GENERATION LOGIC
# ====================================================

def generate_intruder_sentence(essay_section_sentences, existing_sentences, intruder_index):
    """
    Generate one plausible intruder sentence with:
      - upward-biased generation (toward upper word-count bound)
      - slightly loosened length floor
      - asymmetric content-ratio constraint (low punished, high allowed)
      - primary + fallback prompts

    Logs:
      - attempt number
      - prompt type
      - preview
      - word count
      - content ratio
      - acceptance or explicit rejection reason
    """

    if not DEEPSEEK_API_KEY:
        print(f"⚠️ Intruder {intruder_index}: no API key; using fallback sentence.")
        return "This sentence relates to the topic but is not from the original essay."

    # ---------- section statistics ----------
    lengths, densities = [], []
    for s in essay_section_sentences:
        tokens = tokenize_words_lower(s)
        if not tokens:
            continue
        lengths.append(len(tokens))
        content = [t for t in tokens if t not in STOPWORDS]
        densities.append(len(content) / len(tokens))

    avg_len = int(sum(lengths) / max(1, len(lengths)))
    avg_density = sum(densities) / max(1, len(densities))

    # ---------- length & density controls ----------
    CONTENT_RATIO_TOLERANCE = 0.15
    min_density = max(0.0, avg_density - CONTENT_RATIO_TOLERANCE)

    # slightly loosened lower bound; upper bound unchanged
    min_len_1 = max(6, int(avg_len * 0.85))   # was 0.90
    max_len_1 = int(avg_len * 1.15)

    min_len_2 = max(6, int(avg_len * 0.80))   # fallback slightly wider
    max_len_2 = int(avg_len * 1.20)

    prompts = [
        {
            "label": "primary",
            "min_len": min_len_1,
            "max_len": max_len_1,
            "system": (
                "You are a careful academic writing assistant.\n\n"
                "Write ONE detailed sentence that could appear in this essay section.\n\n"
                f"Requirements:\n"
                f"1) {min_len_1}–{max_len_1} words "
                f"(aim for the UPPER end of this range)\n"
                "2) Similar academic level and amount of detail\n"
                "3) May introduce ONE concrete supporting detail if needed\n"
                "4) Avoid discourse markers (e.g., moreover, additionally)\n"
                "5) Do NOT repeat or closely paraphrase any existing sentence\n"
                "6) Output ONE sentence only"
            )
        },
        {
            "label": "fallback",
            "min_len": min_len_2,
            "max_len": max_len_2,
            "system": (
                "You are a careful academic writing assistant.\n\n"
                "Write ONE detailed academic sentence related to this essay section, "
                "with substantial informational content.\n\n"
                f"Requirements:\n"
                f"1) {min_len_2}–{max_len_2} words "
                f"(aim for approximately {max_len_2} words)\n"
                "2) Comparable or slightly higher informational density than the section\n"
                "3) You MAY reframe an idea or add a small contextual detail\n"
                "4) Avoid generic filler or discourse markers\n"
                "5) Do NOT repeat any existing sentence\n"
                "6) Output ONE sentence only"
            )
        }
    ]

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }

    existing_norms = {normalize_sentence(s) for s in existing_sentences}

    for prompt_cfg in prompts:
        if prompt_cfg["label"] == "fallback":
            print(f"\n🔁 Switching to fallback intruder prompt for section {intruder_index}")

        payload = {
            "model": DEEPSEEK_MODEL,
            "messages": [
                {"role": "system", "content": prompt_cfg["system"]},
                {
                    "role": "user",
                    "content": "Essay section:\n" + " ".join(essay_section_sentences)
                },
            ],
            "max_tokens": 220,
            "temperature": 0.75,
        }

        for attempt in range(1, DEEPSEEK_INTRUDER_MAX_RETRIES + 1):
            print("\n----------------------------")
            print(
                f"🔍 DeepSeek intruder attempt {attempt} "
                f"for section {intruder_index} ({prompt_cfg['label']})"
            )

            try:
                resp = requests.post(DEEPSEEK_URL, headers=headers, json=payload, timeout=30)
                raw = resp.json()["choices"][0]["message"]["content"].strip()
                candidate = raw.strip("'\"")

                tokens = tokenize_words_lower(candidate)
                if not tokens:
                    print("❌ Rejected — empty or non-tokenizable output.")
                    continue

                wc = len(tokens)
                content_wc = sum(1 for t in tokens if t not in STOPWORDS)
                density = content_wc / wc

                preview = " ".join(candidate.split()[:6])
                if len(candidate.split()) > 6:
                    preview += "…"

                print(f"✂️ Preview: \"{preview}\"")
                print(
                    f"📏 Word count: {wc} "
                    f"(target {prompt_cfg['min_len']}–{prompt_cfg['max_len']})\n"
                    f"📊 Content ratio: {density:.2f} "
                    f"(min {min_density:.2f}, section avg ≈ {avg_density:.2f})"
                )

                if wc < prompt_cfg["min_len"] or wc > prompt_cfg["max_len"]:
                    print("❌ Rejected — length out of range.")
                    continue

                if density < min_density:
                    print("❌ Rejected — content ratio too low.")
                    continue

                norm = normalize_sentence(candidate)
                if norm in existing_norms:
                    print("❌ Rejected — duplicate of existing sentence.")
                    continue

                if intruder_too_similar(candidate, existing_sentences):
                    print("❌ Rejected — surface similarity to existing sentence.")
                    continue

                print(f"✅ Accepted intruder for section {intruder_index} ({prompt_cfg['label']})")
                return candidate

            except Exception as e:
                print(f"⚠️ Intruder error: {e}")

    print(f"⚠️ No suitable intruder found for section {intruder_index}; using fallback sentence.")
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
      - ONE unified paragraph with numbered sentences
      - A 3-row table:
            1) part of speech
            2) replacements  (blank cells)
            3) originals     (first-letter hints)
    """

    os.makedirs(pdf_dir, exist_ok=True)
    safe_name = sanitize_filename(name)
    pdf_path = os.path.join(pdf_dir, f"{safe_name}.pdf")

    esc_name = html.escape(name)
    esc_number = html.escape(student_id)

    # Build unified paragraph
    unified_paragraph = build_unified_paragraph(sentences)
    esc_paragraph = html.escape(unified_paragraph)

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
        "<b>Extra sentences have been added. Circle the added sentence numbers.<br>",
        f"Five words have been replaced. Find the replacements and provide the originals.</b>",
        "</div>",
        f"<div class='paragraph'>{esc_paragraph}</div>",
        "<table class='syn-table'>",
    ]

    # ======================================================
    # 🔵 ROW 1 — PART OF SPEECH HINTS
    # ======================================================
    html_parts.append("<tr>")
    html_parts.append("<td class='label-cell'>part of speech</td>")
    for idx in range(NUM_WORDS_TO_REPLACE):
        if idx < len(replacements):
            orig, syn, pos_label = replacements[idx]
            html_parts.append(f"<td>{html.escape(pos_label)}</td>")
        else:
            html_parts.append("<td>&nbsp;</td>")
    html_parts.append("</tr>")

    # ======================================================
    # 🔵 ROW 2 — REPLACEMENTS (blank for students)
    # ======================================================
    html_parts.append("<tr>")
    html_parts.append("<td class='label-cell'>replacements</td>")
    for _ in range(NUM_WORDS_TO_REPLACE):
        html_parts.append("<td>&nbsp;</td>")
    html_parts.append("</tr>")

    # ======================================================
    # 🔵 ROW 3 — ORIGINAL WORDS (first-letter hints)
    # ======================================================
    html_parts.append("<tr>")
    html_parts.append("<td class='label-cell'>originals</td>")
    for idx in range(NUM_WORDS_TO_REPLACE):
        if idx < len(replacements):
            orig, syn, pos_label = replacements[idx]
            first_letter = next((ch.lower() for ch in orig if ch.isalpha()), "")
            html_parts.append(f"<td>{html.escape(first_letter)}</td>")
        else:
            html_parts.append("<td>&nbsp;</td>")
    html_parts.append("</tr>")

    html_parts.append("</table>")
    html_parts.append("</body></html>")

    html_doc = "\n".join(html_parts)
    HTML(string=html_doc).write_pdf(pdf_path)
    print(f"📄 PDF created: {pdf_path}")

from rapidfuzz import fuzz

def extract_bracket_sentences(text):
    """
    Extract [ nn ] sentences from final hybrid paragraph.
    Returns list of (num_as_int, sentence_text)
    """
    text = re.sub(r"\s+", " ", text)
    pattern = r"\[\s*([0-9OIl]+)\s*\]\s*(.*?)(?=\[\s*[0-9OIl]+\s*\]|$)"

    out = []
    for m in re.finditer(pattern, text):
        raw = m.group(1)
        cleaned = (
            raw.replace("O","0")
               .replace("o","0")
               .replace("I","1")
               .replace("l","1")
        )
        try:
            num = int(cleaned)
        except ValueError:
            continue
        sent = m.group(2).strip()
        out.append((num, sent))
    return out


def get_pdf_intruder_numbers_from_augmented(augmented_sentences, intruder_texts):
    """
    Compute PDF intruder numbers from the FINAL sentence list.

    For each intruder_text, find its index in augmented_sentences and
    return 1-based indices (these correspond to [ nn ] labels).

    Handles possible duplicates by not reusing the same index twice.
    """
    pdf_nums = []
    used_indices = set()

    for intr in intruder_texts:
        found_idx = None
        for i, s in enumerate(augmented_sentences):
            if i in used_indices:
                continue
            if s.strip() == intr.strip():
                found_idx = i
                break
        if found_idx is not None:
            used_indices.add(found_idx)
            pdf_nums.append(found_idx + 1)  # 1-based for [ nn ]
    return pdf_nums


def detect_intruders_by_fuzz(original_text, hybrid_text, threshold=0.60):
    """
    Similarity-based intruder detection over ALL labeled sentences.

    Returns:
        fuzz_intruder_numbers (list[int])
        fuzz_intruder_sents   (list[str])
        fuzz_intruder_scores  (list[float])
    """
    orig_sents = split_into_sentences(original_text)
    hyb_sents = extract_bracket_sentences(hybrid_text)

    fuzz_nums = []
    fuzz_texts = []
    fuzz_scores = []

    for num, sent in hyb_sents:
        best = 0.0
        for o in orig_sents:
            score = fuzz.ratio(sent.lower(), o.lower()) / 100
            if score > best:
                best = score

        if best < threshold:
            fuzz_nums.append(num)
            fuzz_texts.append(sent)
            fuzz_scores.append(round(best, 3))

    return fuzz_nums, fuzz_texts, fuzz_scores


def process_tsv(input_tsv, output_tsv):
    """
    Final hybrid processor with CORRECT intruder numbering.

    Answer key columns:

        student_id
        name
        original_text
        hybrid_text
        pdf_intruder_numbers      (from actual positions of intruder_texts)
        fuzz_intruder_numbers     (from RapidFuzz < 0.60 on hybrid text)
        fuzz_intruder_scores      (best similarity for those fuzz intruders)
        intruder_sentences        (the inserted intruder texts)
        replacement_words         (synonyms)
        original_words            (pre-replacement)
        pos_labels                (DeepSeek POS)
    """

    freq_ranks = load_frequency_ranks(FREQ_FILE)

    with open(output_tsv, "w", newline="", encoding="utf-8") as keyfile:
        writer = csv.writer(keyfile, delimiter="\t")

        # ---------- HEADER ----------
        writer.writerow([
            "student_id",
            "name",
            "original_text",
            "hybrid_text",
            "pdf_intruder_numbers",
            "fuzz_intruder_numbers",
            "fuzz_intruder_scores",
            "intruder_sentences",
            "replacement_words",
            "original_words",
            "pos_labels",
        ])

        with open(input_tsv, newline="", encoding="utf-8") as infile:
            reader = csv.reader(infile, delimiter="\t")

            for row in reader:
                if len(row) < 3:
                    continue

                student_id, name, text = row[0], row[1], row[2]
                print(f"\n=== Processing {student_id} / {name} ===")

                # 1. original sentences
                orig_sentences = split_into_sentences(text)
                if not orig_sentences:
                    print("⚠️ No sentences found. Skipping.")
                    continue

                # 2. synonym transformation
                synonym_modified_text, replacements = transform_text_with_synonyms(
                    text, freq_ranks
                )
                modified_original_sents = split_into_sentences(synonym_modified_text)

                # 3. insert intruders
                augmented_sents, intruder_positions, intruder_texts = \
                    insert_intruders_into_sentences(modified_original_sents)

                # ✅ CORRECT pdf intruder numbers: based on FINAL augmented_sents
                pdf_intruder_numbers = get_pdf_intruder_numbers_from_augmented(
                    augmented_sents,
                    intruder_texts
                )

                # 4. build unified hybrid paragraph with [ nn ]
                hybrid_paragraph = build_unified_paragraph(augmented_sents)

                # 5. RapidFuzz-based intruder detection (independent checksum)
                fuzz_nums, fuzz_sents, fuzz_scores = detect_intruders_by_fuzz(
                    text,
                    hybrid_paragraph,
                    threshold=0.60
                )

                # 6. generate PDF
                generate_pdf(
                    student_id=student_id,
                    name=name,
                    sentences=augmented_sents,
                    replacements=replacements,
                    pdf_dir=PDF_DIR
                )

                # 7. replacement metadata
                original_words = [orig for (orig, syn, pos) in replacements]
                replacement_words = [syn for (orig, syn, pos) in replacements]
                pos_labels = [pos for (orig, syn, pos) in replacements]

                # 8. write answer key row
                writer.writerow([
                    student_id,
                    name,
                    text,
                    hybrid_paragraph,
                    ",".join(str(n) for n in pdf_intruder_numbers),
                    ",".join(str(n) for n in fuzz_nums),
                    ",".join(str(s) for s in fuzz_scores),
                    " || ".join(intruder_texts),
                    ",".join(replacement_words),
                    ",".join(original_words),
                    ",".join(pos_labels),
                ])

    print(f"\n🎯 Done. Answer key saved to: {output_tsv}")


# ====================================================
# MAIN
# ====================================================

if __name__ == "__main__":
    random.seed()
    process_tsv(INPUT_TSV, ANSWER_KEY)