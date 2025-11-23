#!/usr/bin/env python3
import csv
import os
import re
import random
import time
import html
import json
import requests
from weasyprint import HTML

import nltk
from nltk.tokenize import sent_tokenize

# Ensure punkt tokenizer
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

# =========================
# CONFIGURATION
# =========================

INPUT_TSV = "students.tsv"
PDF_DIR = "PDFs-hybrid-sentence-intruders"
ANSWER_KEY = "answer_key_hybrid_sentence_intruders.tsv"

DEEPSEEK_API_KEY = "YOUR-API-KEY-HERE"
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"
MAX_DEEPSEEK_RETRIES = 5

NUM_INTRUDERS = 3


# =========================
# UTILITIES
# =========================

def split_into_sentences(text):
    if not text:
        return []
    sents = sent_tokenize(str(text))
    return [s.strip() for s in sents if s.strip()]

def tokenize_words(text):
    return re.findall(r"[A-Za-z']+", str(text).lower())

def normalize_sentence(sent):
    return " ".join(tokenize_words(sent)).strip()

def sanitize_filename(name):
    forbidden = r'\/:*?"<>|'
    return "".join(c for c in name if c not in forbidden).strip() or "student"


# =========================
# GENERATE INTRUDERS USING THIRDS
# =========================

def split_text_into_thirds(sentences):
    """Return 3 lists: first third, middle third, last third."""
    n = len(sentences)
    if n < 3:
        return [sentences], [], []   # one part only

    one_third = n // 3

    part1 = sentences[0:one_third]
    part2 = sentences[one_third:2*one_third]
    part3 = sentences[2*one_third:]

    # Ensure no part is empty
    return (
        part1 if part1 else sentences,
        part2 if part2 else sentences,
        part3 if part3 else sentences
    )


def generate_intruder_sentence_with_deepseek(
        essay_section,
        existing_sentences,
        intruder_index,
        model=DEEPSEEK_MODEL,
        max_retries=MAX_DEEPSEEK_RETRIES):

    if not DEEPSEEK_API_KEY:
        raise RuntimeError("DeepSeek API key not set.")

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
        f"{' '.join(essay_section)}\n\n"
        "Write one plausible standalone sentence that matches the STYLE and CONTENT CHARACTERISTICS of THIS SECTION."
    )

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",  "content": user_prompt},
        ],
        "max_tokens": 200,
        "temperature": 0.8,  # increase randomness for variety
    }

    existing_norms = {normalize_sentence(s) for s in existing_sentences}

    attempt = 1
    while attempt <= max_retries:
        try:
            print(f"→ Intruder {intruder_index}, attempt {attempt}   ({intruder_index}-{attempt})")

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


# =========================
# BUILD AUGMENTED TEXT (USING THIRDS)
# =========================

def build_augmented_text_with_intruders(text, num_intruders):

    sentences = split_into_sentences(text)
    if len(sentences) < 1:
        return None, None, None, None

    part1, part2, part3 = split_text_into_thirds(sentences)
    parts = [part1, part2, part3]

    intruders = []

    for idx in range(1, num_intruders + 1):
        target_section = parts[(idx - 1) % len(parts)]
        intruder = generate_intruder_sentence_with_deepseek(
            essay_section=target_section,
            existing_sentences=sentences + intruders,
            intruder_index=idx
        )
        intruders.append(intruder)

    # Combine and shuffle
    all_sentences = sentences + intruders
    flags = [False] * len(sentences) + [True] * len(intruders)

    indices = list(range(len(all_sentences)))
    random.shuffle(indices)

    shuffled = [all_sentences[i] for i in indices]
    shuffled_flags = [flags[i] for i in indices]

    intruder_positions = [i for i, f in enumerate(shuffled_flags) if f]

    # Original sentence order (using scrambled sentence numbers)
    original_order_numbers = []
    for orig_idx in range(len(sentences)):
        # Find where this original sentence ended up in the shuffled list
        shuffled_pos = indices.index(orig_idx)
        original_order_numbers.append(shuffled_pos + 1)  # convert to 1-based

    return shuffled, intruder_positions, intruders, original_order_numbers


# =========================
# PDF GENERATION
# =========================

def generate_pdf_for_student(student_id, name, sentences, num_intruders, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    safe_name = sanitize_filename(name)
    pdf_path = os.path.join(out_dir, f"{safe_name}.pdf")

    esc_name = html.escape(name)
    esc_id   = html.escape(student_id)

    plural = "sentence" if num_intruders == 1 else "sentences"

    html_parts = [
        "<html><head><meta charset='utf-8'><style>",
        "@page { margin: 1.1cm; size: A4; }",
        "body { font-family: Arial, sans-serif; font-size: 12pt; line-height: 1.3; }",
        ".header { font-weight: bold; margin-bottom: 0.5em; }",
        ".text { white-space: pre-wrap; margin: 0.3em 0; }",
        "</style></head><body>",

        f"<div class='header'>Name: {esc_name}<br>Student Number: {esc_id}<br>Sentences Added & Sentence Order</div>",
        f"<div class='text'><b>{num_intruders} extra {plural} sentences have been added to your text.<br>Find the added sentences and provide the correct order of the original sentences.</b><br><br></div>",
    ]

    for i, sent in enumerate(sentences, start=1):
        esc = html.escape(sent)
        html_parts.append(f"<div class='text'><b>{i}.</b> {esc}</div>")

    # Two lines of underscores at the bottom
    html_parts.append("<br>")
    html_parts.append("<div class='text'><b>Added Sentences:</b> _______________________________________</div>")
    html_parts.append("<div class='text'><b>Original Sentence Order:</b> _________________________________</div>")

    html_parts.append("</body></html>")
    html_doc = "\n".join(html_parts)

    HTML(string=html_doc).write_pdf(pdf_path)
    print(f"📄 PDF created: {pdf_path}")


# =========================
# MAIN TSV PROCESSOR
# =========================

def process_tsv(input_tsv, output_pdf_dir, answer_key_path):
    with open(answer_key_path, "w", newline="", encoding="utf-8") as keyfile:
        writer = csv.writer(keyfile, delimiter="\t")
        writer.writerow([
            "student_id",
            "name",
            "intruder_sentence_numbers",
            "intruder_sentences",
            "original_sentence_order_numbers",
            "original_text",
            "augmented_text"
        ])

        with open(input_tsv, encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")

            for row in reader:
                if len(row) < 3:
                    continue

                student_id, name, text = row[0], row[1], row[2]

                print(f"\n=== Processing {student_id} / {name} ===")

                shuffled, intruder_positions, intruders, original_order_numbers = \
                    build_augmented_text_with_intruders(text, NUM_INTRUDERS)

                if shuffled is None:
                    print("⚠️ Not enough text; skipping.")
                    continue

                generate_pdf_for_student(
                    student_id,
                    name,
                    shuffled,
                    len(intruders),
                    output_pdf_dir
                )

                numbers = [pos + 1 for pos in intruder_positions]
                writer.writerow([
                    student_id,
                    name,
                    ",".join(str(n) for n in numbers),
                    " || ".join(intruders),
                    ",".join(str(n) for n in original_order_numbers),
                    text,
                    " ".join(shuffled)
                ])
                keyfile.flush()

    print(f"\n🎯 Done. Answer key saved to: {answer_key_path}")


# =========================
# MAIN
# =========================

if __name__ == "__main__":
    random.seed()
    process_tsv(INPUT_TSV, PDF_DIR, ANSWER_KEY)