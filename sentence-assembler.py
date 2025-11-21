#!/usr/bin/env python3
import csv
import os
import re
import html
from weasyprint import HTML
import nltk
from nltk.tokenize import sent_tokenize

# Ensure punkt
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")


# -------------------------
# STOPWORDS
# -------------------------
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


# -------------------------
# Basic Helpers
# -------------------------

def sanitize_filename(name):
    return "".join(c for c in name if c not in r'\/:*?"<>|').strip() or "student"

def split_into_sentences(text):
    sentences = sent_tokenize(text)
    return [s.strip() for s in sentences if s.strip()]

def tokenize_preserving_hyphens_apostrophes(text):
    """Hyphenated and apostrophized words count as single words."""
    return re.findall(r"[a-z0-9][a-z0-9'\-]*", text.lower())


# -------------------------
# NEW: Expand outward until ≥20 words
# -------------------------

def expand_block(sentences, idx_longest):
    """
    Alternating expansion: prev → next → prev2 → next2 → ...
    Returns the expanded block as a text string AND token list.
    """
    total_sentences = len(sentences)
    block_indices = [idx_longest]

    # Keep expanding until block has >= 20 tokens
    left_offset = 1
    right_offset = 1

    def block_tokens():
        text = " ".join(sentences[i] for i in block_indices)
        return tokenize_preserving_hyphens_apostrophes(text)

    # Expand until enough tokens (or exhaustion)
    while True:
        tokens_now = block_tokens()
        if len(tokens_now) >= 10:
            break

        expanded = False

        # Alternate: previous, then next
        if left_offset <= right_offset:
            prev_idx = idx_longest - left_offset
            if prev_idx >= 0:
                block_indices.insert(0, prev_idx)
                left_offset += 1
                expanded = True
            else:
                left_offset += 1  # skip nonexistent
        else:
            next_idx = idx_longest + right_offset
            if next_idx < total_sentences:
                block_indices.append(next_idx)
                right_offset += 1
                expanded = True
            else:
                right_offset += 1  # skip nonexistent

        # If neither side can expand, break
        if not expanded:
            break

    # Construct final block text + tokens
    block_text = " ".join(sentences[i] for i in block_indices)
    block_tokens = tokenize_preserving_hyphens_apostrophes(block_text)

    return block_text, block_tokens


# -------------------------
# Meaningfulness-based 20-word window selection
# -------------------------

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


# -------------------------
# Replace tokens with WHITE underscores
# -------------------------

def replace_tokens_with_blanks_in_block(block_text, block_tokens, removed_positions):
    """
    Replace ONLY the tokens at `removed_positions` inside block_text.
    Uses <span style='color:white'>__</span> of length*2 underscores.
    Punctuation (including period+space) preserved exactly.
    """

    # Compile token/punct splitter that includes punctuation AND keeps periods.
    pattern = re.compile(r"[A-Za-z0-9][A-Za-z0-9'\-]*|[.,;:!?]")

    parts = pattern.findall(block_text)

    rebuilt = []
    token_i = 0

    for part in parts:
        # Word?
        if re.match(r"[A-Za-z0-9]", part):
            if token_i in removed_positions:
                blank = "_" * (len(part) * 2)
                blank_html = f"<span style='color:white'>{blank}</span>"
                rebuilt.append(blank_html)
            else:
                rebuilt.append(part)
            token_i += 1
        else:
            # punctuation preserved
            rebuilt.append(part)

	# Reconstruct with corrected spacing (fixes missing space after punctuation)
	out = ""
	last = ""
	for piece in rebuilt:
		if last == "":
			out += piece
		else:
			# If the previous token ends with punctuation → force a space before this token.
			# Fixes: "controlled,the" → "controlled, the"
			if last.endswith(('.', ',', ';', ':', '!', '?')):
				out += " " + piece

			# If THIS token *is* punctuation → attach directly to the previous token.
			elif piece in ".,;:!?":
				out += piece

			# Otherwise, normal word → word spacing
			else:
				out += " " + piece

		last = piece

	return out


# -------------------------
# Apply transform to longest sentence (new block-based logic)
# -------------------------

def transform_longest_sentence(text, longest_sentence):
    # Get all sentences
    sentences = split_into_sentences(text)
    idx_longest = sentences.index(longest_sentence)

    # 1. Expand outward until >=20 words
    block_text, block_tokens = expand_block(sentences, idx_longest)

    # 2. Select best 20-word window across entire block
    removed_positions = select_best_window(block_tokens, window=10)

    # 3. Replace window tokens with invisible blanks
    modified_block = replace_tokens_with_blanks_in_block(
        block_text,
        block_tokens,
        set(removed_positions)
    )

    # 4. Replace the original entire block inside the text
    full_text_modified = text.replace(block_text, modified_block)

    # 5. Extract removed words IN ORDER
    removed_words = [block_tokens[i] for i in removed_positions]

    return full_text_modified, removed_words


# -------------------------
# PDF Generator
# -------------------------

def build_word_grid(words):
    words = sorted([w.lower() for w in words])
    while len(words) < 10:
        words.append("")
    words = words[:10]
    return [words[i*2:(i+1)*2] for i in range(5)]

def generate_pdf(student_id, name, full_text_modified, grid, out_dir="PDFs-sentence-assembly"):
    os.makedirs(out_dir, exist_ok=True)
    pdf_path = os.path.join(out_dir, f"{sanitize_filename(name)}.pdf")

    esc_name = html.escape(name)
    esc_id = html.escape(student_id)
    esc_text = full_text_modified  # IMPORTANT: unescaped so spans work

    grid_html = "<table style='width:100%; font-size:14pt; border-collapse:collapse;'>"
    for row in grid:
        grid_html += "<tr>"
        for cell in row:
            grid_html += (
                f"<td style='border:1px solid #ccc; padding:6px; text-align:center;'>"
                f"{html.escape(cell)}</td>"
            )
        grid_html += "</tr>"
    grid_html += "</table>"

    html_doc = f"""
    <html>
    <head>
    <meta charset='utf-8'>
    <style>
        @page {{
            margin: 1.5cm;
            size: A4;
        }}
        body {{
            font-family: Arial, sans-serif;
            font-size: 13pt;
            line-height: 1.4;
        }}
        .header {{
            font-weight: bold;
            margin-bottom: 0.5em;
        }}
        .sentence {{
            margin-top: 1em;
            margin-bottom: 1em;
            white-space: pre-wrap;
        }}
    </style>
    </head>
    <body>

        <div class='header'>
            Name: {esc_name}<br>
            Student Number: {esc_id}<br>
            Sentence Assembler
        </div>

        <div>
            A section of the text has had 20 words removed.  
            Use the 20 words in the grid to reconstruct the text.
        </div>

        <div class='sentence'><b>Text:</b><br>{esc_text}</div>

        <div class='header'>Word Bank (Alphabetized)</div>
        {grid_html}

    </body>
    </html>
    """

    HTML(string=html_doc).write_pdf(pdf_path)
    print(f"📝 PDF created: {pdf_path}")


# -------------------------
# Main TSV Processor
# -------------------------

def process_tsv(input_tsv, output_tsv):
    with open(output_tsv, "w", newline="", encoding="utf-8") as outfile:
        writer = csv.writer(outfile, delimiter="\t")
        writer.writerow(["student_id", "name", "removed_words"])

        with open(input_tsv, newline="", encoding="utf-8") as infile:
            reader = csv.reader(infile, delimiter="\t")

            for row in reader:
                if len(row) < 3:
                    continue

                student_id, name, text = row

                sentences = split_into_sentences(text)
                if not sentences:
                    continue

                # Longest sentence
                longest_sentence = max(sentences, key=lambda s: len(tokenize_preserving_hyphens_apostrophes(s)))

                # Apply new logic
                full_text_modified, removed_words = transform_longest_sentence(
                    text,
                    longest_sentence
                )

                grid = build_word_grid(removed_words)

                generate_pdf(student_id, name, full_text_modified, grid)

                writer.writerow([student_id, name, ", ".join(removed_words)])

    print(f"✓ Done. Answer key written to {output_tsv}")


# -------------------------
if __name__ == "__main__":
    INPUT_TSV = "students.tsv"
    OUTPUT_TSV = "sentence_assembly_answer_key.tsv"
    process_tsv(INPUT_TSV, OUTPUT_TSV)