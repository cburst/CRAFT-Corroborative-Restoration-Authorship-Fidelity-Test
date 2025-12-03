#!/usr/bin/env python3
import shutil
import subprocess
import os
import glob
from datetime import datetime

# -------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------
DOWNLOADS = "/Users/rescreen/Downloads"
TESTDIR = "/Users/rescreen/Downloads/surprise tests"
ARCHIVE_DIR = os.path.join(TESTDIR, "old")
PY = "/opt/homebrew/opt/python@3.11/bin/python3.11"

def log(msg):
    print(msg, flush=True)

# -------------------------------------------------------------
# 1. Find most recent TSV → students.tsv
# -------------------------------------------------------------
log(f"Finding most recent TSV in {DOWNLOADS} ...")
tsv_files = sorted(
    glob.glob(os.path.join(DOWNLOADS, "*.tsv")),
    key=os.path.getmtime,
    reverse=True
)

if not tsv_files:
    raise SystemExit("❌ No TSV files found!")

latest = tsv_files[0]
log(f"Found latest TSV: {latest}")

dest_students = os.path.join(TESTDIR, "students.tsv")

# Clean any previous
if os.path.exists(dest_students):
    os.remove(dest_students)

shutil.copy(latest, dest_students)
log(f"✔ Copied → {dest_students}")

# -------------------------------------------------------------
# 2. Change working directory
# -------------------------------------------------------------
os.chdir(TESTDIR)
log(f"Working directory: {os.getcwd()}")

# Timestamp like Jan27-0938
TIMESTAMP = datetime.now().strftime("%b%d-%H%M")

# -------------------------------------------------------------
# Helper to run python scripts
# -------------------------------------------------------------
def run_script(script, *args):
    cmd = [PY, script] + list(args)
    log(" ".join(cmd))
    subprocess.run(cmd, check=True)

# -------------------------------------------------------------
# Helper: move PDFs safely
# -------------------------------------------------------------
def safe_move_pdfs(src_pattern, dst_dir):
    files = glob.glob(src_pattern)
    if not files:
        log(f"⚠ No PDFs matching {src_pattern}")
        return

    os.makedirs(dst_dir, exist_ok=True)

    for f in files:
        base = os.path.basename(f)
        dst = os.path.join(dst_dir, base)

        try:
            if os.path.exists(dst):
                os.remove(dst)
            shutil.move(f, dst)
        except Exception as e:
            log(f"❌ Failed to move {f} → {dst}: {e}")
        else:
            log(f"Moved {f} → {dst}")

    log(f"✔ Moved {len(files)} files into {dst_dir}")

# -------------------------------------------------------------
# 3. Run hybrid generator (ONE RUN ONLY)
# -------------------------------------------------------------
log("▶ Running hybrid-intruder-synonym.py (real mode) ...")
run_script("hybrid-intruder-synonym.py")

# Delete temporary students.tsv
if os.path.exists("students.tsv"):
    os.remove("students.tsv")
    log("✔ Deleted temporary students.tsv")

# Rename outputs to avoid collisions with test pipeline
old_pdf_dir = "PDFs-hybrid-synonym-intruders"
new_pdf_dir = "real_PDFs-hybrid-synonym-intruders"

old_key = "answer_key_hybrid_synonym_intruders.tsv"
new_key = "real_answer_key_hybrid_synonym_intruders.tsv"

# Rename PDF folder
if os.path.exists(old_pdf_dir):
    if os.path.exists(new_pdf_dir):
        shutil.rmtree(new_pdf_dir)
    shutil.move(old_pdf_dir, new_pdf_dir)
    log(f"✔ Renamed {old_pdf_dir} → {new_pdf_dir}")
else:
    log(f"⚠ No folder {old_pdf_dir} found (expected output missing).")

# Rename answer key
if os.path.exists(old_key):
    if os.path.exists(new_key):
        os.remove(new_key)
    shutil.move(old_key, new_key)
    log(f"✔ Renamed {old_key} → {new_key}")
else:
    log(f"⚠ No answer key {old_key} found.")

# -------------------------------------------------------------
# 4. Run long.py on the real folder
# -------------------------------------------------------------
log(f"▶ Running long.py on {new_pdf_dir} ...")
run_script("long.py", new_pdf_dir)

# Move rearranged PDFs back into the real folder
safe_move_pdfs("long_fixed/*.pdf", new_pdf_dir)

# Cleanup
shutil.rmtree("long_fixed", ignore_errors=True)
shutil.rmtree("long", ignore_errors=True)

# -------------------------------------------------------------
# 5. MERGE ALL PDFs IN THE REAL FOLDER → ONE PDF
# -------------------------------------------------------------
log("▶ Merging all real PDFs into one (merge_pdfs.py) ...")
run_script("merge_pdfs.py", new_pdf_dir)

# merge_pdfs.py saves as: <folder>/<folder_basename>.pdf
folder_basename = os.path.basename(new_pdf_dir)
final_pdf = os.path.join(new_pdf_dir, f"{folder_basename}.pdf")

output_pdf = os.path.join(DOWNLOADS, f"{TIMESTAMP}-real.pdf")

if not os.path.exists(final_pdf):
    raise SystemExit(f"❌ {final_pdf} not found in real folder!")

shutil.copy(final_pdf, output_pdf)
log(f"✔ Copied merged real PDF → {output_pdf}")

# -------------------------------------------------------------
# 6. Archive run contents
# -------------------------------------------------------------
RUN_DIR = os.path.join(ARCHIVE_DIR, f"{TIMESTAMP}-real-run")
os.makedirs(RUN_DIR, exist_ok=True)
log(f"✔ Created archive folder: {RUN_DIR}")

def safe_archive(path):
    if os.path.exists(path):
        shutil.move(path, os.path.join(RUN_DIR, os.path.basename(path)))
        log(f"Archived {path}")
    else:
        log(f"⚠ {path} not found — skipping")

# Archive relevant output
safe_archive(new_pdf_dir)
safe_archive(new_key)

log("✔ Real pipeline complete!")