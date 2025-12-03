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

def run_script(script, *args):
    cmd = [PY, script] + list(args)
    log(" ".join(cmd))
    subprocess.run(cmd, check=True)

def safe_move_pdfs(src_pattern, dst_dir):
    files = glob.glob(src_pattern)
    if not files:
        log(f"⚠ No PDFs in {src_pattern}, skipping.")
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
# FIND THE TWO MOST RECENT TSV FILES
# -------------------------------------------------------------
log(f"Finding 2 most recent TSVs in {DOWNLOADS} ...")
tsv_files = sorted(
    glob.glob(os.path.join(DOWNLOADS, "*.tsv")),
    key=os.path.getmtime,
    reverse=True
)

if len(tsv_files) < 2:
    raise SystemExit("❌ Need at least two TSV files!")

tsv1 = tsv_files[0]
tsv2 = tsv_files[1]

log(f"TSV #1: {tsv1}")
log(f"TSV #2: {tsv2}")

# -------------------------------------------------------------
# COPY TSVs AS students1.tsv AND students2.tsv
# -------------------------------------------------------------
dest1 = os.path.join(TESTDIR, "students1.tsv")
dest2 = os.path.join(TESTDIR, "students2.tsv")

shutil.copy(tsv1, dest1)
shutil.copy(tsv2, dest2)

log(f"Copied TSV 1 → {dest1}")
log(f"Copied TSV 2 → {dest2}")

# -------------------------------------------------------------
# SET WORKING DIRECTORY
# -------------------------------------------------------------
os.chdir(TESTDIR)
log(f"Working directory: {os.getcwd()}")

TIMESTAMP = datetime.now().strftime("%b%d-%H%M")

# -------------------------------------------------------------
# PROCESS A SINGLE TSV (run twice total)
# -------------------------------------------------------------
def process_tsv_run(order_number, tsv_file):
    """
    Process a single TSV (run #1 or run #2).
    Steps:
      1. Copy tsv_file → students.tsv
      2. Run hybrid-intruder-synonym.py
      3. Delete temporary students.tsv
      4. Rename output folders/files with 1_ or 2_ prefix
    """

    log(f"\n===== RUN {order_number}: Processing {tsv_file} =====")

    # --------------------------------------------------------------------
    # 1. Copy TSV to students.tsv (temporary working name)
    # --------------------------------------------------------------------
    if not os.path.exists(tsv_file):
        raise SystemExit(f"❌ TSV missing: {tsv_file}")

    if os.path.exists("students.tsv"):
        os.remove("students.tsv")

    shutil.copy(tsv_file, "students.tsv")
    log(f"✔ Copied {tsv_file} → students.tsv")

    # --------------------------------------------------------------------
    # 2. Run the hybrid generator
    # --------------------------------------------------------------------
    log(f"▶ Running hybrid-intruder-synonym.py for run {order_number} ...")
    run_script("hybrid-intruder-synonym.py")

    # --------------------------------------------------------------------
    # 3. DELETE temporary students.tsv after processing
    # --------------------------------------------------------------------
    if os.path.exists("students.tsv"):
        os.remove("students.tsv")
        log("✔ Deleted temporary students.tsv")
    else:
        log("⚠ Temporary students.tsv not found after processing")

    # --------------------------------------------------------------------
    # 4. Rename output directory + output answer key
    # --------------------------------------------------------------------
    # These names must match EXACTLY what hybrid script creates
    old_pdf_dir = "PDFs-hybrid-synonym-intruders"
    new_pdf_dir = f"{order_number}_PDFs-hybrid-synonym-intruders"

    old_key = "answer_key_hybrid_synonym_intruders.tsv"
    new_key = f"{order_number}_answer_key_hybrid_synonym_intruders.tsv"

    # ---- Rename PDFs folder
    if os.path.exists(old_pdf_dir):
        if os.path.exists(new_pdf_dir):
            shutil.rmtree(new_pdf_dir)
        shutil.move(old_pdf_dir, new_pdf_dir)
        log(f"✔ Renamed {old_pdf_dir} → {new_pdf_dir}")
    else:
        log(f"⚠ No PDF directory {old_pdf_dir} found — skipping rename.")

    # ---- Rename answer key
    if os.path.exists(old_key):
        if os.path.exists(new_key):
            os.remove(new_key)
        shutil.move(old_key, new_key)
        log(f"✔ Renamed {old_key} → {new_key}")
    else:
        log(f"⚠ No answer key {old_key} found — skipping rename.")

    log(f"===== RUN {order_number} COMPLETE =====\n")
# -------------------------------------------------------------
# RUN PIPELINE FOR EACH TSV
# -------------------------------------------------------------
process_tsv_run(1, "students1.tsv")
process_tsv_run(2, "students2.tsv")

# -------------------------------------------------------------
# LONG.PY ON EACH HYBRID PDF FOLDER
# -------------------------------------------------------------
log("Running long.py on 1_PDFs-hybrid-synonym-intruders ...")
run_script("long.py", "1_PDFs-hybrid-synonym-intruders")
safe_move_pdfs("long_fixed/*.pdf", "1_PDFs-hybrid-synonym-intruders")
shutil.rmtree("long", ignore_errors=True)

log("Running long.py on 2_PDFs-hybrid-synonym-intruders ...")
run_script("long.py", "2_PDFs-hybrid-synonym-intruders")
safe_move_pdfs("long_fixed/*.pdf", "2_PDFs-hybrid-synonym-intruders")
shutil.rmtree("long", ignore_errors=True)

# -------------------------------------------------------------
# MERGE PDFs WITH MATCHING NAMES FOR EACH SET
# -------------------------------------------------------------
log("Merging run 1 + run 2 hybrid PDFs ...")
run_script(
    "merge_matchingPDFs.py",
    "1_PDFs-hybrid-synonym-intruders/",
    "2_PDFs-hybrid-synonym-intruders/"
)

# -------------------------------------------------------------
# MERGE ALL PDFs INTO ONE FINAL merged.pdf
# -------------------------------------------------------------
log("Running merge_pdfs.py on merged/")
run_script("merge_pdfs.py", "merged/")

final_pdf = os.path.join(TESTDIR, "merged", "merged.pdf")
dest_pdf = os.path.join(DOWNLOADS, f"{TIMESTAMP}-merged.pdf")

if not os.path.exists(final_pdf):
    log("❌ merged.pdf not found!")
    raise SystemExit()

shutil.copy(final_pdf, dest_pdf)
log(f"Copied merged.pdf → {dest_pdf}")

# -------------------------------------------------------------
# ARCHIVING EVERYTHING
# -------------------------------------------------------------
RUN_DIR = os.path.join(ARCHIVE_DIR, f"{TIMESTAMP}-run")
os.makedirs(RUN_DIR, exist_ok=True)
log(f"✔ Created run archive folder: {RUN_DIR}")

def safe_archive(path):
    if os.path.exists(path):
        shutil.move(path, os.path.join(RUN_DIR, os.path.basename(path)))
        log(f"✔ Archived {path}")
    else:
        log(f"⚠ {path} not found, skipping.")

# Archive PDF folders
safe_archive("1_PDFs-hybrid-synonym-intruders")
safe_archive("2_PDFs-hybrid-synonym-intruders")
safe_archive("merged")

# Archive answer keys
safe_archive("1_answer_key_hybrid_synonym_intruders.tsv")
safe_archive("2_answer_key_hybrid_synonym_intruders.tsv")

# Archive TSVs used
safe_archive("students1.tsv")
safe_archive("students2.tsv")

log("✔ All tasks complete!")