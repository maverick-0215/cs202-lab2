from pydriller import Repository
import csv
"""
bug_fix_keywords = [
   "fix", "bug", "error", "issue", "patch", "crash", "exception",
   "fail", "failure", "broken", "correct", "resolve", "resolved",
   "repair", "defect", "mistake", "incorrect", "typo", "handle",
   "null", "undefined", "invalid", "workaround", "segfault",
   "infinite loop", "hang", "freeze", "stacktrace", "fixes", "fixed",
   "resolves", "patches", "patched", "corrects", "corrected",
]

REPO_URL = "https://github.com/comfyanonymous/ComfyUI"   

with open('bug_fix_commits_py.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['Hash', 'Message', 'Parent Hashes', 'Is a Merge Commit?', 'List of Modified Files'])

    for commit in Repository(REPO_URL, only_modifications_with_file_types=['.py']).traverse_commits():
        msg_lower = commit.msg.lower()
        if any(keyword in msg_lower for keyword in bug_fix_keywords):
            parent_hashes = ', '.join(parent for parent in commit.parents)
            modified_files = ', '.join(mod.new_path for mod in commit.modified_files if mod.new_path)
            writer.writerow([
                commit.hash,
                commit.msg,
                parent_hashes,
                commit.merge,
                modified_files
            ])

print("Filtered bug-fixing commits that modify .py files saved.")
"""
import sys
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
# Huggingface model repo
MODEL_NAME = "mamiksik/CommitPredictorT5"

print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

def infer_fix_type(diff_text):
    """
    Given a diff (or commit message fallback), generate a concise, precise
    imperative commit message via CommitPredictorT5.
    """
    prompt = diff_text

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=32)
    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return prediction


REPO_URL = "https://github.com/comfyanonymous/ComfyUI"

input_csv = "bug_fix_commits_py.csv"
output_csv = "bug_fix_commit_diffs_with_llm.csv"

commit_hashes = []

# Read bug-fixing commit hashes
with open(input_csv, newline='', encoding='utf-8') as infile:
    reader = csv.DictReader(infile)
    for row in reader:
        commit_hashes.append(row['Hash'])

with open(output_csv, 'w', newline='', encoding='utf-8') as outfile:
    writer = csv.writer(outfile)
    writer.writerow([
        'Commit Hash',
        'Commit Message',
        'Filename',
        'Source code(changed lines before)',
        'Souce code(changed lines after)',
        'Diff',
        'LLM Inference (fix type)',
        'Rectified Message'
    ])

    print(f"Processing {len(commit_hashes)} commits...")

    for commit in Repository(REPO_URL, only_commits=commit_hashes).traverse_commits():
        for mod in commit.modified_files:
            diff_text = mod.diff or ""
            # Extract only changed lines
            before_lines = "\n".join(
                f"{ln}: {txt}" for ln, txt in mod.diff_parsed['deleted']
            )
            after_lines = "\n".join(
                f"{ln}: {txt}" for ln, txt in mod.diff_parsed['added']
            )

            # Use diff for inference, fallback to commit message if diff empty
            text_for_inference = diff_text if diff_text else commit.msg

            try:
                fix_type = infer_fix_type(text_for_inference)
            except Exception as e:
                print(f"LLM inference failed on commit {commit.hash}: {e}")
                fix_type = ""

            writer.writerow([
                commit.hash,
                commit.msg,
                mod.new_path or mod.old_path,
                before_lines,
                after_lines,
                diff_text,
                fix_type,
                ''  # Rectified Message placeholder
            ])

print("Done! Output written to", output_csv)