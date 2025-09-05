

# CS202 Labs: Mining & Analyzing Git Repositories

## Overview
This repository contains python code used in the process of **lab 2** and **lab 3** of cs202, focusing on **mining Git repositories** and **analyzing bug-fixing commits** using various tools and techniques.

csv files of bug-fixing commits are not added as they are large in size

---

## **Lab 2: Mining Git Repositories & Bug-Fixing Commits**
Contents:
- **Mining Git repositories** to extract commit data.
- Identifying **bug-fixing commits** and analyzing their properties.
- Understanding the **characteristics of effective commit messages**.
- Exploring how **Large Language Models (LLMs)** can:
  - Infer **fix types**.
  - Generate **commit messages**.
- Introduction to:
  - **PyDriller** — a Python library for mining software repositories.
  - **CommitPredictorT5** — an LLM designed to **predict fix types**

---

## **Lab 3: Analyzing Bug-Fixing Commits**
Building upon the bug-commit data mined in Lab 2, I learned to:
- Analyze properties of code related to bug-fixing commits.
- Concepts introduced:
  - **Cyclomatic Complexity** — measures code complexity based on control flow.
  - **Maintainability Index** — quantifies code maintainability.
  - Calculated using the **Radon** library.
- Learned about:
  - **BERT embeddings** for semantic code representation.
  - **BLEU score** to compare codes semantically **without keyword matching**.
- Focused on **analyzing bug-fixing commits** using these metrics and embeddings.

---

## **Tools Used**
- **Python**
- **PyDriller** — Git repository mining
- **CommitPredictorT5** — LLM for fix type prediction & commit message generation
- **Radon** — cyclomatic complexity & maintainability index calculation
- **BERT embeddings** — semantic code representation
- **BLEU score** — semantic code similarity metric


