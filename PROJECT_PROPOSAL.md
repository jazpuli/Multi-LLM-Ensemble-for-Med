Project Proposal: Multi-LLM Ensemble Methods for Medical Question Answering

(Updated to include GPT-4, LLaMA-2, and Medical-ChatGPT as the three LLMs)

1. Tentative Title

Enhancing Medical Question Answering Through Multi-LLM Ensemble Synergy

2. Abstract

This project investigates ensemble learning techniques that combine three large language models—GPT-4, LLaMA-2, and Medical-ChatGPT—to improve performance on medical question answering (QA). Inspired by the LLM-Synergy framework (Han et al., 2023), we replicate and extend ensemble strategies including a Boosting-based Weighted Majority Vote and a Cluster-based Dynamic Model Selection pipeline. Evaluation will be performed on standard public datasets such as PubMedQA, MedQA-USMLE, and MedMCQA. The goal is to determine whether coordinated multi-model reasoning increases accuracy, stability, and reliability relative to using any model individually. 

Multi Llm Medical Qa (2)

3. Background and Motivation

Medical QA requires high-precision reasoning and domain-specific comprehension. Although individual models such as GPT-4, LLaMA-2, and specialized clinical models like Medical-ChatGPT achieve competitive results, their performance varies significantly by question style, medical domain, and dataset.

No single model is consistently superior across PubMedQA, MedQA-USMLE, and MedMCQA. Ensemble learning is a promising method for leveraging complementary strengths:

GPT-4 provides strong general reasoning.

LLaMA-2 provides transparency and efficiency.

Medical-ChatGPT offers domain-specific clinical grounding.

By combining these capabilities, we aim to build an ensemble that is more robust, more accurate, and less biased than any standalone model.

4. Objectives

Reproduce and extend the LLM-Synergy pipeline using three LLMs: GPT-4, LLaMA-2, and Medical-ChatGPT.

Evaluate each model individually to establish baselines on PubMedQA, MedQA-USMLE, and MedMCQA.

Implement and compare two ensemble approaches:

Boosting-based Weighted Majority Voting

Cluster-based Dynamic Model Selection

Analyze performance improvements, error types, and interpretability implications for medical QA.

5. Methodology
Ensemble Approaches
1. Boosting-Based Weighted Majority Vote

Train a weighting mechanism that increases each model’s influence based on question-level accuracy.

Hard or ambiguous questions receive higher weighting on models that historically perform better on similar examples.

2. Cluster-Based Dynamic Model Selection

Use contextual embeddings (e.g., ClinicalBERT) to cluster questions by topic or reasoning type.

Select the optimal subset or weighting of the three LLMs within each cluster.

Enables specialization (e.g., PubMed-style research reasoning vs. clinical diagnostic reasoning).

Datasets

PubMedQA — yes/no/maybe biomedical research questions

MedQA-USMLE — board-style multiple-choice clinical reasoning

MedMCQA — multi-topic medical exam questions

Evaluation Metric

Accuracy (primary metric due to multiple-choice setups)

Optional secondary analyses: calibration, rationalization quality, model disagreement patterns

6. Expected Results

Based on prior work and the added advantage of incorporating a dedicated medical model:

Ensemble accuracy is expected to exceed individual models by 2–7%, depending on dataset difficulty.

Dynamic selection should outperform static boosting in domains requiring specialized clinical knowledge.

Error analysis will highlight cases where the medical model corrects general-purpose models and vice versa, revealing specialization patterns.

Overall, the ensemble should exhibit higher stability, reduced hallucination rates, and improved domain-specific reasoning.