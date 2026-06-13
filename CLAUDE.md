# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

A structured self-education project in deep learning, built around one concrete anchor project: a **vision-driven GUI test agent** that perceives a Windows diagnostic tool's screen as pixels and emits mouse/keyboard actions — no DOM, no XPath, no accessibility trees. Every phase of learning exists to serve this build.

The owner is a software architect and test engineer with a strong Python background, Fast.AI experience (through Lesson 5), and familiarity with LangChain/LangGraph. Deep learning theory through feedforward networks and CNNs is solid; Transformers onward is the active frontier.

## Repository Structure

Each phase is a self-contained directory with three subdirectories:

- `notes/` — markdown concept notes, one file per topic, plus paper reading notes
- `src/` — Python implementations; files are numbered to reflect the learning progression
- `resources/` — links and reference material

Current phase: `phase-1-transformer/` — building a character-level GPT from scratch, following Karpathy's *Let's build GPT from scratch* video. Files in `src/` progress: bigram baseline → single-head attention → multi-head attention → full GPT.

## Phase Map

```
Phase 1  Transformer internals (self-attention, Q/K/V, multi-head, positional encoding)
Phase 2  LLM internals (tokenization, embeddings, decoder-only architecture, scale)
Phase 3  Vision-Language Models (ViT, CLIP, LLaVA/Qwen-VL)
Phase 4  Agentic perceive→reason→act loop (LangGraph, tool use, structured outputs)
Phase 5  RAG for UI memory (embeddings, vector DBs, semantic retrieval)
Phase 6  Fine-tuning for the domain (LoRA, PEFT, instruction tuning)
```

Phases 1–3 are understanding-focused. Phase 4 is where something first runs. Do not let theory phases expand indefinitely.

## Conventions

- `src/` files in each phase are numbered (`01_`, `02_`, …) to reflect the build-up order — preserve this when adding files.
- Notes files are numbered to match the concepts in build-up order; paper notes come after the implementation notes for that phase.
- No test suite yet — correctness is verified by running the scripts and checking outputs manually.
- Git commits go on `main` directly. Do not add Claude as a co-author in commit messages.
