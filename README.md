# Rigorous AI — Deep Learning for Test Engineers

*Vision-Driven GUI Test Agent — Learn by Building*

---

## 1. Goals

### Goal 1 — Conceptual Mastery

Understand the architecture and internals of modern deep learning methods well enough to read implementations, evaluate design choices, and reason about trade-offs — not just use APIs.

This covers: neural network foundations, transformer architecture and the self-attention mechanism, training dynamics, fine-tuning strategies, and the current landscape of LLMs, vision-language models, and multimodal systems.

---

### Goal 2 — Applied Engineering

The practical dimension is structured around three concrete sub-goals, each grounded in the daily work of a software architect and test engineer.

**2a — AI-Augmented Testing & QA**
Design and build AI-assisted testing pipelines: test generation, anomaly detection in logs, intelligent test selection, and failure classification.

**2b — LLM Integration in Own Tools**
Embed LLMs into services with confidence — prompt engineering, RAG, structured outputs, tool use, evaluation, and knowing *when not to use them*.

**2c — Domain-Specific Model Adaptation**
Understand fine-tuning workflows (LoRA, instruction tuning, RLHF basics) well enough to adapt a model to a specific domain and evaluate whether it worked.

---

### Guiding Principle — Disciplined Learning Habit

This is not a goal but the principle that governs the entire project.

Build a sustainable, thinking-oriented study practice: read primary sources (papers, not just tutorials), implement key concepts from scratch at least once, and maintain a personal knowledge base that compounds over time. The goal is to learn and think — not to collect certificates or finish courses.

---

## 2. Anchor Project

**Vision-Driven GUI Test Agent**

The learning path is structured around one real-world project: building a vision-driven GUI test agent from first principles. The target application is a Windows-based diagnostic tool that interacts with engine control units via UDS diagnostics protocol.

The agent perceives the screen as pixels, reasons about what it sees, and emits mouse/keyboard actions — with no DOM or selector coupling. No XPath. No accessibility trees. Pure vision.

The project serves three roles simultaneously:
- **Motivation** — every theoretical chapter is introduced because the project needs it
- **Implementation ground** — concepts are proven by building, not by reading
- **Exam** — at each phase, something runs

| Deep Learning Concept | Role in the Project |
|---|---|
| Transformer / self-attention | Core reasoning mechanism of the VLM |
| Vision Transformer (ViT) | Screenshot perception — images as patch tokens |
| Vision-Language Models (VLMs) | The agent's perceptual and reasoning system |
| RAG + vector databases | UI element memory across test sessions |
| Agentic loop (perceive → reason → act) | The test execution architecture |
| Fine-tuning / LoRA | Adapting the VLM to the diagnostic tool's specific UI |
| AI evaluation methods | Measuring whether the agent actually works |

---

## 3. Starting Position

### 3.1 Background

- Software development education: 1997–2000 (algorithmic foundation)
- Career as Test Engineer with strong focus on test automation
- Experience: embedded software testing, safety-critical systems, aviation software verification per DO-178B (DAL A through DAL D)
- ISTQB Test Analyst certified
- Current project: testing a Windows-based diagnostic tool (UDS protocol, ECU interaction)
- Strong Python practitioner; advanced Claude Code and GitHub Copilot user
- LangChain / LangGraph — working familiarity with agentic frameworks

**Deep learning study so far:**
- Fast.AI *Practical Deep Learning for Coders* — completed through Lesson 5 (from-scratch model), with own Python implementation
- *Make Your Own Neural Network* by Tariq Rashid — read and applied

---

### 3.2 Strengths That Accelerate the Path

**Mathematical foundation is solid**
Backpropagation implemented from scratch. The training loop, gradient flow, and loss surface are understood mechanistically — not just conceptually. This is the threshold most practitioners never cross.

**Implementation velocity is high**
Deep Python experience combined with AI-assisted tooling means cognitive load can stay on *concepts*, not syntax. Ideas become code quickly.

**Agentic framework literacy**
LangChain/LangGraph familiarity means the agent architecture (perceive → reason → act) is not conceptually foreign. The gap is at the implementation internals, not the pattern.

**Testing rigour as a superpower**
DO-178B at DAL A is one of the most demanding software verification disciplines that exists. The instinct to think in terms of coverage, determinism, failure modes, and traceability is exactly what the AI industry needs — and largely lacks — when it comes to evaluating AI systems. This background will make AI evaluation work deeper than most ML engineers produce.

---

### 3.3 Knowledge Frontier

| Area | Status |
|---|---|
| Backprop, gradient descent, training loop | ✅ Solid — implemented from scratch |
| Feedforward networks, basic CNNs | ✅ Covered (Fast.AI L1–L5) |
| Transformer / self-attention mechanism | 🔶 Not yet reached |
| LLM architecture internals | 🔶 Conceptual awareness only |
| Vision-Language Models (VLMs) | ⬜ New territory |
| Fine-tuning (LoRA, PEFT) | ⬜ New territory |
| RAG architecture | 🔶 Framework-level (LangChain), not internals |
| AI system evaluation methods | 🔶 Strong intuition, formal methods new |

**The entry point in one sentence:**

> You know how a network learns. You don't yet know the self-attention mechanism — the specific computation that makes Transformers capable of context-dependent, relational reasoning across sequences.

---

## 4. Learning Path

Six phases. Each ends with something *built*.

---

### Phase 1 — The Transformer from Scratch

**Core unlock:** Self-attention mechanism, Q/K/V computation, multi-head attention, positional encoding — implemented from scratch.

**Entry point:** You built a neural net from scratch in Fast.AI Lesson 5. Andrej Karpathy's *Let's build GPT from scratch* applies exactly the same pedagogy to the Transformer. One video, ~2 hours, produces a working character-level GPT.

**Project milestone:** Read the architecture diagram of any VLM and understand every component.

**Resources:**
- Karpathy — *Let's build GPT from scratch* (YouTube) — **start here**
- Paper: *Attention is All You Need* (Vaswani et al., 2017) — read *after* the video
- Fast.AI Part 2, Lessons 12–13 (NLP section)

---

### Phase 2 — How Modern LLMs Are Built

**Core unlock:** Tokenization, embeddings, decoder-only vs. encoder architecture, pre-training objectives, the scale dimension.

**Goal:** Load GPT-2 or a small LLaMA variant, trace through every layer, and explain exactly what each one does.

**Project milestone:** Understand the *language reasoning half* of the VLM — what it does with a text instruction like *"click the reset button"*.

**Resources:**
- Hugging Face NLP Course (free)
- Karpathy — *Let's reproduce GPT-2* (follow-up video)

---

### Phase 3 — Vision-Language Models

**Core unlock:** Vision Transformer (ViT) — images as patch tokens. CLIP — aligning vision and language in a shared embedding space. LLaVA / Qwen-VL — how screenshot understanding actually works end to end.

**This is the heart of the project.** The VLM is the agent's perceptual system. Understanding it from the inside determines everything downstream.

**Project milestone:** Run a VLM locally, feed it screenshots of the diagnostic tool, and explain architecturally how it produces *"I see a warning button in the lower panel"*.

**Resources:**
- Paper: *An Image is Worth 16x16 Words* (Dosovitskiy et al.) — ViT
- Paper: *Learning Transferable Visual Models From Natural Language Supervision* (Radford et al.) — CLIP
- Paper: *LLaVA* — focus on the architecture section
- Hands-on: run Qwen-VL or LLaVA locally via Ollama

---

### Phase 4 — The Agentic Perceive → Reason → Act Loop

**Core unlock:** Agent architecture patterns, tool use, structured outputs, state management across steps — understood at implementation depth.

**Head start:** LangGraph is the right framework. This phase deepens existing familiarity to the implementation level. Read the LangGraph source, not just the docs.

**Project milestone:** A working prototype — an agent that opens the diagnostic tool, reads the screen, and executes one deterministic test step reliably.

**Resources:**
- LangGraph documentation + source code
- Anthropic Computer Use API
- Paper: *UI-TARS* (ByteDance) — strong architecture reference, open source

---

### Phase 5 — RAG for UI Memory

**Core unlock:** Embeddings, vector databases, semantic retrieval — understood at the mechanism level, not just the LangChain API level.

**Why this matters:** The agent needs to recognize UI elements it has seen before, even when they shift slightly or change state. A persistent memory of known UI elements retrieved by semantic similarity makes the agent robust.

**Project milestone:** The agent maintains a persistent memory of the diagnostic tool's UI elements and retrieves them by semantic similarity during test execution.

**Resources:**
- Understand approximate nearest neighbour search (FAISS or HNSW) — the math behind vector search
- ChromaDB documentation
- Hugging Face — `sentence-transformers` library

---

### Phase 6 — Fine-Tuning for the Domain

**Core unlock:** LoRA, PEFT, instruction tuning — adapting a pre-trained VLM to the specific UI of the diagnostic tool.

**Why this matters:** A general VLM may struggle with domain-specific UI elements: ECU status widgets, UDS response tables, protocol-specific indicators, custom controls. Fine-tuning closes that gap with a modest amount of annotated data.

**Project milestone:** A VLM fine-tuned on annotated screenshots of the diagnostic tool, measurably outperforming the base model on the target UI elements. Evaluation methodology defined and applied — this is where the DO-178B mindset pays off.

**Resources:**
- Paper: *LoRA: Low-Rank Adaptation of Large Language Models* (Hu et al.)
- Hugging Face PEFT library
- Unsloth (efficient fine-tuning, practical tooling)
- Fast.AI Part 2 — transfer learning lessons

---

### 4.1 Phase Map

```
Phase 1 ──► Phase 2 ──► Phase 3 ──► Phase 4 ──► Phase 5 ──► Phase 6
Transformer   LLMs       VLMs       Agent       UI Memory   Fine-tune
internals    internals  internals   prototype   with RAG    for domain

◄─── Understanding the stack ────────►◄──── Building the agent ──────►

                                     ▲                              ▲
                               First thing                   Production-
                                  runs                       grade agent
```

---

### 4.2 Timing Guidance

Phases 1–3 are primarily *understanding*. Phase 4 is where something first *runs*. Do not let the theory phases expand indefinitely.

- **Phases 1–2** together: target 4–6 weeks of consistent study
- **Phase 3**: 3–4 weeks; it's dense but directly rewarding
- **Phase 4**: open-ended — this is where building starts and learning accelerates

---

*Document version 1.0 — created as the foundation for the self-education project.*
*To be updated as phases are completed and the project evolves.*
*Code: MIT License. Written content: CC BY 4.0.*
