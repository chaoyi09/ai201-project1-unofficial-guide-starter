"""Gradio chat interface for The Unofficial Guide (Milestone 5).

Pipeline: question → retrieve top-k chunks → Groq llama-3.3-70b-versatile
with a strict grounding system prompt → Answer + Sources.

The grounding instruction is explicit about refusing to answer when the
retrieved context is insufficient — this is the property verified by
planning.md's out-of-scope test.
"""

from __future__ import annotations

import os
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv
from groq import Groq

from retrieve import retrieve

load_dotenv(Path(__file__).parent / ".env")

GROQ_MODEL = "llama-3.3-70b-versatile"
TOP_K = 5

SYSTEM_PROMPT = """You are an assistant that answers questions about Northeastern University \
Khoury College graduate courses and professors, using ONLY the document excerpts provided in \
the user message.

Rules — follow them strictly:
1. Answer only from the provided excerpts. Do not use outside knowledge, even if you "know" the answer.
2. If the excerpts do not contain enough information to answer, reply exactly: \
"I don't have enough information in the provided documents to answer that." \
Do not guess, infer beyond what is written, or fabricate professor names, courses, or reviews.
3. Cite the source document filename(s) inline in parentheses next to each claim, \
e.g. "(rmp_smith_cs5800.txt)". If multiple excerpts support the same claim, cite all of them.
4. When excerpts conflict (e.g., one says exams are easy, another says hard), present \
both perspectives rather than picking one.
5. Keep the answer concise — 2–5 sentences unless the question explicitly asks for more."""


def _format_context(hits: list[dict]) -> str:
    """Render retrieved chunks as a numbered list the LLM can quote from."""
    lines = []
    for i, h in enumerate(hits, 1):
        header = f"[Excerpt {i}] source={h['source']}"
        if h.get("professor"):
            header += f" | professor={h['professor']}"
        if h.get("course"):
            header += f" | course={h['course']}"
        lines.append(f"{header}\n{h['text']}")
    return "\n\n".join(lines)


def _format_sources(hits: list[dict]) -> str:
    """Deduplicate source filenames in retrieval order."""
    seen: list[str] = []
    for h in hits:
        s = h["source"]
        if s not in seen:
            seen.append(s)
    return "\n".join(f"- {s}" for s in seen)


def answer_question(question: str) -> tuple[str, str]:
    if not question or not question.strip():
        return "Please enter a question.", ""

    hits = retrieve(question, k=TOP_K)
    if not hits:
        return (
            "I don't have enough information in the provided documents to answer that.",
            "",
        )

    context = _format_context(hits)
    user_message = (
        f"Question: {question}\n\n"
        f"Document excerpts:\n{context}\n\n"
        f"Answer the question using only these excerpts. Cite source filenames."
    )

    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    completion = client.chat.completions.create(
        model=GROQ_MODEL,
        temperature=0.1,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )
    answer = completion.choices[0].message.content.strip()
    return answer, _format_sources(hits)


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="The Unofficial Guide — NEU Khoury") as ui:
        gr.Markdown(
            "# The Unofficial Guide\n"
            "Ask about Northeastern Khoury graduate CS courses and professors. "
            "Answers are grounded in collected Reddit and RateMyProfessors reviews."
        )
        with gr.Row():
            question = gr.Textbox(
                label="Your question",
                placeholder="e.g. What do students say about Prof. Mahsa Derakhshan's exams?",
                lines=2,
            )
        ask_btn = gr.Button("Ask", variant="primary")
        answer = gr.Textbox(label="Answer", lines=8, interactive=False)
        sources = gr.Textbox(label="Sources", lines=4, interactive=False)

        ask_btn.click(answer_question, inputs=question, outputs=[answer, sources])
        question.submit(answer_question, inputs=question, outputs=[answer, sources])

        gr.Examples(
            examples=[
                "What do students say about Prof. Iraklis Tsekourakis's exams in CS5800?",
                "How does Prof. Mahsa Derakhshan grade exams in CS5800?",
                "What are the main criticisms of Prof. Shesh in CS5010?",
                "If I want a better GPA in CS5800, which professor should I choose?",
                "What is the workload like for Prof. Shesh's CS3500 (OOD) course?",
            ],
            inputs=question,
        )
    return ui


if __name__ == "__main__":
    if not os.environ.get("GROQ_API_KEY"):
        raise SystemExit("GROQ_API_KEY missing — copy .env.example to .env and set your key")
    build_ui().launch()
