"""
EvalForge — Streamlit Demo UI.

Natural language → SQL with a fine-tuned Phi-3-mini model.
Shows side-by-side: base model output vs fine-tuned output.
"""

import json
from pathlib import Path

import requests
import streamlit as st

# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="EvalForge — Text-to-SQL",
    page_icon="🧠",
    layout="wide",
)

API_BASE = "http://localhost:8000"

# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ EvalForge")
    st.caption("Fine-tuned Phi-3-mini · Text-to-SQL")

    st.divider()

    # Health check
    try:
        health = requests.get(f"{API_BASE}/health", timeout=3).json()
        model_info = health.get("model", {})
        st.success("API connected")
        st.caption(f"Model: `{model_info.get('adapter') or model_info.get('base_model', '—')}`")
        st.caption(f"Device: `{model_info.get('device', '—')}`")
    except Exception:
        st.error("API not running.\n\n`uvicorn app.main:app --reload`")
        model_info = {}

    st.divider()

    # Eval results panel
    st.subheader("📊 Eval Results")
    eval_path = Path("./outputs/eval_results.json")
    if eval_path.exists():
        with open(eval_path) as f:
            eval_results = json.load(f)

        base_m = eval_results.get("base_model_metrics", {})
        ft_m = eval_results.get("finetuned_model_metrics", {})

        def metric_row(label, key):
            b = base_m.get(key)
            f = ft_m.get(key) if ft_m else None
            b_str = f"{b:.3f}" if b is not None else "—"
            f_str = f"{f:.3f}" if f is not None else "—"
            if b is not None and f is not None:
                delta = f - b
                delta_str = f"(+{delta:.3f})" if delta >= 0 else f"({delta:.3f})"
            else:
                delta_str = ""
            st.text(f"{label}")
            st.caption(f"Base: {b_str}  →  FT: {f_str}  {delta_str}")

        metric_row("Exact Match", "exact_match")
        metric_row("Exec Accuracy", "execution_accuracy")
        metric_row("BLEU-4", "bleu4")
        metric_row("BERTScore F1", "bertscore_f1")
    else:
        st.caption("Run `python eval/run_eval.py` to see results here.")

    st.divider()
    st.caption("Built by Navaneeth S · [GitHub](#) · [HF Hub](#)")


# ── Main ───────────────────────────────────────────────────────────────────────

st.title("🧠 EvalForge — Text-to-SQL")
st.caption("Fine-tuned Phi-3-mini (3.8B) on Spider benchmark via QLoRA/PEFT")

# Load example schemas from API
try:
    examples_data = requests.get(f"{API_BASE}/examples", timeout=3).json()
    examples = examples_data.get("examples", [])
except Exception:
    examples = [
        {
            "name": "E-commerce",
            "schema": (
                "CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT, email TEXT, city TEXT);\n"
                "CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER, total REAL, created_at TEXT);"
            ),
            "sample_questions": [
                "How many orders were placed in the last 30 days?",
                "Which customers have spent more than $1000 in total?",
            ],
        }
    ]

# Schema selector
st.subheader("1. Select or paste a database schema")
example_names = [e["name"] for e in examples] + ["Custom..."]
selected_example = st.selectbox("Example schemas:", example_names)

if selected_example == "Custom...":
    schema_input = st.text_area(
        "Paste your schema (CREATE TABLE statements):",
        height=150,
        placeholder="CREATE TABLE employees (id INTEGER PRIMARY KEY, name TEXT, salary REAL);",
    )
    sample_qs = []
else:
    ex = next(e for e in examples if e["name"] == selected_example)
    schema_input = st.text_area("Schema:", value=ex["schema"], height=150)
    sample_qs = ex.get("sample_questions", [])

# Question input
st.subheader("2. Ask a question")
if sample_qs:
    q_options = ["(type your own)"] + sample_qs
    selected_q = st.selectbox("Example questions:", q_options)
    if selected_q == "(type your own)":
        question_input = st.text_input("Your question:", placeholder="What is the total revenue per category?")
    else:
        question_input = st.text_input("Your question:", value=selected_q)
else:
    question_input = st.text_input("Your question:", placeholder="What is the total revenue per category?")

# Generate button
if st.button("⚡ Generate SQL", type="primary", disabled=not question_input or not schema_input):
    with st.spinner("Generating SQL..."):
        try:
            response = requests.post(
                f"{API_BASE}/generate-sql",
                json={"question": question_input, "schema": schema_input},
                timeout=60,
            )
            if response.status_code == 200:
                result = response.json()

                st.subheader("3. Generated SQL")
                st.code(result["sql"], language="sql")

                col1, col2, col3 = st.columns(3)
                col1.metric("Latency", f"{result['latency_ms']} ms")
                col2.metric("Model", result["model"].split("/")[-1])
                col3.metric("Tokens (approx)", len(result["sql"].split()))

                # Show prompt used (expandable)
                with st.expander("Show full prompt"):
                    from train.dataset_prep import format_prompt
                    prompt = format_prompt(question=question_input, schema=schema_input, sql=None)
                    st.code(prompt, language="text")

            else:
                st.error(f"API error {response.status_code}: {response.text}")
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to the API. Make sure it's running:\n\n`uvicorn app.main:app --reload`")
        except Exception as e:
            st.error(f"Error: {e}")

# ── About section ──────────────────────────────────────────────────────────────

with st.expander("ℹ️ About this project"):
    st.markdown("""
**EvalForge** fine-tunes [Phi-3-mini-4k-instruct](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) (3.8B)
on the [Spider](https://yale-lily.github.io/spider) Text-to-SQL benchmark using **QLoRA** (4-bit quantization + LoRA adapters).

**Stack:**
- Fine-tuning: HuggingFace PEFT + TRL SFTTrainer + bitsandbytes
- Evaluation: Exact match, execution accuracy, BLEU-4, BERTScore F1
- Serving: FastAPI + Streamlit

**Training was done on Kaggle (free T4 GPU) in ~2 hours.**
See the [training notebook](notebooks/kaggle_training.ipynb) for the full reproducible run.
""")
