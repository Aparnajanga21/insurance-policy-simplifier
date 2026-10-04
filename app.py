import os
import tempfile

import streamlit as st

from policy_processor import (
    extract_text_from_pdf,
    clean_text,
    preprocess_text,
    simplify_policy,
    ask_policy,
    calculate_readability_metrics
)


# ============================================================
# 1. Streamlit configuration
# ============================================================

st.set_page_config(
    page_title="Insurance Policy Simplifier",
    page_icon="📄",
    layout="wide"
)


# ============================================================
# 2. Application title
# ============================================================

st.title(
    "📄 Insurance Policy Document Simplifier"
)

st.write(
    """
Upload an insurance policy PDF and convert
complex insurance language into clear,
easy-to-understand English using Generative AI.
"""
)


# ============================================================
# 3. Upload PDF
# ============================================================

uploaded_file = st.file_uploader(
    "Upload Insurance Policy PDF",
    type=["pdf"]
)


if uploaded_file is None:

    st.info(
        "Please upload an insurance policy PDF to begin."
    )

    st.stop()


# ============================================================
# 4. Create document ID
# ============================================================

document_id = (
    f"{uploaded_file.name}_"
    f"{uploaded_file.size}"
)


# ============================================================
# 5. Detect new document
# ============================================================

if (
    "document_id" not in st.session_state
    or
    st.session_state.document_id != document_id
):

    st.session_state.clear()

    st.session_state.document_id = (
        document_id
    )


# ============================================================
# 6. Process uploaded PDF
# ============================================================

if not st.session_state.get(
    "document_processed",
    False
):

    temp_pdf_path = None

    try:

        # ----------------------------------------------------
        # Save PDF temporarily
        # ----------------------------------------------------

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as temp_file:

            temp_file.write(
                uploaded_file.getbuffer()
            )

            temp_pdf_path = (
                temp_file.name
            )


        # ----------------------------------------------------
        # Extract PDF text
        # ----------------------------------------------------

        with st.spinner(
            "Extracting text from PDF..."
        ):

            extracted_text = (
                extract_text_from_pdf(
                    temp_pdf_path
                )
            )


        if not extracted_text.strip():

            st.error(
                "No readable text was found "
                "in the uploaded PDF."
            )

            st.stop()


        # ----------------------------------------------------
        # Clean text
        # ----------------------------------------------------

        with st.spinner(
            "Cleaning policy text..."
        ):

            cleaned_text = (
                clean_text(
                    extracted_text
                )
            )


        # ----------------------------------------------------
        # NLP preprocessing
        # ----------------------------------------------------

        with st.spinner(
            "Processing text using NLP..."
        ):

            processed_text = (
                preprocess_text(
                    cleaned_text
                )
            )


        # ----------------------------------------------------
        # Save to session state
        # ----------------------------------------------------

        st.session_state.document_processed = True

        st.session_state.document_name = (
            uploaded_file.name
        )

        st.session_state.original_text = (
            processed_text
        )

        st.session_state.original_raw_text = (
            cleaned_text
        )


        st.success(
            "Insurance policy uploaded and processed successfully!"
        )


    except Exception as e:

        st.error(
            f"Error while processing the PDF: {e}"
        )

        st.stop()


    finally:

        # ----------------------------------------------------
        # Delete temporary PDF
        # ----------------------------------------------------

        if (
            temp_pdf_path
            and
            os.path.exists(
                temp_pdf_path
            )
        ):

            os.remove(
                temp_pdf_path
            )


# ============================================================
# 7. Document information
# ============================================================

st.divider()

col1, col2 = st.columns(2)


with col1:

    st.write(
        "**Uploaded document:**"
    )

    st.write(
        st.session_state.document_name
    )


with col2:

    st.write(
        "**Processing:**"
    )

    st.write(
        "NLP + Gemini API"
    )


# ============================================================
# 8. Tabs
# ============================================================

simplify_tab, question_tab = st.tabs(
    [
        "✨ Simplify Policy",
        "💬 Ask Questions"
    ]
)


# ============================================================
# TAB 1 — SIMPLIFICATION
# ============================================================

with simplify_tab:

    st.header(
        "✨ Simplify Insurance Policy"
    )

    st.write(
        """
The AI converts complex insurance language
into simple English while preserving important
policy information.
"""
    )


    # --------------------------------------------------------
    # Simplify button
    # --------------------------------------------------------

    if st.button(
        "Simplify Policy",
        type="primary"
    ):

        with st.spinner(
            "Gemini is simplifying the policy..."
        ):

            simplified_text = (
                simplify_policy(
                    st.session_state.original_text
                )
            )


        st.session_state.simplified_text = (
            simplified_text
        )


    # --------------------------------------------------------
    # Display result
    # --------------------------------------------------------

    if st.session_state.get(
        "simplified_text"
    ):

        original_text = (
            st.session_state.original_text
        )

        simplified_text = (
            st.session_state.simplified_text
        )


        st.subheader(
            "Before and After"
        )


        before_col, after_col = st.columns(2)


        # ----------------------------------------------------
        # Original
        # ----------------------------------------------------

        with before_col:

            st.markdown(
                "### 📜 Original Policy"
            )

            st.text_area(
                "Original",
                original_text,
                height=500,
                label_visibility="collapsed"
            )


        # ----------------------------------------------------
        # Simplified
        # ----------------------------------------------------

        with after_col:

            st.markdown(
                "### ✨ Simplified Policy"
            )

            st.text_area(
                "Simplified",
                simplified_text,
                height=500,
                label_visibility="collapsed"
            )


        # ====================================================
        # Readability
        # ====================================================

        st.divider()

        st.subheader(
            "📊 Readability Comparison"
        )


        original_metrics = (
            calculate_readability_metrics(
                original_text
            )
        )


        simplified_metrics = (
            calculate_readability_metrics(
                simplified_text
            )
        )


        original_score = (
            original_metrics[
                "flesch_reading_ease"
            ]
        )


        simplified_score = (
            simplified_metrics[
                "flesch_reading_ease"
            ]
        )


        readability_change = (
            simplified_score
            -
            original_score
        )


        metric_col1, metric_col2 = (
            st.columns(2)
        )


        # ----------------------------------------------------
        # Original metrics
        # ----------------------------------------------------

        with metric_col1:

            st.markdown(
                "### 📜 Original"
            )

            st.metric(
                "Flesch Reading Ease",
                f"{original_score:.1f}"
            )

            st.write(
                f"**Words:** "
                f"{original_metrics['word_count']}"
            )

            st.write(
                f"**Sentences:** "
                f"{original_metrics['sentence_count']}"
            )

            st.write(
                f"**Average words per sentence:** "
                f"{original_metrics['avg_sentence_words']:.1f}"
            )


        # ----------------------------------------------------
        # Simplified metrics
        # ----------------------------------------------------

        with metric_col2:

            st.markdown(
                "### ✨ Simplified"
            )

            st.metric(
                "Flesch Reading Ease",
                f"{simplified_score:.1f}"
            )

            st.write(
                f"**Words:** "
                f"{simplified_metrics['word_count']}"
            )

            st.write(
                f"**Sentences:** "
                f"{simplified_metrics['sentence_count']}"
            )

            st.write(
                f"**Average words per sentence:** "
                f"{simplified_metrics['avg_sentence_words']:.1f}"
            )


        # ----------------------------------------------------
        # Improvement
        # ----------------------------------------------------

        st.divider()


        if readability_change > 0:

            st.success(
                f"Readability improved by "
                f"{readability_change:.1f} points."
            )

        elif readability_change < 0:

            st.warning(
                f"Readability changed by "
                f"{readability_change:.1f} points."
            )

        else:

            st.info(
                "There was no change in the "
                "calculated readability score."
            )


# ============================================================
# TAB 2 — QUESTIONS
# ============================================================

with question_tab:

    st.header(
        "💬 Ask About Your Policy"
    )

    st.write(
        """
Ask a question about the uploaded policy.
The question is answered using the policy
text provided to the Gemini model.
"""
    )


    question = st.text_input(
        "Enter your question"
    )


    if st.button(
        "Ask Question"
    ):

        if not question.strip():

            st.warning(
                "Please enter a question."
            )

        else:

            with st.spinner(
                "Gemini is analyzing the policy..."
            ):

                answer = ask_policy(
                    question,
                    st.session_state.original_text
                )


            st.subheader(
                "Answer"
            )

            st.write(
                answer
            )