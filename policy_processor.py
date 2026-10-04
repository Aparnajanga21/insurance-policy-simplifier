from pypdf import PdfReader
import re
import time
from pathlib import Path
import os
from dotenv import load_dotenv
from google import genai

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize


# ============================================================
# Readability optimization settings
# ============================================================

TARGET_FLESCH_SCORE = 80
MAX_READABILITY_ATTEMPTS = 3


# ============================================================
# 1. NLTK setup
# ============================================================

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)


# ============================================================
# 2. Environment variables
# ============================================================

env_path = Path(__file__).parent / ".env"

load_dotenv(dotenv_path=env_path)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY was not found in the .env file."
    )


# ============================================================
# 3. Configure Gemini
# ============================================================

gemini_client = genai.Client(
    api_key=GEMINI_API_KEY
)


# ============================================================
# 4. Extract text from PDF
# ============================================================

def extract_text_from_pdf(pdf_path):

    reader = PdfReader(pdf_path)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


# ============================================================
# 5. Clean extracted text
# ============================================================

def clean_text(text):

    if not text:
        return ""

    # Remove common page number patterns
    text = re.sub(
        r"Page\s+\d+\s+of\s+\d+",
        "",
        text,
        flags=re.IGNORECASE
    )

    # Remove repeated whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    # Remove unnecessary spaces before punctuation
    text = re.sub(
        r"\s+([,.!?;:])",
        r"\1",
        text
    )

    return text.strip()


# ============================================================
# 6. NLP preprocessing
# ============================================================

def preprocess_text(text):

    """
    Performs basic NLP preprocessing.

    This does NOT generate the simplified text.
    Gemini performs the actual language simplification.
    """

    if not text:
        return ""

    # Sentence segmentation
    sentences = sent_tokenize(text)

    processed_sentences = []

    for sentence in sentences:

        sentence = sentence.strip()

        if not sentence:
            continue

        # Tokenization
        tokens = word_tokenize(sentence)

        # Remove unnecessary spaces
        cleaned_tokens = [
            token.strip()
            for token in tokens
            if token.strip()
        ]

        processed_sentence = " ".join(
            cleaned_tokens
        )

        processed_sentences.append(
            processed_sentence
        )

    return " ".join(
        processed_sentences
    )


# ============================================================
# 7. Split long policy text
# ============================================================

def split_text(
    text,
    chunk_size=6000,
    overlap=300
):

    """
    Splits large policy text into manageable
    sections before sending them to Gemini.

    This is NOT RAG.
    The sections are processed sequentially.
    """

    if not text:
        return []

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# ============================================================
# 8. Gemini response generator
# ============================================================

def generate_gemini_response(
    prompt,
    max_retries=3
):

    for attempt in range(max_retries):

        try:

            response = gemini_client.models.generate_content(
                model="gemini-3.5-flash-lite",
                contents=prompt
            )

            if response.text:
                return response.text.strip()

            return "Gemini returned an empty response."

        except Exception as e:

            error_message = str(e)

            print(
                f"Gemini request failed "
                f"(attempt {attempt + 1}/{max_retries})"
            )

            print(error_message)

            # Retry temporary server errors
            if (
                "503" in error_message
                or
                "UNAVAILABLE" in error_message
            ):

                if attempt < max_retries - 1:

                    wait_time = 5 * (attempt + 1)

                    time.sleep(wait_time)

                    continue

                return (
                    "Gemini is temporarily unavailable. "
                    "Please try again later."
                )

            # Authentication errors
            if (
                "401" in error_message
                or
                "403" in error_message
            ):

                return (
                    "Gemini API authentication failed. "
                    "Please check your API key."
                )

            # Model errors
            if "404" in error_message:

                return (
                    "The configured Gemini model "
                    "could not be found."
                )

            return (
                "An error occurred while communicating "
                "with Gemini."
            )

    return "Unable to generate response."


# ============================================================
# 9. Simplify insurance policy
# ============================================================

def simplify_section_for_readability(
    original_section,
    initial_simplification
):
    """
    Improve a Gemini-generated section until the readability
    score reaches the target, or until the maximum number of
    attempts is reached.

    The function always keeps the version with the highest
    Flesch Reading Ease score.

    This is NOT RAG.
    No embeddings or vector database are used.
    """

    best_text = initial_simplification

    best_metrics = calculate_readability_metrics(
        best_text
    )

    best_score = best_metrics[
        "flesch_reading_ease"
    ]

    print(
        f"Initial section readability: {best_score}"
    )

    # Already good enough
    if best_score >= TARGET_FLESCH_SCORE:
        return best_text

    current_text = best_text

    for attempt in range(
        1,
        MAX_READABILITY_ATTEMPTS + 1
    ):

        revision_prompt = f"""
You are improving an insurance policy explanation
for ordinary customers.

Rewrite the CURRENT SIMPLIFIED TEXT below so that
it is easier to read.

TARGET FLESCH READING EASE SCORE:
{TARGET_FLESCH_SCORE}

CURRENT FLESCH READING EASE SCORE:
{best_score:.2f}

IMPORTANT:

1. Preserve the exact meaning of the original policy.

2. Do not remove important policy information.

3. Preserve:
   - numbers
   - dates
   - percentages
   - monetary amounts
   - coverage limits
   - waiting periods
   - conditions
   - exclusions
   - policy terms

4. Do not invent or add benefits.

5. Use short sentences.

6. Prefer common, simple English words.

7. Avoid unnecessary legal or technical wording.

8. Break long sentences into multiple short sentences.

9. Use bullets when they improve readability.

10. Keep important insurance terms when changing them
    would alter their meaning.

11. Do not provide legal or financial advice.

12. Return ONLY the improved policy text.

ORIGINAL POLICY SECTION:
{original_section}

CURRENT SIMPLIFIED TEXT:
{current_text}
"""

        print(
            f"Readability improvement attempt "
            f"{attempt}/{MAX_READABILITY_ATTEMPTS}..."
        )

        candidate = generate_gemini_response(
            revision_prompt
        )

        if not candidate:
            continue

        candidate_metrics = calculate_readability_metrics(
            candidate
        )

        candidate_score = candidate_metrics[
            "flesch_reading_ease"
        ]

        print(
            f"Candidate readability score: "
            f"{candidate_score}"
        )

        # Keep only the best version.
        if candidate_score > best_score:

            best_text = candidate
            best_score = candidate_score
            current_text = candidate

            print(
                f"Improved score: {best_score}"
            )

        # Stop once target is achieved.
        if best_score >= TARGET_FLESCH_SCORE:

            print(
                f"Target readability reached: "
                f"{best_score}"
            )

            break

    return best_text


def simplify_policy(
    policy_text,
    max_sections=None
):
    """
    Simplifies the insurance policy using Gemini
    and then iteratively improves readability.

    No RAG.
    No embeddings.
    No ChromaDB.
    """

    if not policy_text:

        return "No policy text was provided."


    # --------------------------------------------------------
    # NLP preprocessing
    # --------------------------------------------------------

    processed_text = preprocess_text(
        policy_text
    )


    # --------------------------------------------------------
    # Split large document
    # --------------------------------------------------------

    sections = split_text(
        processed_text,
        chunk_size=6000,
        overlap=300
    )


    if max_sections is not None:

        sections = sections[:max_sections]


    simplified_sections = []


    # --------------------------------------------------------
    # Process every section
    # --------------------------------------------------------

    for i, section in enumerate(sections):

        print(
            f"Simplifying section "
            f"{i + 1}/{len(sections)}..."
        )


        # ----------------------------------------------------
        # First Gemini simplification
        # ----------------------------------------------------

        prompt = f"""
You are an insurance policy document
simplification assistant.

Rewrite the insurance policy section below
in simple, clear English for an ordinary
customer.

IMPORTANT RULES:

1. Preserve the original meaning.

2. Do NOT remove important information.

3. Preserve all:
   - numbers
   - dates
   - percentages
   - monetary amounts
   - coverage limits
   - waiting periods
   - conditions
   - exclusions
   - policy terms

4. Do not invent any information.

5. Do not add benefits that are not present.

6. Replace complicated legal language
   with simple English.

7. Use short sentences.

8. Prefer common words where possible.

9. Use headings or bullet points when useful.

10. Keep the output factually accurate.

11. Do not provide legal or financial advice.

12. Return ONLY the simplified policy section.

Insurance policy section:

{section}
"""


        initial_simplification = (
            generate_gemini_response(
                prompt
            )
        )


        if not initial_simplification:
            continue


        # ----------------------------------------------------
        # Improve readability
        # ----------------------------------------------------

        optimized_section = (
            simplify_section_for_readability(
                section,
                initial_simplification
            )
        )


        simplified_sections.append(
            optimized_section
        )


    # --------------------------------------------------------
    # Combine sections
    # --------------------------------------------------------

    result = "\n\n".join(
        simplified_sections
    )


    # --------------------------------------------------------
    # Calculate final document score
    # --------------------------------------------------------

    final_metrics = (
        calculate_readability_metrics(
            result
        )
    )

    final_score = final_metrics[
        "flesch_reading_ease"
    ]

    print(
        f"Final document readability score: "
        f"{final_score}"
    )


    # --------------------------------------------------------
    # Final document-level optimization
    #
    # Only run this if the combined document is reasonably
    # sized. This avoids sending an extremely large policy
    # back to Gemini.
    # --------------------------------------------------------

    if (
        final_score < TARGET_FLESCH_SCORE
        and len(result) <= 30000
    ):

        best_result = result
        best_result_score = final_score

        for attempt in range(1, 3):

            final_prompt = f"""
You are performing a final readability improvement
on an insurance policy explanation.

TARGET FLESCH READING EASE SCORE:
{TARGET_FLESCH_SCORE}

CURRENT FLESCH READING EASE SCORE:
{best_result_score:.2f}

Rewrite the text below to make it easier for
ordinary customers to understand.

Rules:

1. Preserve the exact meaning.

2. Do not remove important information.

3. Preserve all numbers, dates, percentages,
   monetary amounts, coverage limits, waiting
   periods, conditions, exclusions and policy terms.

4. Do not invent benefits or facts.

5. Use short sentences.

6. Use simple common English.

7. Break long sentences into shorter sentences.

8. Use bullet points when appropriate.

9. Keep necessary insurance terminology.

10. Return ONLY the improved policy text.

CURRENT POLICY TEXT:

{best_result}
"""

            print(
                f"Final readability optimization "
                f"attempt {attempt}/2..."
            )

            candidate = generate_gemini_response(
                final_prompt
            )

            if not candidate:
                continue

            candidate_metrics = (
                calculate_readability_metrics(
                    candidate
                )
            )

            candidate_score = candidate_metrics[
                "flesch_reading_ease"
            ]

            print(
                f"Final candidate score: "
                f"{candidate_score}"
            )

            if candidate_score > best_result_score:

                best_result = candidate
                best_result_score = candidate_score

            if (
                best_result_score
                >= TARGET_FLESCH_SCORE
            ):
                break

        result = best_result


    print(
        "Best final readability score:",
        calculate_readability_metrics(
            result
        )["flesch_reading_ease"]
    )


    return result


# ============================================================
# 10. Ask questions
# ============================================================

def ask_policy(
    question,
    policy_text
):

    """
    Answers a question using the complete
    uploaded policy text.

    IMPORTANT:
    This is NOT RAG.

    The entire available policy text is
    provided directly to Gemini.
    """

    if not question:
        return "Please enter a question."

    if not policy_text:
        return "No policy has been uploaded."


    prompt = f"""
You are an insurance policy assistant.

Answer the user's question using ONLY
the information contained in the insurance
policy below.

Explain the answer in simple English.

Do not invent information.

Do not assume information that is not
present in the policy.

If the answer cannot be found in the
policy, say:

"The information could not be found
in the uploaded policy."

Do not provide legal or financial advice.

INSURANCE POLICY:

{policy_text}

USER QUESTION:

{question}
"""


    return generate_gemini_response(
        prompt
    )


# ============================================================
# 11. Count syllables
# ============================================================

def count_syllables(word):

    word = word.lower()

    word = re.sub(
        r"[^a-z]",
        "",
        word
    )

    if not word:
        return 0

    vowels = "aeiou"

    count = 0

    previous_was_vowel = False

    for char in word:

        is_vowel = char in vowels

        if (
            is_vowel
            and not previous_was_vowel
        ):
            count += 1

        previous_was_vowel = is_vowel


    # Silent e
    if (
        word.endswith("e")
        and count > 1
    ):
        count -= 1


    return max(
        1,
        count
    )


# ============================================================
# 12. Readability metrics
# ============================================================

def calculate_readability_metrics(text):

    if not text:

        return {
            "word_count": 0,
            "sentence_count": 0,
            "avg_sentence_words": 0,
            "flesch_reading_ease": 0
        }


    # Words
    words = re.findall(
        r"\b[a-zA-Z]+\b",
        text
    )


    # Sentences
    sentences = re.split(
        r"[.!?]+",
        text
    )

    sentences = [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


    word_count = len(words)

    sentence_count = len(sentences)


    if word_count == 0:

        return {
            "word_count": 0,
            "sentence_count": 0,
            "avg_sentence_words": 0,
            "flesch_reading_ease": 0
        }


    if sentence_count == 0:
        sentence_count = 1


    # Average sentence length
    avg_sentence_words = (
        word_count /
        sentence_count
    )


    # Syllables
    total_syllables = sum(
        count_syllables(word)
        for word in words
    )


    avg_syllables_per_word = (
        total_syllables /
        word_count
    )


    # Flesch Reading Ease
    readability_score = (
        206.835
        -
        (
            1.015 *
            avg_sentence_words
        )
        -
        (
            84.6 *
            avg_syllables_per_word
        )
    )


    readability_score = max(
        0,
        min(
            100,
            readability_score
        )
    )


    return {

        "word_count":
            word_count,

        "sentence_count":
            sentence_count,

        "avg_sentence_words":
            round(
                avg_sentence_words,
                2
            ),

        "flesch_reading_ease":
            round(
                readability_score,
                2
            )
    }