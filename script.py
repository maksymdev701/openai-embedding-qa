import numpy as np
import openai
import os
import pandas as pd
import pickle
import tiktoken

openai.api_key = os.getenv("OPENAI_API_KEY")

COMPLETION_MODEL = "text-davinci-003"
EMBEDDING_MODEL = "text-embedding-ada-002"

prompt = """Answer the question as truthfully as possible, and if you're unsure of the answer, say "Sorry, I don't know".

Context:
The men's high jump event at the 2020 Summer Olympics took place between 30 July and 1 August 2021 at the Olympic Stadium.
33 athletes from 24 nations competed; the total possible number depended on how many nations would use universality places 
to enter athletes in addition to the 32 qualifying through mark or ranking (no universality places were used in 2021).
Italian athlete Gianmarco Tamberi along with Qatari athlete Mutaz Essa Barshim emerged as joint winners of the event following
a tie between both of them as they cleared 2.37m. Both Tamberi and Barshim agreed to share the gold medal in a rare instance
where the athletes of different nations had agreed to share the same medal in the history of Olympics. 
Barshim in particular was heard to ask a competition official "Can we have two golds?" in response to being offered a 
'jump off'. Maksim Nedasekau of Belarus took bronze. The medals were the first ever in the men's high jump for Italy and 
Belarus, the first gold in the men's high jump for Italy and Qatar, and the third consecutive medal in the men's high jump
for Qatar (all by Barshim). Barshim became only the second man to earn three medals in high jump, joining Patrik Sjöberg
of Sweden (1984 to 1992).

Q: Who won the 2020 Summer Olympics men's high jump?
A:"""


# openai.Completion.create(
#     prompt=prompt,
#     temperature=0,
#     max_tokens=300,
#     model=COMPLETION_MODEL
# )["choices"][0]["text"].strip("\n")

df = pd.read_csv("./olympics_sections_text.csv")
df = df.set_index(["title", "heading"])
# print(f"{len(df)} rows in the data.")


def get_embedding(text: str, model: str = EMBEDDING_MODEL) -> list[float]:
    result = openai.Embedding.create(
        model=model,
        input=text
    )
    return result["data"][0]["embedding"]


def compute_doc_embeddings(df: pd.DataFrame) -> dict[tuple[str, str], list[float]]:
    return {
        idx: get_embedding(r.content) for idx, r in df.iterrows()
    }


def load_embeddings(fname: str) -> dict[tuple[str, str], list[float]]:
    df = pd.read_csv(fname, header=0)
    max_dim = max([int(c) for c in df.columns if c != "title" and c != "heading"])
    return {
        (r.title, r.heading): [r[str(i)] for i in range(max_dim + 1)] for _, r in df.iterrows()
    }


document_embeddings = load_embeddings("./olympics_sections_document_embeddings.csv")
# document_embeddings = compute_doc_embeddings(df)
example_entry = list(document_embeddings.items())[0]
print(f"{example_entry[0]} : {example_entry[1][:5]}... ({len(example_entry[1])} entries)")


def vector_similarity(x: list[float], y: list[float]) -> float:
    return np.dot(np.array(x), np.array(y))


def order_document_sections_by_query_similarity(query: str, contexts: dict[(str, str), np.array]) -> list[
    (float, (str, str))]:
    query_embedding = get_embedding(query)

    document_similarities = sorted([
        (vector_similarity(query_embedding, doc_embedding), doc_index) for doc_index, doc_embedding in contexts.items()
    ], reverse=True)

    return document_similarities


print(order_document_sections_by_query_similarity("Who won the men's high jump?", document_embeddings)[:5])

MAX_SECTION_LEN = 500
SEPERATOR = "\n* "
ENCODING = "gpt2"

encoding = tiktoken.get_encoding(ENCODING)
seperator_len = len(encoding.encode(SEPERATOR))

f"Context separator contains {seperator_len} tokens"

def construct_prompt(question: str, context_embeddings: dict, df:pd.DataFrame) -> str:
    most_relevant_document_sections = order_document_sections_by_query_similarity(question, context_embeddings)

    chosen_sections = []
    chosen_sections_len = 0
    chosen_sections_indexes = []

    for _, section_index in most_relevant_document_sections:
        document_section = df.loc[section_index]

        chosen_sections_len += document_section.tokens + seperator_len
        if chosen_sections_len > MAX_SECTION_LEN:
            break

        chosen_sections.append(SEPERATOR + document_section.content.replace("\n", " "))
        chosen_sections_indexes.append(str(section_index))

    print(f"Selected {len(chosen_sections)} document sections:")
    print("\n".join(chosen_sections_indexes))

    header = """Answer the question as truthfully as possible using the provided context, and if the answer is not contained within the text below, say "I don't know."\n\nContext:\n"""

    return header + "".join(chosen_sections) + "\n\n Q: " + question + "\n A:"

# prompt = construct_prompt(
#     "Who won the 2020 Summer Olympics men's high jump",
#     document_embeddings,
#     df
# )

# print("===\n", prompt)

COMPLETIONS_API_PARAMS = {
    "temperature": 0.0,
    "max_tokens": 300,
    "model": COMPLETION_MODEL
}

def answer_query_with_context(
        query: str,
        df: pd.DataFrame,
        document_embeddings: dict[(str, str), np.array],
        show_prompt: bool = False
) -> str:
    prompt = construct_prompt(
        query,
        document_embeddings,
        df
    )

    if show_prompt:
        print(prompt)

    response = openai.Completion.create(
        prompt=prompt,
        **COMPLETIONS_API_PARAMS
    )

    return response["choices"][0]["text"].strip("\n")

print(answer_query_with_context("Who won the grimblesplatch competition at the 2020 Summer Olympic games?", df, document_embeddings))
