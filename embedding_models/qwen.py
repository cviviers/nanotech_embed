import os
import sys
import traceback
from typing import List, Optional

# Keep this before CUDA initialization. It reduces allocator fragmentation during
# repeated long-sequence reranking requests.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn.functional as F
from torch import Tensor
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

EMBED_MODEL_NAME = os.getenv("QWEN_EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")
RERANK_MODEL_NAME = os.getenv("QWEN_RERANKER_MODEL", "Qwen/Qwen3-Reranker-0.6B")

EMBED_MAX_LENGTH = int(os.getenv("QWEN_EMBED_MAX_LENGTH", "8192"))
RERANK_MAX_LENGTH = int(os.getenv("QWEN_RERANK_MAX_LENGTH", "8192"))
EMBED_BATCH_SIZE = int(os.getenv("QWEN_EMBED_BATCH_SIZE", "16"))
RERANK_BATCH_SIZE = int(os.getenv("QWEN_RERANK_BATCH_SIZE", "2"))
RERANK_LOGITS_TO_KEEP = int(os.getenv("QWEN_RERANK_LOGITS_TO_KEEP", "1"))
CUDA_EMPTY_CACHE_EACH_BATCH = os.getenv("QWEN_CUDA_EMPTY_CACHE_EACH_BATCH", "0").strip().lower() in {
    "1",
    "true",
    "yes",
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _positive_int(value: int, name: str) -> int:
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")
    return value


def _nonnegative_int(value: int, name: str) -> int:
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")
    return value


EMBED_BATCH_SIZE = _positive_int(EMBED_BATCH_SIZE, "QWEN_EMBED_BATCH_SIZE")
RERANK_BATCH_SIZE = _positive_int(RERANK_BATCH_SIZE, "QWEN_RERANK_BATCH_SIZE")
RERANK_LOGITS_TO_KEEP = _nonnegative_int(RERANK_LOGITS_TO_KEEP, "QWEN_RERANK_LOGITS_TO_KEEP")


def _print_local_qwen_error(context: str, exc: Exception) -> None:
    print(f"[Qwen API] {context}: {exc}", file=sys.stderr, flush=True)
    traceback.print_exc(file=sys.stderr)


def _resolve_torch_dtype(env_var: str, default: str = "float16") -> torch.dtype:
    if DEVICE != "cuda":
        return torch.float32
    dtype_name = str(os.getenv(env_var, default)).strip().lower()
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported {env_var}: {dtype_name!r}")


def _batches(items: List[str], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _empty_cuda_cache() -> None:
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


def _is_cuda_oom(exc: Exception) -> bool:
    return "cuda" in str(exc).lower() and "out of memory" in str(exc).lower()


EMBED_TORCH_DTYPE = _resolve_torch_dtype(
    "QWEN_EMBED_TORCH_DTYPE",
    default=str(os.getenv("QWEN_TORCH_DTYPE", "float16")),
)
RERANK_TORCH_DTYPE = _resolve_torch_dtype(
    "QWEN_RERANK_TORCH_DTYPE",
    default=str(os.getenv("QWEN_TORCH_DTYPE", "float16")),
)

EMBED_MODEL_LOAD_KWARGS = {"torch_dtype": EMBED_TORCH_DTYPE} if DEVICE == "cuda" else {}
RERANK_MODEL_LOAD_KWARGS = {"torch_dtype": RERANK_TORCH_DTYPE} if DEVICE == "cuda" else {}

# -------------------------------------------------------------------
# Load models (Embedding + Reranker)
# Implementation follows:
# https://github.com/QwenLM/Qwen3-Embedding/blob/main/README.md
# -------------------------------------------------------------------

# Embedding model
embed_tokenizer = AutoTokenizer.from_pretrained(
    EMBED_MODEL_NAME,
    padding_side="left",
)
embed_model = AutoModel.from_pretrained(EMBED_MODEL_NAME, **EMBED_MODEL_LOAD_KWARGS).to(DEVICE).eval()

# Reranker model
rerank_tokenizer = AutoTokenizer.from_pretrained(
    RERANK_MODEL_NAME,
    padding_side="left",
)
rerank_model = AutoModelForCausalLM.from_pretrained(RERANK_MODEL_NAME, **RERANK_MODEL_LOAD_KWARGS).to(DEVICE).eval()
_RERANK_LOGITS_TO_KEEP_KWARG: Optional[str] = "logits_to_keep" if RERANK_LOGITS_TO_KEEP else None

# Tokens and prompts for reranking (adapted from official README)
token_false_id = rerank_tokenizer.convert_tokens_to_ids("no")
token_true_id = rerank_tokenizer.convert_tokens_to_ids("yes")

prefix = (
    "<|im_start|>system\n"
    " Judge whether the Document meets the requirements based on the Query and the "
    'Instruct provided. Note that the answer can only be "yes" or "no".'
    "<|im_end|>\n<|im_start|>user\n"
)
suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

prefix_tokens = rerank_tokenizer.encode(prefix, add_special_tokens=False)
suffix_tokens = rerank_tokenizer.encode(suffix, add_special_tokens=False)

# -------------------------------------------------------------------
# Helper functions (Embedding)
# -------------------------------------------------------------------

def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """
    Pooling strategy from Qwen3-Embedding README:
    use the last token corresponding to the actual text (considering padding side).
    """
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device),
            sequence_lengths,
        ]


def get_detailed_instruct(task_description: str, query: str) -> str:
    """
    Instruction-aware input formatting, as recommended in Qwen3-Embedding README.
    """
    return f"Instruct: {task_description}\n Query:{query}"


def embed_texts(
    texts: List[str],
    instruction: Optional[str] = None,
    normalize: bool = True,
) -> Tensor:
    """
    Compute embeddings for a list of texts using Qwen3-Embedding-0.6B.
    If `instruction` is provided, apply instruction-aware formatting.
    """
    if len(texts) == 0:
        raise ValueError("texts must be a non-empty list")

    if instruction:
        processed_texts = [get_detailed_instruct(instruction, t) for t in texts]
    else:
        processed_texts = texts

    chunks = []
    for batch_texts in _batches(processed_texts, EMBED_BATCH_SIZE):
        batch_dict = None
        outputs = None
        embeddings = None
        try:
            batch_dict = embed_tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=EMBED_MAX_LENGTH,
                return_tensors="pt",
            )
            batch_dict = {k: v.to(DEVICE) for k, v in batch_dict.items()}

            with torch.inference_mode():
                outputs = embed_model(**batch_dict)
                embeddings = last_token_pool(outputs.last_hidden_state, batch_dict["attention_mask"])
                if normalize:
                    embeddings = F.normalize(embeddings, p=2, dim=1)
                chunks.append(embeddings.detach().cpu())
        except Exception as e:
            if _is_cuda_oom(e):
                _empty_cuda_cache()
            raise
        finally:
            del batch_dict, outputs, embeddings
            if CUDA_EMPTY_CACHE_EACH_BATCH:
                _empty_cuda_cache()

    if len(chunks) == 1:
        return chunks[0]
    return torch.cat(chunks, dim=0)


def cosine_similarity_matrix(a: Tensor, b: Tensor) -> Tensor:
    """
    Assuming a and b are already L2-normalized, cosine sim = dot product.
    a: [N, D], b: [M, D] -> [N, M]
    """
    return a @ b.T


# -------------------------------------------------------------------
# Helper functions (Reranker)
# -------------------------------------------------------------------

def format_instruction(instruction: Optional[str], query: str, doc: str) -> str:
    if instruction is None:
        instruction = "Given a web search query, retrieve relevant passages that answer the query"
    output = (
        "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"
        .format(instruction=instruction, query=query, doc=doc)
    )
    return output


def process_rerank_inputs(pairs: List[str]):
    inputs = rerank_tokenizer(
        pairs,
        padding=False,
        truncation="longest_first",
        return_attention_mask=False,
        max_length=RERANK_MAX_LENGTH - len(prefix_tokens) - len(suffix_tokens),
    )
    for i, ele in enumerate(inputs["input_ids"]):
        inputs["input_ids"][i] = prefix_tokens + ele + suffix_tokens
    inputs = rerank_tokenizer.pad(
        inputs,
        padding=True,
        return_tensors="pt",
        max_length=RERANK_MAX_LENGTH,
    )
    for key in inputs:
        inputs[key] = inputs[key].to(DEVICE)
    return inputs


@torch.no_grad()
def compute_rerank_scores(inputs) -> List[float]:
    """
    Compute reranking scores following the official Qwen3-Reranker usage:
    probability that answer is "yes".
    """
    global _RERANK_LOGITS_TO_KEEP_KWARG
    forward_kwargs = {}
    if _RERANK_LOGITS_TO_KEEP_KWARG:
        forward_kwargs[_RERANK_LOGITS_TO_KEEP_KWARG] = RERANK_LOGITS_TO_KEEP
    try:
        outputs = rerank_model(**inputs, **forward_kwargs)
    except TypeError as e:
        if _RERANK_LOGITS_TO_KEEP_KWARG and _RERANK_LOGITS_TO_KEEP_KWARG in str(e):
            _RERANK_LOGITS_TO_KEEP_KWARG = "num_logits_to_keep"
            try:
                outputs = rerank_model(**inputs, num_logits_to_keep=RERANK_LOGITS_TO_KEEP)
            except TypeError as inner:
                if "num_logits_to_keep" not in str(inner):
                    raise
                _RERANK_LOGITS_TO_KEEP_KWARG = None
                outputs = rerank_model(**inputs)
        else:
            raise
    batch_scores = outputs.logits[:, -1, :]
    true_vector = batch_scores[:, token_true_id]
    false_vector = batch_scores[:, token_false_id]
    yes_no_scores = torch.stack([false_vector, true_vector], dim=1)
    yes_no_scores = torch.nn.functional.log_softmax(yes_no_scores, dim=1)
    scores = yes_no_scores[:, 1].exp().detach().cpu().tolist()
    del outputs, batch_scores, true_vector, false_vector, yes_no_scores
    return scores


def rerank_query_documents(
    query: str,
    documents: List[str],
    instruction: Optional[str] = None,
) -> List[float]:
    if len(documents) == 0:
        raise ValueError("documents must be a non-empty list")
    pairs = [format_instruction(instruction, query, doc) for doc in documents]
    scores: List[float] = []
    for pair_batch in _batches(pairs, RERANK_BATCH_SIZE):
        inputs = None
        try:
            inputs = process_rerank_inputs(pair_batch)
            scores.extend(compute_rerank_scores(inputs))
        except Exception as e:
            if _is_cuda_oom(e):
                _empty_cuda_cache()
            raise
        finally:
            if inputs is not None:
                del inputs
            if CUDA_EMPTY_CACHE_EACH_BATCH:
                _empty_cuda_cache()
    return scores


# -------------------------------------------------------------------
# Pydantic models (API schemas)
# -------------------------------------------------------------------

class EmbeddingRequest(BaseModel):
    texts: List[str]
    instruction: Optional[str] = None
    normalize: bool = True


class EmbeddingResponse(BaseModel):
    embeddings: List[List[float]]
    dimension: int
    model: str
    normalize: bool


class SimilarityRequest(BaseModel):
    texts_a: List[str]
    texts_b: List[str]
    instruction_a: Optional[str] = None
    instruction_b: Optional[str] = None
    normalize: bool = True


class SimilarityResponse(BaseModel):
    similarity: List[List[float]]
    model: str
    normalized: bool


class RankRequest(BaseModel):
    query: str
    documents: List[str]
    instruction: Optional[str] = None
    top_k: Optional[int] = None
    return_embedding_similarity: bool = True
    normalize_embeddings: bool = True


class RankedDocument(BaseModel):
    index: int
    document: str
    reranker_score: float
    embedding_score: Optional[float] = None


class RankResponse(BaseModel):
    query: str
    instruction: Optional[str]
    qwen_model_reranker: str
    qwen_model_embedding: Optional[str] = None
    used_embedding_fallback: bool = False
    results: List[RankedDocument]


# -------------------------------------------------------------------
# FastAPI app
# -------------------------------------------------------------------

app = FastAPI(
    title="Qwen3-0.6B Embedding & Reranker API",
    description=(
        "FastAPI wrapper around Qwen/Qwen3-Embedding-0.6B and "
        "Qwen/Qwen3-Reranker-0.6B.\n"
        "Model usage is based on the official Qwen3-Embedding README."
    ),
    version="0.1.0",
)


@app.get("/")
def read_root():
    return {
        "service": "qwen3-embedding-reranker-api",
        "embedding_model": EMBED_MODEL_NAME,
        "reranker_model": RERANK_MODEL_NAME,
        "device": DEVICE,
        "embed_torch_dtype": str(EMBED_TORCH_DTYPE),
        "rerank_torch_dtype": str(RERANK_TORCH_DTYPE),
        "embed_batch_size": EMBED_BATCH_SIZE,
        "rerank_batch_size": RERANK_BATCH_SIZE,
        "embed_max_length": EMBED_MAX_LENGTH,
        "rerank_max_length": RERANK_MAX_LENGTH,
        "rerank_logits_to_keep": RERANK_LOGITS_TO_KEEP,
        "embedding_dim": 1024,  # per Qwen3-Embedding-0.6B spec
    }


@app.post("/embed", response_model=EmbeddingResponse)
def create_embeddings(payload: EmbeddingRequest):
    if not payload.texts:
        raise HTTPException(status_code=400, detail="texts must be a non-empty list")

    try:
        embeddings = embed_texts(
            texts=payload.texts,
            instruction=payload.instruction,
            normalize=payload.normalize,
        )
    except Exception as e:
        _print_local_qwen_error("/embed failed", e)
        raise HTTPException(status_code=500, detail=str(e))

    return EmbeddingResponse(
        embeddings=embeddings.cpu().tolist(),
        dimension=embeddings.shape[1],
        model=EMBED_MODEL_NAME,
        normalize=payload.normalize,
    )


@app.post("/similarity", response_model=SimilarityResponse)
def compute_similarity(payload: SimilarityRequest):
    if not payload.texts_a:
        raise HTTPException(status_code=400, detail="texts_a must be a non-empty list")
    if not payload.texts_b:
        raise HTTPException(status_code=400, detail="texts_b must be a non-empty list")

    try:
        emb_a = embed_texts(
            texts=payload.texts_a,
            instruction=payload.instruction_a,
            normalize=payload.normalize,
        )
        emb_b = embed_texts(
            texts=payload.texts_b,
            instruction=payload.instruction_b,
            normalize=payload.normalize,
        )
        sims = cosine_similarity_matrix(emb_a, emb_b)
    except Exception as e:
        _print_local_qwen_error("/similarity failed", e)
        raise HTTPException(status_code=500, detail=str(e))

    return SimilarityResponse(
        similarity=sims.cpu().tolist(),
        model=EMBED_MODEL_NAME,
        normalized=payload.normalize,
    )


@app.post("/rank", response_model=RankResponse)
def rank_documents(payload: RankRequest):
    if not payload.documents:
        raise HTTPException(status_code=400, detail="documents must be a non-empty list")

    reranker_failed = False
    reranker_error: Optional[Exception] = None
    rerank_scores: Optional[List[float]] = None

    # Prefer reranker scores when available.
    try:
        rerank_scores = rerank_query_documents(
            query=payload.query,
            documents=payload.documents,
            instruction=payload.instruction,
        )
    except Exception as e:
        _print_local_qwen_error("/rank reranker failed", e)
        reranker_failed = True
        reranker_error = e

    # Optional embedding-based similarity scores.
    # If reranker fails, we force embedding similarity as fallback ranking.
    embedding_scores: Optional[List[float]] = None
    compute_embedding_scores = payload.return_embedding_similarity or reranker_failed
    if compute_embedding_scores:
        try:
            # One query vs many documents
            query_emb = embed_texts(
                texts=[payload.query],
                instruction=payload.instruction,
                normalize=payload.normalize_embeddings,
            )
            doc_embs = embed_texts(
                texts=payload.documents,
                instruction=None,  # typically documents are not instructed
                normalize=payload.normalize_embeddings,
            )
            sims = cosine_similarity_matrix(query_emb, doc_embs)[0]
            embedding_scores = sims.cpu().tolist()
        except Exception as e:
            # Defer failure handling until we know whether fallback is required.
            _print_local_qwen_error("/rank embedding similarity failed", e)
            embedding_scores = None

    # Fallback: if reranker is unavailable, use embedding similarity as rank score.
    if reranker_failed:
        if embedding_scores is None:
            raise HTTPException(
                status_code=500,
                detail=f"reranker error: {reranker_error}; embedding fallback unavailable",
            )
        rerank_scores = [float(score) for score in embedding_scores]

    if rerank_scores is None:
        raise HTTPException(status_code=500, detail="ranker produced no scores")
    if len(rerank_scores) != len(payload.documents):
        raise HTTPException(
            status_code=500,
            detail=(
                f"ranker returned {len(rerank_scores)} scores for "
                f"{len(payload.documents)} documents"
            ),
        )
    if embedding_scores is not None and len(embedding_scores) != len(payload.documents):
        _print_local_qwen_error(
            "/rank embedding similarity length mismatch",
            ValueError(
                f"got {len(embedding_scores)} scores for {len(payload.documents)} documents"
            ),
        )
        embedding_scores = None

    # Build result objects
    results = []
    for idx, (doc, r_score) in enumerate(zip(payload.documents, rerank_scores)):
        emb_score = None
        if embedding_scores is not None:
            emb_score = float(embedding_scores[idx])
        results.append(
            RankedDocument(
                index=idx,
                document=doc,
                reranker_score=float(r_score),
                embedding_score=emb_score,
            )
        )

    # Sort by reranker_score desc, then embedding_score desc (if available)
    results_sorted = sorted(
        results,
        key=lambda r: (r.reranker_score, r.embedding_score if r.embedding_score is not None else -1.0),
        reverse=True,
    )

    # Apply top_k trimming if requested
    if payload.top_k is not None:
        results_sorted = results_sorted[: payload.top_k]

    return RankResponse(
        query=payload.query,
        instruction=payload.instruction,
        qwen_model_reranker=RERANK_MODEL_NAME,
        qwen_model_embedding=EMBED_MODEL_NAME if embedding_scores is not None else None,
        used_embedding_fallback=reranker_error is not None,
        results=results_sorted,
    )

# example usage:

# start the server:
# uvicorn qwen:app --host 0.0.0.0 --port 8000
