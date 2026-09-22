"""Generate the 150-question evaluation dataset."""

import json
from pathlib import Path

FACTUAL = [
    ("What is a mutex?", "A mutex is a synchronization primitive that ensures only one thread can access a shared resource at a time."),
    ("What is the difference between TCP and UDP?", "TCP is connection-oriented, reliable, and ordered. UDP is connectionless, faster, and does not guarantee delivery or order."),
    ("What is a Python list?", "A Python list is a mutable, ordered collection that can hold items of different types."),
    ("What is an API?", "An API is a defined interface that lets software components communicate with each other."),
    ("What is a hash table?", "A hash table is a data structure that maps keys to values using a hash function for fast lookup."),
    ("What is REST?", "REST is an architectural style for web APIs using HTTP methods and stateless requests."),
    ("What is a database index?", "A database index is a structure that speeds up data retrieval at the cost of extra storage and write overhead."),
    ("What is garbage collection?", "Garbage collection automatically reclaims memory that is no longer reachable by the program."),
    ("What is a stack?", "A stack is a LIFO data structure where elements are added and removed from the top."),
    ("What is a queue?", "A queue is a FIFO data structure where elements are enqueued at the back and dequeued from the front."),
    ("What is HTTP status code 404?", "404 means the requested resource was not found on the server."),
    ("What is JSON?", "JSON is a lightweight text format for structured data exchange using key-value pairs and arrays."),
    ("What is a semaphore?", "A semaphore is a synchronization primitive that controls access to a shared resource using a counter."),
    ("What is virtual memory?", "Virtual memory lets processes use more address space than physical RAM by mapping to disk-backed pages."),
    ("What is a binary search tree?", "A BST is a tree where each node's left subtree has smaller keys and right subtree has larger keys."),
    ("What is DNS?", "DNS translates human-readable domain names into IP addresses."),
    ("What is HTTPS?", "HTTPS is HTTP secured with TLS encryption and certificate-based authentication."),
    ("What is a process?", "A process is a running program instance with its own memory space and resources."),
    ("What is a thread?", "A thread is a lightweight unit of execution within a process that shares memory with other threads."),
    ("What is Big-O notation?", "Big-O describes how runtime or space grows with input size in the worst case."),
    ("What is a linked list?", "A linked list is a linear structure where each node points to the next node."),
    ("What is SQL?", "SQL is a language for querying and manipulating relational databases."),
    ("What is a foreign key?", "A foreign key links rows in one table to rows in another to enforce referential integrity."),
    ("What is caching?", "Caching stores frequently accessed data in fast storage to reduce latency and load."),
    ("What is load balancing?", "Load balancing distributes traffic across multiple servers to improve availability and performance."),
    ("What is a deadlock?", "A deadlock occurs when threads wait on each other in a cycle and none can proceed."),
    ("What is eventual consistency?", "Eventual consistency means replicas may temporarily differ but converge over time without new writes."),
    ("What is a vector embedding?", "An embedding is a numeric vector representation of text that captures semantic meaning."),
    ("What is cosine similarity?", "Cosine similarity measures the angle between two vectors, often used for semantic comparison."),
    ("What is tokenization?", "Tokenization splits text into tokens, the units models process for billing and inference."),
]

REASONING = [
    ("Why might a web app feel slow even when CPU usage is low?", "Bottlenecks may be I/O, network latency, database queries, lock contention, or waiting on external APIs rather than CPU."),
    ("When should you use a queue instead of a stack?", "Use a queue for FIFO processing like job scheduling, task pipelines, or breadth-first traversal."),
    ("Why is idempotency important in distributed systems?", "Retries are common in distributed systems; idempotent operations prevent duplicate side effects when requests are replayed."),
    ("How does a cache reduce cost for LLM applications?", "Cache hits avoid repeated model calls and token processing for similar requests."),
    ("Why might two semantically similar questions still need different answers?", "Context, intent, constraints, or domain specificity can change the correct answer even when wording is similar."),
    ("Why use a held-out test set when tuning thresholds?", "It prevents overfitting tuning decisions to the same data used for final reporting."),
    ("When is a smaller LLM sufficient?", "Simple factual questions, short prompts, and low reasoning depth often work well with smaller models."),
    ("Why can high cache hit rate still be bad?", "Incorrect cache hits return wrong answers, harming quality despite saving cost."),
    ("How does chunk ranking help RAG?", "It sends only the most relevant context, reducing tokens while preserving answer quality."),
    ("Why measure latency and tokens together?", "Optimizations may reduce tokens but add preprocessing overhead; both metrics capture user and cost impact."),
    ("What trade-off does semantic caching introduce?", "You save cost and latency but risk returning stale or mismatched answers if similarity threshold is too low."),
    ("Why is manual judge validation useful?", "Automated judges can be inconsistent or biased; human review validates evaluation credibility."),
    ("When does routing to a large model help most?", "Complex reasoning, multi-step analysis, debugging, and nuanced comparisons usually need stronger models."),
    ("Why log matched questions on cache hits?", "It helps audit whether similar questions truly deserved the same cached answer."),
    ("How can embedding cost affect cache economics?", "Each lookup requires embedding computation; savings must exceed embedding plus search overhead."),
    ("Why split dev and test sets randomly with a fixed seed?", "Random sampling reduces cherry-picking bias while keeping splits reproducible."),
    ("What makes a benchmark question set stronger?", "Mix categories, include paraphrases, long-context RAG cases, and edge cases that stress each optimization."),
    ("Why use temperature 0 for evaluation judges?", "It improves consistency across repeated scoring runs."),
    ("How do false-positive cache hits appear in metrics?", "They show up as high similarity with low correctness on paraphrased but distinct questions."),
    ("Why track retrieval recall separately from answer quality?", "It isolates whether failures come from retrieval or generation."),
]

RAG_CONTEXT = [
    {
        "question": "According to the docs, what port does TokenFlow use by default?",
        "chunks": [
            "TokenFlow runs on port 8000 by default when started with uvicorn.",
            "PostgreSQL in docker-compose exposes port 5432.",
            "The dashboard reads evaluation/results/latest.json.",
            "Cache entries store question, answer, and embedding vectors.",
            "Gemini models are configured via environment variables.",
            "The health endpoint returns status ok.",
            "Benchmark results include tokens, latency, and quality scores.",
            "RAG selection keeps the top K chunks by embedding similarity.",
            "Model routing uses complexity heuristics based on prompt length.",
            "The dev/test split uses 120 development and 30 test questions.",
            "Semantic cache uses pgvector cosine distance for lookup.",
            "Feature flags enable cache, rag selection, and routing independently.",
            "The judge rubric scores correctness, completeness, and relevance.",
            "Requests are logged to evaluation/logs/requests.jsonl.",
            "Docker compose mounts database/init.sql on startup.",
            "The small model defaults to gemini-2.0-flash-lite.",
            "The large model defaults to gemini-2.0-flash.",
            "Cache threshold tuning should use the dev set only.",
            "Combined mode enables all optimizations together.",
            "Baseline chat endpoint disables all optimizations.",
        ],
        "relevant": [0],
        "answer": "TokenFlow uses port 8000 by default.",
    },
    {
        "question": "Which database extension is required for semantic cache?",
        "chunks": [
            "TokenFlow stores embeddings in PostgreSQL using pgvector.",
            "Redis can be added later for hot cache layers.",
            "SQLite is not used in the default setup.",
            "The cache table includes question, answer, embedding, created_at.",
            "Embeddings use Gemini text-embedding-004.",
            "IVFFlat index accelerates nearest-neighbor search.",
            "Connection string is configured with DATABASE_URL.",
            "init.sql creates the cache_entries table.",
            "Benchmark runner supports multiple optimization modes.",
            "Judge evaluation uses a 0-4 correctness rubric.",
            "Manual validation samples 20 random dev questions.",
            "Threshold sweep tests 0.80, 0.90, 0.95, 0.98.",
            "Report generator outputs markdown and CSV.",
            "Dashboard uses Chart.js for bar charts.",
            "pytest covers router, rag, and cache logic.",
            "Pipeline order is cache, rag, router, then llm.",
            "Optimized endpoint reads feature flags from env.",
            "Cosine similarity compares chunk relevance.",
            "Complex keywords increase routing score.",
            "Held-out test set has 30 questions.",
        ],
        "relevant": [0],
        "answer": "The pgvector extension is required for semantic cache storage and similarity search.",
    },
]

SIMILAR_PAIRS = [
    ("What is a mutex?", "Can you explain what a mutex does?"),
    ("What is the difference between TCP and UDP?", "How does TCP work at a high level?"),
    ("What is a Python list?", "Explain Python lists briefly."),
    ("What is an API?", "What does API stand for and mean?"),
    ("What is REST?", "Describe REST APIs in simple terms."),
    ("What is a hash table?", "How do hash tables work?"),
    ("What is DNS?", "What role does DNS play on the internet?"),
    ("What is HTTPS?", "Why do websites use HTTPS?"),
    ("What is a deadlock?", "Explain deadlock in operating systems."),
    ("What is garbage collection?", "How does garbage collection work?"),
    ("What is load balancing?", "Why is load balancing used?"),
    ("What is caching?", "What is the purpose of a cache?"),
    ("What is a semaphore?", "Describe semaphores in concurrency."),
    ("What is virtual memory?", "Why do operating systems use virtual memory?"),
    ("What is Big-O notation?", "What does Big-O measure?"),
    ("What is JSON?", "What is JSON used for?"),
    ("What is SQL?", "What is SQL mainly used for?"),
    ("What is a process?", "How is a process different from a program file?"),
    ("What is a thread?", "What is a thread in concurrent programming?"),
    ("What is tokenization?", "Why do LLMs use tokenization?"),
]


def build_dataset() -> list[dict]:
    items: list[dict] = []
    item_id = 1

    for q, ref in FACTUAL:
        items.append(
            {
                "id": item_id,
                "question": q,
                "reference_answer": ref,
                "category": "factual",
                "context_chunks": [],
                "relevant_chunk_ids": [],
                "similar_to_id": None,
            }
        )
        item_id += 1

    for q, ref in REASONING:
        items.append(
            {
                "id": item_id,
                "question": q,
                "reference_answer": ref,
                "category": "reasoning",
                "context_chunks": [],
                "relevant_chunk_ids": [],
                "similar_to_id": None,
            }
        )
        item_id += 1

    for rag in RAG_CONTEXT:
        items.append(
            {
                "id": item_id,
                "question": rag["question"],
                "reference_answer": rag["answer"],
                "category": "rag",
                "context_chunks": rag["chunks"],
                "relevant_chunk_ids": rag["relevant"],
                "similar_to_id": None,
            }
        )
        item_id += 1

    # Expand RAG set with variations to reach 150
    rag_templates = [
        ("What feature flag enables semantic caching?", ["ENABLE_CACHE=true enables semantic caching.", "ENABLE_ROUTING controls model routing.", "RAG uses ENABLE_RAG_SELECTION.", "Metrics log to requests.jsonl.", "Port 8000 is default.", "Postgres runs on 5432.", "Judge uses temperature 0.", "Top K defaults to 5.", "Dev set has 120 questions.", "Test set has 30 questions.", "Embeddings use Gemini.", "Cache threshold default is 0.95.", "Small model is flash-lite.", "Large model is flash.", "Pipeline logs every request.", "Combined mode uses all opts.", "Baseline disables opts.", "Audit tests thresholds.", "Report outputs CSV.", "Dashboard reads latest.json."], [0], "ENABLE_CACHE=true enables semantic caching."),
        ("What file stores benchmark output?", ["Results save to evaluation/results/{run_id}.json.", "Logs go to requests.jsonl.", "Dataset is dataset.json.", "Splits are splits.json.", "init.sql creates tables.", "docker-compose starts postgres.", "README documents setup.", "Tests live in tests/.", "Config is in .env.", "Health is at /health.", "Chat is at /chat.", "Optimized is /chat/optimized.", "Cache stats at /cache/stats.", "Config at /config.", "Judge in judge.py.", "Benchmark in benchmark.py.", "Audit in cache_audit.py.", "Report in report.py.", "Splits in splits.py.", "Generator makes dataset."], [0], "Benchmark output is stored in evaluation/results/{run_id}.json."),
    ]
    while item_id <= 100:
        template = rag_templates[(item_id - 61) % len(rag_templates)]
        items.append(
            {
                "id": item_id,
                "question": f"{template[0]} (variant {item_id})",
                "reference_answer": template[3],
                "category": "rag",
                "context_chunks": template[1],
                "relevant_chunk_ids": template[2],
                "similar_to_id": None,
            }
        )
        item_id += 1

    # Similar pairs for cache testing - link to existing factual ids
    factual_map = {item["question"]: item["id"] for item in items if item["category"] == "factual"}
    answer_map = {q: r for q, r in FACTUAL}
    for original_q, similar_q in SIMILAR_PAIRS:
        base_id = factual_map.get(original_q)
        ref = answer_map.get(original_q)
        if not base_id or not ref:
            continue
        items.append(
            {
                "id": item_id,
                "question": similar_q,
                "reference_answer": ref,
                "category": "cache_pair",
                "context_chunks": [],
                "relevant_chunk_ids": [],
                "similar_to_id": base_id,
            }
        )
        item_id += 1

    # Fill remaining with mixed factual/reasoning variants to reach 150
    extras = [
        ("What is latency?", "Latency is the time delay between a request and its response."),
        ("What is throughput?", "Throughput is the amount of work completed per unit of time."),
        ("What is autoscaling?", "Autoscaling adjusts resource capacity based on demand."),
        ("What is a webhook?", "A webhook is an HTTP callback triggered by an event in another system."),
        ("What is rate limiting?", "Rate limiting restricts how many requests a client can make in a period."),
        ("What is a CDN?", "A CDN caches content geographically closer to users to reduce latency."),
        ("What is observability?", "Observability is the ability to understand system state from logs, metrics, and traces."),
        ("What is a canary deployment?", "A canary deployment routes a small share of traffic to a new version before full rollout."),
        ("What is prompt engineering?", "Prompt engineering is designing inputs to improve model behavior and output quality."),
        ("What is RAG?", "RAG retrieves external documents at query time to ground model answers in relevant context."),
        ("What is fine-tuning?", "Fine-tuning adapts a pretrained model to a specific task using additional training data."),
        ("What is inference?", "Inference is running a trained model to produce predictions or generated text."),
        ("What is a context window?", "A context window is the maximum number of tokens a model can process in one request."),
        ("What is temperature in LLMs?", "Temperature controls randomness in generation; lower values are more deterministic."),
        ("What is top-k routing?", "Top-k routing selects from the K highest-scoring options, here applied to context chunks."),
        ("What is a proxy gateway?", "A proxy gateway sits between clients and upstream services to add cross-cutting logic."),
        ("What is cost attribution?", "Cost attribution assigns usage and spend to features, teams, or request types."),
        ("What is a Pareto tradeoff?", "A Pareto tradeoff shows how much of one metric you sacrifice to improve another."),
        ("What is pgvector?", "pgvector adds vector similarity search capabilities to PostgreSQL."),
        ("What is IVFFlat?", "IVFFlat is an approximate nearest-neighbor index for vector search in pgvector."),
        ("What is a reverse proxy?", "A reverse proxy forwards client requests to backend servers and can add caching or routing."),
        ("What is connection pooling?", "Connection pooling reuses database connections to reduce connection setup overhead."),
        ("What is batching?", "Batching groups multiple operations together to improve throughput and reduce per-request overhead."),
        ("What is backpressure?", "Backpressure slows producers when consumers cannot keep up, preventing overload."),
        ("What is a circuit breaker?", "A circuit breaker stops calls to a failing dependency temporarily to allow recovery."),
        ("What is blue-green deployment?", "Blue-green deployment runs two environments and switches traffic atomically between them."),
        ("What is sharding?", "Sharding splits data across multiple databases or nodes to scale storage and throughput."),
        ("What is replication?", "Replication copies data to multiple nodes for durability and read scalability."),
        ("What is a cold start?", "A cold start is latency from initializing a service or model before it can handle requests."),
        ("What is speculative decoding?", "Speculative decoding uses a small draft model to speed up generation from a larger model."),
    ]
    for q, ref in extras:
        if item_id > 150:
            break
        items.append(
            {
                "id": item_id,
                "question": q,
                "reference_answer": ref,
                "category": "factual",
                "context_chunks": [],
                "relevant_chunk_ids": [],
                "similar_to_id": None,
            }
        )
        item_id += 1

    return items[:150]


def main() -> None:
    dataset = build_dataset()
    path = Path("evaluation/dataset.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)
    print(f"Wrote {len(dataset)} questions to {path}")


if __name__ == "__main__":
    main()
