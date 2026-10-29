"""
TrustGate External Verifier V6
--------------------------------
Evidence-grounded external verification for TrustGate.

V6 focuses on the actual bottleneck observed in V4/V5:
1. rate-limit-safe Wikipedia retrieval;
2. persistent search/page caching;
3. bounded request budget;
4. question-first retrieval;
5. claim/sentence-level evidence scoring;
6. source-aware consensus;
7. conservative verification statuses.

Public compatibility:
    ExternalVerifier(...)
    verifier.verify(question, answer)

Verification statuses:
    SUPPORTED
    CONTRADICTED
    EVIDENCE_FOUND
    INSUFFICIENT_EVIDENCE

Important:
This module is a conservative evidence verifier, not a general-purpose
fact-checking model. Lexical/negation evidence is treated as evidence,
not proof of truth.
"""

import json
import os
import re
import time
from collections import Counter
from urllib.parse import quote

import requests


class ExternalVerifier:
    """Rate-limit-safe, evidence-grounded verifier."""

    def __init__(
        self,
        max_sources=3,
        timeout=10,
        request_delay=4.0,
        max_retries=1,
        max_requests_per_verify=4,
        cache_path="outputs/verifier_v6_cache.json",
        cache_ttl_seconds=7 * 24 * 3600,
        min_supported_score=0.60,
        min_contradicted_score=0.60,
        decision_margin=0.10,
        min_source_consensus=2,
    ):
        self.max_sources = int(max_sources)
        self.timeout = float(timeout)
        self.request_delay = float(request_delay)
        self.max_retries = int(max_retries)
        self.max_requests_per_verify = int(max_requests_per_verify)
        self.cache_path = cache_path
        self.cache_ttl_seconds = float(cache_ttl_seconds)

        self.min_supported_score = float(min_supported_score)
        self.min_contradicted_score = float(min_contradicted_score)
        self.decision_margin = float(decision_margin)
        self.min_source_consensus = int(min_source_consensus)

        self.search_url = "https://en.wikipedia.org/w/api.php"

        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": (
                    "TrustGate-Research/6.0 "
                    "(academic research project; contact via Wikipedia API policy)"
                )
            }
        )

        # In-memory caches.
        self.search_cache = {}
        self.page_cache = {}

        # Persistent cache lets A0/A1 and later runs reuse pages.
        self._load_disk_cache()

        # Request accounting.
        self.last_request_time = 0.0
        self.request_count = 0
        self.verify_request_count = 0
        self.cache_hits = 0
        self.rate_limit_events = 0

    # =========================================================
    # CACHE
    # =========================================================

    def _load_disk_cache(self):
        self.disk_cache = {"search": {}, "pages": {}}

        if not self.cache_path:
            return

        if not os.path.exists(self.cache_path):
            return

        try:
            with open(self.cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, dict):
                self.disk_cache["search"] = data.get("search", {})
                self.disk_cache["pages"] = data.get("pages", {})
        except Exception:
            self.disk_cache = {"search": {}, "pages": {}}

    def _save_disk_cache(self):
        if not self.cache_path:
            return

        directory = os.path.dirname(self.cache_path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        payload = {
            "version": "trustgate_verifier_v6",
            "updated_at": time.time(),
            "search": self.disk_cache.get("search", {}),
            "pages": self.disk_cache.get("pages", {}),
        }

        tmp = self.cache_path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
            os.replace(tmp, self.cache_path)
        except Exception:
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except OSError:
                pass

    @staticmethod
    def _cache_fresh(entry, ttl_seconds):
        if not isinstance(entry, dict):
            return False
        timestamp = entry.get("timestamp")
        if not isinstance(timestamp, (int, float)):
            return False
        return (time.time() - timestamp) <= ttl_seconds

    # =========================================================
    # REQUEST CONTROL
    # =========================================================

    def _wait_before_request(self):
        elapsed = time.time() - self.last_request_time
        if elapsed < self.request_delay:
            time.sleep(self.request_delay - elapsed)

    def _request_budget_available(self):
        return self.verify_request_count < self.max_requests_per_verify

    def _get(self, params):
        """
        Perform a bounded API request.

        A 429 is not hammered repeatedly. Retry-After is respected when
        available; otherwise a conservative backoff is used.
        """
        if not self._request_budget_available():
            print("[Verifier] Request budget exhausted.")
            return None

        for attempt in range(self.max_retries + 1):
            if not self._request_budget_available():
                return None

            self._wait_before_request()

            try:
                self.verify_request_count += 1
                self.request_count += 1

                response = self.session.get(
                    self.search_url,
                    params=params,
                    timeout=self.timeout,
                )
                self.last_request_time = time.time()

                if response.status_code == 429:
                    self.rate_limit_events += 1

                    retry_after = response.headers.get("Retry-After")
                    if retry_after:
                        try:
                            wait_time = min(float(retry_after), 60.0)
                        except ValueError:
                            wait_time = 15.0
                    else:
                        wait_time = min(15.0 * (attempt + 1), 60.0)

                    if attempt >= self.max_retries:
                        print(
                            "[Verifier] HTTP 429 persisted; "
                            "stopping instead of hammering Wikipedia."
                        )
                        return None

                    print(
                        f"[Verifier] HTTP 429; waiting {wait_time:.1f}s "
                        "before one controlled retry."
                    )
                    time.sleep(wait_time)
                    continue

                response.raise_for_status()
                return response

            except requests.RequestException as exc:
                self.last_request_time = time.time()

                if attempt >= self.max_retries:
                    print("[Verifier] Request failed:", str(exc))
                    return None

                wait_time = min(5.0 * (attempt + 1), 20.0)
                print(
                    f"[Verifier] Request error; retrying once after "
                    f"{wait_time:.1f}s."
                )
                time.sleep(wait_time)

        return None

    # =========================================================
    # TEXT NORMALIZATION
    # =========================================================

    @staticmethod
    def normalize_text(text):
        if text is None:
            return ""
        text = str(text).lower()
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    @staticmethod
    def tokenize(text):
        text = ExternalVerifier.normalize_text(text)
        return set(re.findall(r"\b[a-z0-9]{3,}\b", text))

    @staticmethod
    def stopwords():
        return {
            "the", "and", "for", "are", "was", "were", "with", "that",
            "this", "from", "have", "has", "had", "not", "but", "can",
            "could", "would", "should", "does", "did", "how", "what",
            "when", "where", "which", "who", "why", "much", "many",
            "about", "into", "their", "there", "they", "them", "then",
            "than", "your", "you", "its", "it's", "been", "being",
            "also", "some", "such", "more", "most", "very", "only",
            "people", "person", "thing", "things", "tell", "please",
            "following", "answer", "question",
        }

    def important_terms(self, text, max_terms=12):
        tokens = re.findall(r"\b[a-zA-Z][a-zA-Z0-9'-]*\b", self.normalize_text(text))
        stopwords = self.stopwords()

        terms = [
            token
            for token in tokens
            if token not in stopwords and len(token) >= 4
        ]

        counts = Counter(terms)
        return [word for word, _ in counts.most_common(max_terms)]

    # =========================================================
    # SEARCH QUERY GENERATION
    # =========================================================

    def build_search_query(self, question, answer):
        """
        One compact, question-first query.

        The question is the primary retrieval target. Answer terms are used
        only to disambiguate entities that occur in the generated answer.
        """
        question_terms = self.important_terms(question, max_terms=8)
        answer_terms = self.important_terms(answer, max_terms=4)

        selected = list(question_terms[:6])

        for term in answer_terms:
            if term not in selected:
                selected.append(term)
            if len(selected) >= 8:
                break

        if selected:
            return " ".join(selected)

        return self.normalize_text(question)

    def build_search_queries(self, question, answer):
        """
        Compatibility helper.

        V6 deliberately returns one retrieval query by default. Multiple
        search calls were a major contributor to API 429 responses.
        """
        query = self.build_search_query(question, answer)
        return [query] if query else []

    # =========================================================
    # WIKIPEDIA SEARCH
    # =========================================================

    def search_wikipedia(self, query, limit=3):
        query = self.normalize_text(query)
        if not query:
            return []

        cache_key = query

        memory_entry = self.search_cache.get(cache_key)
        if self._cache_fresh(memory_entry, self.cache_ttl_seconds):
            self.cache_hits += 1
            return memory_entry.get("results", [])

        disk_entry = self.disk_cache.get("search", {}).get(cache_key)
        if self._cache_fresh(disk_entry, self.cache_ttl_seconds):
            results = disk_entry.get("results", [])
            self.search_cache[cache_key] = disk_entry
            self.cache_hits += 1
            return results

        if not self._request_budget_available():
            print("[Verifier] Search skipped: request budget exhausted.")
            return []

        params = {
            "action": "query",
            "list": "search",
            "format": "json",
            "utf8": 1,
            "srlimit": int(limit),
            "srsearch": query,
        }

        response = self._get(params)
        if response is None:
            return []

        try:
            data = response.json()
            raw = data.get("query", {}).get("search", [])

            results = []
            for item in raw:
                title = item.get("title", "")
                page_id = item.get("pageid")
                if not title:
                    continue

                results.append(
                    {
                        "title": title,
                        "page_id": page_id,
                        "url": (
                            "https://en.wikipedia.org/wiki/"
                            + quote(title.replace(" ", "_"))
                        ),
                        "source": "Wikipedia",
                        "snippet": item.get("snippet", ""),
                    }
                )

            entry = {"timestamp": time.time(), "results": results}
            self.search_cache[cache_key] = entry
            self.disk_cache.setdefault("search", {})[cache_key] = entry
            self._save_disk_cache()
            return results

        except Exception as exc:
            print("[Verifier] Search parsing error:", str(exc))
            return []

    # =========================================================
    # PAGE RETRIEVAL
    # =========================================================

    def retrieve_wikipedia_page(self, title):
        title = str(title).strip()
        if not title:
            return ""

        memory_entry = self.page_cache.get(title)
        if self._cache_fresh(memory_entry, self.cache_ttl_seconds):
            self.cache_hits += 1
            return memory_entry.get("text", "")

        disk_entry = self.disk_cache.get("pages", {}).get(title)
        if self._cache_fresh(disk_entry, self.cache_ttl_seconds):
            text = disk_entry.get("text", "")
            self.page_cache[title] = disk_entry
            self.cache_hits += 1
            return text

        if not self._request_budget_available():
            print("[Verifier] Page retrieval skipped: request budget exhausted.")
            return ""

        params = {
            "action": "query",
            "prop": "extracts",
            "explaintext": 1,
            "format": "json",
            "utf8": 1,
            "titles": title,
        }

        response = self._get(params)
        if response is None:
            return ""

        try:
            data = response.json()
            pages = data.get("query", {}).get("pages", {})

            for page in pages.values():
                extract = page.get("extract", "") or ""
                if extract:
                    entry = {"timestamp": time.time(), "text": extract}
                    self.page_cache[title] = entry
                    self.disk_cache.setdefault("pages", {})[title] = entry
                    self._save_disk_cache()
                    return extract

        except Exception as exc:
            print("[Verifier] Page parsing error:", str(exc))

        return ""

    # =========================================================
    # SENTENCE PROCESSING
    # =========================================================

    @staticmethod
    def split_sentences(text):
        if not text:
            return []

        text = re.sub(r"\s+", " ", str(text)).strip()
        parts = re.split(r"(?<=[.!?])\s+", text)

        return [
            part.strip()
            for part in parts
            if len(part.strip()) >= 20
        ]

    def sentence_relevance(self, sentence, question_terms, answer_terms):
        sentence_terms = self.tokenize(sentence)

        if not sentence_terms:
            return 0.0

        q = set(question_terms)
        a = set(answer_terms)

        q_overlap = len(sentence_terms & q) / max(len(q), 1)
        a_overlap = len(sentence_terms & a) / max(len(a), 1)

        # Question dominates because it defines what is being checked.
        return (0.65 * q_overlap) + (0.35 * a_overlap)

    def extract_relevant_sentences(
        self,
        question,
        answer,
        evidence_text,
        max_sentences=6,
    ):
        question_terms = self.important_terms(question, max_terms=12)
        answer_terms = self.important_terms(answer, max_terms=10)

        if not (question_terms or answer_terms):
            return []

        scored = []

        for sentence in self.split_sentences(evidence_text):
            score = self.sentence_relevance(
                sentence,
                question_terms,
                answer_terms,
            )

            if score >= 0.12:
                scored.append(
                    {
                        "score": score,
                        "sentence": sentence,
                    }
                )

        scored.sort(key=lambda item: item["score"], reverse=True)

        return [
            item["sentence"]
            for item in scored[:max_sentences]
        ]

    # =========================================================
    # EVIDENCE COLLECTION
    # =========================================================

    def _rank_search_results(self, question, answer, results):
        q_terms = set(self.important_terms(question, max_terms=12))
        a_terms = set(self.important_terms(answer, max_terms=10))

        ranked = []

        for result in results:
            title_terms = self.tokenize(result.get("title", ""))
            snippet_terms = self.tokenize(result.get("snippet", ""))

            title_score = len(title_terms & q_terms) / max(len(q_terms), 1)
            snippet_q = len(snippet_terms & q_terms) / max(len(q_terms), 1)
            snippet_a = len(snippet_terms & a_terms) / max(len(a_terms), 1)

            score = (
                0.55 * title_score
                + 0.30 * snippet_q
                + 0.15 * snippet_a
            )

            ranked.append((score, result))

        ranked.sort(key=lambda item: item[0], reverse=True)
        return [result for _, result in ranked]

    def collect_evidence(self, question, answer):
        queries = self.build_search_queries(question, answer)

        print("[Verifier] Search queries:", queries)

        if not queries:
            return []

        # V6 uses exactly one search request unless the result is cached.
        search_results = self.search_wikipedia(
            queries[0],
            limit=max(self.max_sources, 5),
        )

        if not search_results:
            return []

        ranked_results = self._rank_search_results(
            question,
            answer,
            search_results,
        )

        evidence_records = []
        seen_titles = set()

        for result in ranked_results:
            title = result.get("title", "")
            if not title or title in seen_titles:
                continue

            seen_titles.add(title)

            page_text = self.retrieve_wikipedia_page(title)
            if not page_text:
                continue

            relevant_sentences = self.extract_relevant_sentences(
                question,
                answer,
                page_text,
                max_sentences=6,
            )

            if not relevant_sentences:
                continue

            evidence_records.append(
                {
                    "title": title,
                    "page_id": result.get("page_id"),
                    "source": "Wikipedia",
                    "url": result.get(
                        "url",
                        "https://en.wikipedia.org/wiki/"
                        + quote(title.replace(" ", "_")),
                    ),
                    "sentences": relevant_sentences,
                }
            )

            if len(evidence_records) >= self.max_sources:
                break

            if not self._request_budget_available():
                break

        return evidence_records

    # =========================================================
    # NEGATION
    # =========================================================

    @staticmethod
    def contains_negation(text):
        text = ExternalVerifier.normalize_text(text)

        patterns = [
            r"\bnot\b",
            r"\bno\b",
            r"\bnever\b",
            r"\bneither\b",
            r"\bwithout\b",
            r"\bcannot\b",
            r"\bcan't\b",
            r"\bdoesn't\b",
            r"\bdon't\b",
            r"\bdidn't\b",
            r"\bwon't\b",
            r"\bwasn't\b",
            r"\bisn't\b",
            r"\baren't\b",
        ]

        return any(re.search(pattern, text) for pattern in patterns)

    @staticmethod
    def remove_negation(text):
        text = ExternalVerifier.normalize_text(text)

        patterns = [
            r"\bnot\b",
            r"\bno\b",
            r"\bnever\b",
            r"\bneither\b",
            r"\bwithout\b",
            r"\bcannot\b",
            r"\bcan't\b",
            r"\bdoesn't\b",
            r"\bdon't\b",
            r"\bdidn't\b",
            r"\bwon't\b",
            r"\bwasn't\b",
            r"\bisn't\b",
            r"\baren't\b",
        ]

        for pattern in patterns:
            text = re.sub(pattern, " ", text)

        return re.sub(r"\s+", " ", text).strip()

    # =========================================================
    # CLAIM-LEVEL SCORING
    # =========================================================

    def sentence_support_score(self, answer, evidence_sentence):
        """
        F1-style overlap.

        Precision = fraction of answer terms found in evidence.
        Recall    = fraction of evidence terms found in answer.

        The harmonic mean reduces the tendency of a long Wikipedia
        sentence to receive a high score merely because it shares a
        few common terms.
        """
        answer_words = self.tokenize(answer)
        evidence_words = self.tokenize(evidence_sentence)

        if not answer_words or not evidence_words:
            return 0.0

        overlap = answer_words & evidence_words

        precision = len(overlap) / max(len(answer_words), 1)
        recall = len(overlap) / max(len(evidence_words), 1)

        if precision + recall == 0:
            return 0.0

        return 2.0 * precision * recall / (precision + recall)

    def _claim_score(self, claim, evidence_sentence):
        claim_words = self.tokenize(self.remove_negation(claim))
        evidence_words = self.tokenize(self.remove_negation(evidence_sentence))

        if not claim_words or not evidence_words:
            return 0.0

        overlap = claim_words & evidence_words

        precision = len(overlap) / max(len(claim_words), 1)
        recall = len(overlap) / max(len(evidence_words), 1)

        if precision + recall == 0:
            return 0.0

        return 2.0 * precision * recall / (precision + recall)

    # =========================================================
    # EVIDENCE ASSESSMENT
    # =========================================================

    def assess_evidence(self, answer, evidence_records):
        if not evidence_records:
            return {
                "status": "INSUFFICIENT_EVIDENCE",
                "support_score": 0.0,
                "contradiction_score": 0.0,
                "supporting_sentences": [],
                "contradicting_sentences": [],
                "reason": "No relevant evidence retrieved.",
            }

        answer_sentences = self.split_sentences(answer)
        if not answer_sentences:
            answer_sentences = [answer]

        source_support = {}
        source_contradiction = {}

        supporting_sentences = []
        contradicting_sentences = []

        for record in evidence_records:
            source_name = record.get("title", record.get("source", "unknown"))
            best_support = 0.0
            best_contradiction = 0.0
            best_support_sentence = None
            best_contradiction_sentence = None

            for evidence_sentence in record.get("sentences", []):
                evidence_negated = self.contains_negation(evidence_sentence)

                for claim in answer_sentences:
                    claim_negated = self.contains_negation(claim)
                    score = self._claim_score(claim, evidence_sentence)

                    if score < 0.20:
                        continue

                    if claim_negated == evidence_negated:
                        if score > best_support:
                            best_support = score
                            best_support_sentence = evidence_sentence
                    else:
                        if score > best_contradiction:
                            best_contradiction = score
                            best_contradiction_sentence = evidence_sentence

            source_support[source_name] = best_support
            source_contradiction[source_name] = best_contradiction

            if best_support_sentence is not None:
                supporting_sentences.append(
                    best_support_sentence
                )

            if best_contradiction_sentence is not None:
                contradicting_sentences.append(
                    best_contradiction_sentence
                )

        # Independent-source consensus:
        # the strongest source gets full weight; additional sources add
        # corroboration without allowing one page to dominate.
        support_values = sorted(
            [v for v in source_support.values() if v > 0],
            reverse=True,
        )
        contradiction_values = sorted(
            [v for v in source_contradiction.values() if v > 0],
            reverse=True,
        )

        def consensus(values):
            if not values:
                return 0.0

            top = values[:3]
            weighted = 0.0
            total_weight = 0.0

            for idx, value in enumerate(top):
                weight = 1.0 / (idx + 1)
                weighted += weight * value
                total_weight += weight

            return weighted / total_weight

        support_score = consensus(support_values)
        contradiction_score = consensus(contradiction_values)

        support_sources = sum(
            1 for value in source_support.values()
            if value >= self.min_supported_score
        )
        contradiction_sources = sum(
            1 for value in source_contradiction.values()
            if value >= self.min_contradicted_score
        )

        # Strong contradiction has priority when independently supported.
        if (
            contradiction_score >= self.min_contradicted_score
            and contradiction_score > support_score + self.decision_margin
            and (
                contradiction_sources >= self.min_source_consensus
                or len(evidence_records) == 1
            )
        ):
            return {
                "status": "CONTRADICTED",
                "support_score": support_score,
                "contradiction_score": contradiction_score,
                "supporting_sentences": supporting_sentences,
                "contradicting_sentences": contradicting_sentences,
                "reason": (
                    "Independent evidence has stronger opposing "
                    "claim-level alignment."
                ),
            }

        # Strong support requires corroboration where possible.
        if (
            support_score >= self.min_supported_score
            and support_score > contradiction_score + self.decision_margin
            and (
                support_sources >= self.min_source_consensus
                or len(evidence_records) == 1
            )
        ):
            return {
                "status": "SUPPORTED",
                "support_score": support_score,
                "contradiction_score": contradiction_score,
                "supporting_sentences": supporting_sentences,
                "contradicting_sentences": contradicting_sentences,
                "reason": (
                    "Independent evidence provides stronger "
                    "claim-level support."
                ),
            }

        if support_score > 0.0 or contradiction_score > 0.0:
            return {
                "status": "EVIDENCE_FOUND",
                "support_score": support_score,
                "contradiction_score": contradiction_score,
                "supporting_sentences": supporting_sentences,
                "contradicting_sentences": contradicting_sentences,
                "reason": (
                    "Relevant evidence was retrieved, but the "
                    "evidence is not sufficiently decisive."
                ),
            }

        return {
            "status": "INSUFFICIENT_EVIDENCE",
            "support_score": 0.0,
            "contradiction_score": 0.0,
            "supporting_sentences": [],
            "contradicting_sentences": [],
            "reason": (
                "Retrieved pages did not provide sufficiently "
                "relevant claim-level evidence."
            ),
        }

    # =========================================================
    # VERIFY
    # =========================================================

    def verify(self, question, answer):
        # Reset per-verification request budget.
        self.verify_request_count = 0

        evidence_records = self.collect_evidence(
            question,
            answer,
        )

        assessment = self.assess_evidence(
            answer,
            evidence_records,
        )

        evidence_sentence_count = sum(
            len(record.get("sentences", []))
            for record in evidence_records
        )

        return {
            "question": question,
            "answer": answer,
            "verification_status": assessment["status"],
            "source_count": len(evidence_records),
            "evidence_sentence_count": evidence_sentence_count,
            "support_score": assessment["support_score"],
            "contradiction_score": assessment["contradiction_score"],
            "reason": assessment["reason"],
            "supporting_sentences": assessment.get(
                "supporting_sentences", []
            ),
            "contradicting_sentences": assessment.get(
                "contradicting_sentences", []
            ),
            "sources": evidence_records,
            "verifier_version": "V6",
            "request_count": self.verify_request_count,
            "cache_hits": self.cache_hits,
            "rate_limit_events": self.rate_limit_events,
        }


# =============================================================
# SELF TEST
# =============================================================

if __name__ == "__main__":
    print("=" * 72)
    print("EXTERNAL VERIFIER V6 SELF-TEST")
    print("=" * 72)

    verifier = ExternalVerifier(
        max_sources=3,
        timeout=10,
        request_delay=2.0,
        max_retries=1,
        max_requests_per_verify=4,
    )

    # ---------------------------------------------------------
    # TEST 1: query generation
    # ---------------------------------------------------------
    print("\n[TEST 1] Query generation")

    queries = verifier.build_search_queries(
        "What is the capital of France?",
        "The capital of France is Paris.",
    )

    print("Queries:", queries)

    assert len(queries) == 1
    assert queries[0]

    print("[PASS] One question-first query generated.")

    # ---------------------------------------------------------
    # TEST 2: controlled support
    # ---------------------------------------------------------
    print("\n[TEST 2] Controlled support")

    controlled_support = [
        {
            "title": "France",
            "source": "Test",
            "url": "",
            "sentences": [
                "Paris is the capital and largest city of France."
            ],
        },
        {
            "title": "Paris",
            "source": "Test",
            "url": "",
            "sentences": [
                "Paris is the capital city of France."
            ],
        },
    ]

    support_result = verifier.assess_evidence(
        "Paris is the capital of France.",
        controlled_support,
    )

    print("Status:", support_result["status"])
    print("Support:", round(support_result["support_score"], 4))

    assert support_result["status"] == "SUPPORTED"
    print("[PASS] Multi-source support detected.")

    # ---------------------------------------------------------
    # TEST 3: controlled contradiction
    # ---------------------------------------------------------
    print("\n[TEST 3] Controlled contradiction")

    controlled_contradiction = [
        {
            "title": "Earth",
            "source": "Test",
            "url": "",
            "sentences": [
                "The Earth is not flat."
            ],
        },
        {
            "title": "Shape",
            "source": "Test",
            "url": "",
            "sentences": [
                "Earth is not flat."
            ],
        },
    ]

    contradiction_result = verifier.assess_evidence(
        "The Earth is flat.",
        controlled_contradiction,
    )

    print("Status:", contradiction_result["status"])
    print(
        "Contradiction:",
        round(contradiction_result["contradiction_score"], 4),
    )

    assert contradiction_result["status"] == "CONTRADICTED"
    print("[PASS] Multi-source contradiction detected.")

    # ---------------------------------------------------------
    # TEST 4: ambiguous evidence remains ambiguous
    # ---------------------------------------------------------
    print("\n[TEST 4] Conservative ambiguity")

    ambiguous = [
        {
            "title": "Generic",
            "source": "Test",
            "url": "",
            "sentences": [
                "The topic has been discussed in many different contexts."
            ],
        }
    ]

    ambiguous_result = verifier.assess_evidence(
        "This treatment always cures every infection.",
        ambiguous,
    )

    print("Status:", ambiguous_result["status"])

    assert ambiguous_result["status"] == "INSUFFICIENT_EVIDENCE"
    print("[PASS] Weak evidence is not promoted to a verdict.")

    # ---------------------------------------------------------
    # TEST 5: optional live retrieval
    # ---------------------------------------------------------
    print("\n[TEST 5] Live Wikipedia retrieval")

    live = verifier.verify(
        "What is the capital of France?",
        "The capital of France is Paris.",
    )

    print("Status:", live["verification_status"])
    print("Sources:", live["source_count"])
    print("Evidence sentences:", live["evidence_sentence_count"])
    print("Support:", round(live["support_score"], 4))
    print("Contradiction:", round(live["contradiction_score"], 4))
    print("Requests:", live["request_count"])
    print("Cache hits:", live["cache_hits"])
    print("429 events:", live["rate_limit_events"])

    assert live["verification_status"] in {
        "SUPPORTED",
        "CONTRADICTED",
        "EVIDENCE_FOUND",
        "INSUFFICIENT_EVIDENCE",
    }

    print("[PASS] Live verifier returned a valid status.")

    print("\n" + "=" * 72)
    print("ALL EXTERNAL VERIFIER V6 TESTS PASSED")
    print("=" * 72)

