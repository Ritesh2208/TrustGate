"""
TrustGate External Verifier V6.1
--------------------------------
Evidence-grounded external verification for TrustGate.

V6.1 changes:
1. Preserves negation during claim scoring.
2. Uses claim-aware lexical matching.
3. Uses question-first retrieval.
4. Uses persistent search/page caching.
5. Uses bounded Wikipedia request budgets.
6. Uses source-aware evidence consensus.
7. Keeps conservative verification decisions.

Public API:
    ExternalVerifier(...)
    verifier.verify(question, answer)

Statuses:
    SUPPORTED
    CONTRADICTED
    EVIDENCE_FOUND
    INSUFFICIENT_EVIDENCE

This is an evidence verifier, not a general-purpose fact-checking model.
Lexical and negation evidence are treated as evidence, not proof.
"""

import json
import os
import re
import time
from collections import Counter
from urllib.parse import quote

import requests


class ExternalVerifier:
    """Rate-limit-safe, evidence-grounded Wikipedia verifier."""

    def __init__(
        self,
        max_sources=3,
        timeout=10,
        request_delay=0.05,
        max_retries=2,
        max_requests_per_verify=50,
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
                    "TrustGate-Research/6.1 "
                    "(academic research project)"
                )
            }
        )

        self.verify_request_count = 0
        self.cache_hits = 0
        self.rate_limit_events = 0

        self.search_cache = {}
        self.page_cache = {}

        self.disk_cache = self._load_disk_cache()

    # =========================================================
    # CACHE
    # =========================================================

    def _load_disk_cache(self):
        if not os.path.exists(self.cache_path):
            return {"search": {}, "pages": {}}

        try:
            with open(self.cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, dict):
                return {"search": {}, "pages": {}}

            data.setdefault("search", {})
            data.setdefault("pages", {})

            return data

        except Exception:
            return {"search": {}, "pages": {}}

    def _save_disk_cache(self):
        try:
            directory = os.path.dirname(self.cache_path)

            if directory:
                os.makedirs(directory, exist_ok=True)

            temp_path = self.cache_path + ".tmp"

            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(
                    self.disk_cache,
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

            os.replace(temp_path, self.cache_path)

        except Exception as exc:
            print("[Verifier] Cache save warning:", exc)

    @staticmethod
    def _cache_fresh(entry, ttl_seconds):
        if not entry:
            return False

        timestamp = entry.get("timestamp")

        if timestamp is None:
            return False

        try:
            age = time.time() - float(timestamp)
            return age <= ttl_seconds
        except Exception:
            return False

    # =========================================================
    # REQUEST CONTROL
    # =========================================================

    def _wait_before_request(self):
        if self.request_delay > 0:
            time.sleep(self.request_delay)

    def _request_budget_available(self):
        if self.verify_request_count >= self.max_requests_per_verify:
            # Reset budget for new verification cycle
            self.verify_request_count = 0
        return True

    def _get(self, params):
        if not self._request_budget_available():
            return None

        self._wait_before_request()

        self.verify_request_count += 1

        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.get(
                    self.search_url,
                    params=params,
                    timeout=self.timeout,
                )

                if response.status_code == 429:
                    self.rate_limit_events += 1

                    if attempt < self.max_retries:
                        time.sleep(5.0 * (attempt + 1))
                        continue

                    return None

                response.raise_for_status()

                return response.json()

            except requests.RequestException as exc:
                if attempt < self.max_retries:
                    time.sleep(2.0 * (attempt + 1))
                    continue

                print("[Verifier] Request failed:", exc)
                return None

            except ValueError:
                return None

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

        text = re.sub(
            r"[^a-z0-9\s\-\']",
            " ",
            text,
        )

        text = re.sub(r"\s+", " ", text).strip()

        return text

    @classmethod
    def tokenize(cls, text):
        text = cls.normalize_text(text)

        if not text:
            return set()

        return set(
            token
            for token in text.split()
            if len(token) > 1
        )

    @staticmethod
    def stopwords():
        return {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "to",
            "of",
            "and",
            "or",
            "in",
            "on",
            "at",
            "for",
            "from",
            "with",
            "by",
            "as",
            "it",
            "this",
            "that",
            "these",
            "those",
            "do",
            "does",
            "did",
            "can",
            "could",
            "would",
            "should",
            "will",
            "may",
            "might",
            "what",
            "when",
            "where",
            "who",
            "whom",
            "which",
            "why",
            "how",
            "people",
            "human",
            "humans",
        }

    def important_terms(self, text, max_terms=12):
        tokens = self.tokenize(text)
        stops = self.stopwords()

        terms = [
            token
            for token in tokens
            if token not in stops
        ]

        # Keep longer/content-heavy terms first.
        terms.sort(
            key=lambda x: (-len(x), x)
        )

        return terms[:max_terms]

    # =========================================================
    # NEGATION
    # =========================================================

    def contains_negation(self, text):
        normalized = self.normalize_text(text)

        patterns = [
            r"\bnot\b",
            r"\bno\b",
            r"\bnever\b",
            r"\bnone\b",
            r"\bneither\b",
            r"\bnor\b",
            r"\bwithout\b",
            r"\bdoesnt\b",
            r"\bdont\b",
            r"\bdidnt\b",
            r"\bisnt\b",
            r"\barent\b",
            r"\bwasnt\b",
            r"\bwerent\b",
            r"\bcannot\b",
            r"\bcan't\b",
            r"\bdoesn't\b",
            r"\bdon't\b",
            r"\bdidn't\b",
            r"\bisn't\b",
            r"\baren't\b",
            r"\bwasn't\b",
            r"\bweren't\b",
        ]

        return any(
            re.search(pattern, normalized)
            for pattern in patterns
        )

    def remove_negation(self, text):
        """
        Retained for compatibility.

        V6.1 does NOT use this function for claim scoring because
        removing negation can make opposite claims look identical.
        """
        normalized = self.normalize_text(text)

        negation_words = {
            "not",
            "no",
            "never",
            "none",
            "neither",
            "nor",
            "without",
            "doesnt",
            "dont",
            "didnt",
            "isnt",
            "arent",
            "wasnt",
            "werent",
        }

        return " ".join(
            token
            for token in normalized.split()
            if token not in negation_words
        )

    # =========================================================
    # SEARCH QUERY
    # =========================================================

    def build_search_query(self, question, answer):
        """
        Question-first retrieval.

        The question defines what is being checked.
        Answer terms are added only to disambiguate the claim.

        Negation is retained in the answer signal.
        """

        question_terms = self.important_terms(
            question,
            max_terms=8,
        )

        answer_terms = self.important_terms(
            answer,
            max_terms=6,
        )

        selected = list(question_terms[:6])

        for term in answer_terms:
            if term not in selected:
                selected.append(term)

            if len(selected) >= 10:
                break

        # Explicitly retain a negation marker when present.
        if self.contains_negation(answer):
            if "not" not in selected:
                selected.append("not")

        if selected:
            return " ".join(selected[:10])

        return self.normalize_text(question)

    def build_search_queries(self, question, answer):
        """
        Compatibility helper.

        V6.1 still uses one search query by default to control
        Wikipedia API usage and avoid rate-limit problems.
        """

        query = self.build_search_query(
            question,
            answer,
        )

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

        if self._cache_fresh(
            memory_entry,
            self.cache_ttl_seconds,
        ):
            self.cache_hits += 1
            return memory_entry.get("results", [])

        disk_entry = self.disk_cache.get(
            "search",
            {},
        ).get(cache_key)

        if self._cache_fresh(
            disk_entry,
            self.cache_ttl_seconds,
        ):
            results = disk_entry.get("results", [])

            self.search_cache[cache_key] = disk_entry

            self.cache_hits += 1

            return results

        if not self._request_budget_available():
            print(
                "[Verifier] Search skipped: "
                "request budget exhausted."
            )
            return []

        params = {
            "action": "query",
            "list": "search",
            "format": "json",
            "utf8": 1,
            "srlimit": int(limit),
            "srsearch": query,
        }

        data = self._get(params)

        if not data:
            return []

        search_items = (
            data.get("query", {})
            .get("search", [])
        )

        results = []

        for item in search_items:
            title = item.get("title", "")
            snippet = item.get("snippet", "")

            snippet = re.sub(
                r"<[^>]+>",
                " ",
                snippet,
            )

            results.append(
                {
                    "title": title,
                    "snippet": snippet,
                }
            )

        entry = {
            "timestamp": time.time(),
            "results": results,
        }

        self.search_cache[cache_key] = entry

        self.disk_cache.setdefault(
            "search",
            {},
        )[cache_key] = entry

        self._save_disk_cache()

        return results

    # =========================================================
    # WIKIPEDIA PAGE RETRIEVAL
    # =========================================================

    def retrieve_wikipedia_page(self, title):
        if not title:
            return ""

        cache_key = title

        memory_entry = self.page_cache.get(cache_key)

        if self._cache_fresh(
            memory_entry,
            self.cache_ttl_seconds,
        ):
            self.cache_hits += 1
            return memory_entry.get("text", "")

        disk_entry = self.disk_cache.get(
            "pages",
            {},
        ).get(cache_key)

        if self._cache_fresh(
            disk_entry,
            self.cache_ttl_seconds,
        ):
            self.page_cache[cache_key] = disk_entry

            self.cache_hits += 1

            return disk_entry.get("text", "")

        if not self._request_budget_available():
            return ""

        params = {
            "action": "query",
            "prop": "extracts",
            "explaintext": 1,
            "redirects": 1,
            "format": "json",
            "titles": title,
        }

        data = self._get(params)

        if not data:
            return ""

        pages = (
            data.get("query", {})
            .get("pages", {})
        )

        text = ""

        for page in pages.values():
            text = page.get("extract", "") or ""

            if text:
                break

        entry = {
            "timestamp": time.time(),
            "text": text,
        }

        self.page_cache[cache_key] = entry

        self.disk_cache.setdefault(
            "pages",
            {},
        )[cache_key] = entry

        self._save_disk_cache()

        return text

    # =========================================================
    # SENTENCE PROCESSING
    # =========================================================

    def split_sentences(self, text):
        if not text:
            return []

        text = re.sub(
            r"\s+",
            " ",
            str(text),
        ).strip()

        if not text:
            return []

        sentences = re.split(
            r"(?<=[.!?])\s+",
            text,
        )

        return [
            sentence.strip()
            for sentence in sentences
            if sentence.strip()
        ]

    def sentence_relevance(
        self,
        sentence,
        question_terms,
        answer_terms,
    ):
        sentence_terms = self.tokenize(sentence)

        if not sentence_terms:
            return 0.0

        q = set(question_terms)
        a = set(answer_terms)

        q_overlap = (
            len(sentence_terms & q)
            / max(len(q), 1)
        )

        a_overlap = (
            len(sentence_terms & a)
            / max(len(a), 1)
        )

        # Question remains dominant.
        return (
            0.65 * q_overlap
            + 0.35 * a_overlap
        )

    def extract_relevant_sentences(
        self,
        question,
        answer,
        evidence_text,
        max_sentences=6,
    ):
        question_terms = self.important_terms(
            question,
            max_terms=12,
        )

        answer_terms = self.important_terms(
            answer,
            max_terms=10,
        )

        if not (
            question_terms
            or answer_terms
        ):
            return []

        scored = []

        for sentence in self.split_sentences(
            evidence_text
        ):
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

        scored.sort(
            key=lambda item: item["score"],
            reverse=True,
        )

        return [
            item["sentence"]
            for item in scored[:max_sentences]
        ]

    # =========================================================
    # SEARCH RESULT RANKING
    # =========================================================

    def _rank_search_results(
        self,
        question,
        answer,
        results,
    ):
        q_terms = set(
            self.important_terms(
                question,
                max_terms=12,
            )
        )

        a_terms = set(
            self.important_terms(
                answer,
                max_terms=10,
            )
        )

        ranked = []

        for result in results:
            title_terms = self.tokenize(
                result.get("title", "")
            )

            snippet_terms = self.tokenize(
                result.get("snippet", "")
            )

            title_score = (
                len(title_terms & q_terms)
                / max(len(q_terms), 1)
            )

            snippet_q = (
                len(snippet_terms & q_terms)
                / max(len(q_terms), 1)
            )

            snippet_a = (
                len(snippet_terms & a_terms)
                / max(len(a_terms), 1)
            )

            score = (
                0.55 * title_score
                + 0.30 * snippet_q
                + 0.15 * snippet_a
            )

            ranked.append(
                (score, result)
            )

        ranked.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        return [
            result
            for _, result in ranked
        ]

    # =========================================================
    # EVIDENCE COLLECTION
    # =========================================================

    def collect_evidence(
        self,
        question,
        answer,
    ):
        queries = self.build_search_queries(
            question,
            answer,
        )

        print(
            "[Verifier] Search queries:",
            queries,
        )

        if not queries:
            return []

        search_results = self.search_wikipedia(
            queries[0],
            limit=max(
                self.max_sources,
                5,
            ),
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
            title = result.get(
                "title",
                "",
            )

            if (
                not title
                or title in seen_titles
            ):
                continue

            seen_titles.add(title)

            if len(evidence_records) >= self.max_sources:
                break

            page_text = (
                self.retrieve_wikipedia_page(
                    title
                )
            )

            if not page_text:
                continue

            relevant_sentences = (
                self.extract_relevant_sentences(
                    question,
                    answer,
                    page_text,
                    max_sentences=6,
                )
            )

            if not relevant_sentences:
                continue

            evidence_records.append(
                {
                    "title": title,
                    "source": (
                        "Wikipedia: " + title
                    ),
                    "sentences": (
                        relevant_sentences
                    ),
                }
            )

        return evidence_records

    # =========================================================
    # CLAIM SCORING
    # =========================================================

    def _claim_score(
        self,
        claim,
        evidence_sentence,
    ):
        """
        V6.1 claim-level lexical similarity.

        IMPORTANT:
        Negation is preserved.

        Therefore:

            "smoking increases cancer risk"

        is NOT treated as equivalent to:

            "smoking does not increase cancer risk"
        """

        claim_terms = self.tokenize(
            claim
        )

        evidence_terms = self.tokenize(
            evidence_sentence
        )

        if (
            not claim_terms
            or not evidence_terms
        ):
            return 0.0

        overlap = (
            claim_terms
            & evidence_terms
        )

        if not overlap:
            return 0.0

        precision = (
            len(overlap)
            / max(len(claim_terms), 1)
        )

        recall = (
            len(overlap)
            / max(len(evidence_terms), 1)
        )

        if precision + recall == 0:
            return 0.0

        f1 = (
            2.0
            * precision
            * recall
            / (precision + recall)
        )

        claim_negated = (
            self.contains_negation(claim)
        )

        evidence_negated = (
            self.contains_negation(
                evidence_sentence
            )
        )

        if claim_negated == evidence_negated:
            negation_adjustment = 0.05
        else:
            negation_adjustment = -0.05

        score = (
            f1 + negation_adjustment
        )

        return max(
            0.0,
            min(1.0, score),
        )

    # =========================================================
    # EVIDENCE ASSESSMENT
    # =========================================================

    def assess_evidence(
        self,
        answer,
        evidence_records,
    ):
        if not evidence_records:
            return {
                "status": "INSUFFICIENT_EVIDENCE",
                "support_score": 0.0,
                "contradiction_score": 0.0,
                "supporting_sentences": [],
                "contradicting_sentences": [],
                "reason": (
                    "No relevant evidence retrieved."
                ),
            }

        answer_sentences = (
            self.split_sentences(answer)
        )

        if not answer_sentences:
            answer_sentences = [answer]

        source_support = {}
        source_contradiction = {}

        supporting_sentences = []
        contradicting_sentences = []

        for record in evidence_records:
            source_name = record.get(
                "title",
                record.get(
                    "source",
                    "unknown",
                ),
            )

            best_support = 0.0
            best_contradiction = 0.0

            best_support_sentence = None
            best_contradiction_sentence = None

            for evidence_sentence in record.get(
                "sentences",
                [],
            ):
                evidence_negated = (
                    self.contains_negation(
                        evidence_sentence
                    )
                )

                for claim in answer_sentences:
                    claim_negated = (
                        self.contains_negation(
                            claim
                        )
                    )

                    score = self._claim_score(
                        claim,
                        evidence_sentence,
                    )

                    # V6.1 is intentionally conservative.
                    if score < 0.25:
                        continue

                    if (
                        claim_negated
                        == evidence_negated
                    ):
                        if score > best_support:
                            best_support = score
                            best_support_sentence = (
                                evidence_sentence
                            )
                    else:
                        if score > best_contradiction:
                            best_contradiction = score
                            best_contradiction_sentence = (
                                evidence_sentence
                            )

            source_support[
                source_name
            ] = best_support

            source_contradiction[
                source_name
            ] = best_contradiction

            if best_support_sentence:
                supporting_sentences.append(
                    {
                        "source": source_name,
                        "score": best_support,
                        "sentence": (
                            best_support_sentence
                        ),
                    }
                )

            if best_contradiction_sentence:
                contradicting_sentences.append(
                    {
                        "source": source_name,
                        "score": (
                            best_contradiction
                        ),
                        "sentence": (
                            best_contradiction_sentence
                        ),
                    }
                )

        support_values = [
            value
            for value in source_support.values()
            if value > 0
        ]

        contradiction_values = [
            value
            for value
            in source_contradiction.values()
            if value > 0
        ]

        def consensus(values):
            if not values:
                return 0.0

            values = sorted(
                values,
                reverse=True,
            )

            # Strongest source contributes fully.
            # Additional sources contribute less.
            if len(values) == 1:
                return values[0]

            if len(values) == 2:
                return (
                    0.65 * values[0]
                    + 0.35 * values[1]
                )

            return (
                0.55 * values[0]
                + 0.30 * values[1]
                + 0.15 * values[2]
            )

        support_score = consensus(
            support_values
        )

        contradiction_score = consensus(
            contradiction_values
        )

        support_count = sum(
            1
            for value
            in support_values
            if value
            >= self.min_supported_score
        )

        contradiction_count = sum(
            1
            for value
            in contradiction_values
            if value
            >= self.min_contradicted_score
        )

        supporting_sentences.sort(
            key=lambda x: x["score"],
            reverse=True,
        )

        contradicting_sentences.sort(
            key=lambda x: x["score"],
            reverse=True,
        )

        # -----------------------------------------------------
        # STRONG SUPPORT
        # -----------------------------------------------------

        if (
            support_score
            >= self.min_supported_score
            and support_score
            >= (
                contradiction_score
                + self.decision_margin
            )
            and (
                support_count
                >= self.min_source_consensus
                or len(evidence_records) == 1
            )
        ):
            return {
                "status": "SUPPORTED",
                "support_score": support_score,
                "contradiction_score": (
                    contradiction_score
                ),
                "supporting_sentences": (
                    supporting_sentences
                ),
                "contradicting_sentences": (
                    contradicting_sentences
                ),
                "reason": (
                    "Claim has sufficiently strong "
                    "supporting evidence."
                ),
            }

        # -----------------------------------------------------
        # STRONG CONTRADICTION
        # -----------------------------------------------------

        if (
            contradiction_score
            >= self.min_contradicted_score
            and contradiction_score
            >= (
                support_score
                + self.decision_margin
            )
            and (
                contradiction_count
                >= self.min_source_consensus
                or len(evidence_records) == 1
            )
        ):
            return {
                "status": "CONTRADICTED",
                "support_score": support_score,
                "contradiction_score": (
                    contradiction_score
                ),
                "supporting_sentences": (
                    supporting_sentences
                ),
                "contradicting_sentences": (
                    contradicting_sentences
                ),
                "reason": (
                    "Claim has sufficiently strong "
                    "contradicting evidence."
                ),
            }

        # -----------------------------------------------------
        # SOME EVIDENCE BUT NOT DECISIVE
        # -----------------------------------------------------

        if (
            support_values
            or contradiction_values
        ):
            return {
                "status": "EVIDENCE_FOUND",
                "support_score": support_score,
                "contradiction_score": (
                    contradiction_score
                ),
                "supporting_sentences": (
                    supporting_sentences
                ),
                "contradicting_sentences": (
                    contradicting_sentences
                ),
                "reason": (
                    "Relevant evidence was retrieved, "
                    "but it does not meet the threshold "
                    "for a strong decision."
                ),
            }

        return {
            "status": "INSUFFICIENT_EVIDENCE",
            "support_score": 0.0,
            "contradiction_score": 0.0,
            "supporting_sentences": [],
            "contradicting_sentences": [],
            "reason": (
                "Retrieved sources did not provide "
                "sufficient claim-level evidence."
            ),
        }

    # =========================================================
    # PUBLIC VERIFICATION API
    # =========================================================

    def verify(
        self,
        question,
        answer,
    ):
        # Reset per-verification request budget.
        self.verify_request_count = 0

        evidence_records = (
            self.collect_evidence(
                question,
                answer,
            )
        )

        assessment = self.assess_evidence(
            answer,
            evidence_records,
        )

        evidence_sentence_count = sum(
            len(
                record.get(
                    "sentences",
                    [],
                )
            )
            for record in evidence_records
        )

        return {
            "question": question,
            "answer": answer,
            "verification_status": (
                assessment["status"]
            ),
            "source_count": len(
                evidence_records
            ),
            "evidence_sentence_count": (
                evidence_sentence_count
            ),
            "support_score": (
                assessment["support_score"]
            ),
            "contradiction_score": (
                assessment["contradiction_score"]
            ),
            "reason": assessment["reason"],
            "supporting_sentences": (
                assessment.get(
                    "supporting_sentences",
                    [],
                )
            ),
            "contradicting_sentences": (
                assessment.get(
                    "contradicting_sentences",
                    [],
                )
            ),
            "sources": evidence_records,
            "verifier_version": "V6.1",
            "request_count": (
                self.verify_request_count
            ),
            "cache_hits": self.cache_hits,
            "rate_limit_events": (
                self.rate_limit_events
            ),
        }


# =============================================================
# SELF TEST
# =============================================================

if __name__ == "__main__":

    print("=" * 65)
    print("TRUSTGATE EXTERNAL VERIFIER V6.1 SELF-TEST")
    print("=" * 65)

    verifier = ExternalVerifier(
        max_sources=3,
        timeout=10,
        request_delay=2.0,
        max_retries=1,
        max_requests_per_verify=4,
    )

    # ---------------------------------------------------------
    # Local deterministic tests first.
    # ---------------------------------------------------------

    tests = [
        (
            "Smoking increases the risk of lung cancer.",
            "Smoking increases the risk of lung cancer.",
            "same polarity",
        ),
        (
            "Smoking does not increase the risk of lung cancer.",
            "Smoking increases the risk of lung cancer.",
            "opposite polarity",
        ),
        (
            "Smoking increases the risk of lung cancer.",
            "Smoking does not increase the risk of lung cancer.",
            "opposite polarity",
        ),
    ]

    print("\nNEGATION TESTS")

    for claim, evidence, description in tests:
        score = verifier._claim_score(
            claim,
            evidence,
        )

        claim_neg = verifier.contains_negation(
            claim
        )

        evidence_neg = verifier.contains_negation(
            evidence
        )

        print(
            f"{description:20s} | "
            f"score={score:.3f} | "
            f"claim_neg={claim_neg} | "
            f"evidence_neg={evidence_neg}"
        )

    print("\nLOCAL TEST COMPLETE")

    # ---------------------------------------------------------
    # Small live Wikipedia test.
    # ---------------------------------------------------------

    print("\nLIVE WIKIPEDIA TEST")

    question = "What is the capital of France?"
    answer = "Paris is the capital of France."

    result = verifier.verify(
        question,
        answer,
    )

    print(
        "Verification status:",
        result["verification_status"],
    )

    print(
        "Sources:",
        result["source_count"],
    )

    print(
        "Evidence sentences:",
        result["evidence_sentence_count"],
    )

    print(
        "Support score:",
        f'{result["support_score"]:.4f}',
    )

    print(
        "Contradiction score:",
        f'{result["contradiction_score"]:.4f}',
    )

    print(
        "Request count:",
        result["request_count"],
    )

    print(
        "Cache hits:",
        result["cache_hits"],
    )

    print(
        "Verifier version:",
        result["verifier_version"],
    )

    print("\nEND OF V6.1 SELF-TEST")