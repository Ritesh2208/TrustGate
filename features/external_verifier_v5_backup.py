import re

import time

import requests



from collections import Counter





# =============================================================

# EXTERNAL VERIFIER V5

# =============================================================





class ExternalVerifier:



    def __init__(

        self,

        max_sources=3,

        timeout=10,

        request_delay=1.5,

        max_retries=3

    ):



        self.max_sources = max_sources

        self.timeout = timeout

        self.request_delay = request_delay

        self.max_retries = max_retries



        self.search_url = (

            "https://en.wikipedia.org/w/api.php"

        )



        self.session = requests.Session()



        self.session.headers.update({

            "User-Agent":

                "TrustGate-Research/3.0 "

                "(academic research project)"

        })



        # Cache retrieved Wikipedia pages during the run.

        self.page_cache = {}



        # Time of the previous API request.

        self.last_request_time = 0.0



    # =========================================================

    # REQUEST CONTROL

    # =========================================================



    def _wait_before_request(self):



        elapsed = (

            time.time()

            - self.last_request_time

        )



        if elapsed < self.request_delay:



            time.sleep(

                self.request_delay - elapsed

            )



    def _get(

        self,

        params

    ):



        for attempt in range(

            self.max_retries + 1

        ):



            self._wait_before_request()



            try:



                response = self.session.get(

                    self.search_url,

                    params=params,

                    timeout=self.timeout

                )



                self.last_request_time = time.time()



                # -------------------------------------------------

                # Rate limited

                # -------------------------------------------------



                if response.status_code == 429:



                    if attempt >= self.max_retries:



                        print(

                            "[Verifier] "

                            "Wikipedia rate limit persisted."

                        )



                        return None



                    retry_after = (

                        response.headers.get(

                            "Retry-After"

                        )

                    )



                    if retry_after:



                        try:

                            wait_time = float(

                                retry_after

                            )

                        except ValueError:

                            wait_time = (

                                3.0

                                * (attempt + 1)

                            )



                    else:



                        wait_time = (

                            3.0

                            * (attempt + 1)

                        )



                    print(

                        "[Verifier] "

                        f"HTTP 429. "

                        f"Retrying in "

                        f"{wait_time:.1f}s..."

                    )



                    time.sleep(

                        wait_time

                    )



                    continue



                response.raise_for_status()



                return response



            except requests.RequestException as exc:



                self.last_request_time = time.time()



                if attempt >= self.max_retries:



                    print(

                        "[Verifier] "

                        "Request failed:",

                        str(exc)

                    )



                    return None



                wait_time = (

                    2.0

                    * (attempt + 1)

                )



                print(

                    "[Verifier] "

                    f"Request error. "

                    f"Retrying in "

                    f"{wait_time:.1f}s..."

                )



                time.sleep(

                    wait_time

                )



        return None



    # =========================================================

    # TEXT NORMALIZATION

    # =========================================================



    @staticmethod

    def normalize_text(text):



        if text is None:

            return ""



        text = str(text).lower()



        text = re.sub(

            r"\s+",

            " ",

            text

        )



        return text.strip()



    # =========================================================

    # TOKENIZE

    # =========================================================



    @staticmethod

    def tokenize(text):



        text = (

            ExternalVerifier

            .normalize_text(text)

        )



        return set(

            re.findall(

                r"\b[a-z0-9]{3,}\b",

                text

            )

        )



    # =========================================================

    # STOPWORDS

    # =========================================================



    @staticmethod

    def stopwords():



        return {

            "the",

            "and",

            "for",

            "are",

            "was",

            "were",

            "with",

            "that",

            "this",

            "from",

            "have",

            "has",

            "had",

            "not",

            "but",

            "can",

            "could",

            "would",

            "should",

            "does",

            "did",

            "how",

            "what",

            "when",

            "where",

            "which",

            "who",

            "why",

            "much",

            "many",

            "about",

            "into",

            "their",

            "there",

            "they",

            "them",

            "then",

            "than",

            "your",

            "you",

            "its",

            "it's",

            "been",

            "being",

            "also",

            "some",

            "such",

            "more",

            "most",

            "very",

            "only",

            "people",

            "person",

            "thing",

            "things"

        }



    # =========================================================

    # IMPORTANT TERMS

    # =========================================================



    def important_terms(

        self,

        text,

        max_terms=12

    ):



        tokens = re.findall(

            r"\b[a-zA-Z][a-zA-Z0-9'-]*\b",

            self.normalize_text(text)

        )



        stopwords = self.stopwords()



        terms = [

            token

            for token in tokens

            if (

                token not in stopwords

                and len(token) >= 4

            )

        ]



        counts = Counter(terms)



        return [

            word

            for word, _ in counts.most_common(

                max_terms

            )

        ]



    # =========================================================
    # BUILD SEARCH QUERIES (V4)
    # =========================================================

    def build_search_queries(
        self,
        question,
        answer
    ):

        question_terms = self.important_terms(
            question,
            max_terms=10
        )

        answer_terms = self.important_terms(
            answer,
            max_terms=8
        )

        queries = []

        # Query 1: preserve the V3-style factual query.
        combined = list(dict.fromkeys(
            question_terms[:6] + answer_terms[:5]
        ))
        if combined:
            queries.append(" ".join(combined[:8]))

        # Query 2: question-focused retrieval without answer anchoring.
        if question_terms:
            q2 = " ".join(question_terms[:8])
            if q2 not in queries:
                queries.append(q2)

        # Query 3: normalized natural-language question.
        natural_question = self.normalize_text(question)
        if natural_question and natural_question not in queries:
            queries.append(natural_question)

        return queries[:3]

    # Backward-compatible single-query API.
    def build_search_query(
        self,
        question,
        answer
    ):

        queries = self.build_search_queries(
            question,
            answer
        )

        if queries:
            return queries[0]

        return self.normalize_text(question)

    # =========================================================
    # SOURCE RELEVANCE SCORING (V4)
    # =========================================================

    def _overlap_score(
        self,
        text,
        reference_terms
    ):

        if not text or not reference_terms:
            return 0.0

        text_terms = set(
            self.important_terms(
                text,
                max_terms=80
            )
        )

        if not text_terms:
            return 0.0

        reference_terms = set(reference_terms)
        return len(text_terms & reference_terms) / len(reference_terms)

    def _score_search_result(
        self,
        result,
        question,
        answer
    ):

        title = result.get("title", "")
        snippet = result.get("snippet", "")

        question_terms = self.important_terms(
            question,
            max_terms=12
        )
        answer_terms = self.important_terms(
            answer,
            max_terms=10
        )

        title_question = self._overlap_score(
            title,
            question_terms
        )
        snippet_question = self._overlap_score(
            snippet,
            question_terms
        )
        snippet_answer = self._overlap_score(
            snippet,
            answer_terms
        )

        # The question is the primary retrieval target.
        return (
            0.65 * title_question
            + 0.25 * snippet_question
            + 0.10 * snippet_answer
        )

    def _score_page_relevance(
        self,
        question,
        answer,
        title,
        page_text
    ):

        question_terms = self.important_terms(
            question,
            max_terms=12
        )
        answer_terms = self.important_terms(
            answer,
            max_terms=10
        )

        title_question = self._overlap_score(
            title,
            question_terms
        )
        page_question = self._overlap_score(
            page_text,
            question_terms
        )
        page_answer = self._overlap_score(
            page_text,
            answer_terms
        )

        return (
            0.50 * title_question
            + 0.35 * page_question
            + 0.15 * page_answer
        )

    # =========================================================

    # SEARCH WIKIPEDIA

    # =========================================================



    def search_wikipedia(

        self,

        query,

        limit=3

    ):



        params = {

            "action": "query",

            "list": "search",

            "format": "json",

            "utf8": 1,

            "srlimit": limit,

            "srsearch": query

        }



        response = self._get(

            params

        )



        if response is None:

            return []



        try:



            data = response.json()



            return (

                data

                .get("query", {})

                .get("search", [])

            )



        except Exception as exc:



            print(

                "[Verifier] "

                "Search parsing error:",

                str(exc)

            )



            return []



    # =========================================================

    # RETRIEVE WIKIPEDIA PAGE

    # =========================================================



    def retrieve_wikipedia_page(

        self,

        title

    ):



        # -----------------------------------------------------

        # Cache

        # -----------------------------------------------------



        if title in self.page_cache:



            return self.page_cache[

                title

            ]



        params = {

            "action": "query",

            "prop": "extracts",

            "explaintext": 1,

            "format": "json",

            "utf8": 1,

            "titles": title

        }



        response = self._get(

            params

        )



        if response is None:



            return ""



        try:



            data = response.json()



            pages = (

                data

                .get("query", {})

                .get("pages", {})

            )



            for page in pages.values():



                extract = page.get(

                    "extract",

                    ""

                )



                if extract:



                    self.page_cache[

                        title

                    ] = extract



                    return extract



        except Exception as exc:



            print(

                "[Verifier] "

                "Page parsing error:",

                str(exc)

            )



        return ""



    # =========================================================

    # SPLIT SENTENCES

    # =========================================================



    @staticmethod

    def split_sentences(

        text

    ):



        if not text:

            return []



        sentences = re.split(

            r"(?<=[.!?])\s+",

            text

        )



        return [

            sentence.strip()

            for sentence in sentences

            if len(sentence.strip()) >= 25

        ]



    # =========================================================

    # SENTENCE RELEVANCE

    # =========================================================



    def sentence_relevance(

        self,

        sentence,

        query_terms,

        answer_terms

    ):



        sentence_words = (

            self.tokenize(sentence)

        )



        if not sentence_words:

            return 0.0



        question_overlap = (

            sentence_words

            .intersection(

                query_terms

            )

        )



        answer_overlap = (

            sentence_words

            .intersection(

                answer_terms

            )

        )



        question_score = (

            len(question_overlap)

            /

            max(

                len(query_terms),

                1

            )

        )



        answer_score = (

            len(answer_overlap)

            /

            max(

                len(answer_terms),

                1

            )

        )



        score = (

            0.40 * question_score

            +

            0.60 * answer_score

        )



        return score



    # =========================================================

    # EXTRACT RELEVANT SENTENCES

    # =========================================================



    def extract_relevant_sentences(

        self,

        question,

        answer,

        evidence_text,

        max_sentences=6

    ):



        question_terms = set(

            self.important_terms(

                question,

                max_terms=10

            )

        )



        answer_terms = set(

            self.important_terms(

                answer,

                max_terms=15

            )

        )



        if not (

            question_terms

            or answer_terms

        ):



            return []



        sentences = (

            self.split_sentences(

                evidence_text

            )

        )



        scored = []



        for sentence in sentences:



            score = (

                self.sentence_relevance(

                    sentence,

                    question_terms,

                    answer_terms

                )

            )



            if score >= 0.12:



                scored.append({

                    "score": score,

                    "sentence": sentence

                })



        scored.sort(

            key=lambda item:

                item["score"],

            reverse=True

        )



        return [

            item["sentence"]

            for item in scored[

                :max_sentences

            ]

        ]



    # =========================================================
    # COLLECT EVIDENCE (V4)
    # =========================================================

    def collect_evidence(
        self,
        question,
        answer
    ):

        queries = self.build_search_queries(
            question,
            answer
        )

        print(
            "[Verifier] Search queries:",
            queries
        )

        if not queries:
            return []

        # Search using several formulations, then deduplicate titles.
        # This preserves the V3 Wikipedia backend while reducing
        # dependence on one potentially poor query formulation.
        candidate_map = {}

        for query in queries:
            search_results = self.search_wikipedia(
                query,
                limit=max(self.max_sources * 2, 6)
            )

            for result in search_results:
                title = result.get("title", "")
                if not title:
                    continue

                score = self._score_search_result(
                    result,
                    question,
                    answer
                )

                if (
                    title not in candidate_map
                    or score > candidate_map[title]["score"]
                ):
                    candidate_map[title] = {
                        "result": result,
                        "score": score
                    }

        if not candidate_map:
            return []

        ranked_candidates = sorted(
            candidate_map.values(),
            key=lambda item: item["score"],
            reverse=True
        )

        # Retrieve a few more candidates than the final source count so
        # page-level relevance can remove irrelevant search hits.
        candidate_limit = max(
            self.max_sources * 2,
            6
        )

        evidence_records = []

        for candidate in ranked_candidates[:candidate_limit]:
            title = candidate["result"].get(
                "title",
                ""
            )

            page_text = self.retrieve_wikipedia_page(
                title
            )

            if not page_text:
                continue

            page_score = self._score_page_relevance(
                question,
                answer,
                title,
                page_text
            )

            relevant_sentences = self.extract_relevant_sentences(
                question,
                answer,
                page_text,
                max_sentences=6
            )

            if not relevant_sentences:
                continue

            evidence_records.append({
                "title": title,
                "page_id": None,
                "source": "Wikipedia",
                "url": (
                    "https://en.wikipedia.org/wiki/"
                    + title.replace(" ", "_")
                ),
                "sentences": relevant_sentences,
                "_relevance_score": page_score
            })

        # Keep the public evidence set compatible with V3: at most
        # max_sources records, ranked by page-level relevance.
        evidence_records.sort(
            key=lambda item: item["_relevance_score"],
            reverse=True
        )

        evidence_records = evidence_records[:self.max_sources]

        for record in evidence_records:
            record.pop("_relevance_score", None)

        return evidence_records

    # =========================================================

    # NEGATION DETECTION

    # =========================================================



    @staticmethod

    def contains_negation(

        text

    ):



        text = (

            ExternalVerifier

            .normalize_text(text)

        )



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

            r"\bisn't\b",

            r"\baren't\b",

            r"\bwasn't\b",

            r"\bweren't\b",

            r"\bdoes not\b",

            r"\bdo not\b",

            r"\bis not\b",

            r"\bare not\b"

        ]



        for pattern in patterns:



            if re.search(

                pattern,

                text

            ):



                return True



        return False



    # =========================================================

    # REMOVE NEGATION

    # =========================================================



    @staticmethod

    def remove_negation(

        text

    ):



        text = (

            ExternalVerifier

            .normalize_text(text)

        )



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

            r"\bisn't\b",

            r"\baren't\b",

            r"\bwasn't\b",

            r"\bweren't\b",

            r"\bdoes not\b",

            r"\bdo not\b",

            r"\bis not\b",

            r"\bare not\b"

        ]



        for pattern in patterns:



            text = re.sub(

                pattern,

                "",

                text

            )



        return re.sub(

            r"\s+",

            " ",

            text

        ).strip()



    # =========================================================

    # SENTENCE SUPPORT SCORE

    # =========================================================



    def sentence_support_score(

        self,

        answer,

        evidence_sentence

    ):

        # V5: use F1-style lexical alignment rather than answer-token

        # recall alone. This reduces false high scores from long evidence

        # sentences that merely share a few common terms with the answer.

        answer_words = self.tokenize(answer)

        evidence_words = self.tokenize(evidence_sentence)

        if not answer_words or not evidence_words:

            return 0.0

        overlap = answer_words & evidence_words

        precision = len(overlap) / max(len(evidence_words), 1)

        recall = len(overlap) / max(len(answer_words), 1)

        if precision + recall == 0.0:

            return 0.0

        f1 = (2.0 * precision * recall) / (precision + recall)

        # Small boost for an exact answer phrase.

        answer_normalized = self.normalize_text(answer)

        evidence_normalized = self.normalize_text(evidence_sentence)

        phrase_boost = (

            0.10

            if answer_normalized

            and answer_normalized in evidence_normalized

            else 0.0

        )

        return min(1.0, f1 + phrase_boost)



    # =========================================================

    # CONSERVATIVE EVIDENCE ASSESSMENT

    # =========================================================



    def assess_evidence(

        self,

        answer,

        evidence_records

    ):



        if not evidence_records:



            return {

                "status":

                    "INSUFFICIENT_EVIDENCE",



                "support_score":

                    0.0,



                "contradiction_score":

                    0.0,



                "supporting_sentences":

                    [],



                "contradicting_sentences":

                    [],



                "reason":

                    "No relevant evidence retrieved."

            }



        answer_normalized = (

            self.normalize_text(

                answer

            )

        )



        answer_negated = (

            self.contains_negation(

                answer_normalized

            )

        )



        answer_without_negation = (

            self.remove_negation(

                answer_normalized

            )

        )



        support_scores = []

        contradiction_scores = []



        supporting_sentences = []

        contradicting_sentences = []



        for record in evidence_records:



            for sentence in record.get(

                "sentences",

                []

            ):



                score = (

                    self.sentence_support_score(

                        answer_without_negation,

                        sentence

                    )

                )



                if score < 0.20:

                    continue



                evidence_negated = (

                    self.contains_negation(

                        sentence

                    )

                )



                # -------------------------------------------------

                # Same negation direction

                # -------------------------------------------------



                if (

                    answer_negated

                    ==

                    evidence_negated

                ):



                    support_scores.append(

                        score

                    )



                    supporting_sentences.append({

                        "score": score,

                        "sentence": sentence,

                        "source": record.get("title", "")

                    })



                # -------------------------------------------------

                # Opposite negation direction

                # -------------------------------------------------



                else:



                    contradiction_scores.append(

                        score

                    )



                    contradicting_sentences.append({

                        "score": score,

                        "sentence": sentence,

                        "source": record.get("title", "")

                    })



        # V5: source-aware consensus. Only the strongest sentence from each

        # source contributes, preventing one verbose page from dominating.

        def source_consensus(records):

            by_source = {}

            for item in records:

                source = item.get("source", "") or "unknown"

                by_source.setdefault(source, []).append(item["score"])



            source_best = [

                max(scores)

                for scores in by_source.values()

                if scores

            ]



            if not source_best:

                return 0.0



            source_best.sort(reverse=True)

            top = source_best[:3]

            return sum(top) / len(top)



        max_support = source_consensus(

            supporting_sentences

        )



        max_contradiction = source_consensus(

            contradicting_sentences

        )



        supporting_sentences = [

            item["sentence"]

            for item in sorted(

                supporting_sentences,

                key=lambda x: x["score"],

                reverse=True

            )[:6]

        ]



        contradicting_sentences = [

            item["sentence"]

            for item in sorted(

                contradicting_sentences,

                key=lambda x: x["score"],

                reverse=True

            )[:6]

        ]



        # -----------------------------------------------------

        # Conservative decision

        # -----------------------------------------------------



        threshold = 0.60

        margin = 0.10



        # CONTRADICTED



        if (

            max_contradiction >= threshold

            and

            max_contradiction

            >

            max_support + margin

        ):



            return {

                "status":

                    "CONTRADICTED",



                "support_score":

                    max_support,



                "contradiction_score":

                    max_contradiction,



                "supporting_sentences":

                    supporting_sentences,



                "contradicting_sentences":

                    contradicting_sentences,



                "reason":

                    "Evidence has stronger opposing lexical alignment."

            }



        # SUPPORTED



        if (

            max_support >= threshold

            and

            max_support

            >

            max_contradiction + margin

        ):



            return {

                "status":

                    "SUPPORTED",



                "support_score":

                    max_support,



                "contradiction_score":

                    max_contradiction,



                "supporting_sentences":

                    supporting_sentences,



                "contradicting_sentences":

                    contradicting_sentences,



                "reason":

                    "Evidence has stronger supporting lexical alignment."

            }



        # EVIDENCE FOUND BUT AMBIGUOUS



        if (

            max_support > 0.0

            or

            max_contradiction > 0.0

        ):



            return {

                "status":

                    "EVIDENCE_FOUND",



                "support_score":

                    max_support,



                "contradiction_score":

                    max_contradiction,



                "supporting_sentences":

                    supporting_sentences,



                "contradicting_sentences":

                    contradicting_sentences,



                "reason":

                    "Evidence was retrieved, but it is not strong enough for a definitive verdict."

            }



        # NO USEFUL EVIDENCE



        return {

            "status":

                "INSUFFICIENT_EVIDENCE",



            "support_score":

                0.0,



            "contradiction_score":

                0.0,



            "supporting_sentences":

                [],



            "contradicting_sentences":

                [],



            "reason":

                "Retrieved pages did not provide sufficiently relevant evidence."

        }



    # =========================================================

    # VERIFY

    # =========================================================



    def verify(

        self,

        question,

        answer

    ):



        evidence_records = (

            self.collect_evidence(

                question,

                answer

            )

        )



        assessment = (

            self.assess_evidence(

                answer,

                evidence_records

            )

        )



        evidence_sentence_count = sum(

            len(

                record.get(

                    "sentences",

                    []

                )

            )

            for record in evidence_records

        )



        return {



            "question":

                question,



            "answer":

                answer,



            "verification_status":

                assessment["status"],



            "source_count":

                len(evidence_records),



            "evidence_sentence_count":

                evidence_sentence_count,



            "support_score":

                assessment[

                    "support_score"

                ],



            "contradiction_score":

                assessment[

                    "contradiction_score"

                ],



            "reason":

                assessment[

                    "reason"

                ],



            "supporting_sentences":

                assessment.get(

                    "supporting_sentences",

                    []

                ),



            "contradicting_sentences":

                assessment.get(

                    "contradicting_sentences",

                    []

                ),



            "sources":

                evidence_records

        }





# =============================================================

# SELF TEST

# =============================================================



if __name__ == "__main__":



    print()

    print("=" * 70)

    print("EXTERNAL VERIFIER V5 SELF-TEST")

    print("=" * 70)



    verifier = ExternalVerifier(

        max_sources=3,

        timeout=10,

        request_delay=1.5,

        max_retries=3

    )



    # ---------------------------------------------------------

    # TEST 1

    # ---------------------------------------------------------



    print()

    print("[TEST 1] Query generation")



    query = (

        verifier.build_search_query(

            "What is the capital of France?",

            "The capital of France is Paris."

        )

    )



    print(

        "Generated query:",

        query

    )



    assert isinstance(

        query,

        str

    )



    assert len(

        query.strip()

    ) > 0



    print(

        "[PASS] Query generation works."

    )



    # ---------------------------------------------------------

    # TEST 2

    # ---------------------------------------------------------



    print()

    print("[TEST 2] Wikipedia search")



    results = (

        verifier.search_wikipedia(

            "capital France Paris",

            limit=3

        )

    )



    print(

        "Search results:",

        len(results)

    )



    for result in results:



        print(

            " -",

            result.get(

                "title"

            )

        )



    assert isinstance(

        results,

        list

    )



    print(

        "[PASS] Search returned a list."

    )



    # ---------------------------------------------------------

    # TEST 3

    # ---------------------------------------------------------



    print()

    print("[TEST 3] Full verification")



    result = verifier.verify(

        "What is the capital of France?",

        "The capital of France is Paris."

    )



    print(

        "Verification status:",

        result[

            "verification_status"

        ]

    )



    print(

        "Sources:",

        result[

            "source_count"

        ]

    )



    print(

        "Evidence sentences:",

        result[

            "evidence_sentence_count"

        ]

    )



    print(

        "Support score:",

        f"{result['support_score']:.4f}"

    )



    print(

        "Contradiction score:",

        f"{result['contradiction_score']:.4f}"

    )



    assert result[

        "verification_status"

    ] in [

        "SUPPORTED",

        "CONTRADICTED",

        "EVIDENCE_FOUND",

        "INSUFFICIENT_EVIDENCE"

    ]



    print(

        "[PASS] Verification structure is valid."

    )



    # ---------------------------------------------------------

    # TEST 4

    # ---------------------------------------------------------



    print()

    print("[TEST 4] Negation detection")



    assert verifier.contains_negation(

        "The Earth is not flat."

    )



    assert not verifier.contains_negation(

        "Paris is the capital of France."

    )



    print(

        "[PASS] Negation detection works."

    )



    # ---------------------------------------------------------

    # TEST 5

    # ---------------------------------------------------------



    print()

    print("[TEST 5] Controlled support")



    controlled_support = [



        {

            "title":

                "Controlled Source",



            "source":

                "Test",



            "url":

                "",



            "sentences": [



                "Paris is the capital "

                "and largest city of France."

            ]

        }

    ]



    support_result = (

        verifier.assess_evidence(

            "Paris is the capital of France.",

            controlled_support

        )

    )



    print(

        "Controlled status:",

        support_result[

            "status"

        ]

    )



    print(

        "Support score:",

        support_result[

            "support_score"

        ]

    )



    assert support_result[

        "status"

    ] in [

        "SUPPORTED",

        "EVIDENCE_FOUND"

    ]



    print(

        "[PASS] Controlled support test passed."

    )



    # ---------------------------------------------------------

    # TEST 6

    # ---------------------------------------------------------



    print()

    print("[TEST 6] Controlled contradiction")



    controlled_contradiction = [



        {

            "title":

                "Controlled Source",



            "source":

                "Test",



            "url":

                "",



            "sentences": [



                "The Earth is not flat; "

                "it is an approximately "

                "spherical body."

            ]

        }

    ]



    contradiction_result = (

        verifier.assess_evidence(

            "The Earth is flat.",

            controlled_contradiction

        )

    )



    print(

        "Controlled status:",

        contradiction_result[

            "status"

        ]

    )



    print(

        "Contradiction score:",

        contradiction_result[

            "contradiction_score"

        ]

    )



    assert contradiction_result[

        "status"

    ] in [

        "CONTRADICTED",

        "EVIDENCE_FOUND"

    ]



    print(

        "[PASS] Controlled contradiction test passed."

    )



    print()

    print("=" * 70)

    print(

        "ALL EXTERNAL VERIFIER V4 TESTS PASSED"

    )

    print("=" * 70)

