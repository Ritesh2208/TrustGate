from external_verifier import ExternalVerifier


TEST_CASES = [
    {
        "question": "What is the capital of France?",
        "answer": "The capital of France is Paris."
    },
    {
        "question": "Is the Earth flat?",
        "answer": "No, the Earth is not flat; it is approximately spherical."
    },
    {
        "question": "Can humans live without sleep?",
        "answer": "Humans can live normally without sleep."
    },
    {
        "question": "Does drinking bleach cure infections?",
        "answer": "Drinking bleach cures infections."
    },
    {
        "question": "Do vaccines cause autism?",
        "answer": "Vaccines cause autism."
    },
    {
        "question": "Is water made of hydrogen and oxygen?",
        "answer": "Water is composed of hydrogen and oxygen."
    },
    {
        "question": "Does the Sun orbit the Earth?",
        "answer": "The Sun orbits the Earth."
    },
    {
        "question": "Is Paris the capital of France?",
        "answer": "Paris is the capital of France."
    },
    {
        "question": "Can humans breathe underwater without equipment?",
        "answer": "Humans can breathe underwater without equipment."
    },
    {
        "question": "Does smoking increase the risk of lung cancer?",
        "answer": "Smoking increases the risk of lung cancer."
    },
]


def main():
    verifier = ExternalVerifier(
        max_sources=5,
        timeout=10
    )

    print()
    print("=" * 70)
    print("EXTERNAL VERIFIER V2 DIAGNOSTIC")
    print("=" * 70)

    results = []

    for i, case in enumerate(TEST_CASES, 1):

        print()
        print("=" * 70)
        print(f"[{i}/10]")
        print("=" * 70)

        print("Question:")
        print(case["question"])

        print()
        print("Answer:")
        print(case["answer"])

        try:
            result = verifier.verify(
                question=case["question"],
                answer=case["answer"]
            )

            # -----------------------------------------------------
            # Use the actual field names returned by ExternalVerifier
            # -----------------------------------------------------

            status = result.get(
                "verification_status",
                "UNKNOWN"
            )

            support = result.get(
                "support_score",
                0.0
            )

            contradiction = result.get(
                "contradiction_score",
                0.0
            )

            source_count = result.get(
                "source_count",
                0
            )

            evidence_count = result.get(
                "evidence_sentence_count",
                0
            )

            reason = result.get(
                "reason",
                ""
            )

            print()
            print("Verification status:", status)
            print(
                "Support:",
                f"{float(support):.4f}"
            )
            print(
                "Contradiction:",
                f"{float(contradiction):.4f}"
            )
            print(
                "Sources:",
                source_count
            )
            print(
                "Evidence sentences:",
                evidence_count
            )

            if reason:
                print(
                    "Reason:",
                    reason
                )

            # -----------------------------------------------------
            # Store diagnostic result
            # -----------------------------------------------------

            results.append({
                "question": case["question"],
                "status": status,
                "support": float(support),
                "contradiction": float(contradiction),
                "sources": int(source_count),
                "evidence": int(evidence_count),
                "reason": reason
            })

        except Exception as exc:

            print()
            print(
                "[ERROR]",
                repr(exc)
            )

            results.append({
                "question": case["question"],
                "status": "ERROR",
                "support": 0.0,
                "contradiction": 0.0,
                "sources": 0,
                "evidence": 0,
                "reason": repr(exc)
            })

    # =============================================================
    # SUMMARY
    # =============================================================

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for result in results:

        print(
            f"{str(result['status']):24s} | "
            f"S={result['support']:.3f} | "
            f"C={result['contradiction']:.3f} | "
            f"Sources={result['sources']} | "
            f"Evidence={result['evidence']} | "
            f"{result['question']}"
        )

    # =============================================================
    # STATUS COUNTS
    # =============================================================

    print()
    print("=" * 70)
    print("STATUS COUNTS")
    print("=" * 70)

    status_counts = {}

    for result in results:

        status = result["status"]

        status_counts[status] = (
            status_counts.get(status, 0) + 1
        )

    for status, count in status_counts.items():

        print(
            f"{status:24s}: {count}"
        )

    # =============================================================
    # DIAGNOSTIC INTERPRETATION
    # =============================================================

    total = len(results)

    valid_statuses = {
        "SUPPORTED",
        "CONTRADICTED",
        "EVIDENCE_FOUND",
        "INSUFFICIENT_EVIDENCE"
    }

    valid_count = sum(
        1
        for result in results
        if result["status"] in valid_statuses
    )

    error_count = sum(
        1
        for result in results
        if result["status"] == "ERROR"
    )

    print()
    print("=" * 70)
    print("DIAGNOSTIC CHECK")
    print("=" * 70)

    print(
        f"Cases processed: {total}"
    )

    print(
        f"Valid verifier statuses: {valid_count}"
    )

    print(
        f"Errors: {error_count}"
    )

    if valid_count == total:

        print()
        print(
            "[PASS] All cases returned a valid verifier status."
        )

    elif valid_count > 0:

        print()
        print(
            "[WARNING] Some cases returned unexpected results."
        )

    else:

        print()
        print(
            "[FAIL] No valid verifier statuses were returned."
        )

    print()
    print("=" * 70)
    print("END OF EXTERNAL VERIFIER V2 DIAGNOSTIC")
    print("=" * 70)


if __name__ == "__main__":
    main()