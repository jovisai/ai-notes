"""Test DeepEval structure for evaluation article."""

try:
    from deepeval import evaluate
    from deepeval.test_case import LLMTestCase
    from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
    print("DeepEval imports successful!")
    print(f"LLMTestCase: {LLMTestCase}")
    print(f"AnswerRelevancyMetric: {AnswerRelevancyMetric}")
    print(f"FaithfulnessMetric: {FaithfulnessMetric}")

    # Create a test case (doesn't require API key for structure test)
    test_case = LLMTestCase(
        input="What is the capital of France?",
        actual_output="The capital of France is Paris.",
        retrieval_context=["Paris is the capital and largest city of France."]
    )

    print(f"\nTest case created: {test_case.input}")
    print("DeepEval structure validated!")

except ImportError as e:
    print(f"Import error: {e}")
