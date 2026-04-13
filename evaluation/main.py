from dataclasses import dataclass
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(dotenv_path="../.env")

MODEL = "gpt-4o-mini"
JUDGE_MODEL = "gpt-4o-mini"


@dataclass
class TestCase:
    question: str
    expected: str
    eval_type: str  # exact, contains, or judge


TEST_CASES = [
    TestCase(
        question="What is 2 + 2?",
        expected="4",
        eval_type="contains",
    ),
    TestCase(
        question="What is the capital of France?",
        expected="Paris",
        eval_type="contains",
    ),
    TestCase(
        question="In one word, what color is the sky on a clear day?",
        expected="blue",
        eval_type="exact",
    ),
    TestCase(
        question="Explain what photosynthesis is in one sentence.",
        expected="A correct explanation that mentions plants converting sunlight into energy",
        eval_type="judge",
    ),
    TestCase(
        question="What year did World War 2 end?",
        expected="1945",
        eval_type="contains",
    ),
]


def get_model_answer(client: OpenAI, question: str) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": question}],
        temperature=0,
    )
    return response.choices[0].message.content.strip()


def eval_exact(answer: str, expected: str) -> bool:
    return answer.lower().strip(".!?") == expected.lower().strip(".!?")


def eval_contains(answer: str, expected: str) -> bool:
    return expected.lower() in answer.lower()


def eval_with_judge(client: OpenAI, question: str, answer: str, expected: str) -> bool:
    """Use an LLM to decide if the answer matches the expected criteria."""
    judge_prompt = f"""Question: {question}
Expected: {expected}
Actual answer: {answer}

Does the actual answer meet the expected criteria? Respond with only "yes" or "no"."""

    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": judge_prompt}],
        temperature=0,
    )
    verdict = response.choices[0].message.content.strip().lower()
    return verdict.startswith("yes")


def run_test(client: OpenAI, test: TestCase) -> bool:
    answer = get_model_answer(client, test.question)

    if test.eval_type == "exact":
        passed = eval_exact(answer, test.expected)
    elif test.eval_type == "contains":
        passed = eval_contains(answer, test.expected)
    else:
        passed = eval_with_judge(client, test.question, answer, test.expected)

    status = "PASS" if passed else "FAIL"
    print(f"[{status}] [{test.eval_type}] {test.question}")
    print(f"       Expected: {test.expected}")
    print(f"       Got:      {answer}")
    return passed


def main():
    client = OpenAI()

    results = [run_test(client, test) for test in TEST_CASES]

    passed = sum(results)
    total = len(results)
    print(f"\nResults: {passed}/{total} passed ({100 * passed / total:.0f}%)")


if __name__ == "__main__":
    main()
