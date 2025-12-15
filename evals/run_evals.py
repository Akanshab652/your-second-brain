from evals.test_cases import EVAL_CASES
from evals.llm_judge import llm_judge


def run_evals(chat_fn, llm_client, store):
    print("\n🧪 Running Eval Suite...\n")

    for case in EVAL_CASES:
        print(f"▶️ {case['name']}")

        answer = chat_fn(
            question=case["question"],
            llm_client=llm_client,
            store=store,
            history=[]
        )

        eval_result = llm_judge(
            llm_client=llm_client,
            question=case["question"],
            answer=answer,
            context=""
        )

        print("Answer:", answer)
        print("Eval:", eval_result.model_dump())
        print("-" * 50)
