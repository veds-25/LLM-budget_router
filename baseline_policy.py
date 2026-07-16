from environment import ACTIONS, MODEL_COSTS

def baseline_decide(state):
    if state.queries_remaining == 0:
        return "skip"

    budget_per_query = state.remaining_budget / state.queries_remaining

    if budget_per_query < 10:
        return "skip"
    elif budget_per_query < 40:
        return "qwen06b"
    elif budget_per_query < 60:
        return "ministral8b"
    elif budget_per_query < 90:
        return "qwen30ba3b"
    else:
        return "qwen30b_instruct"