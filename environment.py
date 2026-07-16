import pandas as pd

MODEL_COSTS = {"qwen06b": 6, "ministral8b": 8, "qwen30ba3b": 30,"qwen30b_instruct": 30,}

ACTIONS = ["qwen06b", "ministral8b", "qwen30ba3b", "qwen30b_instruct", "skip"]


def to_discrete_budget(remaining_budget):
    bucket_budget = int((remaining_budget * 4) / 10000)
    if remaining_budget >= 10000:
        bucket_budget = 3
    return bucket_budget


class RoutingState:
    def __init__(self, difficulty, remaining_budget, queries_remaining):
        self.difficulty = difficulty
        self.remaining_budget = remaining_budget
        self.queries_remaining = queries_remaining

    def to_discrete_key(self):
        budget_bucket = to_discrete_budget(self.remaining_budget)
        return (budget_bucket, self.difficulty)


class RoutingEnvironment:
    def __init__(self, dataframe, total_budget=10000):
        self.df = dataframe
        self.total_budget = total_budget
        self.remaining_budget = 0
        self.current_index = 0
        self.query_order = []
        self.quality_history = []

    def reset(self):
        self.remaining_budget = self.total_budget
        self.current_index = 0
        self.quality_history = []

        self.query_order = []
        for i in range(len(self.df)):
            self.query_order.append(i)

        return self._current_state()

    def _current_state(self):
        row_index = self.query_order[self.current_index]
        row = self.df.iloc[row_index]
        queries_left = len(self.query_order) - self.current_index

        return RoutingState(
            difficulty=int(row["difficulty"]),
            remaining_budget=self.remaining_budget,
            queries_remaining=queries_left,
        )

    def step(self, action):
        row_index = self.query_order[self.current_index]
        row = self.df.iloc[row_index]

        if action == "skip":
            quality = 0.0
            cost = 0
        else:
            quality = row[action + "_score"]
            cost = MODEL_COSTS[action]

        if cost > self.remaining_budget:
            reward = -50
        else:
            reward = quality - 0.01 * cost

        self.remaining_budget = self.remaining_budget - cost
        self.quality_history.append(quality)
        self.current_index = self.current_index + 1

        done = False
        if self.current_index >= len(self.query_order):
            done = True
        if self.remaining_budget <= 0:
            done = True

        if done:
            next_state = None
        else:
            next_state = self._current_state()

        info = {"quality": quality, "cost": cost}
        return next_state, reward, done, info

    def episode_summary(self):
        answered = 0
        for q in self.quality_history:
            if q > 0:
                answered = answered + 1

        total_quality = 0.0
        for q in self.quality_history:
            total_quality = total_quality + q

        avg_quality = 0.0
        if len(self.quality_history) > 0:
            avg_quality = total_quality / len(self.quality_history)

        return {
            "queries_answered": answered,
            "total_queries": len(self.query_order),
            "avg_quality": avg_quality,
            "budget_used": self.total_budget - self.remaining_budget,
        }