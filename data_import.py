import pandas as pd
from datasets import load_dataset
import numpy as np
judge_qwen06b = load_dataset( "JiaqiXue/mmr-routing-20k", data_files="data/qwen3-0.6b/judge_scores.jsonl")["train"].to_pandas()
judge_ministral8b = load_dataset( "JiaqiXue/mmr-routing-20k", data_files="data/ministral-8b/judge_scores.jsonl")["train"].to_pandas()
judge_qwen30ba3b = load_dataset( "JiaqiXue/mmr-routing-20k",data_files="data/qwen3-30b-a3b/judge_scores.jsonl")["train"].to_pandas()
judge_qwen30b_instruct = load_dataset("JiaqiXue/mmr-routing-20k",data_files="data/qwen3-30b-a3b-instruct/judge_scores.jsonl")["train"].to_pandas()
judge_features = load_dataset("JiaqiXue/mmr-routing-20k",data_files="data/features/qwen06b_20k.jsonl")["train"].to_pandas()

df_qwen06b = judge_qwen06b.rename(columns = {"score": "qwen06b_score"})
df_ministral8b = judge_ministral8b.rename(columns = {"score": "ministral8b_score"})
df_qwen30ba3b = judge_qwen30ba3b.rename(columns = {"score": "qwen30ba3b_score"})
df_qwen30b_instruct = judge_qwen30b_instruct.rename(columns = {"score": "qwen30b_instruct_score"})

merged = judge_features[["conversation_hash", "turn_idx", "difficulty"]]

score_columns = [(df_qwen06b, "qwen06b_score"), (df_ministral8b, "ministral8b_score"), (df_qwen30ba3b, "qwen30ba3b_score"), (df_qwen30b_instruct, "qwen30b_instruct_score")]

for other, score_col in score_columns:
    merged = merged.merge(other[["conversation_hash", "turn_idx", score_col]], on=["conversation_hash", "turn_idx"],how="inner")


print(merged.shape)
print(merged.head())
print(merged["difficulty"].describe())
print(merged["difficulty"].unique())

MODEL_COSTS = {"quen06b" : 6, "ministral8b" : 8, "qwen30ba3b" : 30, "qwen30b_instruct" : 30}