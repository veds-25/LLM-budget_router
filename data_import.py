import pandas as pd
from datasets import load_dataset
import numpy as np
judge_qwen06b = load_dataset( "JiaqiXue/mmr-routing-20k", data_files="data/qwen3-0.6b/judge_scores.jsonl")["train"].to_pandas()
judge_ministral8b = load_dataset( "JiaqiXue/mmr-routing-20k", data_files="data/ministral-8b/judge_scores.jsonl")["train"].to_pandas()
judge_qwen30ba3b = load_dataset( "JiaqiXue/mmr-routing-20k",data_files="data/qwen3-30b-a3b/judge_scores.jsonl")["train"].to_pandas()
judge_qwen30b_instruct = load_dataset("JiaqiXue/mmr-routing-20k",data_files="data/qwen3-30b-a3b-instruct/judge_scores.jsonl")["train"].to_pandas()
judge_features = load_dataset("JiaqiXue/mmr-routing-20k",data_files="data/features/qwen06b_20k.jsonl")["train"].to_pandas()

print(judge_features.columns.to_list())
print(judge_qwen06b.columns.to_list())
print(judge_features["features"].iloc[0])


df_qwen06b = judge_qwen06b.rename(columns = {"score": "qwen06b_score"})
df_ministral8b = judge_ministral8b.rename(columns = {"score": "ministral8b_score"})
df_qwen30ba3b = judge_qwen30ba3b.rename(columns = {"score": "qwen30ba3b_score"})
df_qwen30b_instruct = judge_qwen30b_instruct.rename(columns = {"score": "qwen30b_instruct_score"})

merged = judge_features[["conversation_hash", "turn_idx", "difficulty"]]
for other in [df_qwen06b, df_ministral8b, df_qwen30ba3b, df_qwen30b_instruct]:
    merged = merged.merge(other[["conversation_hash", "turn_idx", "score"]],
                           on=["conversation_hash", "turn_idx"], how="inner")

print(merged.shape)
merged.head()


rng= np.random.default_rng(seed = 42)

model_names = ["small_model", "medium_model", "large_model"]

"""for i in range(2000):
    difficulty = rng.uniform(0,1)# 0 is easy and 1 is hard 
    row = {"question_id" : i , "difficulty" : difficulty}

    for name,base_quality
    """