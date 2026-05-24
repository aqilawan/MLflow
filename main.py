import mlflow
import os
import pandas as pd
import openai
import dagshub
from dotenv import load_dotenv

load_dotenv()

dagshub.init(repo_owner='aqilawan', repo_name='MLfLow', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/aqilawan/MLfLow.mlflow")
mlflow.set_experiment("LLM Evaluation - Groq")

os.environ["OPENAI_API_KEY"] = os.environ["GROQ_API_KEY"]
os.environ["OPENAI_API_BASE"] = "https://api.groq.com/openai/v1"

eval_data = pd.DataFrame(
    {
        "inputs": [
            "What is LoRA?",
            "What is QLoRA?",
        ],
        "ground_truth": [
            "LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning technique...",
            "QLoRA (Quantized Low-Rank Adaptation) is an advanced extension of LoRA..."
        ],
    }
)

with mlflow.start_run():

    system_prompt = "Answer the following question in two sentences"

    logged_model_info = mlflow.openai.log_model(
        model="llama-3.3-70b-versatile",
        task="chat.completions",
        artifact_path="model",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "{inputs}"}
        ],
    )

    results = mlflow.evaluate(
        logged_model_info.model_uri,
        eval_data,
        targets="ground_truth",
        model_type="question-answering",
        extra_metrics=[
            mlflow.metrics.latency()
        ]
    )

    print("Aggregated metrics:")
    print(results.metrics)

    eval_table = results.tables["eval_results_table"]
    df = pd.DataFrame(eval_table)
    df.to_csv("eval.csv", index=False)
    mlflow.log_artifact("eval.csv") 

    print(eval_table)