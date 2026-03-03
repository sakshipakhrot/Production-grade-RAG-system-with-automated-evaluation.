import os
import pandas as pd
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
from ragas.run_config import RunConfig
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings


from rag_pipeline import setup_rag_chain

load_dotenv()

def run_evaluation():
    print("1. Initializing RAG Pipeline...")
    rag_chain = setup_rag_chain()
    
    # 1. Define Test Cases
    questions = [
        "What is the improved F1-score when MobileNetV2 is used?",
        "What is the R2 score while using XGBoost regression model?"
    ]
    
    ground_truths = [
        "0.87",
        "0.93"
    ]

    print("2. Generating answers and retrieving contexts...")
    answers = []
    contexts = []
    
    for query in questions:
        response = rag_chain.invoke({"input": query})
        answers.append(response["answer"])
        retrieved_texts = [doc.page_content for doc in response["context"]]
        contexts.append(retrieved_texts)

    # 2. Format data for RAGAS
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }
    dataset = Dataset.from_dict(data)

    print("3. Configuring Evaluation Models...")
    
    eval_llm = ChatOpenAI(
        api_key=os.environ.get("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1",
        model="llama-3.3-70b-versatile",
        temperature=0
    )
    eval_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    
    config = RunConfig(max_workers=2, timeout=60)

    print("4. Running RAGAS Evaluation...")
    
    
    answer_relevancy.strictness = 1

    
    result = evaluate(
        dataset=dataset,
        metrics=[
            context_precision,
            context_recall,
            faithfulness,
            answer_relevancy,
        ],
        llm=eval_llm,
        embeddings=eval_embeddings,
        run_config=config 
    )

    
    print("\n=== Evaluation Results ===")
    df = result.to_pandas()
    
    print(df.mean(numeric_only=True))
    
    df.to_csv("evaluation_results.csv", index=False)
    print("\nDetailed results saved to 'evaluation_results.csv'")

if __name__ == "__main__":

    run_evaluation()
