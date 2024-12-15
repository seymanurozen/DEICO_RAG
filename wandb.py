import wandb
import pandas as pd
from ragify import Ragify


def wandb_experiment(config=None):
    qna_df = pd.read_excel(r"documents/Questions_Answers_ContainingParagraph.xlsx")
    questions = qna_df["Question"].tolist()
    reference_responses = qna_df["Answer"].tolist()
    grouped_reference_chunks = qna_df["Containing Paragraph from the Document"].tolist()

    pdf_paths = [
        r"documents/METU_Regulation.pdf",
        r"documents/ISStudentGuide_2023-2024_v1.5.pdf"
    ]

    try:
        with wandb.init(config=config):
            config = wandb.config
            llm_name = config.llm_name
            embedding_name = config.embedding_name
            chunk_size = config.chunk_size

            rag_pipeline = Ragify(
                pdf_paths=pdf_paths,
                llm_name=llm_name,
                embedding_name=embedding_name,
                chunk_size=chunk_size
            )

            chatbot_responses, response_times, precision_k, rouge_scores, bleu_score, avg_response_time = rag_pipeline.evaluate_responses(
                questions, reference_responses, grouped_reference_chunks
            )

            # Log metrics
            metrics_dict = {
                "precision_k_mean": precision_k["mean"],
                "bleu_mean": bleu_score["mean"],
                "bleu_std": bleu_score["std"],
                "rouge_1_precision_mean": rouge_scores["rouge-1"]["mean"]["precision"],
                "rouge_1_recall_mean": rouge_scores["rouge-1"]["mean"]["recall"],
                "rouge_1_f1_mean": rouge_scores["rouge-1"]["mean"]["f1"],
                "rouge_2_precision_mean": rouge_scores.get("rouge-2", {}).get("mean", {}).get("precision", 0),
                "rouge_2_recall_mean": rouge_scores.get("rouge-2", {}).get("mean", {}).get("recall", 0),
                "rouge_2_f1_mean": rouge_scores.get("rouge-2", {}).get("mean", {}).get("f1", 0),
                "rouge_l_precision_mean": rouge_scores.get("rouge-l", {}).get("mean", {}).get("precision", 0),
                "rouge_l_recall_mean": rouge_scores.get("rouge-l", {}).get("mean", {}).get("recall", 0),
                "rouge_l_f1_mean": rouge_scores.get("rouge-l", {}).get("mean", {}).get("f1", 0),
                "average_response_time": avg_response_time,
                "chunk_size": chunk_size,
                "llm_name": llm_name,
                "embedding_name": embedding_name
            }
            print("Logging metrics:", metrics_dict)
            wandb.log(metrics_dict)

            wandb.run.summary.update(metrics_dict)

            results_df = pd.DataFrame({
                "Question": questions,
                "Chatbot Response": chatbot_responses,
                "Reference Answer": reference_responses,
                "Response Time": response_times,
            })
            wandb.log({"results_table": wandb.Table(dataframe=results_df)})

    except Exception as e:
        print(f"Error during training: {e}")
        wandb.log({"error": str(e)})
    finally:
        wandb.finish()


if __name__ == "__main__":
    sweep_config = {
        "method": "grid",
        "parameters":
            {
                "llm_name": {"values": ["llama3.2:1b", "llama3.2:latest"]},
                "embedding_name": {"values": ["nomic-embed-text", "mxbai-embed-large"]},
                "chunk_size": {"values": [1000, 2000]}
            }
        }

    try:
        sweep_id = wandb.sweep(sweep_config, project="ragify")
        wandb.agent(sweep_id, function=wandb_experiment)
    except Exception as e:
        print(f"Error initializing sweep: {e}")