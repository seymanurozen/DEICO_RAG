import wandb
import pandas as pd
from ragify import Ragify


def wandb_experiment(config=None):
    """
    A function designed to run a single experiment using Weights & Biases (wandb).
    It reads a Q&A dataset from an Excel file, initializes a Ragify pipeline with the
    configuration parameters, evaluates the model on the Q&A set, and logs the metrics to wandb.

    Args:
        config (dict, optional): Configuration dictionary for wandb sweep. Defaults to None.
    """
    # Read the Q&A dataset that includes questions, reference answers, and reference paragraph chunks
    qna_df = pd.read_excel(r"documents/Questions_Answers_ContainingParagraph.xlsx")
    questions = qna_df["Question"].tolist()
    reference_responses = qna_df["Answer"].tolist()
    grouped_reference_chunks = qna_df["Containing Paragraph from the Document"].tolist()

    # Paths to PDF documents that serve as the knowledge base
    pdf_paths = [
        r"documents/METU_Regulation.pdf",
        r"documents/ISStudentGuide_2023-2024_v1.5.pdf"
    ]

    try:
        # Begin a wandb run; if config is provided, it will be merged with wandb.config
        with wandb.init(config=config):
            # Retrieve the config values from wandb
            config = wandb.config
            llm_name = config.llm_name
            embedding_name = config.embedding_name
            chunk_size = config.chunk_size

            # Instantiate Ragify with the desired parameters
            rag_pipeline = Ragify(
                pdf_paths=pdf_paths,
                llm_name=llm_name,
                embedding_name=embedding_name,
                chunk_size=chunk_size
            )

            # Evaluate the model on the entire Q&A set
            (
                chatbot_responses,
                response_times,
                precision_k,
                rouge_scores,
                bleu_score,
                avg_response_time
            ) = rag_pipeline.evaluate_responses(
                questions,
                reference_responses,
                grouped_reference_chunks
            )

            # Prepare evaluation metrics to be logged
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

            # Print and log the metrics to wandb
            print("Logging metrics:", metrics_dict)
            wandb.log(metrics_dict)

            # Update the wandb run summary with the metrics
            wandb.run.summary.update(metrics_dict)

            # Create a DataFrame to store and log the model responses
            results_df = pd.DataFrame({
                "Question": questions,
                "Chatbot Response": chatbot_responses,
                "Reference Answer": reference_responses,
                "Response Time": response_times,
            })

            # Log the DataFrame as a wandb Table
            wandb.log({"results_table": wandb.Table(dataframe=results_df)})

    except Exception as e:
        # Handle and log any errors that occur during the experiment
        print(f"Error during training: {e}")
        wandb.log({"error": str(e)})
    finally:
        # Ensure the wandb session is closed gracefully
        wandb.finish()


if __name__ == "__main__":
    # Sweep configuration for hyperparameter tuning or grid-search
    sweep_config = {
        "method": "grid",
        "parameters": {
            "llm_name": {"values": ["llama3.2:1b", "llama3.2:latest"]},
            "embedding_name": {"values": ["nomic-embed-text", "mxbai-embed-large"]},
            "chunk_size": {"values": [1000, 2000]}
        }
    }

    try:
        # Create a sweep on wandb using the defined configuration
        sweep_id = wandb.sweep(sweep_config, project="ragify")
        # Launch the sweep agent to run multiple experiments
        wandb.agent(sweep_id, function=wandb_experiment)
    except Exception as e:
        # Handle errors that might occur during sweep initialization
        print(f"Error initializing sweep: {e}")