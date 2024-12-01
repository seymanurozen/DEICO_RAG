import time
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
# import fitz
# from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
# from rouge import Rouge

class Ragify:
    def __init__(self, pdf_paths,
                 llm_name = "llama3.2:1b",
                 embedding_name = "nomic-embed-text",
                 chunk_size = 1000):
        self.pdf_paths = pdf_paths
        self.llm_name = llm_name
        self.embedding_name = embedding_name
        self.chunk_size = chunk_size

        self.loader = [PyPDFLoader(path) for path in self.pdf_paths]
        self.all_data = [doc for loader in self.loader for doc in loader.load()]

        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=200)
        self.all_splits = self.text_splitter.split_documents(self.all_data)

        self.local_embeddings = OllamaEmbeddings(model=self.embedding_name)
        self.vectorstore = FAISS.from_documents(documents=self.all_splits, embedding=self.local_embeddings)

        self.model = ChatOllama(model=self.llm_name)
        self.rag_prompt = ChatPromptTemplate.from_template("""
            You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. 
            If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

            <context>
            {context}
            </context>

            Answer the following question:

            {question}""")

        self.retriever = self.vectorstore.as_retriever()
        self.qa_chain = (
            {"context": self.retriever | self.format_docs, "question": RunnablePassthrough()}
            | self.rag_prompt
            | self.model
            | StrOutputParser()
        )


    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def generate_response(self, question):
        start_time = time.time()
        response = self.qa_chain.invoke(question)
        time_taken = time.time() - start_time
        return response, time_taken


    # def extract_qa_from_pdf(self, Q_path):
    #     qa_pairs = []
    #     with fitz.open(Q_path) as doc:
    #         for page in doc:
    #             text = page.get_text()
    #             lines = text.splitlines()
    #             for i, line in enumerate(lines):
    #                 if line.startswith("Q:"):
    #                     question = line.replace("Q: ", "").strip()
    #                     if i + 1 < len(lines) and lines[i + 1].startswith("A:"):
    #                         answer = lines[i + 1].replace("A: ", "").strip()
    #                         qa_pairs.append((question, answer))
    #     return qa_pairs
    #
    #
    #
    #
    #
    # def evaluate_responses(self, questions, reference_responses):
    #     chatbot_responses = []
    #     response_times = []
    #
    #     for q in questions:
    #         response, time_taken = self.generate_response(q)
    #         chatbot_responses.append(response)
    #         response_times.append(time_taken)
    #
    #     precision_k = self.calculate_precision_k(chatbot_responses, reference_responses, k=75)
    #     rouge_scores = self.calculate_rouge_scores(chatbot_responses, reference_responses)
    #     bleu_score = self.calculate_bleu_scores(chatbot_responses, reference_responses)
    #     rag_metrics = self.calculate_rag_metrics(chatbot_responses, reference_responses)
    #
    #     return chatbot_responses, response_times, precision_k, rouge_scores, bleu_score, rag_metrics
    #
    #
    # def calculate_precision_k(self, chatbot_responses, reference_responses, k=1):
    #     precisions = []
    #     for i in range(len(chatbot_responses)):
    #         chatbot_tokens = set(chatbot_responses[i].split())
    #         reference_tokens = set(reference_responses[i].split())
    #         intersection = chatbot_tokens.intersection(reference_tokens)
    #         precision = len(intersection) / min(k, len(chatbot_tokens))
    #         precisions.append(precision)
    #     return {"mean": np.mean(precisions), "std": np.std(precisions)}
    #
    #
    # def calculate_rouge_scores(self, chatbot_responses, reference_responses):
    #     rouge_evaluator = Rouge()
    #     metrics = {"rouge-1": [], "rouge-2": [], "rouge-l": []}
    #
    #     for hyp, ref in zip(chatbot_responses, reference_responses):
    #         if hyp and ref:
    #             scores = rouge_evaluator.get_scores(hyp, ref, avg=False)
    #             for metric in metrics.keys():
    #                 metrics[metric].append(scores[0][metric])  # Appending ROUGE-1, ROUGE-2, ROUGE-L
    #
    #     # Compute mean and std for each metric
    #     return {
    #         metric: {
    #             "mean": {
    #                 "precision": np.mean([score["p"] for score in metrics[metric]]),
    #                 "recall": np.mean([score["r"] for score in metrics[metric]]),
    #                 "f1": np.mean([score["f"] for score in metrics[metric]]),
    #             },
    #             "std": {
    #                 "precision": np.std([score["p"] for score in metrics[metric]]),
    #                 "recall": np.std([score["r"] for score in metrics[metric]]),
    #                 "f1": np.std([score["f"] for score in metrics[metric]]),
    #             },
    #         }
    #         for metric in metrics
    #     }
    #
    #
    # def calculate_bleu_scores(self, chatbot_responses, reference_responses):
    #     smoothing = SmoothingFunction().method4
    #     scores = []
    #     for hyp, ref in zip(chatbot_responses, reference_responses):
    #         if hyp and ref:
    #             score = sentence_bleu([ref.split()], hyp.split(), smoothing_function=smoothing)
    #             scores.append(score)
    #     return {"mean": np.mean(scores), "std": np.std(scores)}
    #
    #
    # def calculate_rag_metrics(self, chatbot_responses, reference_responses):
    #     precision_scores, recall_scores, f1_scores = [], [], []
    #     for chatbot_response, reference_response in zip(chatbot_responses, reference_responses):
    #         chatbot_tokens = set(chatbot_response.split())
    #         reference_tokens = set(reference_response.split())
    #
    #         true_positives = len(reference_tokens.intersection(chatbot_tokens))
    #         false_positives = len(chatbot_tokens.difference(reference_tokens))
    #         false_negatives = len(reference_tokens.difference(chatbot_tokens))
    #
    #         precision = true_positives / (true_positives + false_positives + 1e-10)
    #         recall = true_positives / (true_positives + false_negatives + 1e-10)
    #         f1 = 2 * precision * recall / (precision + recall + 1e-10)
    #
    #         precision_scores.append(precision)
    #         recall_scores.append(recall)
    #         f1_scores.append(f1)
    #
    #     return {
    #         "Precision": {"mean": np.mean(precision_scores), "std": np.std(precision_scores)},
    #         "Recall": {"mean": np.mean(recall_scores), "std": np.std(recall_scores)},
    #         "F1 Score": {"mean": np.mean(f1_scores), "std": np.std(f1_scores)},
    #     }


if __name__ == "__main__":
    pdf_paths = [
        r"./documents/METU_Regulation.pdf",
        r"./documents/ISStudentGuide_2023-2024_v1.5.pdf"
    ]
    Q_path = r"./documents/QandA.pdf"

    example_chat_history = [
        {
            'content': 'Who is Tuğba Taşkaya Temizel?',
            'role': 'user'},
        {
            'content': ('Tuğba Taşkaya Temizel is a Professor. Her room number is A-211, phone number is 7782, and email address is ttemizel[at]metu.edu.tr.', 8.58181118965149),
            'role': 'assistant'
        }
    ]

    # # Version 1
    # rag_pipeline = Ragify(
    #     pdf_paths=pdf_paths,
    #     llm_name="llama3.2:1b",
    #     embedding_name="nomic-embed-text",
    #     chunk_size=1000
    # )

    # # Version 2
    # rag_pipeline = Ragify(
    #     pdf_paths=pdf_paths,
    #     llm_name="llama3.2:1b",
    #     embedding_name="mxbai-embed-large",
    #     chunk_size=1000
    # )

    # # Version 3
    # rag_pipeline = Ragify(
    #     pdf_paths=pdf_paths,
    #     llm_name="llama3.2:latest",
    #     embedding_name="nomic-embed-text",
    #     chunk_size=1000
    # )

    # # Version 4
    # rag_pipeline = Ragify(
    #     pdf_paths=pdf_paths,
    #     llm_name="llama3.2:latest",
    #     embedding_name="mxbai-embed-large",
    #     chunk_size=1000
    # )

    # # Version 5
    # rag_pipeline = Ragify(
    #     pdf_paths=pdf_paths,
    #     llm_name="llama3.2:latest",
    #     embedding_name="nomic-embed-text",
    #     chunk_size=500
    # )

    # Version 6
    rag_pipeline = Ragify(
        pdf_paths=pdf_paths,
        llm_name="llama3.2:latest",
        embedding_name="nomic-embed-text",
        chunk_size=2000
    )

    # qa_pairs = rag_pipeline.extract_qa_from_pdf(Q_path)
    # questions = [qa[0] for qa in qa_pairs]
    # reference_responses = [qa[1] for qa in qa_pairs]
    #
    # chatbot_responses, response_times, precision_k, rouge_scores, bleu_score, rag_metrics = rag_pipeline.evaluate_responses(questions, reference_responses)
    #
    # # Detailed Outputs
    # print("Chatbot Responses and Evaluation:")
    # for i, (question, chatbot_response, time_taken) in enumerate(zip(questions, chatbot_responses, response_times)):
    #     print(f"Q{i+1}: {question}")
    #     print(f"Chatbot Response: {chatbot_response}")
    #     print(f"Reference Answer: {reference_responses[i]}")
    #     print(f"Response Time: {time_taken:.4f} seconds")
    #     print()
    #
    # for rouge_type, scores in rouge_scores.items():
    #     print(f"{rouge_type.upper()}:")
    #     print(f"  Precision: Mean={scores['mean']['precision']:.4f}, Std={scores['std']['precision']:.4f}")
    #     print(f"  Recall: Mean={scores['mean']['recall']:.4f}, Std={scores['std']['recall']:.4f}")
    #     print(f"  F1: Mean={scores['mean']['f1']:.4f}, Std={scores['std']['f1']:.4f}")
    #
    # print(f"Precision@k: Mean={precision_k['mean']:.2f}, Std={precision_k['std']:.2f}")
    # print(f"BLEU: Mean={bleu_score['mean']:.4f}, Std={bleu_score['std']:.4f}")
    # print("RAG Metrics:")
    # for metric, stats in rag_metrics.items():
    #     print(f"{metric}: Mean={stats['mean']:.4f}, Std={stats['std']:.4f}")
