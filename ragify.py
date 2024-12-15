import time
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
import numpy as np
import pandas as pd


class Ragify:
    def __init__(self, pdf_paths,
                 llm_name="llama3.2:1b",
                 embedding_name="nomic-embed-text",
                 chunk_size=1000):
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
        You are an assistant specialized in providing information about the 'Rules and Regulations Governing Graduate Studies' at METU.
        Use ONLY the provided context to answer the question. Do not use outside knowledge.
            
         <context>
         {context}
         </context>
         
         Below is the conversation so far:
         {chat_history}
         
         Answer the following question:
         {question}
         """
        )

        self.retriever = self.vectorstore.as_retriever()


    def format_chat_history(self, chat_history):
        formatted_history = ""
        for turn in chat_history:
            role = "User" if turn["role"] == "user" else "Assistant"
            # If the assistant response is a tuple (response, time), extract just the response text
            if isinstance(turn["content"], tuple):
                content = turn["content"][0]
            else:
                content = turn["content"]
            formatted_history += f"{role}: {content}\n"
        return formatted_history.strip()


    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)


    def generate_response(self, question, chat_history=[]):
        start_time = time.time()

        # Do retrieval and formatting
        retrieved_docs = self.retriever.get_relevant_documents(question)
        formatted_context = self.format_docs(retrieved_docs)
        formatted_chat_history = self.format_chat_history(chat_history)

        # Update the prompt
        prompt = self.rag_prompt.format(
            context=formatted_context,
            question=question,
            chat_history=formatted_chat_history
        )

        response = self.model.invoke(prompt)

        time_taken = time.time() - start_time
        return response.content, time_taken


    def evaluate_responses(self, questions, reference_responses, grouped_reference_chunks):
        chatbot_responses = []
        response_times = []

        for q in questions:
            response, time_taken = self.generate_response(q)
            chatbot_responses.append(response)
            response_times.append(time_taken)

        precision_k = rag_pipeline.calculate_precision_k(questions, grouped_reference_chunks, top_k=3)
        rouge_scores = self.calculate_rouge_scores(chatbot_responses, reference_responses)
        bleu_score = self.calculate_bleu_scores(chatbot_responses, reference_responses)

        return chatbot_responses, response_times, precision_k, rouge_scores, bleu_score


    def calculate_precision_k(self, questions, grouped_reference_chunks, top_k=3):
        """
        Calculate Precision@k for retrieved chunks vs. grouped reference chunks.
        """
        match = 0

        for i, (question, ref_chunks) in enumerate(zip(questions, grouped_reference_chunks)):

            # Retrieve relevant chunks for the question
            retrieved_docs = self.retriever.get_relevant_documents(question)
            retrived_chunks = ""
            for doc in retrieved_docs[:min(top_k, len(retrieved_docs))]:
                retrived_chunks += doc.page_content.strip().lower().replace("\n", "")

            # Normalize reference chunks
            ref_chunks_list = [chunk for chunk in ref_chunks.strip().lower().split()]

            if ref_chunks.strip().lower() in retrived_chunks:
                match += 1

        return {"mean": match / len(questions)}


    def calculate_rouge_scores(self, chatbot_responses, reference_responses):
        rouge_evaluator = Rouge()
        metrics = {"rouge-1": [], "rouge-2": [], "rouge-l": []}

        for hyp, ref in zip(chatbot_responses, reference_responses):
            if hyp and ref:
                scores = rouge_evaluator.get_scores(hyp.strip().lower(), ref.strip().lower(), avg=False)
                for metric in metrics.keys():
                    metrics[metric].append(scores[0][metric])

        return {
            metric: {
                "mean": {
                    "precision": np.mean([score["p"] for score in metrics[metric]]),
                    "recall": np.mean([score["r"] for score in metrics[metric]]),
                    "f1": np.mean([score["f"] for score in metrics[metric]]),
                },
                "std": {
                    "precision": np.std([score["p"] for score in metrics[metric]]),
                    "recall": np.std([score["r"] for score in metrics[metric]]),
                    "f1": np.std([score["f"] for score in metrics[metric]]),
                },
            }
            for metric in metrics
        }


    def calculate_bleu_scores(self, chatbot_responses, reference_responses):
        scores = []
        for hyp, ref in zip(chatbot_responses, reference_responses):
            if hyp and ref:
                score = sentence_bleu([ref.strip().lower().split()], hyp.strip().lower().split(),
                                      smoothing_function=SmoothingFunction().method4)
                scores.append(score)
        return {"mean": np.mean(scores), "std": np.std(scores)}


if __name__ == "__main__":

    qna_df = pd.read_excel(r"documents/Questions_Answers_ContainingParagraph.xlsx")
    questions = qna_df["Question"].tolist()
    reference_responses = qna_df["Answer"].tolist()
    grouped_reference_chunks = qna_df["Containing Paragraph from the Document"].tolist()

    pdf_paths = [
        r"documents/METU_Regulation.pdf",
        r"documents/ISStudentGuide_2023-2024_v1.5.pdf"
    ]

    example_chat_history = [
        {
            'content': 'Who is Tuğba Taşkaya Temizel?',
            'role': 'user'},
        {
            'content': ('Tuğba Taşkaya Temizel is a Professor. Her room number is A-211, phone number is 7782, and email address is ttemizel[at]metu.edu.tr.', 8.58181118965149),
            'role': 'assistant'
        }
    ]

    # Version 1
    rag_pipeline = Ragify(
        pdf_paths=pdf_paths,
        llm_name="llama3.2:1b",
        embedding_name="nomic-embed-text",
        chunk_size=1000
    )

    response, time_taken = rag_pipeline.generate_response("What is her email address?", chat_history=example_chat_history)
    print("Response:", response)
    print("Time taken:", time_taken)


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

    # # Version 6
    # rag_pipeline = Ragify(
    #     pdf_paths=pdf_paths,
    #     llm_name="llama3.2:latest",
    #     embedding_name="nomic-embed-text",
    #     chunk_size=2000
    # )
    #
    # # Evaluate Responses
    # chatbot_responses, response_times, precision_k, rouge_scores, bleu_score = rag_pipeline.evaluate_responses(questions, reference_responses, grouped_reference_chunks)
    #
    # for i, (question, chatbot_response, time_taken) in enumerate(zip(questions, chatbot_responses, response_times)):
    #     print(f"Q{i + 1}: {question}")
    #     print(f"Chatbot Response: {chatbot_response}")
    #     print(f"Reference Answer: {reference_responses[i]}")
    #     print(f"Response Time: {time_taken:.4f} seconds")
    #     print()
    #
    # for rouge_type, scores in rouge_scores.items():
    #     print(f"{rouge_type.upper()}:")
    #     print(f"Precision: Mean={scores['mean']['precision']:.4f}, Std={scores['std']['precision']:.4f}")
    #     print(f"Recall: Mean={scores['mean']['recall']:.4f}, Std={scores['std']['recall']:.4f}")
    #     print(f"F1: Mean={scores['mean']['f1']:.4f}, Std={scores['std']['f1']:.4f}")
    #     print(f"Precision@k: Mean={precision_k['mean']:.2f}")
    #     print(f"BLEU: Mean={bleu_score['mean']:.4f}, Std={bleu_score['std']:.4f}")