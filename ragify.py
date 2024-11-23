# import os
# import bs4
# import numpy as np
# import fitz
# from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import Chroma, FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
# from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
# from rouge import Rouge
# from sklearn.metrics import precision_score
# from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


class Ragify:
    def __init__(self, pdf_paths, llm_name):
        self.pdf_paths = pdf_paths
        self.llm_name = llm_name
        self.loader = [PyPDFLoader(path) for path in self.pdf_paths]
        self.all_data = [doc for loader in self.loader for doc in loader.load()]
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.all_splits = self.text_splitter.split_documents(self.all_data)
        self.local_embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url="https://ollama-947388510747.us-central1.run.app")
        # self.vectorstore = Chroma.from_documents(documents=self.all_splits, embedding=self.local_embeddings)
        self.vectorstore = FAISS.from_documents(documents=self.all_splits, embedding=self.local_embeddings)
        self.model = ChatOllama(model=self.llm_name, base_url="https://ollama-947388510747.us-central1.run.app")
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

    def generate_response(self, question):
        print(question)
        answer = self.qa_chain.invoke(question)
        print(answer)
        return answer

    # def evaluate_responses(self, questions, reference_responses):
    #     chatbot_responses = [self.generate_response(q) for q in questions]
    #
    #     precision_k = self.calculate_precision_k(chatbot_responses, reference_responses, k=75)
    #     rouge_scores = self.calculate_rouge_scores(chatbot_responses, reference_responses)
    #     bleu_score = self.calculate_bleu_score(chatbot_responses, reference_responses)
    #
    #     return chatbot_responses, precision_k, rouge_scores, bleu_score
    #
    # def calculate_precision_k(self, chatbot_responses, reference_responses, k=1):
    #     precisions = []
    #     for i in range(len(chatbot_responses)):
    #         chatbot_tokens = set(chatbot_responses[i].split())
    #         reference_tokens = set(reference_responses[i].split())
    #         intersection = chatbot_tokens.intersection(reference_tokens)
    #         precision = len(intersection) / k
    #         precisions.append(precision)
    #     return np.mean(precisions)
    #
    # def calculate_rouge_scores(self, chatbot_responses, reference_responses):
    #     rouge_evaluator = Rouge()
    #     valid_pairs = [(hyp, ref) for hyp, ref in zip(chatbot_responses, reference_responses) if hyp and ref]
    #
    #     if not valid_pairs:
    #         raise ValueError("No valid pairs to evaluate. Ensure responses are not empty.")
    #
    #     hyps, refs = zip(*valid_pairs)
    #     scores = rouge_evaluator.get_scores(hyps, refs, avg=True)
    #     return scores
    #
    # def calculate_bleu_score(self, chatbot_responses, reference_responses):
    #
    #     bleu_scores = []
    #     smoothing = SmoothingFunction().method4
    #     for hyp, ref in zip(chatbot_responses, reference_responses):
    #         reference_tokens = [ref.split()]
    #         hypothesis_tokens = hyp.split()
    #         score = sentence_bleu(reference_tokens, hypothesis_tokens, smoothing_function=smoothing)
    #         bleu_scores.append(score)
    #     return np.mean(bleu_scores)


if __name__ == "__main__":
    pdf_paths = [
        r"./documents/METU_Regulation.pdf",
        r"./documents/ISStudentGuide_2023-2024_v1.5.pdf"
    ]


    rag_pipeline = Ragify(
        pdf_paths=pdf_paths,
        llm_name="llama3.2:1b"
    )

    print(rag_pipeline.generate_response(question="What is the maximum number of semester for masters program?"))

    # Q_path = r"C:\Users\PoyaSystem\Desktop\QandA.pdf"
    # qa_pairs = rag_pipeline.extract_qa_from_pdf(Q_path)
    # questions = [qa[0] for qa in qa_pairs]
    # reference_responses = [qa[1] for qa in qa_pairs]
    #
    # chatbot_responses, precision_k, rouge_scores, bleu_score = rag_pipeline.evaluate_responses(questions, reference_responses)
    #
    # print("Chatbot Responses and Evaluation:")
    # for i, (question, chatbot_response) in enumerate(zip(questions, chatbot_responses)):
    #     print(f"Q{i+1}: {question}")
    #     print(f"Chatbot Response: {chatbot_response}")
    #     print(f"Reference Answer: {reference_responses[i]}")
    #     print()
    #
    # print(f"Precision@k: {precision_k:.2f}")
    # print(f"ROUGE Scores: {rouge_scores}")
    # print(f"BLEU Score: {bleu_score:.4f}")

