import time
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
import numpy as np
import pandas as pd


class Ragify:
    """
    A class that implements a Retrieval-Augmented Generation (RAG) pipeline, integrating
    LLM-based question answering with a local document database.

    Attributes:
        pdf_paths (list): Paths to the PDF files containing the knowledge base.
        llm_name (str): Name of the Large Language Model to be used for generating answers.
        embedding_name (str): Name of the model used to produce text embeddings.
        chunk_size (int): The size of text chunks for indexing.

    Typical usage:
        rag_pipeline = Ragify(pdf_paths=["doc1.pdf"], llm_name="llama3.2:1b", ...)
        response, time_taken = rag_pipeline.generate_response("Your question")
    """

    def __init__(self, pdf_paths,
                 llm_name="llama3.2:1b",
                 embedding_name="nomic-embed-text",
                 chunk_size=1000):
        """
        Initializes the Ragify object by loading documents from PDF paths,
        splitting them into chunks, creating embeddings, and initializing the LLM and retriever.

        Args:
            pdf_paths (list): A list of file paths to PDFs containing the knowledge base.
            llm_name (str, optional): The name of the LLM model to be used. Defaults to "llama3.2:1b".
            embedding_name (str, optional): The name of the embedding model. Defaults to "nomic-embed-text".
            chunk_size (int, optional): The size of text chunks. Defaults to 1000.
        """

        # Store constructor parameters
        self.pdf_paths = pdf_paths
        self.llm_name = llm_name
        self.embedding_name = embedding_name
        self.chunk_size = chunk_size

        # Load PDF documents
        self.loader = [PyPDFLoader(path) for path in self.pdf_paths]
        # Flatten the list of documents
        self.all_data = [doc for loader in self.loader for doc in loader.load()]

        # Split documents into manageable chunks
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=200)
        self.all_splits = self.text_splitter.split_documents(self.all_data)

        # Create embeddings for chunks and build a FAISS vector store
        self.local_embeddings = OllamaEmbeddings(model=self.embedding_name)
        self.vectorstore = FAISS.from_documents(documents=self.all_splits, embedding=self.local_embeddings)

        # Initialize the Ollama LLM model
        self.model = ChatOllama(model=self.llm_name)

        # Define a base prompt for RAG
        self.rag_prompt = ChatPromptTemplate.from_template(
            """
            You are an assistant specialized in providing information about the 'Rules and Regulations Governing Graduate Studies' at METU.
            Use ONLY the provided context to answer the question. Do not use outside knowledge."Answer only if the question aligns with the provided context. For out-of-scope questions, respond with 'I’d be happy to help with any questions about the 'METU Regulations and IS Student Guide'!"

            <context>
            {context}
            </context>

            Below is the conversation so far:
            {chat_history}

            Answer the following question:
            {question}
            """
        )

        # Create a retriever from the FAISS vector store
        self.retriever = self.vectorstore.as_retriever()

    def format_chat_history(self, chat_history):
        """
        Converts a list of dictionaries (representing user and assistant turns)
        into a readable string format for the prompt.

        Args:
            chat_history (list): A list of dictionaries with keys "role" and "content".

        Returns:
            str: A string that represents the conversation.
        """
        formatted_history = ""
        for turn in chat_history:
            # Determine whether user or assistant
            role = "User" if turn["role"] == "user" else "Assistant"

            # In some cases, assistant content might be a tuple (response, time)
            # so we extract only the text portion
            if isinstance(turn["content"], tuple):
                content = turn["content"][0]
            else:
                content = turn["content"]

            formatted_history += f"{role}: {content}\n"

        return formatted_history.strip()

    def format_docs(self, docs):
        """
        Formats a list of document objects into a single string containing the text of each document.

        Args:
            docs (list): A list of documents, each having a 'page_content' attribute.

        Returns:
            str: A concatenated string of all document contents.
        """
        return "\n\n".join(doc.page_content for doc in docs)

    def generate_response(self, question, chat_history=[]):
        """
        Generates a response to a given question by retrieving relevant context from
        the vector store and then passing it to the LLM.

        Args:
            question (str): The user's query.
            chat_history (list, optional): Conversation history up to this point. Defaults to [].

        Returns:
            tuple(str, float): A tuple containing the answer from the LLM and the time taken to generate it.
        """

        # Record the start time for performance tracking
        start_time = time.time()

        # Retrieve context relevant to the question
        retrieved_docs = self.retriever.get_relevant_documents(question)
        formatted_context = self.format_docs(retrieved_docs)

        # Format conversation history (if any) into a readable string
        formatted_chat_history = self.format_chat_history(chat_history)

        # Create the final prompt by injecting the context, question, and history
        prompt = self.rag_prompt.format(
            context=formatted_context,
            question=question,
            chat_history=formatted_chat_history
        )

        # Invoke the LLM to generate a response
        response = self.model.invoke(prompt)

        # Calculate the time taken to generate the response
        time_taken = time.time() - start_time

        return response.content, time_taken

    def evaluate_responses(self, questions, reference_responses, grouped_reference_chunks):
        """
        Evaluates chatbot responses on a set of questions against reference answers
        using metrics like precision@k, ROUGE, and BLEU, as well as measures response times.

        Args:
            questions (list): A list of question strings.
            reference_responses (list): The reference (ground-truth) answers.
            grouped_reference_chunks (list): The reference text chunks relevant to each question.

        Returns:
            tuple: Contains:
                - chatbot_responses (list): The chatbot's answers
                - response_times (list): Time taken for each answer
                - precision_k (dict): Precision@k score
                - rouge_scores (dict): ROUGE metrics
                - bleu_score (dict): BLEU metric (mean and std)
                - avg_response_time (float): Average response time for the set
        """

        chatbot_responses = []
        response_times = []

        # Generate responses for each question
        for q in questions:
            response, time_taken = self.generate_response(q)
            chatbot_responses.append(response)
            response_times.append(time_taken)

        # Calculate precision@k
        precision_k = self.calculate_precision_k(questions, grouped_reference_chunks, top_k=3)
        # Calculate ROUGE scores
        rouge_scores = self.calculate_rouge_scores(chatbot_responses, reference_responses)
        # Calculate BLEU scores
        bleu_score = self.calculate_bleu_scores(chatbot_responses, reference_responses)
        # Calculate average response time
        avg_response_time = np.mean(response_times)

        return chatbot_responses, response_times, precision_k, rouge_scores, bleu_score, avg_response_time

    def calculate_precision_k(self, questions, grouped_reference_chunks, top_k=3):
        """
        Calculates Precision@k by comparing retrieved chunks to known reference chunks.

        Args:
            questions (list): A list of questions.
            grouped_reference_chunks (list): Each entry holds the reference text chunk for that question.
            top_k (int, optional): The number of retrieved documents to consider. Defaults to 3.

        Returns:
            dict: A dictionary with "mean" key indicating the average precision@k across all questions.
        """
        match = 0

        # Iterate through each question and its corresponding reference chunks
        for i, (question, ref_chunks) in enumerate(zip(questions, grouped_reference_chunks)):

            # Retrieve relevant chunks for the question
            retrieved_docs = self.retriever.get_relevant_documents(question)
            retrived_chunks = ""
            for doc in retrieved_docs[:min(top_k, len(retrieved_docs))]:
                retrived_chunks += doc.page_content.strip().lower().replace("\n", "")

            # Convert the reference chunks into a normalized list of tokens
            ref_chunks_list = [chunk for chunk in ref_chunks.strip().lower().split()]

            # Check if the reference chunks appear in the retrieved text
            if ref_chunks.strip().lower() in retrived_chunks:
                match += 1

        # Calculate the mean of matched cases
        return {"mean": match / len(questions)}

    def calculate_rouge_scores(self, chatbot_responses, reference_responses):
        """
        Calculates ROUGE-1, ROUGE-2, and ROUGE-L scores for the chatbot responses
        against the reference responses.

        Args:
            chatbot_responses (list): A list of chatbot-generated answers.
            reference_responses (list): A list of reference answers (ground truth).

        Returns:
            dict: A dictionary containing the mean and standard deviation for precision,
                  recall, and F1 for each ROUGE metric.
        """

        # Initialize a Rouge evaluator
        rouge_evaluator = Rouge()
        metrics = {"rouge-1": [], "rouge-2": [], "rouge-l": []}

        # Calculate ROUGE scores for each (hypothesis, reference) pair
        for hyp, ref in zip(chatbot_responses, reference_responses):
            if hyp and ref:
                scores = rouge_evaluator.get_scores(
                    hyp.strip().lower(),
                    ref.strip().lower(),
                    avg=False
                )

                # Collect metrics for each ROUGE type
                for metric in metrics.keys():
                    metrics[metric].append(scores[0][metric])

        # Compute mean and std for each metric
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
        """
        Calculates the BLEU score for each chatbot response compared to a reference answer.
        Uses the NLTK implementation of BLEU with a smoothing function.

        Args:
            chatbot_responses (list): Chatbot's predicted responses.
            reference_responses (list): Reference ground-truth answers.

        Returns:
            dict: A dictionary containing the mean and std of BLEU scores.
        """
        scores = []
        for hyp, ref in zip(chatbot_responses, reference_responses):
            if hyp and ref:
                # Convert text to lowercase tokens
                score = sentence_bleu(
                    [ref.strip().lower().split()],
                    hyp.strip().lower().split(),
                    smoothing_function=SmoothingFunction().method4
                )
                scores.append(score)

        return {"mean": np.mean(scores), "std": np.std(scores)}


if __name__ == "__main__":
    # Example: reading from an Excel file that contains questions,
    # reference answers, and reference paragraphs.
    qna_df = pd.read_excel(r"documents/Questions_Answers_ContainingParagraph.xlsx")
    questions = qna_df["Question"].tolist()
    reference_responses = qna_df["Answer"].tolist()
    grouped_reference_chunks = qna_df["Containing Paragraph from the Document"].tolist()

    # Paths to PDF documents that will form the knowledge base
    pdf_paths = [
        r"documents/METU_Regulation.pdf",
        r"documents/ISStudentGuide_2023-2024_v1.5.pdf"
    ]

    # Example chat history between a user and the assistant
    example_chat_history = [
        {
            'content': 'Who is Tuğba Taşkaya Temizel?',
            'role': 'user'
        },
        {
            'content': (
                "Tuğba Taşkaya Temizel is a Professor. Her room number is A-211, "
                "phone number is 7782, and email address is ttemizel[at]metu.edu.tr.",
                8.58181118965149
            ),
            'role': 'assistant'
        }
    ]

    # Instantiate Ragify with your desired parameters
    rag_pipeline = Ragify(
        pdf_paths=pdf_paths,
        llm_name="llama3.2:1b",
        embedding_name="nomic-embed-text",
        chunk_size=1000
    )

    # Generate a response based on a question and chat history
    response, time_taken = rag_pipeline.generate_response(
        "What is her email address?",
        chat_history=example_chat_history
    )
    print("Response:", response)
    print("Time taken:", time_taken)

    # The following lines show how you could try different configurations (commented out):
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