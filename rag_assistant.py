from utils import load_json
from llm_client import run_llm
from database import VectorDB
import time

class RAGAssistant:
    def __init__(self, retriever: VectorDB, config_path="config.json", prompts_path="prompt_chains.json"):
        self.retriever = retriever
        self.config = load_json(config_path)
        self.prompt_chain = load_json(prompts_path)["medical_rag_chain"]

    def serial_chain_workflow(self, input_query: str):
        context = "\n\n".join(self.retriever.retrieve(input_query))
        response_chain = []
        response = f"Context:\n{context}\nQuestion:\n{input_query}"

        for step_prompt in self.prompt_chain:
            combined_prompt = f"{step_prompt}\nInput:\n{response}"
            response = run_llm(combined_prompt, model=self.config["together_model"])
            response_chain.append(response)
            time.sleep(3)  # Prevent rate limit

        return response_chain

    def generate(self, query: str):
        responses = self.serial_chain_workflow(query)
        return responses[-1]  # Final answer
