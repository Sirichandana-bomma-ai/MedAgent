class MemoryAgent:
    def __init__(self):
        self.history = []

    def add_interaction(self, question, answer):
        self.history.append({"question": question, "answer": answer})

    def get_context(self):
        return "\n".join([f"Q: {h['question']} A: {h['answer']}" for h in self.history])
