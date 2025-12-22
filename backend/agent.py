# # from langchain_classic.agents import create_react_agent, AgentExecutor
# from langchain.agents import create_agent
# from langchain_classic.agents import AgentExecutor
# from langchain_classic.prompts import PromptTemplate
# from tools import knowledge_base_search
# from langchain_huggingface import HuggingFaceEndpoint
# from dotenv import load_dotenv
# import os
# load_dotenv()

# class SupportAgent:
#     def __init__(self):
#         self.llm = HuggingFaceEndpoint(
#             repo_id="meta-llama/Llama-3.1-8B-Instruct",
#             huggingfacehub_api_token=os.getenv("HF_API_TOKEN"),
#             temperature=0.1,
#             max_new_tokens=200
#             # task="conversational"
#         )
        
#         self.tools = [knowledge_base_search]
        
#         # ReAct prompt
#         react_prompt = PromptTemplate(
#     template="""
#         Answer the following questions as best you can. You have access to the following tools:

#         {tools}

#         Use the following format EXACTLY:

#         Question: the input question you must answer
#         Thought: reasoning
#         Action: one of [{tool_names}]
#         Action Input: string
#         Observation: tool result
#         Thought: final reasoning
#         Final Answer: answer

#         Begin!

#         Question: {input}
#         {agent_scratchpad}
#         """,
#             input_variables=[
#                 "input",
#                 "tools",
#                 "tool_names",
#                 "agent_scratchpad"
#             ]
# )

        
#         self.agent = create_agent(
#             model=self.llm,
#             tools=self.tools,
#             system_prompt=react_prompt
#         )
        
#         self.agent_executor = AgentExecutor(
#             agent=self.agent,
#             tools=self.tools,
#             verbose=True,  # Shows the agent's thought process
#             max_iterations=3,
#             handle_parsing_errors=False
#         )
    
#     def run(self, question: str) -> dict:
#         try:
#             result = self.agent_executor.invoke({"input": question})
#             return {
#                 "answer": result["output"],
#                 "status": "success"
#             }
#         except Exception as e:
#             return {
#                 "answer": "I'm sorry, I couldn't process that request.",
#                 "status": "error",
#                 "error": str(e)
#             }

# # Test the agent
# if __name__ == "__main__":
#     agent = SupportAgent()
    
#     print("Testing agent...")
#     result = agent.run("How do I reset my coffee maker?")
#     print(f"Answer: {result['answer']}")









from langchain_classic.agents import create_react_agent
from langchain_classic.agents import AgentExecutor
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_classic import hub 
from langchain_classic.prompts import PromptTemplate
from tools import knowledge_base_search
from dotenv import load_dotenv
import os

load_dotenv()
# meta-llama/Llama-3.1-8B-Instruct
class SupportAgent:
    def __init__(self):
        # Use HuggingFaceEndpoint directly (not ChatHuggingFace)
        self.llm = HuggingFaceEndpoint(
            repo_id="meta-llama/Llama-3.1-8B-Instruct",
            huggingfacehub_api_token=os.getenv("HF_API_TOKEN"),
            temperature=0.1,
            max_new_tokens=512,
            stop_sequences=["Observation:", "Observation", "\nQuestion:"]
        )
        self.llm = ChatHuggingFace(llm=self.llm)
        self.tools = [knowledge_base_search]

        # Simplified ReAct Prompt (Works better for Llama-3.1 in 2025)
        template = """Answer the following questions as best you can. You have access to the following tools:

                    {tools}

                    Use the following format:

                    Question: the input question you must answer
                    Thought: you should always think about what to do
                    Action: the action to take, should be one of [{tool_names}]
                    Action Input: the input to the action
                    Observation: the result of the action
                    ... (this Thought/Action/Action Input/Observation can repeat N times)
                    Thought: I now know the final answer
                    Final Answer: the final answer to the original input question

                    Begin!

                    Question: {input}
                    Thought: {agent_scratchpad}"""

        custom_prompt = PromptTemplate.from_template(template)

        
        # Pull a standard ReAct prompt from the hub that Llama-3.1 understands
        # This prompt explicitly handles the "Thought/Action/Action Input" loop
        # prompt = hub.pull("hwchase17/react")

        # Use create_react_agent which is compatible with standard LLM endpoints
        self.agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=custom_prompt
        )
        
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True, # Critical for 400-error prevention
            max_iterations=3,
            early_stopping_method="force"
        )
    
    def run(self, question: str) -> dict:
        # try:
        result = self.agent_executor.invoke({"input": question})
        return {
            "answer": result["output"],
            "status": "success"
        }
        # except Exception as e:
        #     return {
        #         "answer": "I'm sorry, I couldn't process that request.",
        #         "status": "error",
        #         "error": str(e)
        #     }


if __name__ == "__main__":
    agent = SupportAgent()
    print("Testing agent...")
    result = agent.run("How do I reset my coffee maker?")
    print(f"Answer: {result['answer']}")
        