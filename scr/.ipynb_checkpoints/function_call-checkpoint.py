import openai
from haystack.nodes.base import BaseComponent
from typing import List
import json

class OpenAIFunctionCall(BaseComponent):
    outgoing_edges = 1
    def __init__(self, API_KEY):
        self.API_KEY = API_KEY

    def run(self, documents: List[str]):
        
        try:
            document_content_list = [doc.content for doc in documents]
            print("documents extracted")
            document_content = " ".join(document_content_list)
        except Exception as e:
            print("Error extracting content:", e)
            return
        functions = [
            {
                "name": "update_dataframe",
                "description": "write the fund details to a dataframe",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "Product_reference_num": {
                            "type": "string",
                            "description": "The FCA product reference number which will be six or seven digits"
                        },
                        "investment_objective": {
                            "type": "string",
                            "description": "You should return the investment objective of the fund. This is likely to be something like this: The Fund aims to grow your investment over t – t + delta t years"
                        },
                        "investment_policy": {
                            "type": "string",
                            "description": "Return a summary of the fund's investment policy, you will summarise the main points. Maximum 250 words."
                        },
                        "investment_strategy": {
                            "type": "string",
                            "description": "Return a summary of the fund's investment strategy, you will summarise the main points. Maximum 250 words."
                        },
                        "ESG": {
                            "type": "string",
                            "description": "Return either True, or False. True if the fund is an ESG fund, False otherwise."
                        },
                        "fund_name": {
                            "type": "string",
                            "description": "Return the name of the fund"
                        },
                    },
                },
                "required": ["Product_reference_num", 
                             "investment_objective", 
                             "investment_policy",
                             "investment_strategy",
                             "ESG",
                             "fund_name"]
            }
        ]
        
        openai.api_key = self.API_KEY
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "system", "content": document_content}],
            functions=functions,
            function_call="auto",  # auto is default, but we'll be explicit
        )

        function_call_args = json.loads(response["choices"][0]["message"]["function_call"]["arguments"])
        
        return function_call_args, "output_1"

    def run_batch(self, documents: List[str]):
        # You can either process multiple documents in a batch here or simply loop over the run method
        results = []
        for document_content in document_content:
            result, _ = self.run(document_content)
            results.append(result)
        return results, "output_1"