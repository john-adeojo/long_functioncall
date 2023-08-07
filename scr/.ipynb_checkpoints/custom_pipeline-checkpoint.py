from function_call import OpenAIFunctionCall

def create_pipeline(retriever):
    p = Pipeline()
    p.add_node(component=retriever, name="retriever", inputs=["Query"])
    p.add_node(component=OpenAIFunctionCall(), name="OpenAIFunctionCall", inputs=["retriever"])
    return p