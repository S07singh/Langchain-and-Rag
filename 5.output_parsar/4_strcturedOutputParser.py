from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task='text-generation'
)

model = ChatHuggingFace(llm=llm)

schema = [
    ResponseSchema(name="fact_1", description="Fact 1 about the topic"),
    ResponseSchema(name="fact_2", description="Fact 2 about the topic"),
    ResponseSchema(name="fact_3", description="Fact 3 about the topic")
    ]
parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template='Give me 3 facts about the topic {topic} \n {format_instructions}',
    input_variables=["topic"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# prompt = template.format(topic="Black Holes")
# prompt = template.invoke({"topic": "Black Holes"}) # we can also use invoke to format the prompt
# result = model.invoke(prompt)
# parsed_result = parser.parse(result.content)
# print(parsed_result)  

# Instead of invoking the model directly, we can use a chain to handle the parsing automatically.
# This is a more general approach that can be used with any model and parser.
chain = template | model | parser
result = chain.invoke({"topic": "Black Holes"})
print(result)  # This will print the parsed output as a dictionary

# limitations of structured output parser:
#  we can validate the data type of variables. for validation we need to use pydantic.