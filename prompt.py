
from typing import List
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

class AgendaItem(BaseModel):
    name: str = Field(..., description="Agenda item name")
    points: List[str] = Field(..., description="Discussion points for this agenda")


class ActionItem(BaseModel):
    task: str = Field(..., description="Description of the action item")


class Summarize(BaseModel):
    summary: str = Field(..., description="Concise meeting summary (~100 words)")


class SummarizeResponse(BaseModel):
    summary: str = Field(..., description="Concise meeting summary (~100 words)")
    agendas: List[AgendaItem] = Field(..., description="List of agendas with discussion points and at least 3")
    action_items: List[ActionItem] = Field(..., description="List of to-do items from the meeting")
    
    
    
    
parser=PydanticOutputParser(pydantic_object=SummarizeResponse)



SUMMARIZE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "human",
            """You are a highly analytical assistant that summarizes meeting transcripts into a structured format.

Please provide your response in **{language}**.

Your summary must contain the following three sections:

1. **General Summary**: A concise paragraph describing the main topics discussed and key decisions made.
2. **Agendas**: A list of agenda items, each with bullet points highlighting the core discussion points.
3. **Action Items**: A bullet-point list of specific, actionable tasks or follow-ups agreed upon during the meeting.

Use clear and formal language. Structure the output using Markdown formatting.

Meeting Transcript:
{context}"""
        ),
    ]
)

REFINE_TEMPLATE = """
You are refining a previously generated summary using new meeting content.

Please respond in **{language}**.

### Original Summary:
{existing_answer}

### Additional Transcript:
------------------------
{context}
------------------------

### Instructions:
Update and improve the existing summary by incorporating the new information. 
Ensure that the final summary remains structured into:
1. General Summary
2. Agendas
3. Action Items

If any corrections are needed in the original summary, fix them as well.
Use bullet points where appropriate.
"""
REFINE_PROMPT = ChatPromptTemplate.from_messages([("human", REFINE_TEMPLATE)])

PARSER_PROMPT_TEMPLATE = """
You are an intelligent assistant. Your task is to extract structured insights from a meeting summary.

### Language:
Output must be in **{language}**.

### Summary to Analyze:
------------------------
{summary}
------------------------

### Your Task:
Parse the above summary and return a **JSON** object matching the following schema:
{format_instructions}

### Important Guidelines:
- Only return a valid JSON object.
- Do **not** include any explanations, comments, or surrounding text.
- If any value is not present in the summary, return an empty string ("").
- Ensure all field names match the format exactly.

Strictly output JSON only.
"""

PARSER_PROMPT = PromptTemplate(
    template=PARSER_PROMPT_TEMPLATE,
    input_variables=["summary", "language"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)
