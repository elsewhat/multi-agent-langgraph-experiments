
#Load .env with OPENAI_API_KEY
from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List
import operator
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage, ChatMessage
from langchain_openai import ChatOpenAI

#For extracting code blocks from llm 
from mdextractor import extract_md_blocks

#From https://github.com/langchain-ai/langchain/issues/21479#issuecomment-2105618237
model = ChatOpenAI(
    api_key="ollama",
    #model="llama3.1:8b-instruct-q8_0",
    model="mistral-nemo-32kb:12b-instruct-2407-q8_0",
    base_url="http://host.docker.internal:11434/v1",
)

# model = ChatOpenAI(model="gpt-4o-mini") 

# The state to use in the graph
# ora_tables_definition will be initialized when the graph is started
class AgentState(TypedDict):
    ora_tables_definition: str
    plan_output: str
    draft_ouput: str
    draft_output_code_block: str
    feedback_output: str
    revision_number: int
    max_revisions: int 

PLAN_PROMPT = """You are an expert technical architect working on modernizing enterprise sofware for bowling. \
Create a plan for how to convert the provided Oracle table definitions into OpenAPI 3.1 components in YAML \
Do not provide any YAML in the response!
"""

CONVERT_PROMPT = """You are an expert technical architect working on modernizing enterprise sofware for bowling. \
You are given a set of Oracle table definitions in the system prompt and will be given a plan in the user prompt.\
Only output YAML code representing the OpenAPI 3.1 components.
Start the response with ```yml and end with ``` \
---
{ora_tables_definition}
"""

REFINE_SYSTEM_PROMPT = """You are an expert developer working on modernizing enterprise sofware for bowling. 
You will be given .
1. An OpenAPI specification to improve through revision
2. Validation output of this OpenAPI specifiction (from stoplight spectral)

Your goal and only output should be to improve the OpenAPI specification. Only output YAML code representing the OpenAPI 3.1 
Start the response with ```yml and end with ```
---OpenAPI specification
{openapi_specification}
"""

REFINE_USER_PROMPT = """
OpenAPI validation results
---
{openapi_validation}
"""

def node_plan(state: AgentState):
    messages = [
        SystemMessage(content=PLAN_PROMPT), 
        HumanMessage(content=state['ora_tables_definition'])
    ]
    response = model.invoke(messages)
    return {"plan_output": response.content}

def node_generate_openapi_components(state: AgentState):
    user_message = HumanMessage(
        content=f"Generate OpenAPI 3.1 from this plan\n\n{state['plan_output']}")
    messages = [
        SystemMessage(
            content=CONVERT_PROMPT.format(ora_tables_definition=state['ora_tables_definition'])
        ),
        user_message
        ]
    response = model.invoke(messages)
    return {
        "draft_ouput": response.content, 
        "draft_output_code_block": extract_md_blocks(response.content)[0],
    }


def node_feedback(state: AgentState):
    # Write draft_code_block from earlier stage
    with open('gen_openapi.yaml', 'w') as file:
        file.write(state['draft_output_code_block'])


    from langchain_experimental.llm_bash.bash import BashProcess
    from langchain_community.tools import ShellTool

    shell_tool = ShellTool(ask_human_input=False)
    validationOutput = shell_tool.run({"commands": ["spectral lint gen_openapi.yaml"]})
    return {
        "feedback_output": validationOutput
    }

def node_refine(state: AgentState):
    messages = [
        SystemMessage(
            content=REFINE_SYSTEM_PROMPT.format(openapi_specification=state['draft_output_code_block'])
        ),
        HumanMessage(
            content=REFINE_USER_PROMPT.format(openapi_validation=state['feedback_output'])
        )
        ]
    response = model.invoke(messages)

    if len(extract_md_blocks(response.content)) == 0:
        print(f"Refinement failed to produce any code blocks {response.content}")
        return {
            "revision_number": state.get("revision_number", 1) + 1
        }        
    else:
        return {
            "draft_output": response.content, 
            "draft_output_code_block": extract_md_blocks(response.content)[0],
            "revision_number": state.get("revision_number", 1) + 1
        }



def should_continue(state):
    if state["revision_number"] > state["max_revisions"]:
        return END
    elif "No results with a severity of 'error' found!" in state["feedback_output"]:
        return END
    else:
        return "refine"


builder = StateGraph(AgentState)

builder.add_node("plan", node_plan)
builder.add_node("draft", node_generate_openapi_components)
builder.add_node("feedback", node_feedback)
builder.add_node("refine", node_refine)

builder.set_entry_point("plan")
builder.add_edge("plan", "draft")
builder.add_edge("draft", "feedback")
#builder.add_edge("feedback", "refine")
builder.add_conditional_edges(
    "feedback", 
    should_continue, 
    {END: END, "refine": "refine"}
)
builder.add_edge("refine", "feedback")
graph = builder.compile()

# From https://resources.oreilly.com/examples/0636920024859/blob/master/bowlerama_tables.sql
ORA_TABLES_DEFINITION = """/*-- bowlerama_tables.sql */

CREATE TABLE frame
(
   bowler_id      NUMBER
 , game_id        NUMBER
 , frame_number   NUMBER
 , strike         VARCHAR2 (1) DEFAULT 'N'
 , spare          VARCHAR2 (1) DEFAULT 'N'
 , score          NUMBER
 , CONSTRAINT frame_pk PRIMARY KEY (bowler_id, game_id, frame_number)
);

CREATE TABLE frame_audit
(
   bowler_id      NUMBER
 , game_id        NUMBER
 , frame_number   NUMBER
 , old_strike     VARCHAR2 (1)
 , new_strike     VARCHAR2 (1)
 , old_spare      VARCHAR2 (1)
 , new_spare      VARCHAR2 (1)
 , old_score      NUMBER
 , new_score      NUMBER
 , change_date    DATE
 , operation      VARCHAR2 (6)
);
"""

for event in graph.stream({
    'ora_tables_definition': ORA_TABLES_DEFINITION,
    "max_revisions": 10,
    "revision_number": 1,
}, stream_mode="updates"):
    print(event)


# print(f"Plan:\n{result['plan_output']}")
#print(f"Draft:\n{result['draft']}")
# print(f"Draft:\n{result['draft_output_code_block']}")
#print(f"Latest feedback:\n{result['feedback_output']}")
# with open('gen_openapi.yaml', 'w') as file:
#     file.write(result['draft_code_block'])


# from langchain_experimental.llm_bash.bash import BashProcess
# from langchain_community.tools import ShellTool

# shell_tool = ShellTool(ask_human_input=False)
# validationOutput = shell_tool.run({"commands": ["spectral lint gen_openapi.yaml"]})

# print(validationOutput)