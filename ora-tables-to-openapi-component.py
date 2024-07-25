
#Load .env with OPENAI_API_KEY
from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List
import operator
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage, ChatMessage
from langchain_openai import ChatOpenAI



#From https://github.com/langchain-ai/langchain/issues/21479#issuecomment-2105618237
model = ChatOpenAI(
    api_key="ollama",
    #model="llama3.1:8b-instruct-q8_0",
    model="mistral-nemo:12b-instruct-2407-q8_0",
    base_url="http://host.docker.internal:11434/v1",
)

# model = ChatOpenAI(model="gpt-4o-mini") 

# The state to use in the graph
# ora_tables_definition will be initialized when the graph is started
class AgentState(TypedDict):
    ora_tables_definition: str
    plan: str
    draft: str
    feedback: str
    revision_number: int 

PLAN_PROMPT = """You are an expert technical architect working on modernizing enterprise sofware for bowling \
Create a plan for how to convert the provided Oracle table definitions into OpenAPI 3.1 components in YAML \
Do not provide any YAML in the response!
"""

CONVERT_PROMPT = """You are an expert technical architect working on modernizing enterprise sofware for bowling \
You are given a set of Oracle table definitions in the system prompt and will be given a plan in the user prompt.\
Only output YAML code representing the OpenAPI 3.1 components.
Start the response with ```yml and end with ``` \
---
{ora_tables_definition}
"""

def node_plan(state: AgentState):
    messages = [
        SystemMessage(content=PLAN_PROMPT), 
        HumanMessage(content=state['ora_tables_definition'])
    ]
    response = model.invoke(messages)
    return {"plan": response.content}

def node_generate_openapi_components(state: AgentState):
    user_message = HumanMessage(
        content=f"Here is my plan:\n\n{state['plan']}")
    messages = [
        SystemMessage(
            content=CONVERT_PROMPT.format(ora_tables_definition=state['ora_tables_definition'])
        ),
        user_message
        ]
    response = model.invoke(messages)
    return {
        "draft": response.content, 
        "revision_number": state.get("revision_number", 1) + 1
    }

def should_continue(state):
    if state["revision_number"] > state["max_revisions"]:
        return END
    return "reflect"

builder = StateGraph(AgentState)


builder.add_node("planner", node_plan)
builder.add_node("generate", node_generate_openapi_components)

builder.set_entry_point("planner")
builder.add_edge("planner", "generate")
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

result=graph.invoke({
    'ora_tables_definition': ORA_TABLES_DEFINITION,
    "max_revisions": 2,
    "revision_number": 1,
})
print(f"Plan:\n{result['plan']}")
print(f"Draft:\n{result['draft']}")


with open('gen_openapi.yaml', 'w') as file:
    file.write(result['draft'])


from langchain_experimental.llm_bash.bash import BashProcess
from langchain_community.tools import ShellTool

shell_tool = ShellTool(ask_human_input=False)
validationOutput = shell_tool.run({"commands": ["spectral lint gen_openapi.yaml"]})

print(validationOutput)