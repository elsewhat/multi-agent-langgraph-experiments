
#Load .env with OPENAI_API_KEY
from dotenv import load_dotenv
load_dotenv()
from langchain_ollama import ChatOllama
import os

#from langchain.globals import set_debug
#set_debug(True)

from langchain_core.utils.function_calling import convert_to_openai_function
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

#For extracting code blocks from llm 
from mdextractor import extract_md_blocks

#From https://github.com/langchain-ai/langchain/issues/21479#issuecomment-2105618237
#model = ChatOllama(
#   api_key="ollama",
#   model="llama3.1:8b-instruct-q8_0",
   #model="mistral-nemo-32kb:12b-instruct-2407-q8_0",
   #host="host.docker.internal:11434",
#)
# model = ChatOpenAI(
#    api_key="ollama",
#    model="llama3.1:8b-instruct-q8_0",
#    #model="mistral-nemo-32kb:12b-instruct-2407-q8_0",
#    base_url="http://host.docker.internal:11434/v1",
# )

#model = ChatOpenAI(model="gpt-4o-mini") 
# See https://python.langchain.com/v0.2/docs/integrations/chat/ollama_functions/
from langchain_experimental.llms.ollama_functions import OllamaFunctions

model = OllamaFunctions(
   api_key="ollama",
   model="llama3.1:8b-instruct-q8_0",
#    #model="mistral-nemo-32kb:12b-instruct-2407-q8_0",
   base_url="http://host.docker.internal:11434",
)



class OraclePLSQLStoredProcedureClassification(BaseModel):
    """Tag the Oracle PLSQL stored procedure based on its functionality"""
    createsData: bool = Field(description="Creates new rows of data through the INSERT keyword")
    selectsData: bool = Field(description="Select rows of data through the SELECT keyword")
    updatesData: bool = Field(description="Updates data through the UPDATE keyword")
    deletesData: bool = Field(description="Updates data through the DELETE keyword")
    tables: str = Field(description="Comma-separated list of database tables the stored procedure interacts with")

tagging_functions = [convert_to_openai_function(OraclePLSQLStoredProcedureClassification)]

prompt_plsql = ChatPromptTemplate.from_messages([
    ("system", "You are an expert Oracle database developer tasked with categorizing advanced Oracle PLSQL stored procedures. You will be provided with a PLSQL stored procedure and must tag it through function calling"),
    ("user", "{oracle_plsql_storedprocedure}")
])

model_plsql_analyzer= model.bind(
    functions=tagging_functions,
    function_call={"name": "OraclePLSQLStoredProcedureClassification"}
)

tagging_chain = prompt_plsql | model_plsql_analyzer

# From https://resources.oreilly.com/examples/0636920024859/-/blob/master/givebonus1.sp?ref_type=heads
ORA_STORED_PROCEDURE_DEFINITION = """CREATE OR REPLACE PROCEDURE give_bonus (
   dept_in IN employees.department_id%TYPE,
   bonus_in IN NUMBER)
/*
|| Give the same bonus to each employee in the
|| specified department, but only if they have
|| been with the company for at least 6 months.
*/
IS
   l_name VARCHAR2(50);

   CURSOR by_dept_cur 
   IS
      SELECT *
        FROM employees
       WHERE department_id = dept_in;

   fdbk INTEGER;
BEGIN
   /* Retrieve all information for the specified department. */
   SELECT department_name
     INTO l_name
     FROM departments
    WHERE department_id = dept_in;

   /* Make sure the department ID was valid. */
   IF l_name IS NULL
   THEN
      DBMS_OUTPUT.PUT_LINE (
         'Invalid department ID specified: ' || dept_in);   
   ELSE
      /* Display the header. */
      DBMS_OUTPUT.PUT_LINE (
         'Applying Bonuses of ' || bonus_in || 
         ' to the ' || l_name || ' Department');
   END IF;

   /* For each employee in the specified department... */
   FOR rec IN by_dept_cur
   LOOP
      -- Function in rules package (rp) determines if
	  -- employee should get a bonus. 
	  -- Note: this program is NOT IMPLEMENTED! 
      IF employee_rp.eligible_for_bonus (rec)  
      THEN
         /* Update this column. */

         UPDATE employees
            SET salary = rec.salary + bonus_in
          WHERE employee_id = rec.employee_id;

         /* Make sure the update was successful. */
         IF SQL%ROWCOUNT = 1
         THEN
            DBMS_OUTPUT.PUT_LINE (
               '* Bonus applied to ' ||
               rec.last_name); 
         ELSE
            DBMS_OUTPUT.PUT_LINE (
               '* Unable to apply bonus to ' ||
               rec.last_name); 
         END IF;
      END IF;
   END LOOP;
END;
/

"""
result = tagging_chain.invoke({"oracle_plsql_storedprocedure": ORA_STORED_PROCEDURE_DEFINITION})

print(result)
#Actual output
# content='' id='run-02e35817-8d21-430b-a2b4-0c564490bff9-0' tool_calls=[{'name': 'OraclePLSQLStoredProcedureClassification', 'args': {'createsData': False, 'selectsData': True, 'updatesData': True, 'deletesData': False, 'tables': 'employees, departments'}, 'id': 'call_9d490bb24d4d45d497f4f2f4be68952b', 'type': 'tool_call'}]