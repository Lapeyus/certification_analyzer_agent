 
import asyncio
import time
import json
import traceback
import os
import httpx
from dotenv import load_dotenv

from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.agents.llm_agent import LlmAgent
from google.adk.sessions import InMemorySessionService
from google.adk.sessions import DatabaseSessionService
from google.adk.models.lite_llm import LiteLlm


from google.adk.runners import Runner
from google.adk.tools import google_search
# import google.generativeai as genai

# --- CONFIGURE YOUR API KEYS HERE ---
load_dotenv()
# genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# --- Constants ---
# MODEL_NAME = "gemini-1.5-flash"
# MODEL_NAME = "gemini-2.0-flash"
MODEL_NAME = "gemini-2.5-pro"

# --- Helper Classes and Functions ---
class Part:
    def __init__(self, text):
        self.text = text

class Content:
    def __init__(self, role, parts):
        self.role = role
        self.parts = parts

def clean_and_parse_json(text_output: str) -> dict | None:
    """Cleans markdown fences and attempts to parse JSON."""
    if not text_output:
        return None
    cleaned_output_text = text_output.strip()
    if cleaned_output_text.startswith("```json"):
        cleaned_output_text = cleaned_output_text[len("```json"):].strip()
    if cleaned_output_text.startswith("```"):
         cleaned_output_text = cleaned_output_text[len("```"):].strip()
    if cleaned_output_text.endswith("```"):
        cleaned_output_text = cleaned_output_text[:-len("```")].strip()
    try:
        return json.loads(cleaned_output_text)
    except json.JSONDecodeError:
        start_index = cleaned_output_text.find('{')
        end_index = cleaned_output_text.rfind('}')
        if start_index != -1 and end_index != -1 and end_index > start_index:
            potential_json = cleaned_output_text[start_index : end_index + 1]
            try:
                return json.loads(potential_json)
            except json.JSONDecodeError:
                print(f"\n(Warning: Output could not be parsed as JSON after extensive cleaning. Attempted: '{potential_json}')")
                return None
        else:
             print(f"\n(Warning: Output could not be parsed as JSON, invalid structure. Started with: '{cleaned_output_text[:100]}...')")
             return None
    except Exception as parse_err:
        print(f"\n(Warning: Unexpected error parsing JSON: {parse_err})")
        return None

# --- Agent Definitions ---

# Agent 1: Extract Claims for Fact-Checking
extract_claims_fact_check_agent = LlmAgent(
    name="ExtractClaimsFactCheckAgent",
    description="Identifies and extracts verifiable factual claims from text for the fact-checking process.",
    # model=MODEL_NAME,
    model=LiteLlm(model="openai/magistral:24b"), #gemma3n:e2b
    instruction="""
You are a Claim Identification and Contextualization Specialist, the first step in a fact-checking pipeline. Your task is to meticulously analyze the input text, distinguish verifiable claims from other language, enrich those claims, and suggest how to verify them.

**Your process for analyzing the text:**
1.  **Deconstruct:** Mentally break down the input text into its core statements or semantic chunks.
2.  **Classify Each Statement:** For each statement, determine if it is a 'Verifiable Claim' or an 'Ignored Statement'.
    *   **Verifiable Claim:** A statement that can be proven true or false with objective evidence. This includes statistics, specific quantities, dates, events, and factual assertions.
    *   **Ignored Statement:** Anything that is not a verifiable claim. This includes:
        *   **Opinion & Subjective Language:** "This was a great success."
        *   **Future Promises & Intentions:** "We will double this number by 2030."
        *   **Vague Statements & Puffery:** "We are making the country stronger."
        *   **Rhetorical Questions.**
3.  **Contextualize Verifiable Claims:** For every statement you classify as a 'Verifiable Claim', create an enhanced, standalone version for searching. Add essential context (like names, locations, or topics mentioned elsewhere in the text) so it can be understood and verified on its own.
4.  **Create a Verification Guide:** For each verifiable claim, add a brief, actionable guide on how one could verify that specific assertion. Suggest the ideal type of source or data to search for.

**Output Format:**
*   You MUST output ONLY a single, valid JSON object.
*   The object will have two keys: `"verifiable_claims"` and `"ignored_statements"`.
*   `"verifiable_claims"`: A list of objects, where each object has:
    *   `"original_statement"` (string): The exact claim as it appeared in the text.
    *   `"contextualized_claim"` (string): The enriched, standalone version of the claim for searching.
    *   `"verification_guide"` (string): A brief guide on how to fact-check the claim.
*   `"ignored_statements"`: A list of objects, where each object has:
    *   `"statement"` (string): The exact phrase or sentence being ignored.
    *   `"reason"` (string): The category for why it was ignored (e.g., "Opinion", "Future Promise", "Vague Statement").
*   If no verifiable claims are found, `"verifiable_claims"` must be an empty list `[]`.

**Example Input:**
"Under the Chaves administration, Costa Rica has built 200 new schools. This investment has been a great success and we promise to double this number by 2030. We also lowered the national unemployment rate to 5%."

**Example Output:**
```json
{
  "verifiable_claims": [
    {
      "original_statement": "has built 200 new schools",
      "contextualized_claim": "The Chaves administration in Costa Rica has built 200 new schools.",
      "verification_guide": "Check official reports from Costa Rica's Ministry of Public Education for school construction data during the specified administration."
    },
    {
      "original_statement": "lowered the national unemployment rate to 5%",
      "contextualized_claim": "The Chaves administration lowered the national unemployment rate in Costa Rica to 5%.",
      "verification_guide": "Consult the latest reports from the National Institute of Statistics and Census (INEC) of Costa Rica for official unemployment data."
    }
  ],
  "ignored_statements": [
    {
      "statement": "This investment has been a great success",
      "reason": "Opinion"
    },
    {
      "statement": "we promise to double this number by 2030",
      "reason": "Future Promise"
    }
  ]
}
```
    """,
    output_key='claims'
)

# Agent 2: Search for Fact-Checking Evidence
evidence_search_fact_check_agent = LlmAgent(
    name="EvidenceSearchFactCheckAgent",
    description="Gathers and summarizes evidence for each claim as part of the fact-checking process.",
    model=MODEL_NAME,
    tools=[google_search],
    instruction="""
You are an AI Research Analyst. Your purpose is to gather neutral, verifiable evidence for a fact-checking pipeline. You will receive a JSON object from the previous agent containing a list of claims to investigate.

**Your Goal:**
For EACH object in the `verifiable_claims` list from the input, you must find and synthesize evidence.

**Instructions:**
1.  **Formulate a Search Query:**
    *   Use the `contextualized_claim` as the primary source for your search query.
    *   Use the `verification_guide` as a strategic hint to help you target the most reliable sources (e.g., official agencies, statistical reports).
2.  **Execute Search:** Use the `google_search` tool with your formulated query.
3.  **Synthesize Evidence:** Analyze the search results from high-quality sources. Prioritize official data, reports from non-partisan organizations, and reputable news outlets as suggested by the verification guide.
4.  **Summarize Findings:** Write a concise, 1-3 sentence objective summary of the evidence. If the evidence is conflicting or can't be found, state that clearly.
5.  **List Sources:** Provide the direct URLs of the sources that informed your summary.

**Output Format:**
*   You MUST output ONLY a single, valid JSON object.
*   The **keys** of this object MUST be the exact `contextualized_claim` string from the input.
*   The **value** for each key must be a JSON object containing:
    *   `"evidence_summary"` (string)
    *   `"sources"` (a list of URL strings)
    *   `"search_query"` (the query string you used)

**Example Input:**
```json
{
  "verifiable_claims": [
    {
      "original_statement": "has built 200 new schools",
      "contextualized_claim": "The Chaves administration in Costa Rica has built 200 new schools.",
      "verification_guide": "Check official reports from Costa Rica's Ministry of Public Education for school construction data during the specified administration."
    },
    {
      "original_statement": "lowered the national unemployment rate to 5%",
      "contextualized_claim": "The Chaves administration lowered the national unemployment rate in Costa Rica to 5%.",
      "verification_guide": "Consult the latest reports from the National Institute of Statistics and Census (INEC) of Costa Rica for official unemployment data."
    }
  ],
  "ignored_statements": []
}
```

**Example Output:**
```json
{
    "The Chaves administration in Costa Rica has built 200 new schools.": {
        "evidence_summary": "According to the Ministry of Public Education's annual report, 195 new educational infrastructure projects were completed under the Chaves administration. News reports corroborate this number, citing the official ministry data.",
        "sources": ["https://mep.go.cr/anual-report-2023.pdf", "https://news.cr/education-projects-completed"],
        "search_query": "Ministry of Public Education Costa Rica school construction data Chaves administration"
    },
    "The Chaves administration lowered the national unemployment rate in Costa Rica to 5%.": {
        "evidence_summary": "The latest data from the National Institute of Statistics and Census (INEC) shows the open unemployment rate was 7.8%. Historical data shows the rate was higher previously, but did not reach the specific 5% figure.",
        "sources": ["https://inec.cr/employment/quarterly-report", "https://centralbank.cr/economic-indicators/unemployment"],
        "search_query": "Costa Rica unemployment rate INEC Chaves administration"
    }
}
```
""",
    output_key='google_search_results'
)

# Agent 3: Analyze Evidence and Finalize Fact-Check
claim_analysis_fact_check_agent = LlmAgent(
    name="ClaimAnalysisFactCheckAgent",
    # model=MODEL_NAME,
    model=LiteLlm(model="openai/magistral:24b"), #gemma3n:e2b

    instruction="""
You are a Lead Fact-Check Judge, the final, decisive step in the fact-checking pipeline. Your task is to render a final verdict for each claim by synthesizing the initial claim analysis with the evidence gathered by the research analyst.

**Your Inputs:**
1.  **Initial Analysis (`claims_analysis`):** A JSON object containing the `verifiable_claims` list from the first agent.
2.  **Research Findings (`google_search_results`):** A JSON object where keys are the `contextualized_claim` and values are the evidence packages from the second agent.

**Your Thinking Process:**
1.  Iterate through each claim object in the `verifiable_claims` list from the initial analysis.
2.  For each claim, use its `contextualized_claim` string as a key to find the matching evidence package in the research findings.
3.  Carefully compare the `contextualized_claim` against the `evidence_summary` provided in the research findings.
4.  Assign a final `status` based on this strict comparison:
    *   **'Supported'**: The evidence directly and unambiguously confirms the claim.
    *   **'Contradicted'**: The evidence directly and unambiguously refutes the claim.
    *   **'Unsubstantiated'**: The evidence is insufficient, ambiguous, conflicting, or does not address the specific details of the claim.
5.  If the status is 'Contradicted' or 'Unsubstantiated', write a brief, one-sentence `reasoning` explaining *why* the evidence does not support the claim. For 'Supported', the reasoning must be an empty string.

**Output Format:**
*   You MUST output ONLY a valid JSON object with a single key, `"fact_check_results"`.
*   The value will be a list of objects. Each object is a complete report for a single claim and must contain:
    *   `original_statement`: The claim as it first appeared.
    *   `claim`: The full `contextualized_claim` that was investigated.
    *   `status`: Your final verdict ('Supported', 'Contradicted', or 'Unsubstantiated').
    *   `reasoning`: Your explanation (empty string if 'Supported').
    *   `evidence_summary`: The summary provided by the research analyst.
    *   `sources`: The list of source URLs.
    *   `search_query`: The search query used to find the evidence.

**Example Inputs:**
*(Implicitly passed the `claims_analysis` from Agent 1 and `google_search_results` from Agent 2)*

**Example Output:**
```json
{
  "fact_check_results": [
    {
      "original_statement": "has built 200 new schools",
      "claim": "The Chaves administration in Costa Rica has built 200 new schools.",
      "status": "Contradicted",
      "reasoning": "The evidence states that 195 educational infrastructure projects were completed, not 200 schools.",
      "evidence_summary": "According to the Ministry of Public Education's annual report, 195 new educational infrastructure projects were completed under the Chaves administration. News reports corroborate this number, citing the official ministry data.",
      "sources": ["https://mep.go.cr/anual-report-2023.pdf", "https://news.cr/education-projects-completed"],
      "search_query": "Ministry of Public Education Costa Rica school construction data Chaves administration"
    },
    {
      "original_statement": "lowered the national unemployment rate to 5%",
      "claim": "The Chaves administration lowered the national unemployment rate in Costa Rica to 5%.",
      "status": "Unsubstantiated",
      "reasoning": "The evidence shows a drop in unemployment but indicates the most recent figure is 7.8%, not the 5% claimed.",
      "evidence_summary": "The latest data from the National Institute of Statistics and Census (INEC) shows the open unemployment rate was 7.8%. Historical data shows the rate was higher previously, but did not reach the specific 5% figure.",
      "sources": ["https://inec.cr/employment/quarterly-report", "https://centralbank.cr/economic-indicators/unemployment"],
      "search_query": "Costa Rica unemployment rate INEC Chaves administration"
    }
  ]
}
reply in spanish
```
""",
    description="Analyzes the gathered evidence to determine the final fact-check status of each claim.",
    output_key='final_results'
)


# --- Pipeline Definition ---
agent = SequentialAgent(
    name="FactCheckingPipeline",
    sub_agents=[
        extract_claims_fact_check_agent,
        # evidence_search_fact_check_agent,
        claim_analysis_fact_check_agent
    ],
    description="A 3-step pipeline of fact-checking agents that extracts claims, gathers evidence, and analyzes the findings."
)

# --- Runner and Interaction Function ---
APP_NAME_FACTCHECK = "political_factcheck_app"
# session_service = InMemorySessionService()
# Example using a local SQLite file:
db_url = "sqlite:///./my_agent_data.db"
session_service = DatabaseSessionService(db_url=db_url)

try:
    factcheck_runner = Runner(
        agent=agent,
        app_name=APP_NAME_FACTCHECK,
        session_service=session_service
    )
    print("✅ FactCheck Runner initialized successfully.")
except Exception as e:
    print(f"❌ Error initializing FactCheck Runner: {e}")
    traceback.print_exc()
    exit()

async def call_fact_check_pipeline(political_text: str, user_id: str, session_id: str):
    """Runs the political content fact-checking pipeline."""
    print(f"\n>>> Starting Fact-Checking Pipeline for Text:")
    print(f"'''\n{political_text[:500].strip()}...\n'''")
    print(f">>> User: {user_id}, Session: {session_id}")

    session = await session_service.create_session(app_name=APP_NAME_FACTCHECK, user_id=user_id, session_id=session_id)
    print(f"FactCheck session created/retrieved with ID: {session_id}")

    try:
        content = Content(role='user', parts=[Part(text=political_text)])
        print("Running fact-checking pipeline...")
        start_run_time = time.time()
        final_event = None
        async for event in factcheck_runner.run_async(user_id=user_id, session_id=session_id, new_message=content):
            print(f"  ...Event from: {event.author}")
            if event.is_final_response():
                final_event = event
        end_run_time = time.time()
        print(f"Fact-checking pipeline run complete in {end_run_time - start_run_time:.2f} seconds.")

        print("\n--- Final Fact-Checking Results ---")
        if final_event and final_event.content and final_event.content.parts:
            final_output_text = final_event.content.parts[0].text
            print(f"Raw Output (FactChecker):\n{final_output_text}")
            final_json = clean_and_parse_json(final_output_text)
            if final_json and 'fact_check_results' in final_json:
                # Add original text to the final JSON
                final_json['original_text'] = political_text

                # Write the output to a file
                output_filename = "fact_check_results.json"
                with open(output_filename, "w") as f:
                    json.dump(final_json, f, indent=2)
                print(f"\n✅ Results saved to {output_filename}")

                print("\n--- Parsed Fact-Check Status ---")
                results = final_json['fact_check_results']
                if isinstance(results, list):
                    # Resolve redirect URLs before printing
                    async def resolve_url(client: httpx.AsyncClient, url: str) -> str:
                        """Resolves a redirect URL if it's a vertexai search link."""
                        if "vertexaisearch.cloud.google.com" not in url:
                            return url
                        try:
                            resp = await client.head(url, follow_redirects=True, timeout=10)
                            return str(resp.url)
                        except httpx.RequestError as e:
                            print(f"\n(Warning: Could not resolve URL {url}: {e})")
                            return url

                    async with httpx.AsyncClient() as client:
                        update_tasks = []
                        for item in results:
                            if isinstance(item, dict) and 'sources' in item:
                                sources = item.get('sources', [])
                                if isinstance(sources, list):
                                    resolve_tasks = [resolve_url(client, url) for url in sources]
                                    async def update_item_sources(item_to_update, tasks_to_await):
                                        item_to_update['sources'] = await asyncio.gather(*tasks_to_await)
                                    update_tasks.append(update_item_sources(item, resolve_tasks))
                        await asyncio.gather(*update_tasks)

                    if not results:
                        print("(No verifiable claims were extracted from the input text)")
                    for item in results:
                        if isinstance(item, dict):
                            claim = item.get('claim', 'N/A')
                            status = item.get('status', 'N/A')
                            reasoning = item.get('reasoning', '')
                            sources = item.get('sources', [])
                            search_query = item.get('search_query', 'N/A')
                            print(f"- Claim: \"{claim}\"")
                            print(f"  Status: {status}")
                            if reasoning:
                                print(f"  Reasoning: {reasoning}")
                            print(f"  Search Query: \"{search_query}\"")
                            if sources:
                                print("  Sources:")
                                for source in sources:
                                    print(f"    - {source}")
                        else:
                            print(f"  (Error: Result item expected dictionary, got {type(item)})")
                else:
                    print("(Error: 'fact_check_results' key found, but value is not a list)")
            elif final_json:
                print("(Error: Parsed JSON, but missing 'fact_check_results' key)")
        elif final_event and final_event.error_message:
             print(f"❌ Pipeline ended with error: {final_event.error_message}")
        else:
            print("❌ Final FactCheck Output: (No final event captured or content missing)")

    except Exception as e:
        print(f"❌ An error occurred during the fact-checking pipeline execution: {e}")
        traceback.print_exc()

async def main(political_text_to_check):
    """Runs an example for the fact-checking pipeline."""
    user_id = "political_dept_01"
    factcheck_session_id = f"factcheck_run_{time.time()}_{time.monotonic_ns()}"

    await call_fact_check_pipeline(political_text_to_check, user_id, factcheck_session_id)

if __name__ == "__main__":
    political_text_to_check = """
Al tico le gusta trabajar. 

Lo que no le gusta es que le pongan amarras y que otros traten de decidir cuál jornada laboral se le impondrá para llevar el sustento a su casa. 

Es por eso que el proyecto de ley de Jornadas cuatro por tres busca generar más y mejores empleos. 

Administración Chaves Robles República de Costa Rica logro disminuir el desempleo a la cifra histórica de 6,9%, la más baja desde que existen datos y queremos ir por más. 

Esta es una necesidad del país para poder continuar atrayendo inversión extranjera. 

Esto es clave para competir de manera eficaz por atraer y retener inversión extranjera directa. 

Nos parece inconcebible lo que están haciendo varios diputados que desde la comodidad de su curul y con un salario millonario garantizado, no quieren darle a usted la oportunidad de elegir cuál es el mejor horario para trabajar y cuáles son las mejores condiciones. 

Son las personas que quieren tener, como dijo Rodrigo Arias Sánchez a Costa Rica, pequeñita, mansa. 

Y el Frente Amplio pareciera que le agrega pobre como un garbancito en mi mano. 

Insisten en ponerle palos a la carreta del progreso. 

2180 Esa es la cantidad astronómica de mociones que presentó el Frente Amplio al proyecto de ley de jornadas cuatro por tres. 

A ellos los acompañan las diputadas y liberacionistas Moserrath Ruíz, Dinorah Barquero y Sonia Rojas. 

También Vanessa Castro, del PUSC, las Independiente, Kattia Cambronero, Cynthia Córdoba y Gloria Navas, así como LuzMary Alpízar. 

Entre todas y todos ellos presentaron más de 2500 nefastas mociones a un proyecto que solo busca generar más y mejores empleos. 

Reprocho completamente la actitud que asumen esos. 

Eso no, eso es incorrecto, eso es incorrecto. 

El día que eso acabe, empieza a arreglarse este país. 

Compatriotas, vean lo absurdas que son gran cantidad de esas mociones que aunque usted no lo crea, están atrasando la discusión responsable y sensata. 

Solo por cambiar la palabra cuido por cuidado en el texto de la ley. 

Otra de sus grandes preocupaciones es que el texto dice trabajador y ellos quieren cambiarlo por persona trabajadora. 

En el peor de los escenarios, la discusión de las 2500 mociones implicaría que lleguen hasta diciembre en este trámite. 

¿Sabe cuánto le cuesta eso al país? Nada más y nada menos que 17.762.000.000 de colones. 

Plata con la que se podrían construir casi tres colegios como el Liceo de Chacarita o 17 modernas delegaciones policiales. 

Compatriota con ese dinero despilfarrado, gracias a varios diputados podríamos construir dos nuevos pasos a desnivel como el de Cartago o comprar casi 54.000 computadoras para niños y jóvenes estudiantes. 

Necesitamos darle esta oportunidad a estas compañías para que no se queden atrás, para que puedan seguir compitiendo en los mercados internacionales y sigamos siendo tierra fértil para la inversión extranjera directa que viene a generar empleos de calidad en Costa Rica. 

Costarricense no coma cuento. 

Algunos lo asustan con la vaina vacía y le quieren meter ideas en la cabeza que no son ciertas, como por ejemplo que las jornadas cuatro por tres serán obligatorias. 

El trabajador va a tener derecho a decidir si la usa o no la usa, que regirán para todos los trabajos del país. 

La ley claramente establece que será para manufactura, para servicios corporativos que laboren en diferentes husos horarios, para industria médica y para servicios de apoyo de estas actividades. 

Y hasta han inventado que quienes trabajen en esta modalidad ya no se les pagarán horas extra al trabajador de jornada diurna se le pagará un 17% más por hora laborada y al trabajador de jornada nocturna un 25% más por hora laborada. 

El daño que ocasionan las 2500 mociones presentadas es tan grande que atrasarán por varios meses la discusión de otros proyectos importantes como regular por fin el desastre en Crucitas. 

La reforma a la injusta ley de radio y televisión, el proyecto para aumentar las penas de cárcel por sicariato o el préstamo con el Banco Mundial para bajar la deuda y tener más recursos que podríamos invertir en educación, salud y seguridad. 

Le están quitando al trabajador costarricense su derecho a elegir cuál jornada le sirve más. 

Si pudiera escoger tomar una jornada de horario cuatro tres lo tomaría inmediatamente. 

A mí me gustaría trabajar en una jornada cuatro por tres, porque el tiempo que quedan es bastante y queda tiempo para mandados, para lo que uno necesite. 

Señores diputados, ustedes se han dado el gusto de tener tres años trabajando en jornada cuatro por tres, ya que los viernes no sesionan. 

Permítanle al país avanzar y al costarricense elegir la jornada que prefiera. 

Ustedes ya despertaron, ya entendieron que nadie debe ni puede decidir por ustedes. 

Y si el futuro está en manos del pueblo, yo creo que lo mejor está por venir. 

Por cada una y cada uno de nuestros jefes, hoy más que nunca no vamos a aflojar. 
 
    """
    print("🚀 Starting Fact-Checking Agentic Workflow...")
    try:
        asyncio.run(main(political_text_to_check))
    except RuntimeError as e:
         if "cannot run event loop while another loop is running" in str(e):
             print("\n--------------------------------------------------------------------")
             print("INFO: Detected running event loop (e.g., Jupyter/Colab).")
             print("✅ Please execute 'await main()' in a cell directly.")
             print("--------------------------------------------------------------------")
         else:
             raise e
    print("\n✅ Fact-Checking Agentic Workflow finished.")
