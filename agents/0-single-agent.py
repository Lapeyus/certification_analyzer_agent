import asyncio
import time
import os
import sys

from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.cli.utils import logs
from google.adk.runners import InMemoryRunner
from google.adk.sessions import Session
from google.genai import types

 

# # --- CONFIGURATION ---
load_dotenv(override=True)
logs.log_to_tmp_folder()

FILE_PATH = os.environ.get("FILE_PATH")
# GUIDE_FILE_PATH = os.environ.get("GUIDE_FILE_PATH")
# OUTPUT_FILE_PATH = os.environ.get("OUTPUT_FILE_PATH")
# STATE_ANALYZED_JSON = "analyzed_json"
TEXT = "question_payload"

try:
  with open(FILE_PATH, 'r') as f:
    TEXT = f.read()

except FileNotFoundError as e:
  print(f"** ERROR: Could not find a required file. Please check your paths in the CONFIGURATION section.", file=sys.stderr)
  print(f"** Details: {e}", file=sys.stderr)
  sys.exit(1)


# # --- END CONFIGURATION ---

root_agent = Agent(
    name='certification_analyzer_agent',
    
    model="gemini-2.0-flash",
    description=(
        'An agent that analyzes cloud certification questions against an exam'
        ' guide to produce structured JSON study materials.'
    ),
    instruction=f"""
break down the input text into its core statements or semantic chunks.
{TEXT}
""",
    output_key='out'
)

async def main():
  my_app = 'certification_analyzer_agent'
  my_user_id = 'user1'
  runner = InMemoryRunner(
      agent=root_agent,
      app_name=my_app,
  )

   # my_session: Session,prompt_text: str,
  async def run_analysis(text: dict):
    content = types.Content(role='user', parts=[types.Part(text="hi")])
    print(f"\n▶️  Running analysis...")

    my_session = await runner.session_service.create_session(
        app_name=my_app,
        user_id=my_user_id,
        state={
           'text': text
        }
    )

    async for event in runner.run_async(
        user_id=my_user_id,
        session_id=my_session.id,
        new_message=content,
    ):
        if (event.author != "User" and event.content.parts and event.content.parts[0].text):
            print(f"  🧠  Thought from {event.author}: {event.content.parts[0].text.strip()}...")
            sys.stdout.flush()

        if event.content.parts and event.content.parts[0].text:
            response_text = event.content.parts[0].text
            print(response_text)

            
  start_time = time.time()
  print('Start time:', time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(start_time)))
  print('------------------------------------')
 
  await run_analysis(
      TEXT
  )

  # print(
  #     await runner.artifact_service.list_artifact_keys(
  #         app_name=my_app,
  #         user_id=my_user_id,
  #         session_id=my_session.id
  #     )
  # )
  end_time = time.time()
  print('------------------------------------')
  print('End time:', time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(end_time)))
  print('Total time (seconds):', end_time - start_time)


if __name__ == '__main__':
  asyncio.run(main())
