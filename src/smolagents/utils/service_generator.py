import argparse
import json
import os
import shutil
from pathlib import Path

MAIN_PY_TEMPLATE = """\
# main.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any, AsyncGenerator, Union
import json
import logging
import os # Added os import

# Import MultiStepAgent. Ensure smolagents is installed in the environment.
try:
    from smolagents import MultiStepAgent
except ImportError:
    # This is a fallback for local testing if smolagents is not in the Python path
    # but is available in a subfolder (e.g., when testing the generator script itself).
    # In a Docker container, smolagents should be installed via requirements.txt.
    import sys
    # Adjust the path if main.py is not in the root of the project during local test
    # For example, if your smolagents source is in a 'src' folder:
    # sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src')) 
    # Or more generally, adjust based on your project structure for local testing.
    # This path adjustment is primarily for when running main.py directly for testing,
    # not for the Docker container where smolagents is expected to be in PYTHONPATH.
    sys.path.append(".") 
    from smolagents import MultiStepAgent


# --- Configuration ---
AGENT_FOLDER_PATH = "./agent_data"  # This path will be inside the Docker container
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# --- Logging Setup ---
# Uvicorn typically handles logging configuration.
# If running standalone, basicConfig might be useful.
# logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger("uvicorn.error") # Aligns with Uvicorn's default logger
logger.setLevel(LOG_LEVEL)


# --- Pydantic Models ---
class AgentRunRequest(BaseModel):
    task: str
    stream: Optional[bool] = False
    additional_args: Optional[Dict[str, Any]] = None

class AgentRunResponse(BaseModel):
    output: Any
    error: Optional[str] = None

# --- FastAPI App ---
app = FastAPI()
agent: Optional[MultiStepAgent] = None

@app.on_event("startup")
async def startup_event():
    global agent
    try:
        logger.info(f"Attempting to load agent from: {AGENT_FOLDER_PATH}")
        agent = MultiStepAgent.from_folder(AGENT_FOLDER_PATH)
        logger.info(f"Agent loaded successfully from {AGENT_FOLDER_PATH}")
    except Exception as e:
        logger.error(f"Error loading agent from {AGENT_FOLDER_PATH}: {e}", exc_info=True)
        agent = None

async def stream_agent_output_generator(agent_instance: MultiStepAgent, task: str, additional_args: Optional[Dict[str, Any]]) -> AsyncGenerator[str, None]:
    try:
        logger.debug(f"Streaming agent output for task: '{task}'")
        # agent_instance.run with stream=True is expected to be a synchronous generator
        for chunk in agent_instance.run(task=task, stream=True, additional_args=additional_args, reset=True):
            if isinstance(chunk, (dict, list)): 
                yield f"data: {json.dumps(chunk)}\\n\\n"
            else:
                try:
                    yield f"data: {json.dumps({'output': chunk})}\\n\\n"
                except TypeError: 
                    yield f"data: {json.dumps({'output': str(chunk)})}\\n\\n"
        yield f"data: {json.dumps({'event': 'done'})}\\n\\n"
        logger.debug(f"Finished streaming agent output for task: '{task}'")
    except Exception as e:
        logger.error(f"Error during agent run (streaming): {e}", exc_info=True)
        yield f"data: {json.dumps({'error': str(e), 'error_type': type(e).__name__})}\\n\\n"

@app.post("/run") 
async def run_agent_task(request: AgentRunRequest):
    global agent
    if agent is None:
        logger.error("Agent not loaded or failed to load. Returning 503.")
        raise HTTPException(status_code=503, detail="Agent not loaded or failed to load.")

    logger.info(f"Received request to /run: task='{request.task}', stream={request.stream}")
    try:
        if request.stream:
            logger.debug("Streaming response requested.")
            return StreamingResponse(
                stream_agent_output_generator(agent, request.task, request.additional_args), 
                media_type="text/event-stream"
            )
        else:
            logger.debug("Non-streaming response requested.")
            # If agent.run is an async def, it should be awaited.
            # If it's synchronous, FastAPI runs it in a thread pool.
            # Assuming MultiStepAgent.run is synchronous for non-streaming:
            result = agent.run(task=request.task, stream=False, additional_args=request.additional_args, reset=True)
            logger.info(f"Agent run completed. Output type: {type(result)}")
            return AgentRunResponse(output=result)
    except Exception as e:
        logger.error(f"Error during agent run: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    global agent
    status = "loaded" if agent else "not loaded"
    return {"message": f"Smolagent FastAPI Wrapper. Agent status: {status}"}

# For local debugging of the generated main.py:
# if __name__ == "__main__":
#     import uvicorn
#     # Ensure 'agent_data' directory is relative to this main.py when running locally.
#     uvicorn.run(app, host="0.0.0.0", port=8000)
"""

DOCKERFILE_TEMPLATE = """\
# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Copy the FastAPI application code
COPY main.py .
COPY requirements.txt .

# Copy the agent data into a specific directory in the container
# The content of the host's agent_folder_path will be copied into /app/agent_data
COPY ./agent_data ./agent_data

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 8000
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
"""

def generate_mcp_manifest(agent_name: str, output_dir: Path, agent_requirements: list[str]):
    docker_image_name = agent_name.lower().replace("-", "_").replace(" ", "_") + ":latest"
    
    mcp_manifest = {
      "mcp_version": "0.1.0",
      "tool_name": agent_name,
      "description": "A smolagent exposed via a FastAPI service.",
      "protocol_details": {
        "type": "http_api",
        "base_url": "http://localhost:8000", # Default, user might need to change if not local Docker
        "endpoints": [
          {
            "path": "/run",
            "method": "POST",
            "description": "Runs the agent with a given task.",
            "request_schema": {
              "type": "object",
              "properties": {
                "task": {"type": "string", "description": "The task for the agent."},
                "stream": {"type": "boolean", "description": "Whether to stream the response.", "default": False},
                "additional_args": {"type": "object", "description": "Additional arguments for the agent's run method.", "additionalProperties": True}
              },
              "required": ["task"]
            },
            "response_schema": {
              "description": "The agent's output. If stream=true, this will be a Server-Sent Events stream. Otherwise, a JSON object with an 'output' field.",
              "content_types": ["application/json", "text/event-stream"]
            }
          }
        ]
      },
      "execution_environment": {
        "type": "docker",
        "image": docker_image_name,
        "requirements_file": "requirements.txt" # Points to the requirements within the Docker context
      },
      "dependencies": {
        # These are agent-specific python packages beyond the FastAPI app's base_requirements
        "python": sorted(list(set(agent_requirements))) 
      }
    }

    mcp_manifest_path = output_dir / "mcp_tool.json"
    with open(mcp_manifest_path, "w", encoding="utf-8") as f:
        json.dump(mcp_manifest, f, indent=2)
    print(f"Generated MCP manifest at {mcp_manifest_path}")


def generate_agent_service_files(agent_folder_path_str: str, output_dir_str: str):
    agent_folder_path = Path(agent_folder_path_str)
    output_dir = Path(output_dir_str)

    if not agent_folder_path.is_dir():
        # In CLI context, this might be handled by argparse or a prior check.
        # For direct calls, this check is useful.
        print(f"Error: Agent folder path '{agent_folder_path}' does not exist or is not a directory.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Ensured output directory exists: {output_dir}")

    agent_json_path = agent_folder_path / "agent.json"
    agent_requirements = []
    if agent_json_path.is_file():
        try:
            with open(agent_json_path, "r", encoding="utf-8") as f:
                agent_config = json.load(f)
            # Ensure "requirements" key exists and is a list
            raw_requirements = agent_config.get("requirements", [])
            if isinstance(raw_requirements, list):
                agent_requirements = [str(req) for req in raw_requirements] # Ensure all are strings
            else:
                print(f"Warning: 'requirements' in {agent_json_path} is not a list. Ignoring.")
            print(f"Successfully read agent requirements from {agent_json_path}")
        except Exception as e:
            print(f"Warning: Could not read or parse agent.json at {agent_json_path}: {e}")
            print("Proceeding without agent-specific requirements.")
    else:
        print(f"Warning: agent.json not found in {agent_folder_path}. Agent-specific requirements will not be included.")

    base_requirements = [
        "fastapi",
        "uvicorn[standard]",
        "smolagents" 
    ]
    # Filter out any potential non-string items from agent_requirements before combining
    valid_agent_requirements = [req for req in agent_requirements if isinstance(req, str)]
    all_requirements = list(set(base_requirements + valid_agent_requirements))
    
    requirements_txt_path = output_dir / "requirements.txt"
    with open(requirements_txt_path, "w", encoding="utf-8") as f:
        for req in sorted(all_requirements):
            f.write(req + "\\n")
    print(f"Generated requirements.txt at {requirements_txt_path}")

    main_py_path = output_dir / "main.py"
    with open(main_py_path, "w", encoding="utf-8") as f:
        f.write(MAIN_PY_TEMPLATE)
    print(f"Generated main.py at {main_py_path}")

    docker_agent_data_path = output_dir / "agent_data"
    if docker_agent_data_path.exists():
        print(f"Removing existing agent_data directory: {docker_agent_data_path}")
        shutil.rmtree(docker_agent_data_path)
    
    print(f"Copying agent data from {agent_folder_path} to {docker_agent_data_path}")
    shutil.copytree(agent_folder_path, docker_agent_data_path)
    print(f"Agent data copied successfully.")

    dockerfile_path = output_dir / "Dockerfile"
    with open(dockerfile_path, "w", encoding="utf-8") as f:
        f.write(DOCKERFILE_TEMPLATE)
    print(f"Generated Dockerfile at {dockerfile_path}")
    
    agent_name_for_manifest = agent_folder_path.name
    generate_mcp_manifest(agent_name_for_manifest, output_dir, valid_agent_requirements)

    # This part is for the CLI user, not for the library function's print.
    # The CLI handler will print these.
    # print(f"\\nFastAPI wrapper generation complete. Files are in: {output_dir}")
    # print(f"MCP Tool Manifest generated at {output_dir / 'mcp_tool.json'}")
    # print("\\nTo build and run the Docker container:")
    # print(f"1. cd {output_dir}")
    # print(f"2. docker build -t {agent_name_for_manifest.lower().replace('-', '_')}_service .") # Suggest a more specific tag
    # print(f"3. docker run -p 8000:8000 {agent_name_for_manifest.lower().replace('-', '_')}_service")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a FastAPI wrapper and MCP manifest for a smolagent.")
    parser.add_argument("--agent_folder_path", type=str, required=True,
                        help="Path to the folder where the smolagent has been saved (using agent.save()).")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Path to the directory where the FastAPI app and MCP manifest files will be generated.")
    
    args = parser.parse_args()
    
    # Call the refactored main function
    generate_agent_service_files(args.agent_folder_path, args.output_dir)
    
    # Print instructions here for standalone script execution
    print(f"\\nFastAPI wrapper generation complete. Files are in: {args.output_dir}")
    print(f"MCP Tool Manifest generated at {Path(args.output_dir) / 'mcp_tool.json'}")
    agent_name_cli = Path(args.agent_folder_path).name
    docker_image_cli_tag = agent_name_cli.lower().replace("-", "_").replace(" ", "_") + "_service"
    print("\\nTo build and run the Docker container:")
    print(f"1. cd {args.output_dir}")
    print(f"2. docker build -t {docker_image_cli_tag} .")
    print(f"3. docker run -p 8000:8000 {docker_image_cli_tag}")
"""
