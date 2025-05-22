#!/usr/bin/env python
# coding=utf-8

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from smolagents import CodeAgent, InferenceClientModel, LiteLLMModel, Model, OpenAIServerModel, Tool, TransformersModel
from smolagents.default_tools import TOOL_MAPPING
from smolagents.utils.service_generator import generate_agent_service_files


leopard_prompt = "How many seconds would it take for a leopard at full speed to run through Pont des Arts?"


def _parse_run_arguments(parser):
    """Helper function to add arguments for the 'run' command."""
    parser.add_argument(
        "prompt",
        type=str,
        nargs="?",  # Makes it optional
        default=leopard_prompt,
        help="The prompt to run with the agent",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="InferenceClientModel",
        help="The model type to use (e.g., InferenceClientModel, OpenAIServerModel, LiteLLMModel, TransformersModel)",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="Qwen/Qwen2.5-Coder-32B-Instruct",
        help="The model ID to use for the specified model type",
    )
    parser.add_argument(
        "--imports",
        nargs="*",  # accepts zero or more arguments
        default=[],
        help="Space-separated list of imports to authorize (e.g., 'numpy pandas')",
    )
    parser.add_argument(
        "--tools",
        nargs="*",
        default=["web_search"],
        help="Space-separated list of tools that the agent can use (e.g., 'tool1 tool2 tool3')",
    )
    parser.add_argument(
        "--verbosity-level",
        type=int,
        default=1,
        help="The verbosity level, as an int in [0, 1, 2].",
    )
    group = parser.add_argument_group("api options", "Options for API-based model types")
    group.add_argument(
        "--provider",
        type=str,
        default=None,
        help="The inference provider to use for the model",
    )
    group.add_argument(
        "--api-base",
        type=str,
        help="The base URL for the model",
    )
    group.add_argument(
        "--api-key",
        type=str,
        help="The API key for the model",
    )
    parser.set_defaults(func=_handle_run_agent)

def _parse_wrap_fastapi_arguments(parser):
    """Helper function to add arguments for the 'wrap-fastapi' command."""
    parser.add_argument(
        "--agent_folder_path", 
        type=str, 
        required=True, 
        help="Path to the folder where the smolagent has been saved (using agent.save())."
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        required=True, 
        help="Path to the directory where the FastAPI app files will be generated."
    )
    parser.set_defaults(func=_handle_wrap_fastapi)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Smolagents CLI tool.")
    subparsers = parser.add_subparsers(dest="command", help="Available commands", required=True)

    # Subparser for the 'run' command
    run_parser = subparsers.add_parser("run", help="Run a CodeAgent with specified parameters.")
    _parse_run_arguments(run_parser)

    # Subparser for the 'wrap-fastapi' command
    wrap_parser = subparsers.add_parser("wrap-fastapi", help="Wrap a saved smolagent into a FastAPI service.")
    _parse_wrap_fastapi_arguments(wrap_parser)
    
    return parser.parse_args()

def _handle_run_agent(args):
    """Handles the 'run' command."""
    run_smolagent(
        args.prompt,
        args.tools,
        args.model_type,
        args.model_id,
        provider=args.provider,
        api_base=args.api_base,
        api_key=args.api_key,
        imports=args.imports,
        verbosity_level=args.verbosity_level,
    )

def _handle_wrap_fastapi(args):
    """Handles the 'wrap-fastapi' command."""
    generate_agent_service_files(args.agent_folder_path, args.output_dir)
    print(f"\\nFastAPI wrapper generation complete. Files are in: {args.output_dir}")
    print(f"MCP Tool Manifest generated at {Path(args.output_dir) / 'mcp_tool.json'}")
    agent_name_cli = Path(args.agent_folder_path).name
    docker_image_cli_tag = agent_name_cli.lower().replace("-", "_").replace(" ", "_") + "_service"
    print("\\nTo build and run the Docker container:")
    print(f"1. cd {args.output_dir}")
    print(f"2. docker build -t {docker_image_cli_tag} .")
    print(f"3. docker run -p 8000:8000 {docker_image_cli_tag}")

def load_model(
    model_type: str,
    model_id: str,
    api_base: str | None = None,
    api_key: str | None = None,
    provider: str | None = None,
) -> Model:
    if model_type == "OpenAIServerModel":
        return OpenAIServerModel(
            api_key=api_key or os.getenv("FIREWORKS_API_KEY"),
            api_base=api_base or "https://api.fireworks.ai/inference/v1",
            model_id=model_id,
        )
    elif model_type == "LiteLLMModel":
        return LiteLLMModel(
            model_id=model_id,
            api_key=api_key,
            api_base=api_base,
        )
    elif model_type == "TransformersModel":
        return TransformersModel(model_id=model_id, device_map="auto")
    elif model_type == "InferenceClientModel":
        return InferenceClientModel(
            model_id=model_id,
            token=api_key or os.getenv("HF_API_KEY"),
            provider=provider,
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def run_smolagent(
    prompt: str,
    tools: list[str],
    model_type: str,
    model_id: str,
    api_base: str | None = None,
    api_key: str | None = None,
    imports: list[str] | None = None,
    provider: str | None = None,
    verbosity_level: int = 1,
) -> None:
    load_dotenv()

    model = load_model(model_type, model_id, api_base=api_base, api_key=api_key, provider=provider)

    available_tools = []
    for tool_name in tools:
        if "/" in tool_name:
            available_tools.append(Tool.from_space(tool_name))
        else:
            if tool_name in TOOL_MAPPING:
                available_tools.append(TOOL_MAPPING[tool_name]())
            else:
                raise ValueError(f"Tool {tool_name} is not recognized either as a default tool or a Space.")

    print(f"Running agent with these tools: {tools}")
    agent = CodeAgent(
        tools=available_tools, 
        model=model, 
        additional_authorized_imports=imports,
        verbosity_level=verbosity_level 
    )

    agent.run(prompt)


def main() -> None:
    args = parse_arguments()
    if hasattr(args, 'func'):
        args.func(args)
    else:
        # This case might happen if no subcommand is given and the main parser
        # doesn't have a default action. However, with 'required=True' for subparsers,
        # argparse should handle it.
        print("Please specify a command: run or wrap-fastapi")


if __name__ == "__main__":
    main()
