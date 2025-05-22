import requests
import json
from typing import Optional, Dict, Any, Generator

from smolagents.tools import Tool # Assuming Tool is in smolagents.tools

class FastApiAgentTool(Tool):
    def __init__(
        self,
        base_url: str,
        name: str = "remote_agent_service",
        description: str = "A tool to interact with a remotely hosted smolagent via FastAPI.",
    ):
        self.base_url = base_url.rstrip('/') # Ensure no trailing slash
        self.name = name
        self.description = description
        
        self.inputs = {
            "task": {"type": "string", "description": "The task for the remote agent."},
            "stream": {"type": "boolean", "description": "Whether to stream the response.", "default": False, "nullable": True},
            "additional_args": {"type": "object", "description": "Additional arguments for the agent.", "nullable": True}
        }
        self.output_type = "any" # Can be string for text, object for JSON, or any

        super().__init__()
        # Skip validation because we use **kwargs in forward to capture all defined inputs
        self.skip_forward_signature_validation = True

    def setup(self):
        """
        Placeholder for future enhancements.
        For example, this method could fetch and parse /openapi.json from self.base_url
        to dynamically populate self.inputs, self.output_type, and self.description.
        """
        pass

    def forward(self, task: str, stream: Optional[bool] = False, additional_args: Optional[Dict[str, Any]] = None, **kwargs) -> Any:
        payload = {
            "task": task,
            "stream": stream,
            # Only include additional_args if it's provided and not None
            **(additional_args if additional_args is not None else {}) 
        }
        
        # The **kwargs might capture other inputs if defined in self.inputs and passed.
        # However, the current self.inputs only defines task, stream, additional_args.
        # If self.inputs were dynamic, this would be more relevant.
        # For now, we primarily rely on the explicitly defined parameters.
        # Let's ensure all keys from self.inputs are considered for the payload if passed via kwargs.
        for key in self.inputs:
            if key in kwargs and key not in payload: # Avoid overwriting explicit params
                payload[key] = kwargs[key]
        
        # Clean up payload: remove keys with None values if their definition allows null,
        # but ensure keys with default values are present if not nullable.
        # For "additional_args", if it's None, it should not be in the payload.
        # The FastAPI server will use its default for "stream" if not provided.
        final_payload = {"task": task}
        if stream is not None: # stream has a default in FastAPI, but send if specified
            final_payload["stream"] = stream
        if additional_args is not None:
            final_payload["additional_args"] = additional_args


        if stream:
            # Ensure the server is expecting 'stream' in the payload to differentiate
            response = requests.post(f"{self.base_url}/run", json=final_payload, stream=True, headers={'Accept': 'text/event-stream'})
            response.raise_for_status()  # Check for HTTP errors
            
            def stream_generator() -> Generator[Any, None, None]:
                for line in response.iter_lines():
                    if line:
                        decoded_line = line.decode('utf-8')
                        if decoded_line.startswith('data: '):
                            try:
                                data_json_str = decoded_line[len('data: '):]
                                data_json = json.loads(data_json_str)
                                
                                if 'output' in data_json:
                                    yield data_json['output']
                                elif data_json.get('event') == 'done':
                                    break
                                elif 'error' in data_json:
                                    # Log or handle error streamed from server
                                    # For now, we'll yield it as an error object or message
                                    yield {"error": data_json['error'], "error_type": data_json.get('error_type', 'StreamError')}
                                    break 
                            except json.JSONDecodeError:
                                # Log if a data line is not valid JSON
                                print(f"Warning: Could not decode JSON from stream: {decoded_line}")
                                pass 
            return stream_generator()
        else:
            response = requests.post(f"{self.base_url}/run", json=final_payload)
            response.raise_for_status() # Check for HTTP errors
            
            try:
                response_json = response.json()
                return response_json.get("output")
            except json.JSONDecodeError as e:
                raise Exception(f"Failed to decode JSON response from server: {e}. Response text: {response.text}")


__all__ = ["FastApiAgentTool"]
