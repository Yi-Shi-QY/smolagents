import unittest
from unittest.mock import patch, MagicMock
import json
import requests # For requests.exceptions

# Adjust import path
from src.smolagents.service_tools import FastApiAgentTool

class TestFastApiAgentTool(unittest.TestCase):

    def test_initialization(self):
        base_url = "http://localhost:8000/"
        tool = FastApiAgentTool(base_url=base_url, name="my_remote_agent", description="Test description.")
        
        self.assertEqual(tool.base_url, "http://localhost:8000") # Check trailing slash removal
        self.assertEqual(tool.name, "my_remote_agent")
        self.assertEqual(tool.description, "Test description.")
        
        expected_inputs = {
            "task": {"type": "string", "description": "The task for the remote agent."},
            "stream": {"type": "boolean", "description": "Whether to stream the response.", "default": False, "nullable": True},
            "additional_args": {"type": "object", "description": "Additional arguments for the agent.", "nullable": True}
        }
        self.assertEqual(tool.inputs, expected_inputs)
        self.assertEqual(tool.output_type, "any")
        self.assertTrue(tool.skip_forward_signature_validation)

    @patch('src.smolagents.service_tools.requests.post')
    def test_forward_non_streaming_successful(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"output": "mocked agent response"}
        mock_post.return_value = mock_response

        tool = FastApiAgentTool(base_url="http://testservice.com")
        result = tool.forward(task="test task", stream=False, additional_args={"param": "value"})

        expected_payload = {"task": "test task", "stream": False, "additional_args": {"param": "value"}}
        mock_post.assert_called_once_with(f"{tool.base_url}/run", json=expected_payload)
        self.assertEqual(result, "mocked agent response")

    @patch('src.smolagents.service_tools.requests.post')
    def test_forward_non_streaming_no_additional_args(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"output": "mocked agent response"}
        mock_post.return_value = mock_response

        tool = FastApiAgentTool(base_url="http://testservice.com")
        result = tool.forward(task="test task") # additional_args is None, stream is default False

        expected_payload = {"task": "test task", "stream": False} # additional_args should not be in payload
        mock_post.assert_called_once_with(f"{tool.base_url}/run", json=expected_payload)
        self.assertEqual(result, "mocked agent response")


    @patch('src.smolagents.service_tools.requests.post')
    def test_forward_streaming_successful(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        # Simulate iter_lines for SSE
        sse_lines = [
            b"data: " + json.dumps({"output": "chunk1"}).encode('utf-8'),
            b"data: " + json.dumps({"output": "chunk2"}).encode('utf-8'),
            b"data: " + json.dumps({"event": "done"}).encode('utf-8'),
        ]
        mock_response.iter_lines.return_value = iter(sse_lines)
        mock_post.return_value = mock_response

        tool = FastApiAgentTool(base_url="http://testservice.com")
        generator = tool.forward(task="test stream task", stream=True, additional_args={"detail": "high"})
        
        results = list(generator)

        expected_payload = {"task": "test stream task", "stream": True, "additional_args": {"detail": "high"}}
        mock_post.assert_called_once_with(f"{tool.base_url}/run", json=expected_payload, stream=True, headers={'Accept': 'text/event-stream'})
        self.assertEqual(results, ["chunk1", "chunk2"])

    @patch('src.smolagents.service_tools.requests.post')
    def test_forward_http_error(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("Test HTTP Error")
        mock_post.return_value = mock_response

        tool = FastApiAgentTool(base_url="http://testservice.com")
        with self.assertRaises(requests.exceptions.HTTPError):
            tool.forward(task="test error task") # Non-streaming
        
        # Also test for streaming
        with self.assertRaises(requests.exceptions.HTTPError):
            # The generator won't be consumed, error happens at requests.post
             tool.forward(task="test error task stream", stream=True)


    @patch('src.smolagents.service_tools.requests.post')
    def test_forward_non_streaming_json_decode_error(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.side_effect = json.JSONDecodeError("Error decoding JSON", "doc", 0)
        mock_response.text = "Invalid JSON response" # For the error message
        mock_post.return_value = mock_response

        tool = FastApiAgentTool(base_url="http://testservice.com")
        with self.assertRaisesRegex(Exception, "Failed to decode JSON response from server"):
            tool.forward(task="test json error")

    @patch('src.smolagents.service_tools.requests.post')
    @patch('src.smolagents.service_tools.print') # Mock print to check warnings
    def test_forward_streaming_malformed_json_line(self, mock_print, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        sse_lines = [
            b"data: not a valid json",
            b"data: " + json.dumps({"output": "chunk1"}).encode('utf-8'),
            b"data: " + json.dumps({"event": "done"}).encode('utf-8'),
        ]
        mock_response.iter_lines.return_value = iter(sse_lines)
        mock_post.return_value = mock_response

        tool = FastApiAgentTool(base_url="http://testservice.com")
        generator = tool.forward(task="test stream malformed", stream=True)
        
        results = list(generator)
        
        self.assertEqual(results, ["chunk1"])
        mock_print.assert_called_with("Warning: Could not decode JSON from stream: data: not a valid json")

    @patch('src.smolagents.service_tools.requests.post')
    def test_forward_streaming_error_in_stream(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        sse_lines = [
            b"data: " + json.dumps({"output": "chunk1"}).encode('utf-8'),
            b"data: " + json.dumps({"error": "Something went wrong on server", "error_type": "ServerError"}).encode('utf-8'),
            b"data: " + json.dumps({"event": "done"}).encode('utf-8'), # Should not be reached if error terminates
        ]
        mock_response.iter_lines.return_value = iter(sse_lines)
        mock_post.return_value = mock_response

        tool = FastApiAgentTool(base_url="http://testservice.com")
        generator = tool.forward(task="test stream error in data", stream=True)
        
        results = list(generator)
        
        # The current implementation yields the error and breaks
        self.assertEqual(results, ["chunk1", {"error": "Something went wrong on server", "error_type": "ServerError"}])

if __name__ == '__main__':
    unittest.main()
