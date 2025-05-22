import unittest
import tempfile
import json
from pathlib import Path
import shutil
import os

# Adjust import path based on where the test is run from and project structure
# This assumes tests are run from the project root.
from src.smolagents.utils.service_generator import generate_agent_service_files, MAIN_PY_TEMPLATE, DOCKERFILE_TEMPLATE

class TestGenerateAgentServiceFiles(unittest.TestCase):

    def test_successful_generation_with_agent_json(self):
        with tempfile.TemporaryDirectory() as tmp_agent_folder_str, \
             tempfile.TemporaryDirectory() as tmp_output_dir_str:
            
            agent_folder_path = Path(tmp_agent_folder_str)
            output_dir_path = Path(tmp_output_dir_str)

            # Create a minimal agent.json
            agent_config_data = {
                "name": "MyTestAgent", # For mcp_tool.json tool_name
                "requirements": ["specific_agent_dep==1.0", "another_dep>=2.0"]
            }
            with open(agent_folder_path / "agent.json", "w") as f:
                json.dump(agent_config_data, f)
            
            # Create a dummy tool file to check if agent_data is copied
            (agent_folder_path / "tools").mkdir(exist_ok=True)
            with open(agent_folder_path / "tools" / "dummy_tool.py", "w") as f:
                f.write("# dummy tool")

            generate_agent_service_files(str(agent_folder_path), str(output_dir_path))

            # Assertions for file creation
            self.assertTrue((output_dir_path / "main.py").is_file())
            self.assertTrue((output_dir_path / "Dockerfile").is_file())
            self.assertTrue((output_dir_path / "requirements.txt").is_file())
            self.assertTrue((output_dir_path / "mcp_tool.json").is_file())
            
            # Assert agent_data copying
            self.assertTrue((output_dir_path / "agent_data").is_dir())
            self.assertTrue((output_dir_path / "agent_data" / "agent.json").is_file())
            self.assertTrue((output_dir_path / "agent_data" / "tools" / "dummy_tool.py").is_file())

            # Spot-check requirements.txt
            with open(output_dir_path / "requirements.txt", "r") as f:
                requirements_content = f.read()
                self.assertIn("fastapi", requirements_content)
                self.assertIn("uvicorn[standard]", requirements_content)
                self.assertIn("smolagents", requirements_content)
                self.assertIn("specific_agent_dep==1.0", requirements_content)
                self.assertIn("another_dep>=2.0", requirements_content)

            # Spot-check mcp_tool.json
            with open(output_dir_path / "mcp_tool.json", "r") as f:
                mcp_content = json.load(f)
                # agent_name in mcp_tool.json is derived from the agent_folder_path's name
                expected_agent_name = agent_folder_path.name
                self.assertEqual(mcp_content["tool_name"], expected_agent_name)
                self.assertIn("specific_agent_dep==1.0", mcp_content["dependencies"]["python"])
                self.assertIn("another_dep>=2.0", mcp_content["dependencies"]["python"])
                self.assertEqual(mcp_content["execution_environment"]["image"], f"{expected_agent_name.lower().replace('-', '_')}:latest")


            # Spot-check Dockerfile
            with open(output_dir_path / "Dockerfile", "r") as f:
                dockerfile_content = f.read()
                self.assertIn("COPY ./agent_data ./agent_data", dockerfile_content)
                self.assertIn("CMD [\"uvicorn\", \"main:app\"", dockerfile_content)
                
            # Spot-check main.py (very basic check)
            with open(output_dir_path / "main.py", "r") as f:
                main_py_content = f.read()
                self.assertIn("AGENT_FOLDER_PATH = \"./agent_data\"", main_py_content)
                self.assertIn("app = FastAPI()", main_py_content)


    def test_generation_missing_agent_json(self):
        with tempfile.TemporaryDirectory() as tmp_agent_folder_str, \
             tempfile.TemporaryDirectory() as tmp_output_dir_str:
            
            agent_folder_path = Path(tmp_agent_folder_str)
            output_dir_path = Path(tmp_output_dir_str)

            # No agent.json is created in agent_folder_path

            generate_agent_service_files(str(agent_folder_path), str(output_dir_path))

            # Assertions for file creation (should still create them with defaults)
            self.assertTrue((output_dir_path / "main.py").is_file())
            self.assertTrue((output_dir_path / "Dockerfile").is_file())
            self.assertTrue((output_dir_path / "requirements.txt").is_file())
            self.assertTrue((output_dir_path / "mcp_tool.json").is_file())
            self.assertTrue((output_dir_path / "agent_data").is_dir()) # agent_data is copied (empty in this case)

            # Spot-check requirements.txt (should only have base requirements)
            with open(output_dir_path / "requirements.txt", "r") as f:
                requirements_content = f.read()
                self.assertIn("fastapi\n", requirements_content) # Check for newline to be more specific
                self.assertIn("uvicorn[standard]\n", requirements_content)
                self.assertIn("smolagents\n", requirements_content)
                self.assertNotIn("specific_agent_dep==1.0", requirements_content)

            # Spot-check mcp_tool.json (dependencies should be empty)
            with open(output_dir_path / "mcp_tool.json", "r") as f:
                mcp_content = json.load(f)
                expected_agent_name = agent_folder_path.name
                self.assertEqual(mcp_content["tool_name"], expected_agent_name)
                self.assertEqual(mcp_content["dependencies"]["python"], [])

if __name__ == '__main__':
    unittest.main()
