
from __future__ import annotations

import time
import json
from typing import List
from datetime import datetime
from pathlib import Path

import pandas as pd
from google import genai

TIME_SLEEP = 20 # seconds between polling for batch job status

GEMINI_MODEL_CONFIGS = {
    'gemini-2.0-flash-exp': {
        'temperature': 0.7,
        'description': 'Fast, state-of-the-art experimental model'
    },
    'gemini-1.5-pro': {
        'temperature': 0.7,
        'description': 'Most capable with 1M token context window'
    },
    'gemini-1.5-flash': {
        'temperature': 0.8,  # Slightly higher for faster, more creative output
        'description': 'Balanced speed and cost-efficiency'
    },
    'gemini-3-flash-preview': {
        'temperature': 0.7,
        'description': 'Preview of upcoming Gemini Flash 3 model'
    },
    'gemini-3-pro-preview': {
        'temperature': 0.7,
        'description': 'Preview of upcoming Gemini 3 Pro model'
    },
}

def format_message_json(prompt_template: List, input_cv: str, job_desc: str = '') -> List[dict]:
    return [
        {
            'role': item['role'],
            'content': item['content'].replace("{original_cv}", input_cv).replace("{job_description}", job_desc)
        }
        for item in prompt_template
    ]


def format_message_txt(prompt_template, input_cv: str, job_desc: str = '') -> List[dict]:
    messages_string = prompt_template.format(original_cv=input_cv, job_description=job_desc)
    return [{"role": "user", "content": messages_string}]


class GeminiClient:
    """Kan je google modelletjes mee gebruiken"""

    def __init__(
        self,
        input_api_key: str,
        model: str,
        prompt_template_path: str,
        prompt_template,
        prompt_job_description: str
    ):
        # Initialize client with API key
        self.client = genai.Client(api_key=input_api_key)

        if not model: 
            raise ValueError("Model must be specified for GeminiClient")

        self.model = model

        if self.model not in GEMINI_MODEL_CONFIGS:
            raise ValueError(
                f"Unsupported Gemini model: {self.model}. "
                f"Supported models: {list(GEMINI_MODEL_CONFIGS.keys())}"
            )

        self.model_config = GEMINI_MODEL_CONFIGS[self.model]

        self.prompt_template_path = prompt_template_path
        self.prompt_template = prompt_template
        self.prompt_job_description = prompt_job_description if prompt_job_description else ''

        current_datetime = datetime.now()
        self.time_marker = current_datetime.strftime('%Y_%m_%d_%H_%M_%S')

    def format_messages(self, input_cv: str) -> str:
        """Format prompt for Gemini (expects single string, not message list)."""
        if self.prompt_template_path.endswith('.json'):
            messages = format_message_json(self.prompt_template, input_cv, self.prompt_job_description)
            # Combine all messages into single prompt
            return "\n\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
        return format_message_txt(self.prompt_template, input_cv, self.prompt_job_description)[0]['content']

    def _create_batch_requests(self, cv_s_dataframe: pd.DataFrame, to_be_modified_col: str) -> List[dict]:
        """Create batch requests for all CVs."""
        batch_requests = []

        for index, row in cv_s_dataframe.iterrows():
            prompt = self.format_messages(input_cv=row[to_be_modified_col])

            # Create batch request following Gemini's format
            batch_request = {
                'key': f"resume-request-{str(index)}",
                'request': {
                    'contents': [
                        {
                            'parts': [{'text': prompt}],
                            'role': 'user'
                        }
                    ],
                    'generation_config': {
                        'temperature': self.model_config['temperature'],
                    }
                }
            }
            batch_requests.append(batch_request)

        return batch_requests

    def _process_batch(self, batch_requests: List[dict]) -> dict:
        """Process a batch and return results mapping index -> content."""
        # Create temporary JSONL file
        temp_filename = f"gemini_batch_{self.time_marker}_{id(batch_requests)}.jsonl"

        with open(temp_filename, "w") as f:
            for item in batch_requests:
                f.write(json.dumps(item) + "\n")

        # Upload file
        from google.genai import types
        uploaded_file = self.client.files.upload(
            file=temp_filename,
            config=types.UploadFileConfig(
                display_name=f'batch-requests-{self.time_marker}',
                mime_type='application/jsonl'
            )
        )

        # Verify file was uploaded successfully
        if not uploaded_file.name:
            raise Exception("Failed to upload batch requests file")

        # Create batch job
        batch_job = self.client.batches.create(
            model=f"models/{self.model}",
            src=uploaded_file.name,
            config={'display_name': f"{self.model}_resume_generation_{self.time_marker}"}
        )

        # Clean up temp file
        Path(temp_filename).unlink(missing_ok=True)

        # Wait for batch completion
        completed_states = {
            'JOB_STATE_SUCCEEDED',
            'JOB_STATE_FAILED',
            'JOB_STATE_CANCELLED',
            'JOB_STATE_EXPIRED',
        }

        batch_job_name = batch_job.name
        if not batch_job_name:
            raise Exception("Batch job was created but has no name")

        print(f"Batch job created: {batch_job_name}")

        while batch_job.state and batch_job.state.name not in completed_states:
            print(f"Batch {batch_job_name} is still {batch_job.state.name}...")
            time.sleep(TIME_SLEEP)  # Poll every 20 seconds
            batch_job = self.client.batches.get(name=batch_job_name)

        # Verify we have a valid final state
        if not batch_job.state:
            raise Exception(f"Batch {batch_job_name} has no state information")

        print(f"Batch {batch_job_name}: {batch_job.state.name}")

        # Process results
        if batch_job.state.name != 'JOB_STATE_SUCCEEDED':
            raise Exception(f"Batch {batch_job_name} finished with state: {batch_job.state.name}")

        results = {}

        # Handle file-based results
        if batch_job.dest and batch_job.dest.file_name:
            result_file_name = batch_job.dest.file_name
            file_content = self.client.files.download(file=result_file_name)
            json_data = file_content.decode('utf-8')

            # Parse results
            for line in json_data.splitlines():
                if not line.strip():
                    continue
                json_record = json.loads(line)
                key = json_record.get("key")
                if key is None:
                    continue

                # Extract index from key "resume-request-{index}"
                custom_id_no = int(key.split("-")[-1])

                # Extract response text
                response = json_record.get("response", {})
                candidates = response.get("candidates", [])
                content = ""

                if candidates:
                    candidate = candidates[0]
                    content_parts = candidate.get("content", {}).get("parts", [])
                    if content_parts:
                        content = content_parts[0].get("text", "")

                results[custom_id_no] = content if content else "not_successfully_modified"

        # Handle inline results
        elif batch_job.dest and batch_job.dest.inlined_responses:
            for batch_request, inline_response in zip(batch_requests, batch_job.dest.inlined_responses):
                key = batch_request.get("key")
                if key is None:
                    continue

                custom_id_no = int(key.split("-")[-1])
                content = ""

                if inline_response.response:
                    content = inline_response.response.text

                results[custom_id_no] = content if content else "not_successfully_modified"

        return results

    def start_batch(self, cv_s_dataframe: pd.DataFrame) -> str:
        """
        Start a batch job and return the batch ID without waiting for completion.
        Expects a DataFrame with a single column of CVs.
        Returns the batch job ID. 
        Use get_batch_results(batch_id, cv_s_dataframe) to retrieve results later.
        """
        if len(cv_s_dataframe.columns) > 1:
            raise Exception("More than one column of resumes inputted. Please reformat input to only contain one column")

        to_be_modified_col = cv_s_dataframe.columns[-1]

        batch_requests = self._create_batch_requests(cv_s_dataframe, to_be_modified_col)
        print(f"Creating batch job for {len(cv_s_dataframe)} CVs...")

        # Create temporary JSONL file
        temp_filename = f"gemini_batch_{self.time_marker}_{id(batch_requests)}.jsonl"

        with open(temp_filename, "w") as f:
            for item in batch_requests:
                f.write(json.dumps(item) + "\n")

        # Upload file
        from google.genai import types
        uploaded_file = self.client.files.upload(
            file=temp_filename,
            config=types.UploadFileConfig(
                display_name=f'batch-requests-{self.time_marker}',
                mime_type='application/jsonl'
            )
        )

        # Verify file was uploaded successfully
        if not uploaded_file.name:
            raise Exception("Failed to upload batch requests file")

        # Create batch job
        batch_job = self.client.batches.create(
            model=f"models/{self.model}",
            src=uploaded_file.name,
            config={'display_name': f"{self.model}_resume_generation_{self.time_marker}"}
        )

        # Clean up temp file
        Path(temp_filename).unlink(missing_ok=True)

        batch_job_name = batch_job.name
        if not batch_job_name:
            raise Exception("Batch job was created but has no name")

        print(f"Batch job created: {batch_job_name}")
        print(f"Use get_batch_results('{batch_job_name}') to retrieve results when complete")

        return batch_job_name

    def get_batch_results(self, batch_id: str, cv_s_dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Retrieve results from a completed batch job.
        Expects the original DataFrame used to start the batch job to map results correctly.
        And the batch ID returned from start_batch().
        Returns a DataFrame with the modified CVs as one column.
        """
        if len(cv_s_dataframe.columns) > 1:
            raise Exception("More than one column of resumes inputted. Please reformat input to only contain one column")

        to_be_modified_col = cv_s_dataframe.columns[-1]
        modified_col_name = f"Modified_{self.model}_of_{to_be_modified_col}_Model{self.model}"

        # Get batch job status
        batch_job = self.client.batches.get(name=batch_id)

        if not batch_job.state:
            raise Exception(f"Batch {batch_id} has no state information")

        print(f"Batch {batch_id} current state: {batch_job.state.name}")

        # Check if batch is complete
        if batch_job.state.name != 'JOB_STATE_SUCCEEDED':
            if batch_job.state.name in ['JOB_STATE_FAILED', 'JOB_STATE_CANCELLED', 'JOB_STATE_EXPIRED']:
                raise Exception(f"Batch {batch_id} finished with state: {batch_job.state.name}")
            else:
                raise Exception(f"Batch {batch_id} is not yet complete. Current state: {batch_job.state.name}")

        results = {}

        # Handle file-based results
        if batch_job.dest and batch_job.dest.file_name:
            result_file_name = batch_job.dest.file_name
            file_content = self.client.files.download(file=result_file_name)
            json_data = file_content.decode('utf-8')

            # Parse results
            for line in json_data.splitlines():
                if not line.strip():
                    continue
                json_record = json.loads(line)
                key = json_record.get("key")
                if key is None:
                    continue

                # Extract index from key "resume-request-{index}"
                custom_id_no = int(key.split("-")[-1])

                # Extract response text
                response = json_record.get("response", {})
                candidates = response.get("candidates", [])
                content = ""

                if candidates:
                    candidate = candidates[0]
                    content_parts = candidate.get("content", {}).get("parts", [])
                    if content_parts:
                        content = content_parts[0].get("text", "")

                results[custom_id_no] = content if content else "not_successfully_modified"

        # Handle inline results
        elif batch_job.dest and batch_job.dest.inlined_responses:
            # Recreate batch_requests to match with inline responses
            batch_requests = self._create_batch_requests(cv_s_dataframe, to_be_modified_col)
            for batch_request, inline_response in zip(batch_requests, batch_job.dest.inlined_responses):
                key = batch_request.get("key")
                if key is None:
                    continue

                custom_id_no = int(key.split("-")[-1])
                content = ""

                if inline_response.response:
                    content = inline_response.response.text

                results[custom_id_no] = content if content else "not_successfully_modified"

        # Build output column in original order
        output_resumes = [
            results.get(idx, "not_successfully_modified")
            for idx in cv_s_dataframe.index
        ]

        result_df = cv_s_dataframe.copy()
        result_df[modified_col_name] = output_resumes
        return result_df[[modified_col_name]]

    def generate_group_of_cv_s(self, cv_s_dataframe: pd.DataFrame) -> pd.DataFrame:
        """Generate modified CVs using batch processing."""
        if len(cv_s_dataframe.columns) > 1:
            raise Exception("More than one column of resumes inputted. Please reformat input to only contain one column")

        to_be_modified_col = cv_s_dataframe.columns[-1]
        modified_col_name = f"Modified_{self.model}_of_{to_be_modified_col}_Model{self.model}"

        # Create batch requests
        batch_requests = self._create_batch_requests(cv_s_dataframe, to_be_modified_col)
        print(f"Processing {len(cv_s_dataframe)} CVs in a single batch")

        # Process batch
        all_results = self._process_batch(batch_requests)

        # Build output column in original order
        output_resumes = [
            all_results.get(idx, "not_successfully_modified")
            for idx in cv_s_dataframe.index
        ]

        cv_s_dataframe[modified_col_name] = output_resumes
        return cv_s_dataframe[[modified_col_name]]
