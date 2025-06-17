# main.py

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import vertexai
from vertexai.generative_models import GenerativeModel, SafetySetting, Part
import os
import zipfile
import io
import logging
import tempfile
import time
from typing import List, Dict, Any, Optional
import pandas as pd
import uuid
import shutil
import json
import re
import uvicorn
import concurrent.futures
import traceback
from google.api_core import exceptions as google_exceptions
from datetime import datetime

# ==============================================================================
# 1. LOGGING AND APP CONFIGURATION
# ==============================================================================

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Cheque Field Extraction API",
    description="An API to extract structured data (Date, Amount) from cropped cheque images using a two-step AI process.",
    version="1.0.0"
)

# Configure CORS to allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================================================================
# 2. VERTEX AI AND CONSTANTS CONFIGURATION
# ==============================================================================

# Configure your Google Cloud project details here
GCP_PROJECT_ID = "hbl-uat-ocr-fw-app-prj-spk-4d"
GCP_LOCATION = "asia-south1"
GCP_API_ENDPOINT = "asia-south1-aiplatform.googleapis.com"

# Initialize Vertex AI
try:
    vertexai.init(project=GCP_PROJECT_ID, location=GCP_LOCATION, api_endpoint=GCP_API_ENDPOINT)
    logger.info(f"Vertex AI initialized successfully for project '{GCP_PROJECT_ID}' in '{GCP_LOCATION}'.")
except Exception as e:
    logger.error(f"Fatal: Could not initialize Vertex AI. Please check credentials and project settings. Error: {e}")
    # This is a critical error, the application might not be functional.
    # Consider exiting or implementing a health check endpoint that reports this failure.


# Define safety settings to allow processing of financial documents
SAFETY_SETTINGS = [
    SafetySetting(category=SafetySetting.HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=SafetySetting.HarmBlockThreshold.BLOCK_NONE),
    SafetySetting(category=SafetySetting.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=SafetySetting.HarmBlockThreshold.BLOCK_NONE),
    SafetySetting(category=SafetySetting.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=SafetySetting.HarmBlockThreshold.BLOCK_NONE),
    SafetySetting(category=SafetySetting.HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=SafetySetting.HarmBlockThreshold.BLOCK_NONE),
]

# --- SCALABLE FIELD DEFINITIONS ---
# To add a new field, add its definition here. The prompts and processing logic will adapt.
FIELDS = [
    {"id": 1, "name": "date"},
    {"id": 2, "name": "amount"}
]

# --- PERFORMANCE TUNING CONSTANTS ---
MAX_CONCURRENT_WORKERS = 100  # Max threads for processing images in parallel
API_CALL_BATCH_SIZE = 30     # Number of images to process in one batch
REASK_CONFIDENCE_THRESHOLD = 0.90 # Trigger re-ask if confidence is below this score
MAX_API_RETRIES = 5          # Max retries for a failing API call

# --- GLOBAL JOB MANAGEMENT ---
# Using a thread pool executor for background tasks
executor = concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT_WORKERS)
active_tasks = {}
processed_jobs = {}


# ==============================================================================
# 3. CHEQUE PROCESSOR CLASS (CORE LOGIC)
# ==============================================================================

class ChequeProcessor:
    """Encapsulates all logic for processing cheque images with Vertex AI."""

    def __init__(self, model_name: str = "gemini-1.5-flash-001"):
        self.model = GenerativeModel(model_name, safety_settings=SAFETY_SETTINGS)
        self.current_year = datetime.now().year
        self.previous_year = self.current_year - 1

    def _call_vertex_ai_with_retry(self, prompt_parts: List[Any]) -> Any:
        """
        Calls the Vertex AI model with a robust exponential backoff retry mechanism.
        """
        delay = 1.0
        retryable_errors = (
            google_exceptions.ResourceExhausted,
            google_exceptions.TooManyRequests,
            google_exceptions.ServiceUnavailable,
            google_exceptions.DeadlineExceeded,
        )

        for attempt in range(MAX_API_RETRIES):
            try:
                return self.model.generate_content(prompt_parts, generation_config={"temperature": 0.0})
            except retryable_errors as e:
                logger.warning(f"API Error ({type(e).__name__}) on attempt {attempt + 1}/{MAX_API_RETRIES}. Retrying in {delay:.2f}s...")
                time.sleep(delay + (uuid.uuid4().int % 1000) / 1000.0) # Add jitter
                delay *= 2.0
            except Exception as e:
                logger.error(f"A non-retryable API error occurred: {e}")
                raise  # Re-raise for the calling function to handle

        logger.error(f"API call failed after {MAX_API_RETRIES} attempts.")
        raise google_exceptions.RetryError("API call failed after max retries.", None)


    @staticmethod
    def _extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
        """
        Robustly extracts a JSON object from a string, cleaning markdown fences.
        """
        # Find text between ```json and ```
        match = re.search(r'```json\s*(\{[\s\S]*?\})\s*```', text, re.DOTALL)
        if match:
            json_str = match.group(1)
        else:
            # Fallback: find the first '{' and the last '}'
            start = text.find('{')
            end = text.rfind('}')
            if start == -1 or end == -1 or end < start:
                logger.warning(f"Could not find a valid JSON structure in text: {text[:200]}...")
                return None
            json_str = text[start:end+1]

        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            logger.error(f"Failed to decode JSON from extracted string: {json_str[:200]}...")
            return None

    # --- STEP 1: OCR (Image to Markdown) ---
    def generate_markdown_from_image(self, file_data: bytes, mime_type: str) -> str:
        """
        Performs the first step: high-fidelity OCR to convert image to text.
        """
        prompt = (
            "You are a state-of-the-art Optical Character Recognition (OCR) engine. "
            "Your sole purpose is to transcribe the text from the provided image with the highest possible fidelity. "
            "Analyze the image and convert all visible characters—handwritten or printed—into a clean text string. "
            "Preserve line breaks as you see them. Do not interpret, correct, or add any information that is not visually present. "
            "Your output must be only the transcribed text and nothing else."
        )
        image_part = Part.from_data(data=file_data, mime_type=mime_type)
        response = self._call_vertex_ai_with_retry([prompt, image_part])
        return response.text.strip()

    # --- STEP 2: Extraction (Image + Markdown to JSON) ---
    def extract_fields(self, file_data: bytes, mime_type: str, markdown_text: str, file_path: str) -> Dict[str, Any]:
        """
        Performs the main extraction using both image and text context.
        """
        try:
            # === Initial Extraction ===
            initial_prompt = self._build_extraction_prompt(self._get_field_descriptions())
            response = self._call_vertex_ai_with_retry([initial_prompt, Part.from_data(file_data, mime_type), f"OCR Text Hint:\n{markdown_text}"])
            initial_result = self._extract_json_from_text(response.text)

            if not initial_result or "extracted_fields" not in initial_result:
                logger.error(f"Initial extraction failed to produce valid JSON for {file_path}. Response: {response.text}")
                # Create a default structure to allow re-ask to proceed
                initial_result = {"extracted_fields": [{"field_name": f["name"], "value": "", "confidence": 0.0} for f in FIELDS]}


            final_results_map = {field.get("field_name"): field for field in initial_result.get("extracted_fields", [])}

            # === Re-Ask Logic ===
            for field_info in FIELDS:
                field_name = field_info["name"]
                current_field = final_results_map.get(field_name)

                # Condition to trigger a re-ask attempt
                should_reask = not current_field or not current_field.get("value") or current_field.get("confidence", 0.0) < REASK_CONFIDENCE_THRESHOLD

                if should_reask:
                    logger.warning(f"Confidence for '{field_name}' is low/null for {file_path}. Triggering re-ask.")
                    reask_prompt = self._build_reask_prompt(field_name)
                    reask_response = self._call_vertex_ai_with_retry([reask_prompt, Part.from_data(file_data, mime_type), f"OCR Text Hint:\n{markdown_text}"])
                    reask_result = self._extract_json_from_text(reask_response.text)

                    if reask_result and reask_result.get("extracted_fields"):
                        new_field = reask_result["extracted_fields"][0]
                        # Replace if the new result has a higher confidence score
                        if new_field.get("confidence", 0.0) > final_results_map.get(field_name, {}).get("confidence", 0.0):
                            final_results_map[field_name] = new_field
                            logger.info(f"Re-ask for '{field_name}' succeeded with higher confidence for {file_path}.")

            return {"extracted_fields": list(final_results_map.values())}

        except Exception as e:
            logger.error(f"An error occurred during field extraction for {file_path}: {e}")
            logger.error(traceback.format_exc())
            return {"error": str(e), "extracted_fields": []}

    # --- PROMPT ENGINEERING HELPERS ---
    def _get_field_descriptions(self) -> Dict[str, str]:
        """
        Returns a dictionary of detailed extraction instructions for each field.
        This makes it easy to scale and manage prompts.
        """
        return {
            "date": (
                f"**Objective**: Extract the 8-digit date from the cheque's date grid. "
                f"**Rules**: 1. Focus only on handwritten ink, ignore grid lines. 2. If a date is corrected (struck-out), use the new, valid date. "
                f"3. **Temporal Logic**: The year MUST be {self.current_year} or {self.previous_year}. Use this rule to correct OCR errors (e.g., if the last digit looks like '{str(self.current_year + 1)[-1]}', it must be '{str(self.current_year)[-1]}'). "
                f"4. **Format**: The final output MUST be a single string in 'DD-MM-YYYY' format. If unreadable, return an empty string."
            ),
            "amount": (
                "**Objective**: Extract the numeric 'courtesy amount' from the box next to the Rupee symbol ('₹'). "
                "**Rules**: 1. Locate the digits to the right of the '₹'. 2. **Filter Strictly**: Discard ALL non-numeric characters (commas, '₹', '/-'). Only a single decimal point is allowed. "
                "3. If corrected, use the final value. "
                "4. **Format**: Standardize the final number to have exactly two decimal places (e.g., '5000' becomes '5000.00'). If unreadable, return an empty string."
            )
        }

    def _build_extraction_prompt(self, descriptions: Dict[str, str]) -> str:
        """Dynamically builds the main extraction prompt from field definitions."""
        fields_str = "\n\n".join([f"### {field['name']}\n{descriptions.get(field['name'], '')}" for field in FIELDS])

        return f"""
        You are a financial document analysis expert. You are given a cropped image of a cheque and its OCR text. Your task is to extract specific fields with extreme accuracy. Use BOTH the image and the provided text to cross-verify your findings.

        **Field Extraction Guidelines:**
        {fields_str}

        **Critical Instructions:**
        1.  **Confidence Score:** Provide a confidence score (float from 0.0 to 1.0) for each field. 1.0 is absolute certainty. Scores below 0.9 require a `reason`.
        2.  **Reasoning:** If confidence is low, briefly explain why (e.g., "Handwriting for '7' is ambiguous", "Ink smudge obscures last two digits").
        3.  **Strict JSON Output:** Your entire response MUST be a single, valid JSON object with NO other text or markdown.

        **JSON Output Structure:**
        ```json
        {{
            "extracted_fields": [
                {{
                    "field_name": "date",
                    "value": "DD-MM-YYYY",
                    "confidence": 0.99,
                    "reason": null
                }},
                {{
                    "field_name": "amount",
                    "value": "12345.00",
                    "confidence": 0.85,
                    "reason": "Final digit is poorly written, could be a 0 or 6."
                }}
            ]
        }}
        ```
        """

    def _build_reask_prompt(self, field_name: str) -> str:
        """Dynamically builds a targeted re-ask prompt."""
        description = self._get_field_descriptions().get(field_name, "No description available.")
        return f"""
        A previous attempt to extract the '{field_name}' field was unsuccessful or had low confidence. Please re-examine the provided image and OCR text with intense focus.

        **Correction Instruction for `{field_name}`:**
        {description}

        **Strict JSON Output Format:**
        Your response MUST be a single, valid JSON object containing only the "extracted_fields" key. This key must hold an array with a SINGLE object for the '{field_name}' you were asked to re-examine. Example:
        ```json
        {{
            "extracted_fields": [
                {{
                    "field_name": "{field_name}",
                    "value": "...",
                    "confidence": 0.95,
                    "reason": "Re-evaluation clarified the ambiguous digit."
                }}
            ]
        }}
        ```
        """

    def process_single_image(self, file_info: Dict) -> Dict:
        """
        Orchestrates the full two-step process for a single image.
        """
        file_path = file_info['path']
        logger.info(f"Processing image: {file_path}")
        try:
            # Step 1: Get Markdown from Image
            markdown = self.generate_markdown_from_image(file_info['data'], file_info['type'])
            logger.debug(f"[{file_path}] OCR Result: {markdown}")

            # Step 2: Extract fields using Image + Markdown
            result = self.extract_fields(file_info['data'], file_info['type'], markdown, file_path)
            result['file_path'] = file_path
            return result

        except Exception as e:
            logger.error(f"Failed to process image {file_path}. Error: {e}")
            return {"file_path": file_path, "error": str(e), "extracted_fields": []}


# ==============================================================================
# 4. BACKGROUND TASK AND BATCH PROCESSING
# ==============================================================================

def generate_excel_report(results: List[Dict], output_dir: str, job_id: str) -> str:
    """Creates a consolidated Excel report from the extraction results."""
    excel_path = os.path.join(output_dir, f"cheque_extraction_report_{job_id}.xlsx")
    
    data_for_df = []
    for item in results:
        row = {"filepath": item.get("file_path", "Unknown")}
        if "error" in item:
            row["error"] = item["error"]
        
        for field in item.get("extracted_fields", []):
            name = field.get("field_name")
            if name:
                row[name] = field.get("value")
                row[f"{name}_confidence"] = field.get("confidence")
                row[f"{name}_reason"] = field.get("reason")
        data_for_df.append(row)

    if not data_for_df:
        logger.warning(f"Job {job_id}: No data was processed to generate a report.")
        return "" # Return empty if no data

    df = pd.DataFrame(data_for_df)
    
    # Define column order for clarity
    ordered_cols = ["filepath"]
    for field in FIELDS:
        name = field["name"]
        ordered_cols.extend([name, f"{name}_confidence", f"{name}_reason"])
    ordered_cols.append("error")
    
    # Filter to only include columns that actually exist in the DataFrame
    df = df.reindex(columns=[col for col in ordered_cols if col in df.columns])

    df.to_excel(excel_path, index=False, engine='xlsxwriter')
    logger.info(f"Excel report generated successfully at: {excel_path}")
    return excel_path


def background_processing_task(file_contents: List[bytes], file_names: List[str], job_id: str):
    """
    The main background task that handles unzipping, processing, and reporting.
    """
    job_start_time = time.time()
    logger.info(f"Starting background job {job_id} for {len(file_names)} zip file(s).")

    # Create a temporary directory for this job
    temp_dir = tempfile.mkdtemp(prefix=f"cheque_job_{job_id}_")
    
    try:
        # --- 1. Unzip all files and collect images ---
        images_to_process = []
        for zip_content, zip_name in zip(file_contents, file_names):
            with zipfile.ZipFile(io.BytesIO(zip_content)) as zf:
                for member in zf.infolist():
                    if member.is_dir() or member.filename.startswith('__MACOSX'):
                        continue # Skip directories and macOS metadata

                    _, ext = os.path.splitext(member.filename)
                    supported_ext = {'.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png', '.tiff': 'image/tiff'}
                    if ext.lower() in supported_ext:
                        images_to_process.append({
                            'path': member.filename,
                            'data': zf.read(member),
                            'type': supported_ext[ext.lower()]
                        })

        total_images = len(images_to_process)
        if total_images == 0:
            raise ValueError("No supported image files (.jpg, .png, .tiff) found in the uploaded zip archives.")

        logger.info(f"Job {job_id}: Found {total_images} images to process.")
        processed_jobs[job_id]["total_files"] = total_images

        # --- 2. Process images in batches ---
        all_results = []
        processor = ChequeProcessor()
        
        for i in range(0, total_images, API_CALL_BATCH_SIZE):
            batch = images_to_process[i:i + API_CALL_BATCH_SIZE]
            logger.info(f"Job {job_id}: Processing batch {i//API_CALL_BATCH_SIZE + 1} of { (total_images + API_CALL_BATCH_SIZE - 1)//API_CALL_BATCH_SIZE }...")
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(batch)) as batch_executor:
                future_to_image = {batch_executor.submit(processor.process_single_image, img): img for img in batch}
                for future in concurrent.futures.as_completed(future_to_image):
                    all_results.append(future.result())
            
            processed_jobs[job_id]["processed_files"] = len(all_results)
            logger.info(f"Job {job_id}: Progress {len(all_results)}/{total_images} images.")

        # --- 3. Generate final report ---
        output_file_path = generate_excel_report(all_results, temp_dir, job_id)

        # --- 4. Update job status to 'completed' ---
        job_end_time = time.time()
        processed_jobs[job_id].update({
            "status": "completed",
            "end_time": job_end_time,
            "output_file_path": output_file_path,
            "processing_duration_seconds": round(job_end_time - job_start_time, 2),
            "results": f"/download/{job_id}"
        })
        logger.info(f"Job {job_id} completed in {processed_jobs[job_id]['processing_duration_seconds']}s.")

    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}", exc_info=True)
        processed_jobs[job_id].update({
            "status": "failed",
            "end_time": time.time(),
            "error_message": str(e)
        })
    # Note: We do not clean up the temp_dir here to allow for downloading the file.
    # A separate cleanup mechanism would be needed for a production system.


# ==============================================================================
# 5. FASTAPI ENDPOINTS
# ==============================================================================

@app.post("/upload", status_code=202)
async def upload_cheque_zips(background_tasks: BackgroundTasks, files: List[UploadFile] = File(...)):
    """
    Upload one or more ZIP files containing cropped cheque images.
    This endpoint initiates a background job for processing.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files were uploaded.")
    
    file_contents = []
    file_names = []
    for file in files:
        if not file.filename.lower().endswith('.zip'):
            raise HTTPException(status_code=400, detail=f"Invalid file type: '{file.filename}'. Only .zip files are allowed.")
        file_contents.append(await file.read())
        file_names.append(file.filename)

    job_id = str(uuid.uuid4())
    
    # Set up initial job status
    processed_jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "start_time": datetime.now().isoformat(),
        "input_files": file_names,
        "total_files": 0,
        "processed_files": 0
    }

    # Add the long-running task to the background
    background_tasks.add_task(background_processing_task, file_contents, file_names, job_id)
    
    return {
        "message": "Upload successful. Processing has started in the background.",
        "job_id": job_id,
        "status_url": f"/status/{job_id}"
    }


@app.get("/status/{job_id}")
async def get_job_status(job_id: str):
    """Check the status of a specific processing job."""
    job = processed_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job ID '{job_id}' not found.")
    return job


@app.get("/download/{job_id}")
async def download_excel_report(job_id: str):
    """Download the final Excel report for a completed job."""
    job = processed_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job ID '{job_id}' not found.")
    
    if job.get("status") != "completed":
        raise HTTPException(status_code=400, detail=f"Job is not yet complete. Current status: '{job.get('status')}'.")
        
    output_path = job.get("output_file_path")
    if not output_path or not os.path.exists(output_path):
        logger.error(f"Output file for completed job {job_id} not found at path: {output_path}")
        raise HTTPException(status_code=404, detail="Result file not found, it may have been cleaned up.")

    return FileResponse(
        path=output_path,
        filename=os.path.basename(output_path),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

@app.get("/", include_in_schema=False)
def root():
    return {"message": "Cheque Extraction API is running. See /docs for API documentation."}


if __name__ == "__main__":
    # To run this app:
    # 1. Make sure you have the required libraries:
    #    pip install "fastapi[all]" uvicorn pandas openpyxl xlsxwriter google-cloud-aiplatform
    # 2. Set up Google Cloud authentication in your environment:
    #    gcloud auth application-default login
    # 3. Run the server:
    #    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)