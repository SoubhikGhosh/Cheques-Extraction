# main.py

from fastapi import FastAPI, UploadFile, File, HTTPException
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
from PIL import Image, ImageFile
from dateutil import parser # Import for robust date parsing

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ==============================================================================
# 1. LOGGING AND APP CONFIGURATION
# ==============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Standardized Cheque Extraction API",
    description="Uses dynamic cropping, strict confidence scoring, and robust post-processing to extract and standardize cheque data.",
    version="3.5.0" # Version updated for enhanced date standardization
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# ==============================================================================
# 2. VERTEX AI AND CONSTANTS CONFIGURATION
# ==============================================================================

GCP_PROJECT_ID = "hbl-uat-ocr-fw-app-prj-spk-4d"
GCP_LOCATION = "asia-south1"
GCP_API_ENDPOINT = "asia-south1-aiplatform.googleapis.com"

try:
    vertexai.init(project=GCP_PROJECT_ID, location=GCP_LOCATION, api_endpoint=GCP_API_ENDPOINT)
    logger.info(f"Vertex AI initialized successfully for project '{GCP_PROJECT_ID}'.")
except Exception as e:
    logger.error(f"Fatal: Could not initialize Vertex AI. Error: {e}")

SAFETY_SETTINGS = [
    SafetySetting(category=c, threshold=SafetySetting.HarmBlockThreshold.BLOCK_NONE)
    for c in SafetySetting.HarmCategory
]

FIELDS_TO_EXTRACT = ["date", "amount"]
MAX_CONCURRENT_WORKERS = 50
API_CALL_BATCH_SIZE = 10
MAX_API_RETRIES = 5
executor = concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT_WORKERS)
active_tasks = {}
processed_jobs = {}

# ==============================================================================
# 3. CHEQUE PROCESSOR CLASS (CORE LOGIC)
# ==============================================================================

class ChequeProcessor:
    """Handles image cropping, field extraction, and data standardization."""

    def __init__(self, model_name: str = "gemini-1.5-flash-002"):
        self.model = GenerativeModel(model_name, safety_settings=SAFETY_SETTINGS)
        self.current_year = datetime.now().year
        self.previous_year = self.current_year - 1

    def _call_vertex_ai_with_retry(self, prompt: str, image_part: Part) -> Any:
        delay = 1.0
        retryable_errors = (google_exceptions.ResourceExhausted, google_exceptions.TooManyRequests)
        for attempt in range(MAX_API_RETRIES):
            try:
                return self.model.generate_content([prompt, image_part], generation_config={"temperature": 0.0})
            except retryable_errors as e:
                logger.warning(f"API Error ({type(e).__name__}) on attempt {attempt + 1}. Retrying in {delay:.1f}s...")
                time.sleep(delay + (uuid.uuid4().int % 1000) / 1000.0)
                delay *= 2.0
            except Exception as e:
                logger.error(f"Non-retryable API error: {e}", exc_info=True)
                raise
        raise google_exceptions.RetryError(f"API call failed after {MAX_API_RETRIES} retries.", None)

    @staticmethod
    def _extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
        match = re.search(r'```json\s*(\{[\s\S]*?\})\s*```', text, re.DOTALL)
        json_str = match.group(1) if match else text[text.find('{'):text.rfind('}')+1]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            logger.error(f"Failed to decode JSON from response: {json_str[:250]}...")
            return None

    def _get_field_prompt(self, field_name: str) -> str:
        confidence_instructions = (
            f"**Confidence Scoring (Strict, Character-Informed, and Defensible):**\n"
            f"1.  **Core Principle:** The confidence score is a calculated metric of certainty. A field's overall confidence is **limited by the lowest confidence of any of its individual characters.**\n"
            f"2.  **Calculation Basis:** Your score must integrate: a) Visual quality (blur, smudge), b) Handwriting legibility (clear print vs. messy cursive), c) Character ambiguity ('O' vs '0', '1' vs '7'), and d) Adherence to format rules.\n"
            f"3.  **Strict Benchmarks (Non-Negotiable):**\n"
            f"    - **0.98 - 1.00 (Perfect/Production Ready):** Absolute certainty. Machine-printed or exceptionally clear handwriting. No plausible alternative for any character.\n"
            f"    - **0.90 - 0.97 (High/Review Recommended):** Strong confidence, but with minor imperfections, like slight character ambiguity that context strongly overrules.\n"
            f"    - **0.75 - 0.89 (Moderate/Human Review Required):** Reasonable confidence, but with documented uncertainty. Use this if one or two characters are genuinely ambiguous (e.g., a handwritten '4' that resembles a '9').\n"
            f"    - **< 0.75 (Low/Unreliable):** Significant uncertainty. Multiple characters are ambiguous, text is smudged, or handwriting is barely legible.\n"
            f"4.  **Mandatory Justification:** For any confidence score below **0.95**, you are **REQUIRED** to provide a concise `reason` field in the JSON, pinpointing the source of uncertainty."
        )
        
        field_specific_prompts = {
            "date": (
                f"You are a hyper-precise OCR engine specializing in messy, handwritten financial documents. Your task is to extract the 8-digit date from a pre-cropped image of a cheque's date grid.\n\n"
                f"**CRITICAL INTERNAL PROCESS:**\n"
                f"1.  **Digit-by-Digit Analysis:** Your primary challenge is to **mentally erase the printed box lines** and focus *only* on the handwritten ink that forms the digits. A vertical box line next to a '1' does NOT make it a '4' and similarly a line in front of a '3' does NOT make it an '8'.\n"
                f"2.  **Apply Temporal Rule:** The current year is **{self.current_year}**. A valid cheque will be for **{self.current_year}** or late **{self.previous_year}**. A year like '{self.current_year + 1}' is invalid. Use this rule to disambiguate OCR errors. If the last digit of the year is ambiguous between a '{str(self.current_year)[-1]}' and '{str(self.current_year+1)[-1]}', you MUST conclude it is '{str(self.current_year)[-1]}'.\n\n"

                f"{confidence_instructions}\n\n"
                f"**FINAL OUTPUT:** A single, valid JSON object with 'value' (in 'DD-MM-YYYY' format or empty string), 'confidence' (float), and an optional 'reason' (string)."
            ),
            "amount": (
                f"You are an expert financial OCR system. Your task is to extract the numeric 'courtesy amount' from a pre-cropped image of a cheque's amount box.\n\n"
                f"**CRITICAL INTERNAL PROCESS:**\n"
                f"1.  **Handwriting Analysis:** Pay extremely close attention to the handwritten numbers. Differentiate common confusions: '1' vs '7', '5' vs 'S', '0' vs '6', '2' vs 'Z'.\n"
                f"2.  **Apply Strict Filtering Rule:** Discard ALL non-numeric characters ('₹', commas, '/-'). The ONLY allowed non-digit character is a single decimal point (.).\n"
                f"3.  **Standardize Format:** Format the cleaned number to have exactly two decimal places (e.g., '887' becomes '887.00').\n\n"
                f"{confidence_instructions}\n\n"
                f"**FINAL OUTPUT:** A single, valid JSON object with 'value' (string, e.g., '5000.00' or empty), 'confidence' (float), and an optional 'reason' (string)."
            )
        }
        
        base_prompt = (
            f"{field_specific_prompts.get(field_name, 'Extract the text visible in the image.')}\n\n"
            "Example JSON Output: {\"value\": \"1234.50\", \"confidence\": 0.88, \"reason\": \"Handwritten '4' is unclear and could be a '9'.\"}"
        )
        return base_prompt

    def _extract_field_from_crop(self, cropped_image_bytes: bytes, field_name: str) -> Dict[str, Any]:
        """Sends a cropped image to Vertex AI and standardizes the result."""
        prompt = self._get_field_prompt(field_name)
        image_part = Part.from_data(data=cropped_image_bytes, mime_type='image/png')
        
        try:
            response = self._call_vertex_ai_with_retry(prompt, image_part)
            extracted_data = self._extract_json_from_text(response.text)
            
            if extracted_data and 'value' in extracted_data:
                # --- Post-processing and Standardization Logic ---
                if field_name == "date":
                    raw_date_value = extracted_data.get("value")
                    if raw_date_value and isinstance(raw_date_value, str) and len(raw_date_value) >= 6:
                        
                        # --- NEW: Pre-processing for separator-less DDMMYYYY format ---
                        if len(raw_date_value) == 8 and raw_date_value.isdigit():
                            # Help the parser by formatting DDMMYYYY to DD-MM-YYYY
                            raw_date_value = f"{raw_date_value[:2]}-{raw_date_value[2:4]}-{raw_date_value[4:]}"
                        
                        try:
                            # Use dateutil.parser for robust parsing. dayfirst=True is crucial.
                            parsed_date = parser.parse(raw_date_value, dayfirst=True)
                            standardized_date = parsed_date.strftime('%d-%m-%Y')
                            extracted_data["value"] = standardized_date
                            logger.info(f"Standardized date '{raw_date_value}' to '{standardized_date}'")
                        except (parser.ParserError, TypeError, ValueError) as e:
                            logger.warning(f"Could not parse date '{raw_date_value}'. Keeping original from AI. Error: {e}")

                return {
                    "field_name": field_name, "value": extracted_data.get("value", ""),
                    "confidence": extracted_data.get("confidence", 0.0), "reason": extracted_data.get("reason")
                }
        except Exception as e:
            logger.error(f"Extraction failed for field '{field_name}'. Error: {e}")
        
        return {"field_name": field_name, "value": "", "confidence": 0.0, "reason": "Extraction failed"}

    def process_full_cheque_image(self, file_info: Dict) -> Dict:
        """Orchestrates the full process for a single image: crop -> extract -> aggregate."""
        file_path = file_info['path']
        logger.info(f"Processing full cheque image: {file_path}")
        results = {"file_path": file_path, "extracted_fields": []}
        try:
            image = Image.open(io.BytesIO(file_info['data']))
            image = image.convert("RGB")
            width, height = image.size
            date_coords = (int(2 * width / 3), 0, width, int(height / 3))
            date_crop = image.crop(date_coords)
            with io.BytesIO() as byte_arr:
                date_crop.save(byte_arr, format='PNG')
                results["extracted_fields"].append(self._extract_field_from_crop(byte_arr.getvalue(), "date"))
            amount_coords = (int(2 * width / 3), int(height / 3), width, int(2 * height / 3))
            amount_crop = image.crop(amount_coords)
            with io.BytesIO() as byte_arr:
                amount_crop.save(byte_arr, format='PNG')
                results["extracted_fields"].append(self._extract_field_from_crop(byte_arr.getvalue(), "amount"))
            return results
        except Exception as e:
            logger.error(f"Failed to process image {file_path}. Error: {e}", exc_info=True)
            results["error"] = str(e)
            return results

# ==============================================================================
# 4. BACKGROUND TASK AND BATCH PROCESSING
# ==============================================================================

def generate_excel_report(results: List[Dict], output_dir: str, job_id: str) -> str:
    excel_path = os.path.join(output_dir, f"cheque_extraction_report_{job_id}.xlsx")
    data_for_df = []
    for item in results:
        row = {"filepath": item.get("file_path", "Unknown")}
        if "error" in item:
            row["error"] = item["error"]
        for field_result in item.get("extracted_fields", []):
            name = field_result.get("field_name")
            if name:
                row[name] = field_result.get("value")
                row[f"{name}_confidence"] = field_result.get("confidence")
                row[f"{name}_reason"] = field_result.get("reason")
        data_for_df.append(row)
    if not data_for_df:
        logger.warning(f"Job {job_id}: No data to generate a report.")
        return ""
    df = pd.DataFrame(data_for_df)
    ordered_cols = ["filepath"] + [col for name in FIELDS_TO_EXTRACT for col in (name, f"{name}_confidence", f"{name}_reason")] + ["error"]
    df = df.reindex(columns=[col for col in ordered_cols if col in df.columns])
    df.to_excel(excel_path, index=False, engine='xlsxwriter')
    logger.info(f"Excel report generated: {excel_path}")
    return excel_path

def background_processing_task(file_contents: List[bytes], file_names: List[str], job_id: str):
    job_start_time = time.time()
    logger.info(f"Starting background job {job_id} for {len(file_names)} zip file(s).")
    temp_dir = tempfile.mkdtemp(prefix=f"cheque_job_{job_id}_")
    try:
        images_to_process = []
        supported_ext = {'.jpg', '.jpeg', '.png', '.tiff', '.tif'}
        for zip_content, zip_name in zip(file_contents, file_names):
            zip_extract_path = os.path.join(temp_dir, os.path.splitext(zip_name)[0])
            with zipfile.ZipFile(io.BytesIO(zip_content)) as zf:
                zf.extractall(zip_extract_path)
            for root, _, files in os.walk(zip_extract_path):
                if '__MACOSX' in root.split(os.sep):
                    continue
                for file in files:
                    if file.startswith('._'):
                        continue
                    if os.path.splitext(file)[1].lower() in supported_ext:
                        full_path = os.path.join(root, file)
                        with open(full_path, 'rb') as f_img:
                            images_to_process.append({
                                'path': os.path.relpath(full_path, temp_dir),
                                'data': f_img.read()
                            })
        total_images = len(images_to_process)
        if total_images == 0:
            raise ValueError("No valid image files found in the zip archives after filtering.")
        logger.info(f"Job {job_id}: Found {total_images} valid images to process.")
        processed_jobs[job_id]["total_files"] = total_images
        all_results = []
        processor = ChequeProcessor()
        for i in range(0, total_images, API_CALL_BATCH_SIZE):
            batch = images_to_process[i:i + API_CALL_BATCH_SIZE]
            logger.info(f"Job {job_id}: Processing batch {i//API_CALL_BATCH_SIZE + 1}/{ (total_images + API_CALL_BATCH_SIZE - 1)//API_CALL_BATCH_SIZE }...")
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(batch)) as batch_executor:
                future_to_image = {batch_executor.submit(processor.process_full_cheque_image, img): img for img in batch}
                for future in concurrent.futures.as_completed(future_to_image):
                    all_results.append(future.result())
            processed_jobs[job_id]["processed_files"] = len(all_results)
            logger.info(f"Job {job_id}: Progress {len(all_results)}/{total_images} images.")
        output_file_path = generate_excel_report(all_results, temp_dir, job_id)
        job_end_time = time.time()
        processed_jobs[job_id].update({
            "status": "completed", "end_time": job_end_time, "output_file_path": output_file_path,
            "processing_duration_seconds": round(job_end_time - job_start_time, 2),
            "results_url": f"/download/{job_id}"
        })
        logger.info(f"Job {job_id} completed successfully.")
    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}", exc_info=True)
        processed_jobs[job_id].update({"status": "failed", "end_time": time.time(), "error_message": str(e)})

# ==============================================================================
# 5. FASTAPI ENDPOINTS
# ==============================================================================

@app.post("/upload", status_code=202)
async def upload_and_process(files: List[UploadFile] = File(...)):
    file_contents = [await f.read() for f in files]
    file_names = [f.filename for f in files]
    job_id = str(uuid.uuid4())
    processed_jobs[job_id] = {
        "job_id": job_id, "status": "queued", "start_time": datetime.now().isoformat(),
        "input_files": file_names, "total_files": 0, "processed_files": 0
    }
    future = executor.submit(background_processing_task, file_contents, file_names, job_id)
    active_tasks[job_id] = future
    return {"message": "Upload successful. Processing has started.", "job_id": job_id, "status_url": f"/status/{job_id}"}

@app.get("/status/{job_id}")
async def get_job_status(job_id: str):
    job = processed_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job ID '{job_id}' not found.")
    if job.get("status") in ["queued", "processing"]:
        future = active_tasks.get(job_id)
        if future and future.running():
            job["status"] = "processing"
        elif future and future.done() and job.get("status") != "completed":
            job = processed_jobs.get(job_id, job)
    return job

@app.get("/download/{job_id}")
async def download_report(job_id: str):
    job = processed_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job.get("status") != "completed":
        raise HTTPException(status_code=400, detail=f"Job not complete. Status: '{job.get('status')}'.")
    output_path = job.get("output_file_path")
    if not output_path or not os.path.exists(output_path):
        raise HTTPException(status_code=404, detail="Result file not found.")
    return FileResponse(path=output_path, filename=os.path.basename(output_path))

@app.get("/", include_in_schema=False)
def root():
    return {"message": "Standardized Cheque Extraction API is running. See /docs for details."}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)