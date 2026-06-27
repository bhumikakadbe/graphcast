# cloud_access.py
import os
from typing import List, Optional
from google.cloud import storage
import requests
from production_pipeline.utils import logger, retry

class CloudAccessClient:
    """Manages secure access, authentication, and data retrieval from Cloud Storage buckets."""
    
    def __init__(self, google_creds_path: Optional[str] = None):
        self.google_creds_path = google_creds_path or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        self.gcs_client = self._init_gcs_client()
        
    def _init_gcs_client(self) -> storage.Client:
        """Initializes GCS Client. Falls back to anonymous client if credentials are not found."""
        if self.google_creds_path and os.path.exists(self.google_creds_path):
            try:
                logger.info(f"Authenticating with GCS using service account key: {self.google_creds_path}")
                return storage.Client.from_service_account_json(self.google_creds_path)
            except Exception as e:
                logger.warning(f"Failed to authenticate with service account, falling back to anonymous: {e}")
                
        logger.info("Initializing anonymous Google Cloud Storage Client (Public Buckets Only).")
        return storage.Client.create_anonymous_client()

    @retry(max_retries=5, initial_backoff=2.0)
    def download_gcs_blob(self, bucket_name: str, blob_name: str, destination_path: str):
        """Downloads a file from a Google Cloud Storage bucket with fault tolerance."""
        logger.info(f"Downloading from GCS: gs://{bucket_name}/{blob_name} -> {destination_path}")
        os.makedirs(os.path.dirname(destination_path), exist_ok=True)
        
        bucket = self.gcs_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        
        # Download and write chunk by chunk to prevent loading huge files into memory
        with open(destination_path, "wb") as f:
            blob.download_to_file(f)
            
        logger.info(f"Successfully downloaded GCS blob. File size: {os.path.getsize(destination_path) / (1024**2):.2f} MB")

    @retry(max_retries=3, initial_backoff=1.0)
    def list_gcs_blobs(self, bucket_name: str, prefix: str) -> List[str]:
        """Lists blobs in a GCS bucket under a specific prefix."""
        logger.info(f"Listing blobs in gs://{bucket_name}/{prefix} ...")
        bucket = self.gcs_client.bucket(bucket_name)
        blobs = bucket.list_blobs(prefix=prefix)
        blob_names = [blob.name for blob in blobs]
        logger.info(f"Found {len(blob_names)} blobs.")
        return blob_names

    @retry(max_retries=5, initial_backoff=3.0)
    def download_http_file(self, url: str, destination_path: str):
        """Downloads file from AWS S3 (NOAA public datasets) or ECMWF via HTTP."""
        logger.info(f"Downloading via HTTP: {url} -> {destination_path}")
        os.makedirs(os.path.dirname(destination_path), exist_ok=True)
        
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(destination_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1MB chunk size
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        if int(percent) % 20 == 0:  # Log every 20%
                            logger.debug(f"Download progress: {percent:.1f}%")
                            
        logger.info(f"HTTP download completed. Total size: {os.path.getsize(destination_path) / (1024**2):.2f} MB")

    def validate_local_dataset(self, file_path: str, expected_min_size_mb: float = 1.0) -> bool:
        """Validates if downloaded file exists, is not empty, and exceeds minimum size."""
        if not os.path.exists(file_path):
            logger.error(f"Validation failed: File {file_path} does not exist.")
            return False
        
        file_size_mb = os.path.getsize(file_path) / (1024**2)
        if file_size_mb < expected_min_size_mb:
            logger.error(f"Validation failed: File {file_path} size ({file_size_mb:.2f}MB) is smaller than expected ({expected_min_size_mb}MB).")
            return False
            
        logger.info(f"Validation successful for {file_path} ({file_size_mb:.2f}MB).")
        return True

# Singleton Instance for internal library usage
cloud_client = CloudAccessClient()
