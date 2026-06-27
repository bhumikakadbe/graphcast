# Setup & Environment Configuration Guide

This guide explains how to set up the development environment for the GraphCast weather forecasting pipeline.

---

## 0. ERA5 CDS API Configuration (Required for Data Downloads)

ERA5 reanalysis data is downloaded from the **Copernicus Climate Data Store (CDS)** using the `cdsapi` Python library.

### Step 1: Register & Obtain API Key

1. Register at [https://cds.climate.copernicus.eu/](https://cds.climate.copernicus.eu/)
2. Log in and accept the **ERA5 licence** at [https://cds.climate.copernicus.eu/cdsapp#!/terms/licence-to-use-copernicus-products](https://cds.climate.copernicus.eu/cdsapp#!/terms/licence-to-use-copernicus-products)
3. Visit [https://cds.climate.copernicus.eu/api-how-to](https://cds.climate.copernicus.eu/api-how-to) to find your **UID** and **API Key**

### Step 2: Create the API Credentials File

Create the file `~/.cdsapirc` (Linux/macOS) or `%USERPROFILE%\.cdsapirc` (Windows):

```
url: https://cds.climate.copernicus.eu/api/v2
key: <YOUR-UID>:<YOUR-API-KEY>
verify: 0
```

> ⚠️ **Security**: Never commit this file to GitHub. It is already in `.gitignore`.

### Step 3: Install the CDS Client

```bash
pip install cdsapi
```

### Step 4: Test the Connection

```bash
# Test with a 1-day download (fast, ~2 MB)
python data_collection/era5_downloader.py --test --region nagpur
```

Expected output:
```
✅ CDS API test PASSED. Variables: ['t2m', 'msl']
   Grid: lat=33, lon=45
```

### Step 5: Estimate Storage Before Downloading

```bash
# How much space does 2015–2018 Nagpur data need?
python data_collection/era5_downloader.py --estimate --start-year 2015 --end-year 2018 --region nagpur
```

### Step 6: Download ERA5 2015 (Nagpur Region)

```bash
# Download all of 2015
python data_collection/era5_downloader.py --year 2015 --region nagpur

# Download a single month (for testing)
python data_collection/era5_downloader.py --year 2015 --month 6 --region nagpur

# Download multiple years
python data_collection/era5_downloader.py --start-year 2015 --end-year 2018 --region nagpur
```

Downloaded files are saved to: `data/ERA5/raw/{year}/`

### ERA5 Variables Downloaded

| Type | Variables |
|---|---|
| **Pressure levels (13 levels)** | Temperature, Specific Humidity, U/V Wind, Geopotential, Vertical Velocity |
| **Surface** | 2m Temperature, 10m Wind U/V, MSLP, Total Precipitation, Cloud Cover, Land-Sea Mask |
| **Temporal** | 6-hourly: 00:00, 06:00, 12:00, 18:00 UTC |
| **Region** | Nagpur: 17°N–25°N, 74°E–85°E (default) |

---

## 1. CUDA & GPU Setup Instructions

GraphCast utilizes JAX under the hood, which relies on the XLA (Accelerated Linear Algebra) compiler for compilation and execution on GPUs. To enable GPU acceleration, ensure the following are installed on your host system:

*   **NVIDIA Driver**: Version `520.x` or higher.
*   **CUDA Toolkit**: Version `11.8` or `12.x`.
*   **cuDNN**: Version `8.6` or higher (compatible with your CUDA version).

### JAX GPU Memory Preallocation

By default, JAX preallocates 75% of the total GPU memory when the first JAX operation is run. In a multi-tenant or pipeline environment, this can lead to Out-Of-Memory (OOM) errors. You can control this behavior using the following environment variables:

```bash
# Disable JAX preallocation (allocate memory dynamically as needed)
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Alternatively, set a specific preallocation fraction (e.g., 40%)
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.40
```

---

## 2. Windows WSL2 Setup Guide

Running JAX with GPU support is highly recommended under Windows Subsystem for Linux (WSL2).

### Step 1: Install WSL2 and Ubuntu
Open PowerShell as Administrator and run:
```powershell
wsl --install -d Ubuntu-22.04
```
Ensure you have the latest NVIDIA Drivers installed on the Windows host. WSL2 will automatically bridge the GPU.

### Step 2: Install Miniconda in WSL2
Inside your WSL2 Ubuntu terminal:
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda
echo 'export PATH="$HOME/miniconda/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

### Step 3: Install Compiler Tools & ecCodes (For PyGRIB/Cartopy)
```bash
sudo apt-get update && sudo apt-get install -y \
    build-essential \
    libeccodes-dev \
    libproj-dev \
    proj-data \
    proj-bin \
    libgeos-dev
```

### Step 4: Build Conda Environment
```bash
conda env create -f environment.yml
conda activate graphcast
```

### Step 5: Verify JAX GPU Access
```python
python -c "import jax; print('GPU Device Count:', jax.device_count(), 'Devices:', jax.devices())"
```

---

## 3. Linux Production Server Setup Guide

For dedicated Linux weather servers, installing through Docker ensures maximum reproducibility and isolates drivers.

### Using Docker
1. Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) to pass host GPUs into the container.
2. Build the Docker image:
   ```bash
   docker build -t graphcast-pipeline .
   ```
3. Run the container:
   ```bash
   docker run --gpus all -p 8000:8000 -p 5000:5000 \
     -e GOOGLE_APPLICATION_CREDENTIALS=/app/gcs-key.json \
     -v /path/to/credentials.json:/app/gcs-key.json \
     graphcast-pipeline
   ```

---

## 4. Google Colab Compatibility Guide

To run this pipeline inside a free or premium Google Colab environment:

1. Create a new notebook with a **GPU** runtime (T4, V100, or A100).
2. Install Cartopy and ecCodes dependencies:
   ```bash
   !apt-get install -y libeccodes-dev libproj-dev proj-data proj-bin libgeos-dev
   ```
3. Install required Python packages:
   ```bash
   !pip install dm-haiku dm-tree jraph chex optax trimesh xarray-tensorstore pygrib gcsfs google-cloud-storage fastapi uvicorn flask
   ```
4. Verify CUDA:
   ```python
   import jax
   print(jax.devices()) # Should print [GpuDevice(id=0)]
   ```

---

## 5. Distributed Training & TPU Compatibility

GraphCast supports multi-device and TPU configurations through JAX's native model parallelism.

### Distributed multi-node training:
On an HPC cluster running SLURM, configure JAX to connect devices across nodes:

```python
import jax

# Initialize JAX distributed API
jax.distributed.initialize()

print("Global device count:", jax.device_count())
print("Local device count:", jax.local_device_count())
```

### SLURM Submission Script template:
```bash
#!/bin/bash
#SBATCH --job-name=graphcast-train
#SBATCH --nodes=2
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu

# Multi-node JAX environment config
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=12345
export NODE_RANK=$SLURM_NODEID

# Run the training script
python production_pipeline/training.py
```
