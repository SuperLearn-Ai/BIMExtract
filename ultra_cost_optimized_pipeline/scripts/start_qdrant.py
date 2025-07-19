"""
Script to start Qdrant vector database for ultra-cost-optimized pipeline
"""

import os
import sys
import subprocess
import time
import logging
import requests
from pathlib import Path
import yaml

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_config():
    """Load pipeline configuration"""
    config_path = Path(__file__).parent.parent / "config" / "pipeline_config.yaml"
    
    if not config_path.exists():
        logging.error(f"Configuration file not found: {config_path}")
        return None
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config.get('vector_storage', {}).get('qdrant', {})


def check_docker_installed():
    """Check if Docker is installed"""
    try:
        result = subprocess.run(['docker', '--version'], 
                              capture_output=True, text=True, check=True)
        logging.info(f"Docker found: {result.stdout.strip()}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        logging.error("Docker is not installed or not in PATH")
        return False


def check_qdrant_running(host="localhost", port=6333):
    """Check if Qdrant is already running"""
    try:
        response = requests.get(f"http://{host}:{port}")
        if response.status_code == 200:
            logging.info(f"Qdrant is already running on {host}:{port}")
            return True
    except requests.exceptions.ConnectionError:
        pass
    
    return False


def start_qdrant_docker(config):
    """Start Qdrant using Docker"""
    host = config.get('host', 'localhost')
    port = config.get('port', 6333)
    
    # Create data directory for persistence
    data_dir = Path.cwd() / "qdrant_data"
    data_dir.mkdir(exist_ok=True)
    
    # Docker command to start Qdrant
    docker_cmd = [
        'docker', 'run', '-d',
        '--name', 'qdrant_pipeline',
        '-p', f'{port}:6333',
        '-v', f'{data_dir.absolute()}:/qdrant/storage',
        '--restart', 'unless-stopped',
        'qdrant/qdrant:v1.7.0'
    ]
    
    try:
        # Stop existing container if running
        subprocess.run(['docker', 'stop', 'qdrant_pipeline'], 
                      capture_output=True, check=False)
        subprocess.run(['docker', 'rm', 'qdrant_pipeline'], 
                      capture_output=True, check=False)
        
        # Start new container
        logging.info("Starting Qdrant container...")
        result = subprocess.run(docker_cmd, capture_output=True, text=True, check=True)
        
        container_id = result.stdout.strip()
        logging.info(f"Qdrant container started: {container_id}")
        
        # Wait for Qdrant to be ready
        logging.info("Waiting for Qdrant to be ready...")
        for i in range(30):  # Wait up to 30 seconds
            if check_qdrant_running(host, port):
                logging.info("Qdrant is ready!")
                return True
            time.sleep(1)
        
        logging.error("Qdrant failed to start within 30 seconds")
        return False
        
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to start Qdrant container: {e}")
        logging.error(f"Error output: {e.stderr}")
        return False


def start_qdrant_binary(config):
    """Start Qdrant using binary (if available)"""
    host = config.get('host', 'localhost')
    port = config.get('port', 6333)
    
    # Check if qdrant binary is available
    try:
        subprocess.run(['qdrant', '--version'], 
                      capture_output=True, text=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        logging.error("Qdrant binary not found in PATH")
        return False
    
    # Create storage directory
    storage_dir = Path.cwd() / "qdrant_storage"
    storage_dir.mkdir(exist_ok=True)
    
    # Start Qdrant
    try:
        logging.info("Starting Qdrant binary...")
        process = subprocess.Popen([
            'qdrant',
            '--config-path', str(storage_dir / 'config.yaml'),
            '--storage-path', str(storage_dir)
        ])
        
        # Wait for Qdrant to be ready
        logging.info("Waiting for Qdrant to be ready...")
        for i in range(30):
            if check_qdrant_running(host, port):
                logging.info("Qdrant is ready!")
                return True
            time.sleep(1)
        
        logging.error("Qdrant failed to start within 30 seconds")
        process.terminate()
        return False
        
    except Exception as e:
        logging.error(f"Failed to start Qdrant binary: {e}")
        return False


def create_qdrant_config(config):
    """Create optimized Qdrant configuration"""
    storage_dir = Path.cwd() / "qdrant_storage"
    storage_dir.mkdir(exist_ok=True)
    
    config_content = f"""
# Ultra-Cost-Optimized Qdrant Configuration
storage:
  storage_path: {storage_dir.absolute()}
  snapshots_path: {storage_dir.absolute()}/snapshots
  temp_path: {storage_dir.absolute()}/temp
  
  # Optimize for cost-efficiency
  on_disk_payload: true
  wal_capacity_mb: 32
  wal_segments_ahead: 0
  
  # Performance optimizations for self-hosted
  performance:
    max_search_threads: 0  # Use all available cores
    max_optimization_threads: 2
    
service:
  host: {config.get('host', 'localhost')}
  http_port: {config.get('port', 6333)}
  grpc_port: {config.get('port', 6333) + 1}
  
  # Logging
  log_level: INFO
  
  # Telemetry (disable for privacy)
  telemetry_disabled: true

# Cluster configuration (for single node)
cluster:
  enabled: false
  
# Optimizations for cost efficiency
collection_settings:
  # Default settings for new collections
  vectors:
    on_disk: true
  
  # Quantization enabled by default
  quantization:
    scalar:
      type: int8
      quantile: 0.99
      always_ram: false
"""
    
    config_path = storage_dir / "config.yaml"
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    logging.info(f"Created Qdrant config: {config_path}")


def main():
    """Main function to start Qdrant"""
    logging.info("Starting Qdrant for ultra-cost-optimized pipeline...")
    
    # Load configuration
    config = load_config()
    if not config:
        sys.exit(1)
    
    host = config.get('host', 'localhost')
    port = config.get('port', 6333)
    
    # Check if already running
    if check_qdrant_running(host, port):
        logging.info("Qdrant is already running, no action needed")
        return
    
    # Create optimized configuration
    create_qdrant_config(config)
    
    # Try Docker first, then binary
    success = False
    
    if check_docker_installed():
        logging.info("Attempting to start Qdrant using Docker...")
        success = start_qdrant_docker(config)
    
    if not success:
        logging.info("Attempting to start Qdrant using binary...")
        success = start_qdrant_binary(config)
    
    if success:
        logging.info(f"✅ Qdrant started successfully on {host}:{port}")
        logging.info("You can access the Qdrant dashboard at http://localhost:6333/dashboard")
        logging.info("Use Ctrl+C to stop the service when running in foreground")
    else:
        logging.error("❌ Failed to start Qdrant")
        logging.error("Please install Docker or Qdrant binary manually")
        logging.error("Installation instructions:")
        logging.error("  Docker: https://docs.docker.com/get-docker/")
        logging.error("  Qdrant: https://qdrant.tech/documentation/quick-start/")
        sys.exit(1)


if __name__ == "__main__":
    main()