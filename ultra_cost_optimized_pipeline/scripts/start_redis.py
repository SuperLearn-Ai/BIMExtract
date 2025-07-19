"""
Script to start Redis cache service for ultra-cost-optimized pipeline
"""

import os
import sys
import subprocess
import time
import logging
import redis
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
    
    return config.get('local_llm', {}).get('caching', {})


def check_redis_running(host="localhost", port=6379, password=None):
    """Check if Redis is already running"""
    try:
        client = redis.Redis(
            host=host, 
            port=port, 
            password=password,
            socket_connect_timeout=2,
            socket_timeout=2
        )
        client.ping()
        logging.info(f"Redis is already running on {host}:{port}")
        return True
    except (redis.ConnectionError, redis.TimeoutError):
        return False


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


def start_redis_docker(config):
    """Start Redis using Docker"""
    host = config.get('host', 'localhost')
    port = config.get('port', 6379)
    password = config.get('password')
    
    # Create data directory for persistence
    data_dir = Path.cwd() / "redis_data"
    data_dir.mkdir(exist_ok=True)
    
    # Docker command to start Redis
    docker_cmd = [
        'docker', 'run', '-d',
        '--name', 'redis_pipeline',
        '-p', f'{port}:6379',
        '-v', f'{data_dir.absolute()}:/data',
        '--restart', 'unless-stopped'
    ]
    
    # Add password if configured
    if password:
        docker_cmd.extend(['--env', f'REDIS_PASSWORD={password}'])
        docker_cmd.append('redis:7-alpine')
        docker_cmd.extend(['redis-server', '--requirepass', password, '--appendonly', 'yes'])
    else:
        docker_cmd.append('redis:7-alpine')
        docker_cmd.extend(['redis-server', '--appendonly', 'yes'])
    
    try:
        # Stop existing container if running
        subprocess.run(['docker', 'stop', 'redis_pipeline'], 
                      capture_output=True, check=False)
        subprocess.run(['docker', 'rm', 'redis_pipeline'], 
                      capture_output=True, check=False)
        
        # Start new container
        logging.info("Starting Redis container...")
        result = subprocess.run(docker_cmd, capture_output=True, text=True, check=True)
        
        container_id = result.stdout.strip()
        logging.info(f"Redis container started: {container_id}")
        
        # Wait for Redis to be ready
        logging.info("Waiting for Redis to be ready...")
        for i in range(30):  # Wait up to 30 seconds
            if check_redis_running(host, port, password):
                logging.info("Redis is ready!")
                return True
            time.sleep(1)
        
        logging.error("Redis failed to start within 30 seconds")
        return False
        
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to start Redis container: {e}")
        logging.error(f"Error output: {e.stderr}")
        return False


def start_redis_native(config):
    """Start Redis using native installation"""
    host = config.get('host', 'localhost')
    port = config.get('port', 6379)
    password = config.get('password')
    
    # Check if redis-server is available
    try:
        subprocess.run(['redis-server', '--version'], 
                      capture_output=True, text=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        logging.error("redis-server binary not found in PATH")
        return False
    
    # Create configuration file
    config_dir = Path.cwd() / "redis_config"
    config_dir.mkdir(exist_ok=True)
    
    data_dir = Path.cwd() / "redis_data"
    data_dir.mkdir(exist_ok=True)
    
    config_content = f"""
# Ultra-Cost-Optimized Redis Configuration
bind {host}
port {port}
dir {data_dir.absolute()}

# Persistence settings for cost optimization
save 900 1
save 300 10
save 60 10000

# AOF (Append Only File) for durability
appendonly yes
appendfilename "appendonly.aof"
appendfsync everysec

# Memory optimization
maxmemory 2gb
maxmemory-policy allkeys-lru

# Disable dangerous commands
rename-command FLUSHDB ""
rename-command FLUSHALL ""
rename-command DEBUG ""

# Network optimization
tcp-keepalive 300
timeout 0

# Logging
loglevel notice
logfile {config_dir.absolute()}/redis.log

# Background save optimization
stop-writes-on-bgsave-error yes
rdbcompression yes
rdbchecksum yes

# Client optimization
tcp-backlog 511
"""
    
    # Add password if configured
    if password:
        config_content += f"\nrequirepass {password}\n"
    
    config_path = config_dir / "redis.conf"
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    # Start Redis
    try:
        logging.info("Starting Redis server...")
        process = subprocess.Popen([
            'redis-server',
            str(config_path)
        ])
        
        # Wait for Redis to be ready
        logging.info("Waiting for Redis to be ready...")
        for i in range(30):
            if check_redis_running(host, port, password):
                logging.info("Redis is ready!")
                return True
            time.sleep(1)
        
        logging.error("Redis failed to start within 30 seconds")
        process.terminate()
        return False
        
    except Exception as e:
        logging.error(f"Failed to start Redis server: {e}")
        return False


def optimize_redis_for_caching(host="localhost", port=6379, password=None):
    """Apply runtime optimizations for caching workload"""
    try:
        client = redis.Redis(
            host=host,
            port=port,
            password=password,
            socket_connect_timeout=5,
            socket_timeout=5
        )
        
        # Test connection
        client.ping()
        
        # Apply optimizations
        optimizations = [
            # Memory optimization
            ("CONFIG SET maxmemory-policy allkeys-lru", "Set LRU eviction policy"),
            ("CONFIG SET maxmemory 2gb", "Set memory limit"),
            
            # Performance optimizations
            ("CONFIG SET save '900 1 300 10 60 10000'", "Configure background saves"),
            ("CONFIG SET appendfsync everysec", "Set AOF sync policy"),
            
            # Network optimizations
            ("CONFIG SET tcp-keepalive 300", "Set TCP keepalive"),
            ("CONFIG SET timeout 0", "Disable idle timeout"),
        ]
        
        for command, description in optimizations:
            try:
                result = client.execute_command(*command.split())
                logging.info(f"✅ {description}: {result}")
            except Exception as e:
                logging.warning(f"⚠️  Failed to apply {description}: {e}")
        
        # Get memory info
        memory_info = client.info('memory')
        logging.info(f"Redis memory usage: {memory_info.get('used_memory_human', 'unknown')}")
        
        return True
        
    except Exception as e:
        logging.error(f"Failed to optimize Redis: {e}")
        return False


def create_redis_monitoring_script():
    """Create a simple monitoring script for Redis"""
    script_content = """#!/usr/bin/env python3
import redis
import time
import json

def monitor_redis(host='localhost', port=6379, password=None):
    client = redis.Redis(host=host, port=port, password=password)
    
    while True:
        try:
            info = client.info()
            stats = {
                'connected_clients': info.get('connected_clients', 0),
                'used_memory_human': info.get('used_memory_human', 'unknown'),
                'keyspace_hits': info.get('keyspace_hits', 0),
                'keyspace_misses': info.get('keyspace_misses', 0),
                'instantaneous_ops_per_sec': info.get('instantaneous_ops_per_sec', 0),
                'hit_rate': 0
            }
            
            # Calculate hit rate
            hits = stats['keyspace_hits']
            misses = stats['keyspace_misses']
            if hits + misses > 0:
                stats['hit_rate'] = hits / (hits + misses) * 100
            
            print(f"Redis Stats: {json.dumps(stats, indent=2)}")
            
        except Exception as e:
            print(f"Error monitoring Redis: {e}")
        
        time.sleep(10)

if __name__ == "__main__":
    monitor_redis()
"""
    
    script_path = Path.cwd() / "scripts" / "monitor_redis.py"
    script_path.parent.mkdir(exist_ok=True)
    
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    # Make executable
    script_path.chmod(0o755)
    
    logging.info(f"Created Redis monitoring script: {script_path}")


def main():
    """Main function to start Redis"""
    logging.info("Starting Redis for ultra-cost-optimized pipeline...")
    
    # Load configuration
    config = load_config()
    if not config:
        # Use defaults
        config = {
            'host': 'localhost',
            'port': 6379,
            'password': None
        }
    
    host = config.get('host', 'localhost')
    port = config.get('port', 6379)
    password = config.get('password')
    
    # Check if already running
    if check_redis_running(host, port, password):
        logging.info("Redis is already running, no action needed")
        return
    
    # Try Docker first, then native
    success = False
    
    if check_docker_installed():
        logging.info("Attempting to start Redis using Docker...")
        success = start_redis_docker(config)
    
    if not success:
        logging.info("Attempting to start Redis using native installation...")
        success = start_redis_native(config)
    
    if success:
        logging.info(f"✅ Redis started successfully on {host}:{port}")
        
        # Apply optimizations
        time.sleep(2)  # Give Redis a moment to fully start
        optimize_redis_for_caching(host, port, password)
        
        # Create monitoring script
        create_redis_monitoring_script()
        
        logging.info("Redis is optimized for caching workload")
        logging.info("Use 'python scripts/monitor_redis.py' to monitor performance")
        logging.info("Use Ctrl+C to stop the service when running in foreground")
    else:
        logging.error("❌ Failed to start Redis")
        logging.error("Please install Docker or Redis manually")
        logging.error("Installation instructions:")
        logging.error("  Docker: https://docs.docker.com/get-docker/")
        logging.error("  Redis: https://redis.io/download")
        sys.exit(1)


if __name__ == "__main__":
    main()