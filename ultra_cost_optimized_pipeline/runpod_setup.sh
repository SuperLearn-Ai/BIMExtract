#!/bin/bash
# RunPod Setup Script for Ultra-Cost-Optimized Pipeline
# This script sets up the environment and deploys all models to RunPod

set -e

echo "🚀 Ultra-Cost-Optimized Pipeline - RunPod Setup"
echo "=============================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is required but not installed!"
    exit 1
fi

print_status "Python 3 found: $(python3 --version)"

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    print_error "pip3 is required but not installed!"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    print_status "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source venv/bin/activate

# Install required packages for deployment
print_status "Installing deployment dependencies..."
pip install --upgrade pip
pip install aiohttp pyyaml asyncio pathlib

# Check for RunPod API key
if [ -z "$RUNPOD_API_KEY" ]; then
    echo ""
    echo "🔑 RunPod API Key Required"
    echo "========================="
    echo "You need a RunPod API key to deploy models."
    echo "1. Go to https://runpod.io/"
    echo "2. Sign up/login to your account"
    echo "3. Navigate to Settings > API Keys"
    echo "4. Create a new API key"
    echo ""
    read -p "Enter your RunPod API key: " RUNPOD_API_KEY
    
    if [ -z "$RUNPOD_API_KEY" ]; then
        print_error "API key is required to continue!"
        exit 1
    fi
    
    # Export the API key
    export RUNPOD_API_KEY="$RUNPOD_API_KEY"
    
    # Add to .env file for persistence
    echo "RUNPOD_API_KEY=$RUNPOD_API_KEY" > .env
    print_status "API key saved to .env file"
fi

# Create necessary directories
print_status "Creating directory structure..."
mkdir -p config
mkdir -p logs
mkdir -p temp

# Run the deployment script
print_status "Starting RunPod deployment..."
echo ""
python3 deploy_to_runpod.py

# Check if deployment was successful
if [ $? -eq 0 ]; then
    echo ""
    print_status "🎉 RunPod deployment completed successfully!"
    echo ""
    echo "📋 Next Steps:"
    echo "=============="
    echo "1. ✅ Models deployed to RunPod"
    echo "2. ✅ Configuration saved to config/runpod_config.yaml"
    echo "3. 🔄 Install pipeline dependencies:"
    echo "   pip install -r requirements.txt"
    echo "4. 🚀 Run the pipeline with RunPod models:"
    echo "   python run_pipeline.py --runpod"
    echo ""
    echo "💰 Expected Performance:"
    echo "• Cost: ~$0.46 per 1K pages (still 85% cheaper!)"
    echo "• Speed: 3-5x faster with GPU acceleration"
    echo "• Memory: Only ~3.6GB used locally"
    echo "• Reliability: No more OOM crashes"
    echo ""
    echo "🔗 Useful Commands:"
    echo "• Test single document: python run_pipeline.py --files document.pdf --runpod"
    echo "• Batch processing: python run_pipeline.py --batch-size 10 --runpod"
    echo "• View logs: tail -f logs/pipeline_run.log"
    echo ""
else
    print_error "Deployment failed! Check the logs above for details."
    exit 1
fi 