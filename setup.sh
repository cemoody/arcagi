#!/bin/bash
# Setup script for fresh GCP GPU instance
# This script installs all dependencies and sets up the arcagi project

set -e  # Exit on error

echo "=== ARC-AGI GPU Instance Setup Script ==="
echo "This script will set up everything needed to run experiments"
echo ""

# Update system packages
echo "1. Updating system packages..."
sudo apt-get update -y
sudo apt-get upgrade -y

# Install essential tools
echo "2. Installing essential tools..."
sudo apt-get install -y \
    git \
    curl \
    wget \
    build-essential \
    python3-dev \
    python3-pip \
    htop \
    tmux \
    vim \
    nano

# Install uv (Python package manager)
echo "3. Installing uv..."
if command -v uv &> /dev/null; then
    echo "uv is already installed ($(uv --version))"
else
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.cargo/env  # Add uv to PATH
fi

# Ensure uv is in PATH for this session
if ! command -v uv &> /dev/null; then
    export PATH="$HOME/.cargo/bin:$PATH"
fi

# Verify CUDA is available (for GPU instances)
echo "4. Checking CUDA availability..."
if command -v nvidia-smi &> /dev/null; then
    echo "CUDA is available:"
    nvidia-smi
else
    echo "WARNING: CUDA/nvidia-smi not found. This might not be a GPU instance."
fi

# Clone the repository
echo "5. Setting up repository..."
PROJECT_DIR="$HOME/arcagi"

# Check if repository already exists and is valid
if [ -d "$PROJECT_DIR/.git" ]; then
    echo "Repository already exists at $PROJECT_DIR"
    cd "$PROJECT_DIR"
    echo "Pulling latest changes..."
    git pull || echo "Warning: Could not pull changes (possibly authentication issue)"
    auth_choice="existing"
else
    echo "Choose authentication method for private repo:"
    echo "1) SSH key (recommended)"
    echo "2) Personal access token"
    echo "3) Skip clone (manual setup)"
    read -p "Enter choice (1-3): " auth_choice
fi

case $auth_choice in
    existing)
        echo "Using existing repository..."
        ;;
    1)
        echo "Setting up SSH key for GitHub..."
        if [ ! -f ~/.ssh/id_ed25519 ]; then
            ssh-keygen -t ed25519 -C "$(whoami)@$(hostname)" -f ~/.ssh/id_ed25519 -N ""
        fi
        
        echo ""
        echo "=== ADD THIS SSH KEY TO GITHUB ==="
        echo "Go to: https://github.com/settings/ssh/new"
        echo "Title: $(hostname)"
        echo "Key:"
        cat ~/.ssh/id_ed25519.pub
        echo "================================="
        echo ""
        read -p "Press Enter after adding the SSH key to GitHub..."
        
        # Test SSH connection
        ssh -T git@github.com || true
        
        REPO_URL="git@github.com:cemoody/arcagi.git"
        ;;
    2)
        read -p "Enter your GitHub personal access token: " token
        REPO_URL="https://$token@github.com/cemoody/arcagi.git"
        ;;
    3)
        echo "Skipping repository clone. You'll need to clone manually:"
        echo "git clone <your-repo-url> $PROJECT_DIR"
        mkdir -p "$PROJECT_DIR"
        cd "$PROJECT_DIR"
        echo "Created empty project directory. Skipping dependency installation."
        echo "After cloning, run: cd $PROJECT_DIR && uv venv --python 3.12 && source .venv/bin/activate && uv sync"
        exit 0
        ;;
esac

# Clone repository if needed (only for new setups)
if [ "$auth_choice" != "existing" ] && [ "$auth_choice" != "3" ]; then
    if [ ! -d "$PROJECT_DIR" ]; then
        echo "Cloning repository..."
        git clone "$REPO_URL" "$PROJECT_DIR"
        cd "$PROJECT_DIR"
    else
        cd "$PROJECT_DIR"
    fi
fi

# Set up Python environment with uv
echo "6. Setting up Python environment..."
if [ -d ".venv" ]; then
    echo "Virtual environment already exists"
    source .venv/bin/activate
    echo "Activated existing virtual environment"
else
    echo "Creating new virtual environment..."
    uv venv --python 3.12
    source .venv/bin/activate
fi

# Install dependencies
echo "7. Installing project dependencies..."
if [ -f "pyproject.toml" ]; then
    echo "Installing/updating dependencies with uv sync..."
    uv sync
else
    echo "Warning: No pyproject.toml found, skipping dependency installation"
fi

# Set up Weights & Biases
echo "8. Setting up Weights & Biases (wandb)..."

# Check if wandb is already configured
if wandb status &> /dev/null; then
    echo "wandb is already configured and logged in"
    wandb_choice="existing"
else
    echo "wandb is needed for experiment tracking and logging."
    echo ""
    echo "Choose wandb setup option:"
    echo "1) Login with API key (recommended)"
    echo "2) Login with browser"
    echo "3) Skip wandb setup"
    read -p "Enter choice (1-3): " wandb_choice
fi

case $wandb_choice in
    existing)
        echo "Using existing wandb configuration..."
        ;;
    1)
        read -p "Enter your wandb API key: " wandb_key
        echo "$wandb_key" | wandb login
        ;;
    2)
        wandb login
        ;;
    3)
        echo "Skipping wandb setup. You can set it up later with 'wandb login'"
        ;;
esac

# Verify PyTorch can see GPU
echo "9. Verifying PyTorch GPU access..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device count: {torch.cuda.device_count()}')"

# Verify wandb setup
echo "10. Verifying wandb setup..."
if [ "$wandb_choice" != "3" ]; then
    python -c "import wandb; print(f'wandb version: {wandb.__version__}'); print('wandb setup verified')"
else
    echo "wandb setup was skipped"
fi

# Set up auto-shutdown after 3 hours
echo "11. Setting up auto-shutdown (3 hours)..."
if atq | grep -q "shutdown"; then
    echo "Auto-shutdown is already scheduled:"
    atq | grep "shutdown"
else
    echo "sudo shutdown -h +180" | at now
    echo "Auto-shutdown scheduled for 3 hours from now"
fi

echo ""
echo "=== Setup Complete! ==="
echo ""
if atq | grep -q "shutdown"; then
    echo "Auto-shutdown: Instance will shut down based on scheduled task"
    echo "To cancel shutdown: sudo shutdown -c"
else
    echo "No auto-shutdown scheduled"
fi
echo ""
echo "Current environment:"
echo "- Project directory: $PROJECT_DIR"
echo "- Python version: $(python --version)"
echo "- uv version: $(uv --version)"
echo ""
echo "To start training immediately:"
echo "cd ~/arcagi && source .venv/bin/activate && python arcagi/color_mapping/ex33.py"