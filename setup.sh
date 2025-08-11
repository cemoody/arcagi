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
PROJECT_DIR="$HOME/arcagi/arcagi"

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
        # Note: This will clone to $HOME/arcagi, then we cd to $HOME/arcagi/arcagi
        ;;
    2)
        read -p "Enter your GitHub personal access token: " token
        REPO_URL="https://$token@github.com/cemoody/arcagi.git"
        # Note: This will clone to $HOME/arcagi, then we cd to $HOME/arcagi/arcagi
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
    if [ ! -d "$HOME/arcagi" ]; then
        echo "Cloning repository..."
        git clone "$REPO_URL" "$HOME/arcagi"
    fi
    cd "$PROJECT_DIR"
else
    # For existing repos, make sure we're in the right directory
    cd "$PROJECT_DIR"
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
    if ! uv sync; then
        echo "Error: Failed to install dependencies with uv sync"
        echo "This might be due to dependency conflicts or network issues"
        echo "Trying to install basic requirements manually..."
        uv pip install torch torchvision torchaudio wandb --index-url https://download.pytorch.org/whl/cu121
    fi
else
    echo "Warning: No pyproject.toml found, installing basic requirements..."
    echo "Installing PyTorch with CUDA support and wandb..."
    uv pip install torch torchvision torchaudio wandb --index-url https://download.pytorch.org/whl/cu121
fi

# Set up Weights & Biases
echo "8. Setting up Weights & Biases (wandb)..."

# Ensure wandb is available (install if needed)
if ! command -v wandb &> /dev/null; then
    echo "wandb not found, installing with uv..."
    uv pip install wandb
fi

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

# Ensure PyTorch is available (install if needed)
if ! python -c "import torch" &> /dev/null; then
    echo "PyTorch not found, installing with CUDA support..."
    # Install PyTorch with CUDA support for GPU instances
    uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
fi

# Ensure we're in the virtual environment
source .venv/bin/activate
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device count: {torch.cuda.device_count()}')"

# Verify wandb setup
echo "10. Verifying wandb setup..."
if [ "$wandb_choice" != "3" ]; then
    # Ensure we're in the virtual environment
    source .venv/bin/activate
    python -c "import wandb; print(f'wandb version: {wandb.__version__}'); print('wandb setup verified')"
else
    echo "wandb setup was skipped"
fi

# Set up auto-shutdown after 3 hours
echo "11. Setting up auto-shutdown (3 hours)..."

# Install and start the at daemon if not available
if ! command -v at &> /dev/null; then
    echo "Installing at daemon for scheduling..."
    sudo apt-get update -qq
    sudo apt-get install -y at
fi

# Ensure atd service is running
if ! systemctl is-active --quiet atd; then
    echo "Starting at daemon..."
    sudo systemctl enable atd
    sudo systemctl start atd
fi

# Schedule the shutdown
if atq | grep -q "shutdown"; then
    echo "Auto-shutdown is already scheduled:"
    atq | grep "shutdown"
else
    echo "sudo shutdown -h +180" | at now
    if [ $? -eq 0 ]; then
        echo "Auto-shutdown scheduled for 3 hours from now"
    else
        echo "Failed to schedule with 'at', falling back to manual reminder"
        echo "To manually shutdown in 3 hours, run: sudo shutdown -h +180"
    fi
fi

echo ""
echo "=== Setup Complete! ==="
echo ""
if command -v atq &> /dev/null && atq | grep -q "shutdown"; then
    echo "Auto-shutdown: Instance will shut down based on scheduled task"
    echo "To cancel shutdown: sudo shutdown -c"
else
    echo "No auto-shutdown scheduled (or 'at' daemon not available)"
fi
# Set up uv in bash profile for persistent access
echo "12. Setting up uv in bash profile..."
UV_PATH_LINE='export PATH="$HOME/.cargo/bin:$PATH"'
CARGO_ENV_LINE='source "$HOME/.cargo/env"'

# Add to .bashrc if not already present
if ! grep -q "\.cargo/bin" ~/.bashrc 2>/dev/null; then
    echo "Adding uv to .bashrc..."
    echo "# Added by arcagi setup script" >> ~/.bashrc
    echo "$UV_PATH_LINE" >> ~/.bashrc
    echo "$CARGO_ENV_LINE" >> ~/.bashrc
else
    echo "uv PATH already configured in .bashrc"
fi

# Also add to .bash_profile if it exists or create it
if [ ! -f ~/.bash_profile ]; then
    echo "Creating .bash_profile..."
    cat > ~/.bash_profile << 'EOF'
# .bash_profile

# Get the aliases and functions
if [ -f ~/.bashrc ]; then
    . ~/.bashrc
fi

# User specific environment and startup programs
EOF
fi

if ! grep -q "\.cargo/bin" ~/.bash_profile 2>/dev/null; then
    echo "Adding uv to .bash_profile..."
    echo "# Added by arcagi setup script" >> ~/.bash_profile
    echo "$UV_PATH_LINE" >> ~/.bash_profile
    echo "$CARGO_ENV_LINE" >> ~/.bash_profile
else
    echo "uv PATH already configured in .bash_profile"
fi

echo ""
echo "Current environment:"
echo "- Project directory: $PROJECT_DIR"
# Ensure we show the virtual environment's python version
source .venv/bin/activate
echo "- Python version: $(python --version)"
echo "- uv version: $(uv --version)"
echo ""
echo "Environment setup:"
echo "- uv is now available in new shell sessions"
echo "- Virtual environment: .venv (activate with: source .venv/bin/activate)"
echo ""
echo "To start training immediately:"
echo "cd ~/arcagi/arcagi && source .venv/bin/activate && python arcagi/color_mapping/ex33.py"