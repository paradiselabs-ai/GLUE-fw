#!/bin/bash
# Function to add to your .zshrc file

# Add this to your ~/.zshrc file:
#
# # Auto-activate virtual environment for glue-fw project
# source ~/Desktop/AI_ML/Creating/glue-fw/auto_activate.sh

# Function to detect and activate the glue-fw virtual environment
glue_fw_auto_activate() {
  # Check if we're in the glue-fw project directory or any subdirectory
  if [[ "$PWD" == "$HOME/Desktop/AI_ML/Creating/glue-fw"* ]]; then
    # Only activate if not already in the virtual environment
    if [[ -z "$VIRTUAL_ENV" || "$VIRTUAL_ENV" != *"glue-fw/venv"* ]]; then
      echo "Activating glue-fw virtual environment..."
      source "$HOME/Desktop/AI_ML/Creating/glue-fw/venv/bin/activate"
    fi
  fi
}

# Add the function to the directory change hook
autoload -U add-zsh-hook
add-zsh-hook chpwd glue_fw_auto_activate

# Run once when the shell starts
glue_fw_auto_activate
