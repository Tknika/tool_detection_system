#!/bin/bash
# Tmux session manager for YOLO Pipeline

SESSION_NAME="yolo_pipeline"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Function to check if session exists
session_exists() {
    tmux has-session -t "$SESSION_NAME" 2>/dev/null
}

# Function to create new session
create_session() {
    echo "Creating new tmux session: $SESSION_NAME"
    tmux new-session -d -s "$SESSION_NAME" -c "$SCRIPT_DIR"
    
    # Activate virtual environment
    tmux send-keys -t "$SESSION_NAME" "source activate_env.sh" Enter
    tmux send-keys -t "$SESSION_NAME" "echo 'Environment activated. Ready to run pipeline.'" Enter
}

# Function to attach to existing session
attach_session() {
    echo "Attaching to existing session: $SESSION_NAME"
    tmux attach-session -t "$SESSION_NAME"
}

# Function to run pipeline in background
run_pipeline() {
    local n_trials=${1:-20}
    echo "Starting YOLO pipeline with $n_trials trials in background..."
    
    if ! session_exists; then
        create_session
    fi
    
    # Run the pipeline
    tmux send-keys -t "$SESSION_NAME" "python complete_yolo_pipeline.py --n-trials $n_trials" Enter
    
    echo "Pipeline started in tmux session: $SESSION_NAME"
    echo "To monitor progress: tmux attach -t $SESSION_NAME"
    echo "To detach: Ctrl+B, then D"
}

# Function to start TensorBoard
start_tensorboard() {
    echo "Starting TensorBoard..."
    
    if ! session_exists; then
        create_session
    fi
    
    # Start TensorBoard in a new window
    tmux new-window -t "$SESSION_NAME" -n "tensorboard"
    tmux send-keys -t "$SESSION_NAME:tensorboard" "source activate_env.sh" Enter
    tmux send-keys -t "$SESSION_NAME:tensorboard" "tensorboard --logdir yolo/runs/optuna/ --host 0.0.0.0 --port 6006" Enter
    
    echo "TensorBoard started on http://localhost:6006"
    echo "To access remotely: http://YOUR_SERVER_IP:6006"
}

# Function to show session status
show_status() {
    if session_exists; then
        echo "Session $SESSION_NAME is running"
        tmux list-windows -t "$SESSION_NAME"
    else
        echo "Session $SESSION_NAME is not running"
    fi
}

# Function to kill session
kill_session() {
    if session_exists; then
        echo "Killing session: $SESSION_NAME"
        tmux kill-session -t "$SESSION_NAME"
    else
        echo "Session $SESSION_NAME is not running"
    fi
}

# Main script logic
case "$1" in
    "create")
        if session_exists; then
            echo "Session $SESSION_NAME already exists"
            attach_session
        else
            create_session
            attach_session
        fi
        ;;
    "attach")
        if session_exists; then
            attach_session
        else
            echo "Session $SESSION_NAME does not exist. Use 'create' first."
        fi
        ;;
    "run")
        run_pipeline "$2"
        ;;
    "tensorboard")
        start_tensorboard
        ;;
    "status")
        show_status
        ;;
    "kill")
        kill_session
        ;;
    "help"|*)
        echo "YOLO Pipeline Tmux Manager"
        echo "Usage: $0 {create|attach|run|tensorboard|status|kill|help}"
        echo ""
        echo "Commands:"
        echo "  create      - Create new tmux session and attach"
        echo "  attach      - Attach to existing session"
        echo "  run [trials] - Run pipeline in background (default: 20 trials)"
        echo "  tensorboard - Start TensorBoard monitoring"
        echo "  status      - Show session status"
        echo "  kill        - Kill the session"
        echo "  help        - Show this help"
        echo ""
        echo "Examples:"
        echo "  $0 create                    # Create and attach to session"
        echo "  $0 run 30                    # Run pipeline with 30 trials"
        echo "  $0 tensorboard               # Start TensorBoard"
        echo "  $0 status                    # Check status"
        ;;
esac
