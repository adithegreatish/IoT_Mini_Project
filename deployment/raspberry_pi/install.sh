#!/usr/bin/env bash
# ==============================================================================
# Smart Parking & HSRP: Automated Raspberry Pi 5 Deployment Installer
# Target: Raspberry Pi OS 64-bit (Debian Bookworm aarch64)
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}==================================================================${NC}"
echo -e "${BLUE} Smart Parking & HSRP: Raspberry Pi 5 Automated Setup Installer ${NC}"
echo -e "${BLUE} Project Root: ${REPO_ROOT}${NC}"
echo -e "${BLUE}==================================================================${NC}"

# 1. Architecture Check
echo ""
echo -e "${BLUE}[1/10] Checking System Architecture & OS...${NC}"
ARCH="$(uname -m)"
if [ "${ARCH}" != "aarch64" ] && [ "${ARCH}" != "arm64" ]; then
    echo -e "${RED}[ERROR] Unsupported CPU architecture: ${ARCH}${NC}"
    echo "This deployment package requires a 64-bit ARM OS (aarch64)."
    echo "Please install 64-bit Raspberry Pi OS (Debian Bookworm 64-bit)."
    exit 1
fi
echo -e "${GREEN}[OK] 64-bit ARM architecture confirmed (${ARCH}).${NC}"

# 2. Python Version Check
echo ""
echo -e "${BLUE}[2/10] Checking Python Installation...${NC}"
PYTHON_BIN=""
for candidate in python3.12 python3.11 python3; do
    if command -v "${candidate}" &> /dev/null; then
        PYTHON_BIN="$(command -v "${candidate}")"
        break
    fi
done

if [ -z "${PYTHON_BIN}" ]; then
    echo -e "${RED}[ERROR] Python 3 not found.${NC}"
    echo "Install python with: sudo apt update && sudo apt install -y python3 python3-venv python3-pip"
    exit 1
fi

PY_VER="$("${PYTHON_BIN}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
echo -e "${GREEN}[OK] Using Python binary: ${PYTHON_BIN} (v${PY_VER})${NC}"

# 3. Hardware Resource Check (RAM & Disk Space)
echo ""
echo -e "${BLUE}[3/10] Checking RAM and Available Storage...${NC}"
TOTAL_RAM_MB="$(free -m | awk '/^Mem:/{print $2}')"
AVAIL_DISK_MB="$(df -m "${REPO_ROOT}" | awk 'NR==2 {print $4}')"

echo "Detected Total RAM: ${TOTAL_RAM_MB} MB"
echo "Detected Available Disk Space: ${AVAIL_DISK_MB} MB"

if [ "${TOTAL_RAM_MB}" -lt 1800 ]; then
    echo -e "${YELLOW}[WARN] Less than 2 GB RAM detected. Recommended: Raspberry Pi 5 (8 GB RAM).${NC}"
else
    echo -e "${GREEN}[OK] Sufficient RAM available (${TOTAL_RAM_MB} MB).${NC}"
fi

if [ "${AVAIL_DISK_MB}" -lt 1500 ]; then
    echo -e "${RED}[ERROR] Less than 1.5 GB free disk space available (${AVAIL_DISK_MB} MB).${NC}"
    echo "Please free up storage on the microSD card or NVMe drive."
    exit 1
fi
echo -e "${GREEN}[OK] Sufficient storage available.${NC}"

# 4. System Package Dependencies
echo ""
echo -e "${BLUE}[4/10] Checking System Shared Libraries (OpenCV, V4L2, Git LFS)...${NC}"
MISSING_PKGS=()
for pkg in libgl1 libglib2.0-0 v4l-utils git-lfs; do
    if ! dpkg -s "${pkg}" &> /dev/null; then
        MISSING_PKGS+=("${pkg}")
    fi
done

if [ ${#MISSING_PKGS[@]} -gt 0 ]; then
    echo -e "${YELLOW}[INFO] Installing missing system packages: ${MISSING_PKGS[*]}...${NC}"
    echo "Running: sudo apt update && sudo apt install -y ${MISSING_PKGS[*]}"
    sudo apt update && sudo apt install -y "${MISSING_PKGS[@]}"
else
    echo -e "${GREEN}[OK] Required system libraries are installed.${NC}"
fi

# Ensure Git LFS hooks are active and pull models if needed
if command -v git-lfs &> /dev/null; then
    git lfs install --skip-repo &> /dev/null || true
fi

# 5. Linux Virtual Environment Setup
echo ""
echo -e "${BLUE}[5/10] Configuring Clean Linux Virtual Environment (.venv-pi)...${NC}"
VENV_DIR="${REPO_ROOT}/.venv-pi"
if [ ! -d "${VENV_DIR}" ]; then
    echo "Creating new virtual environment at ${VENV_DIR}..."
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

# Activate virtualenv
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip

echo "Installing ARM64 CPU dependencies from requirements-pi.txt..."
pip install -r "${REPO_ROOT}/requirements-pi.txt"
echo -e "${GREEN}[OK] Virtual environment ready and dependencies installed.${NC}"

# 6. Writable Directories
echo ""
echo -e "${BLUE}[6/10] Verifying Project Directory Hierarchy...${NC}"
mkdir -p "${REPO_ROOT}/data/test_reports"
mkdir -p "${REPO_ROOT}/logs"
mkdir -p "${REPO_ROOT}/models"
echo -e "${GREEN}[OK] Directories data/, logs/, models/ verified.${NC}"

# 7. Model Integrity Check
echo ""
echo -e "${BLUE}[7/10] Validating Model Files & SHA-256 Checksums...${NC}"
bash "${SCRIPT_DIR}/check_models.sh"

# 8. Database Initialization
echo ""
echo -e "${BLUE}[8/10] Initializing SQLite Parking Database...${NC}"
python -c "
from app.infrastructure.paths import paths
from app.database.connection import DatabaseManager
db_path = paths.resolve('data/parking_events.sqlite')
db = DatabaseManager(str(db_path))
db.initialize_schema()
print(f'Successfully initialized database at {db_path}')
"
echo -e "${GREEN}[OK] Database initialized successfully.${NC}"

# 9. USB Webcam Hardware Discoverability
echo ""
echo -e "${BLUE}[9/10] Discovering USB Video Devices (/dev/video*)...${NC}"
if command -v v4l2-ctl &> /dev/null; then
    echo "Available V4L2 Video Devices:"
    v4l2-ctl --list-devices || true
else
    ls -l /dev/video* 2>/dev/null || echo -e "${YELLOW}[WARN] No /dev/video* devices currently connected.${NC}"
fi

# Ensure user is in 'video' group
CURRENT_USER="$(id -un)"
if ! groups "${CURRENT_USER}" | grep -q '\bvideo\b'; then
    echo -e "${YELLOW}[ACTION REQUIRED] Adding user '${CURRENT_USER}' to 'video' group for webcam access...${NC}"
    sudo usermod -a -G video "${CURRENT_USER}"
    echo "Note: You may need to log out and log back in for video permissions to take effect."
fi

# 10. Preflight Diagnostics
echo ""
echo -e "${BLUE}[10/10] Running Preflight Self-Diagnostic Check...${NC}"
python "${REPO_ROOT}/scripts/preflight_raspberry_pi.py" || true

echo ""
echo -e "${GREEN}==================================================================${NC}"
echo -e "${GREEN} RASPBERRY PI 5 DEPLOYMENT INSTALLATION COMPLETE!               ${NC}"
echo -e "${GREEN}==================================================================${NC}"
echo ""
echo "To start the application, run:"
echo -e "    ${BLUE}bash deployment/raspberry_pi/start.sh${NC}"
echo ""
echo "Or start manually using:"
echo -e "    ${BLUE}source .venv-pi/bin/activate${NC}"
echo -e "    ${BLUE}python app/main.py --config configs/raspberry_pi.yaml${NC}"
echo ""
echo "Dashboard will be accessible at: http://localhost:8000"
echo "=================================================================="
