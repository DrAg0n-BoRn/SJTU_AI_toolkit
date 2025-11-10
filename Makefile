# ==============================================================================
# Makefile for project packaging using Nuitka
# ==============================================================================

# --- Variables ---
PYTHON      := python
APP_NAME    := visual_ccc
EXE_NAME    := VisualCCC
# Nuitka:WARNING: To compile a package with a '__main__' module, specify its containing directory but, not the '__main__.py' itself.
MAIN_SCRIPT := $(APP_NAME)
FILES_DIR   := $(APP_NAME)/resources
ICON_MAC    := $(FILES_DIR)/icon.icns
ICON_WIN    := $(FILES_DIR)/icon.ico

# --- Targets ---

.DEFAULT_GOAL := help

help:
	@echo "Packaging with Nuitka"
	@echo "----------------------------------------------"
	@echo "Usage: make [target]"
	@echo ""
	@echo "Available targets:"
	@echo "  help           - Shows this help message."
	@echo "  build_mac      - Builds the macOS .app bundle (must be run on macOS)."
	@echo "  build_windows  - Builds the Windows .exe (must be run on Windows)."
	@echo "  build_linux    - Builds the Linux executable (must be run on Linux)."

# Deprecated plugins
# --enable-plugin=numpy
# --enable-plugin=torch

# Use for fast C-based JSON handling if needed
# --include-module=_json

# Build target for macOS
build_mac:
	@echo "--- Building macOS .app bundle with Nuitka ---"
	$(PYTHON) -m nuitka \
	  --standalone \
	  --output-dir=dist \
	  --output-filename="$(EXE_NAME)" \
	  --product-name="$(EXE_NAME)" \
	  --macos-create-app-bundle \
	  --macos-app-icon="$(ICON_MAC)" \
	  --include-data-dir="$(FILES_DIR)=$(FILES_DIR)" \
	  --enable-plugin=tk-inter \
	  --module-parameter=torch-disable-jit=yes \
	  --include-module=_json \
	  $(MAIN_SCRIPT)

# Build target for Windows
build_windows:
	@echo "--- Building Windows executable with Nuitka ---"
	$(PYTHON) -m nuitka \
	  --standalone \
	  --output-dir=dist \
	  --output-filename="$(EXE_NAME)" \
	  --product-name="$(EXE_NAME)" \
	  --windows-disable-console \
	  --windows-icon-from-ico="$(ICON_WIN)" \
	  --include-data-dir="$(FILES_DIR)=$(FILES_DIR)" \
	  --enable-plugin=tk-inter \
	  --module-parameter=torch-disable-jit=yes \
	  --include-module=_json \
	  $(MAIN_SCRIPT)

# Build target for Linux
build_linux:
	@echo "--- Building Linux executable with Nuitka ---"
	$(PYTHON) -m nuitka \
	  --standalone \
	  --output-dir=dist \
	  --output-filename="$(EXE_NAME)" \
	  --product-name="$(EXE_NAME)" \
	  --include-data-dir="$(FILES_DIR)=$(FILES_DIR)" \
	  --enable-plugin=tk-inter \
	  --module-parameter=torch-disable-jit=yes \
	  --include-module=_json \
	  $(MAIN_SCRIPT)

# Declare targets that are not files
.PHONY: help build_mac build_windows build_linux
