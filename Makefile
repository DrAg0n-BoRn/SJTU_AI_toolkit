# ==============================================================================
# Makefile for project packaging using Nuitka
# ==============================================================================

# --- Variables ---
APP_NAME    := visual_ccc
EXE_NAME    := VisualCCC
# Nuitka:WARNING: To compile a package with a '__main__' module, specify its containing directory but, not the '__main__.py' itself.
MAIN_SCRIPT := $(APP_NAME)
FILES_DIR   := $(APP_NAME)/resources
ICON_MAC    := $(FILES_DIR)/icon.icns
ICON_WIN    := $(FILES_DIR)/icon.ico
ICON_DMG    := $(FILES_DIR)/apple.icns
DMG_NAME    := $(APP_NAME)-mac
WIN_VERSION := 4.0.0.0

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
	uv run python -m nuitka \
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

# Windows
# --windows-console-mode=disable | attach
# --include-module=_json is built-in in Windows nuitka

# Build target for Windows
build_windows:
	@echo "--- Building Windows executable with Nuitka ---"
	uv run python -m nuitka \
	  --standalone \
	  --output-dir=dist \
	  --output-filename="$(EXE_NAME)" \
	  --product-name="$(EXE_NAME)" \
	  --product-version="$(WIN_VERSION)" \
	  --windows-console-mode=disable \
	  --windows-icon-from-ico="$(ICON_WIN)" \
	  --include-data-dir="$(FILES_DIR)=$(FILES_DIR)" \
	  --enable-plugin=tk-inter \
	  --module-parameter=torch-disable-jit=yes \
	  $(MAIN_SCRIPT)

# Build target for Linux
build_linux:
	@echo "--- Building Linux executable with Nuitka ---"
	uv run python -m nuitka \
	  --standalone \
	  --output-dir=dist \
	  --output-filename="$(EXE_NAME)" \
	  --product-name="$(EXE_NAME)" \
	  --include-data-dir="$(FILES_DIR)=$(FILES_DIR)" \
	  --enable-plugin=tk-inter \
	  --module-parameter=torch-disable-jit=yes \
	  --include-module=_json \
	  $(MAIN_SCRIPT)

# Make a disk image to distribute in macOS
# Automatically finds the .app file in the 'dist' directory.
make_dmg:
	@echo "--- Locating .app bundle ---"
	$(eval APP_PATH := $(shell find dist -maxdepth 1 -name "*.app"))
	$(eval APP_FILENAME := $(shell basename $(APP_PATH)))

	@if [ -z "$(APP_PATH)" ]; then \
		echo "Error: Could not find an .app bundle in the 'dist' directory."; \
		exit 1; \
	fi
	@echo "Found: $(APP_FILENAME)"

	@echo "--- Preparing DMG source directory ---"
	@rm -rf "dist/dmg_source"
	@mkdir -p "dist/dmg_source"
	@cp -R "$(APP_PATH)" "dist/dmg_source/"

	@echo "--- Creating distributable .dmg file ---"
	create-dmg \
	  --volname "$(APP_NAME)" \
	  --volicon "$(ICON_DMG)" \
	  --window-pos 200 120 \
	  --window-size 600 400 \
	  --icon-size 128 \
	  --icon "$(APP_FILENAME)" 175 120 \
	  --hide-extension "\$(APP_FILENAME)" \
	  --app-drop-link 425 120 \
	  "dist/$(DMG_NAME).dmg" \
	  "dist/dmg_source/"

	@echo "--- Cleaning up ---"
	@rm -rf "dist/dmg_source"


# Declare targets that are not files
.PHONY: help build_mac build_windows build_linux make_dmg 
