# SpriteSheetDetector

A high-performance C++ tool designed to detect sprite sheets and determine their grid layout (rows × columns). It utilizes OpenCV for image processing and analysis to robustly identify regular grid patterns in images.

## Features

- **Dual Algorithms**:
  - `GridSpriteDetector`: Exhaustive search algorithm for maximum robustness.
  - `SpriteDetector`: Fast estimation algorithm based on projection analysis.
- **Batch Processing**: Supports processing single files or entire directories.
- **Format Support**: Handles PNG, JPG, JPEG, and BMP images.
- **JSON Output**: Generates detailed analysis results in JSON format, including execution time and confidence metrics for both algorithms.
- **Fixed Grid Verification**: Optional mode to verify if an image fits a specific grid layout.

## Prerequisites

- **C++ Compiler**: C++20 compatible compiler (MSVC, GCC, Clang).
- **CMake**: Version 3.21 or later.
- **vcpkg**: For dependency management.

## Dependencies

The project relies on the following libraries (managed via vcpkg):
- [OpenCV](https://opencv.org/): For image processing.
- [nlohmann-json](https://github.com/nlohmann/json): For JSON serialization.

## Build Instructions

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/<YOUR_USERNAME>/SpriteSheetDetector.git
    cd SpriteSheetDetector
    ```

2.  **Configure with CMake** (assuming vcpkg is installed):
    ```bash
    cmake -B build -S . -DCMAKE_TOOLCHAIN_FILE=[path/to/vcpkg]/scripts/buildsystems/vcpkg.cmake
    ```

3.  **Build**:
    ```bash
    cmake --build build --config Release
    ```

## Usage

Run the executable from the command line:

```bash
SpriteSheetDetector.exe <input_path> [rows cols]
```

### Arguments

- `<input_path>`: Path to a single image file or a directory containing images.
- `[rows cols]` (Optional): Force verification against a specific grid size (e.g., `4 4`).

### Examples

**Analyze a single image:**
```bash
SpriteSheetDetector.exe "C:/Assets/character_sheet.png"
```

**Batch process a directory:**
```bash
SpriteSheetDetector.exe "C:/Assets/Sprites/"
```

**Verify if an image is a 4x4 grid:**
```bash
SpriteSheetDetector.exe "C:/Assets/grid.png" 4 4
```

## Output

The tool outputs detection logs to the console and saves a full report to a JSON file.

- **Single File Input**: Generates `<input_filename>.results.json`.
- **Directory Input**: Generates `SpriteSheetDetector_results.json` inside the directory.

**Sample JSON Output:**
```json
{
  "file": "character.png",
  "ok": true,
  "algorithms": {
    "GridSpriteDetector": {
      "isSpriteSheet": true,
      "rows": 4,
      "cols": 8,
      "timeMs": 45
    },
    "SpriteDetectorFast": {
      "isSpriteSheet": true,
      "rows": 4,
      "cols": 8,
      "timeMs": 5
    }
  }
}
```

## License

MIT License

