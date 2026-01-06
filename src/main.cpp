#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <nlohmann/json.hpp>
#include <fstream>
#include <filesystem>
#include <algorithm>
#include <cctype>
#include <chrono>
#include "SpriteDetector.h"
#include "GridSpriteDetector.h" // 引入新检测器

using json = nlohmann::json;
namespace fs = std::filesystem;

int main(int argc, char** argv) {
    // Keep console output focused on detection results (suppress OpenCV INFO logs).
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);

    if (argc > 1) {
        std::string inputPath = argv[1];

        // Optional: fixed grid
        // Usage:
        //   SpriteSheetDetector.exe <file_or_directory_path> [rows cols]
        int fixedRows = 0;
        int fixedCols = 0;
        if (argc >= 4) {
            try {
                fixedRows = std::max(0, std::stoi(argv[2]));
                fixedCols = std::max(0, std::stoi(argv[3]));
            } catch (...) {
                fixedRows = 0;
                fixedCols = 0;
            }
        }
        
        std::vector<std::string> filesToProcess;
        
        if (fs::is_directory(inputPath)) {
            std::cout << "Processing directory: " << inputPath << std::endl;
            for (const auto& entry : fs::directory_iterator(inputPath)) {
                if (entry.is_regular_file()) {
                    std::string ext = entry.path().extension().string();
                    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                    if (ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".bmp") {
                        filesToProcess.push_back(entry.path().string());
                    }
                }
            }
        } else {
            filesToProcess.push_back(inputPath);
        }

        std::sort(filesToProcess.begin(), filesToProcess.end());

        int successCount = 0;
        json out;
        out["input"] = inputPath;
        out["results"] = json::array();
        long long totalMs = 0;
        long long totalGridMs = 0;
        long long totalSpriteMs = 0;
        int detectedGridCount = 0;
        int detectedSpriteCount = 0;

        for (const auto& filePath : filesToProcess) {
            std::cout << "  Analyzing: " << fs::path(filePath).filename().string() << "... ";
            
            cv::Mat image = cv::imread(filePath, cv::IMREAD_UNCHANGED);
            if (image.empty()) {
                std::cout << "[ERROR] Failed to load image." << std::endl;
                out["results"].push_back({
                    {"file", fs::path(filePath).filename().string()},
                    {"path", filePath},
                    {"imageWidth", 0},
                    {"imageHeight", 0},
                    {"ok", false},
                    {"error", "Failed to load image"}
                });
                continue;
            }

            // 规则网格 a*b：对比两种算法（GridSpriteDetector / 优化后的 SpriteDetector）
            SpriteSheetInfo gridInfo;
            const auto t0 = std::chrono::steady_clock::now();
            if (fixedRows > 0 && fixedCols > 0) {
                gridInfo = GridSpriteDetector::detectFixedGrid(image, fixedRows, fixedCols, /*generateSprites=*/false);
            } else {
                gridInfo = GridSpriteDetector::detect(image, /*generateSprites=*/false);
            }
            const auto t1 = std::chrono::steady_clock::now();
            const long long gridMs = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

            SpriteSheetInfo spriteInfo;
            const auto t2 = std::chrono::steady_clock::now();
            if (fixedRows > 0 && fixedCols > 0) {
                spriteInfo = SpriteDetector::detectFixedGrid(image, fixedRows, fixedCols, /*generateSprites=*/false);
            } else {
                spriteInfo = SpriteDetector::detect(image, /*generateSprites=*/false);
            }
            const auto t3 = std::chrono::steady_clock::now();
            const long long spriteMs = std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count();

            totalGridMs += gridMs;
            totalSpriteMs += spriteMs;
            totalMs += (gridMs + spriteMs);

            if (gridInfo.isSpriteSheet) detectedGridCount++;
            if (spriteInfo.isSpriteSheet) detectedSpriteCount++;

            // 默认“总体 detectedCount”以 GridSpriteDetector 为准（更稳）；同时输出两者结果供对比
            const bool detected = gridInfo.isSpriteSheet;
            if (detected) successCount++;

            std::cout << (detected ? "[DETECTED]" : "[NOT A SHEET]")
                      << " Grid=" << (gridInfo.isSpriteSheet ? (std::to_string(gridInfo.rows) + "x" + std::to_string(gridInfo.cols)) : std::string("NA"))
                      << " (" << gridMs << " ms)"
                      << ", Sprite=" << (spriteInfo.isSpriteSheet ? (std::to_string(spriteInfo.rows) + "x" + std::to_string(spriteInfo.cols)) : std::string("NA"))
                      << " (" << spriteMs << " ms)"
                      << std::endl;

            json algorithms;
            algorithms["GridSpriteDetector"] = {
                {"isSpriteSheet", gridInfo.isSpriteSheet},
                {"rows", gridInfo.rows},
                {"cols", gridInfo.cols},
                {"cellWidth", gridInfo.cellWidth},
                {"cellHeight", gridInfo.cellHeight},
                {"timeMs", gridMs}
            };
            algorithms["SpriteDetectorFast"] = {
                {"isSpriteSheet", spriteInfo.isSpriteSheet},
                {"rows", spriteInfo.rows},
                {"cols", spriteInfo.cols},
                {"cellWidth", spriteInfo.cellWidth},
                {"cellHeight", spriteInfo.cellHeight},
                {"timeMs", spriteMs}
            };

            out["results"].push_back({
                {"file", fs::path(filePath).filename().string()},
                {"path", filePath},
                {"imageWidth", image.cols},
                {"imageHeight", image.rows},
                {"ok", true},
                {"algorithms", algorithms}
            });
        }

        out["totalTimeMs"] = totalMs;
        out["detectedCount"] = successCount;
        out["fileCount"] = static_cast<int>(filesToProcess.size());
        out["algorithmsSummary"] = {
            {"GridSpriteDetector", {{"detectedCount", detectedGridCount}, {"totalTimeMs", totalGridMs}}},
            {"SpriteDetectorFast", {{"detectedCount", detectedSpriteCount}, {"totalTimeMs", totalSpriteMs}}}
        };

        std::string outPath;
        if (fs::is_directory(inputPath)) {
            outPath = (fs::path(inputPath) / "SpriteSheetDetector_results.json").string();
        } else {
            outPath = (fs::path(inputPath).string() + ".results.json");
        }

        std::ofstream o(outPath);
        o << out.dump(2) << std::endl;
        std::cout << "\nBatch processing complete. Detected " << successCount << "/" << filesToProcess.size()
                  << " files as sprite sheets. Total " << totalMs << " ms. -> " << fs::path(outPath).filename().string() << std::endl;
        return 0;
    }

    std::cout << "Usage: SpriteSheetDetector.exe <file_or_directory_path> [rows cols]" << std::endl;
    return 1;
}
