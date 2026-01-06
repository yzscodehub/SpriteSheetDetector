#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <optional>
#include <nlohmann/json.hpp>

struct SpriteInfo {
    int x;
    int y;
    int width;
    int height;
    
    NLOHMANN_DEFINE_TYPE_INTRUSIVE(SpriteInfo, x, y, width, height)
};

struct SpriteSheetInfo {
    // 检测结果。
    bool isSpriteSheet = false;
    // 网格形状（仅当 isSpriteSheet == true 时有意义）。
    int rows = 0;
    int cols = 0;
    // 名义上的格子尺寸（像素）。当图像尺寸无法被 rows/cols 整除时，
    // 实际每个格子的宽高可能因为取整而相差 1px 左右。
    int cellWidth = 0;
    int cellHeight = 0;
    // 可选：每个格子的矩形坐标列表（左上角 x/y + width/height）。
    // 注意：当前流程通常会关闭坐标生成以减少输出体积；
    // 若未生成，即使 isSpriteSheet==true，该列表也可能为空。
    std::vector<SpriteInfo> sprites;

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(SpriteSheetInfo, isSpriteSheet, rows, cols, cellWidth, cellHeight, sprites)
};

class SpriteDetector {
public:
    /// 规则网格(a*b)检测器（相对 `GridSpriteDetector` 更“快”，但候选更少）。
    ///
    /// ----------------------------
    /// 原理（完整版，面向工程实现）
    /// ----------------------------
    ///
    /// 该检测器与 `GridSpriteDetector` 共享同一套“证据验证”思想：
    /// - 线证据（grid lines vs gutters）
    /// - 格子占用证据（cell occupancy）
    /// - sparse-grid 支持（允许空格很多）
    ///
    /// 区别在于：候选 (rows, cols) 的生成策略不同。
    ///
    /// 伪代码（高层流程）：
    /// ```
    /// mask = makeForegroundMask(image)
    /// rawIntegral  = integral(mask>0)
    /// cutsIntegral = integral(erode(mask)>0)
    ///
    /// // 1) 估计步长（pitch）：用前景投影“段距”推断出大致的行/列周期
    /// stepX = estimatePitchByGaps(rawIntegral, axis=X)
    /// stepY = estimatePitchByGaps(rawIntegral, axis=Y)
    ///
    /// // 2) 基于步长生成小范围候选：围绕 (H/stepY, W/stepX) 做邻域枚举
    /// candidates += neighborhood(r0, c0)
    ///
    /// // 3) 加少量常见网格兜底（避免步长估计失败时完全没候选）
    /// candidates += commonGrids()
    ///
    /// // 4) 对每个候选做验证（与 GridSpriteDetector 基本一致）
    /// for (r,c) in candidates:
    ///   lineErr = checkGridLines(cutsIntegral, rawIntegral, r, c)
    ///   occ     = cellOccupancy(rawIntegral, r, c)
    ///   if passEvidence(lineErr, occ, componentsCount): accepted += (r,c)
    ///
    /// // 5) 从 accepted 中挑 best（以 lineErr 为主，结合步长 tie-break，倾向更密网格）
    /// best = selectBest(accepted)
    /// ```
    ///
    /// 为什么“更快”：
    /// - `GridSpriteDetector` 枚举所有可能 rows/cols（上限由 minCellPx 决定）。
    /// - 本检测器先用投影估计周期，把搜索空间收敛到很小的候选集合（典型几十到几百），所以更快。
    ///
    /// 为什么要保留 `GridSpriteDetector`：
    /// - 当步长估计失败（背景复杂/掩码质量差/特殊排布）时，这个 detector 可能漏检；
    /// - 全枚举版更稳健，但代价更高。
    ///
    /// 输出与限制：
    /// - 仅接受 rows>=2 且 cols>=2 的二维网格（1xN / Nx1 直接视为失败）
    /// - 默认不生成每格坐标（generateSprites=false），用于减少输出与加速
    static SpriteSheetInfo detect(const cv::Mat& image, bool generateSprites = false);

    // 校验指定网格(rows x cols)是否匹配（失败则 isSpriteSheet=false）
    static SpriteSheetInfo detectFixedGrid(const cv::Mat& image, int rows, int cols, bool generateSprites = false);

private:
    // 计算行列投影
    static void computeProjections(const cv::Mat& binaryMask, std::vector<int>& rowProj, std::vector<int>& colProj);
    
    // 分析投影数据，尝试找到规则的周期
    // 返回 {count, size}，如果没找到规律则返回 nullopt
    static std::optional<std::pair<int, int>> analyzeProjection(const std::vector<int>& projection, int totalLength);
    
    // 获取二值掩码（区分背景和前景）
    static cv::Mat getBinaryMask(const cv::Mat& image);

    // 提取所有的 Bounding Box (基于轮廓)
    static std::vector<SpriteInfo> extractSpritesByContours(const cv::Mat& mask);
};
