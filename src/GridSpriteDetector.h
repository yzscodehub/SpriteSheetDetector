#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <optional>
#include <nlohmann/json.hpp>
#include "SpriteDetector.h" // 复用结构体定义

class GridSpriteDetector {
public:
    /// 从输入图像中检测“规则网格(a*b)”精灵图。
    ///
    /// ----------------------------
    /// 原理（完整版，面向“规则网格切分”场景）
    /// ----------------------------
    ///
    /// 目标：
    /// - 给定一张图片，判断它是否是“规则网格 a*b”的精灵表（spritesheet），并输出 rows/cols。
    /// - “规则网格”指：所有网格线在全图范围内遵循同一组 (rows, cols) 的均分规则，且格子间通常存在背景缝隙(gutter)。
    ///
    /// 本检测器是 **规则驱动 / 证据验证** 的，而不是“通用分割器”：
    /// - 只接受真正的二维网格：rows>=2 且 cols>=2（1xN / Nx1 直接视为失败）。
    /// - 网格定义在 **整张图片范围内**（不会先裁剪 content bounds 再均分），原因：
    ///   - 很多精灵格内部有 padding/对齐空白；如果先按内容裁剪，网格线会整体错位。
    ///
    /// 核心证据（需要同时成立）：
    /// 1) 网格线证据（Line Evidence）：候选 (rows, cols) 的所有内部分割线应尽量落在“空缝/gutter”上。
    ///    - 通过二值前景掩码的积分图 integral01 快速统计“分割线 1px 宽条带”上的前景像素数；
    ///    - 允许在每条线附近 ±radius 小范围搜索最空的位置（容忍取整误差/轻微抖动/抗锯齿）。
    /// 2) 格子占用证据（Cell Evidence）：把整图均分成 rows*cols 个格子后，大多数格子应该有“足够多”的前景像素。
    ///    - 这是为了抑制误报：一张普通插画/单个角色往往也能碰巧让几条线很“空”，但格子占用不会像精灵表那样规律。
    ///    - 但：有些表允许大量空帧/空列（例如动画缺帧），因此这里实现了 sparse-grid 支持（见下）。
    ///
    /// 前景掩码（Foreground Mask）：
    /// - 见 `SpriteMask::makeForegroundMask()`：
    ///   - 若 PNG 有 alpha：前景 = alpha > threshold（最稳健）。
    ///   - 否则：从四角 flood-fill 背景后取反（假设背景与边界连通且相对均匀）。
    /// - 线证据会同时使用两种 mask：
    ///   - “腐蚀后的 maskCuts”：更容忍抗锯齿边缘/描边，避免把边缘当成“切线”；
    ///   - “原始 maskRaw”：防止过度腐蚀导致误认为线很干净。
    ///
    /// sparse-grid 支持（允许很多空格但不影响其它结果）：
    /// - 对 occ.nonEmptyRatio < 0.70 的情况，不直接否决，而是要求更严格的组合条件：
    ///   - 组件数量（connected components）足够大（避免单个对象/插画误报）；
    ///   - 线误差 lineErr 更小（网格线更“干净”）；
    ///   - 网格格子数 rows*cols 不能远大于“有效组件数”（避免超细网格/过切）。
    /// - 同时对组件做自适应面积阈值（参考 p90）过滤粒子/噪点，避免特效碎片抬高组件数。
    ///
    /// 候选生成与选择：
    /// - 本检测器是“全枚举 + 多重剪枝”：
    ///   - 枚举 rows, cols 直到格子最小尺寸小于 minCellPx；
    ///   - 先用 cuts-mask 的线检查做廉价剪枝，再用 raw+cuts 组合得分；
    ///   - 通过 Cell Evidence 与 sparse 规则后进入 accepted 集合。
    /// - 最终选择：
    ///   - 先最小化 lineErr（网格线一致性）；
    ///   - 在接近最小误差的候选中，用步长估计（投影/组件中心）作为 tie-break，
    ///     避免因子分解歧义（如 10x10 vs 5x20）；
    ///   - 再在 remaining 候选中倾向更密的网格（避免子谐波过粗网格）；
    ///   - 若存在“密集网格候选”，优先在密集候选中选 best（避免 superharmonic 网格例如 9x5 赢过真实 5x5）。
    ///
    /// 复杂度：
    /// - O(R*C*(rows+cols)) 的枚举式搜索（R/C为最大行列候选），但大量剪枝使得常见精灵表很快。
    ///
    /// 返回值：
    /// - 成功：isSpriteSheet=true，填充 rows/cols 与 cellWidth/cellHeight。
    /// - 失败：isSpriteSheet=false，其余字段保持默认值。
    ///
    /// @param image           输入图像（推荐带 alpha 的 PNG；JPEG 等无 alpha 也支持，但依赖 flood-fill 掩码质量）。
    /// @param generateSprites 是否生成每个格子的坐标列表：
    ///                        - true：填充 SpriteSheetInfo::sprites（每格一个矩形）。
    ///                        - false（默认）：不生成坐标，sprites 为空，用于减少输出体积。
    static SpriteSheetInfo detect(const cv::Mat& image, bool generateSprites = false);
    /// 验证指定的规则网格 (rows x cols) 是否匹配该图像，并输出基础网格信息。
    ///
    /// 适用场景：你已经知道网格是 a*b，只想做一致性校验并拿到 rows/cols/cellWidth/cellHeight。
    /// 规则/假设同 detect()：只接受 rows>=2 且 cols>=2。
    ///
    /// @param image           输入图像。
    /// @param rows            期望行数（>=2）。
    /// @param cols            期望列数（>=2）。
    /// @param generateSprites 同 detect()。
    static SpriteSheetInfo detectFixedGrid(const cv::Mat& image, int rows, int cols, bool generateSprites = false);

private:
    // 获取有效内容区域 (Auto-Crop)
    static cv::Rect getContentBounds(const cv::Mat& mask);
    
    // 检查网格线是否干净
    // 返回：切到的非透明像素比例 (0.0 = 完美干净, 1.0 = 全是像素)
    static double checkGridLines(const cv::Mat& mask, const cv::Mat& integral01, const cv::Rect& bounds, int rows, int cols, int searchRadiusPx);
    static double computeFilledCellRatio(const cv::Mat& integral01, const cv::Rect& bounds, int rows, int cols, double perCellFillThreshold);
};

