#pragma once

#include <opencv2/opencv.hpp>

namespace SpriteMask {

/// 为规则网格(a*b)精灵图检测生成前景二值掩码（CV_8UC1，像素值为 0 或 255）。
///
/// 网格检测器将“前景”视为精灵内容，并期望在网格分割线附近的背景/间隙(gutter)大多为空(0)。
///
/// ----------------------------
/// 原理（为什么需要这一步）
/// ----------------------------
/// 后续两套网格检测器（`GridSpriteDetector` / `SpriteDetector`）都不直接在彩色图上判断网格线，
/// 而是把问题化简为“网格线附近是否存在前景像素”：
/// - 若候选网格线落在格子间的背景缝隙(gutter)上，那么沿线 1px 条带的前景像素应很少；
/// - 反之，如果线切到了精灵内容，沿线前景像素会明显变多。
///
/// 因此我们需要一个可靠的前景二值掩码（0/255），并且尽量做到：
/// - 前景包含精灵主体（允许少量漏检/误检，但不能把整张图都当前景）
/// - 背景/空白区域尽量为 0（尤其是 gutter 区域）
///
/// 规则：
/// - 若图像带 alpha（4 通道）：前景 = (alpha > alphaThreshold)。
/// - 否则：从四个角点对背景进行 flood-fill（假设背景与图像边界连通），再取反得到前景。
///
/// 备注：
/// - 对于无 alpha 图，该方法在“背景相对均匀且与边界连通”时效果最好；若背景复杂/被包围，
///   可能导致掩码不准，从而影响网格判断。
///
/// @param image          输入图像（任意深度，1/3/4 通道均可）。
/// @param alphaThreshold 有 alpha 时使用的阈值。
/// @param floodDiff      无 alpha 时 flood-fill 的容差（每通道允许的强度差）。
cv::Mat makeForegroundMask(const cv::Mat& image, int alphaThreshold = 20, int floodDiff = 5);

} // namespace SpriteMask



