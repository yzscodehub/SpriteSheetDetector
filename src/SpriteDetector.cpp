#include "SpriteDetector.h"
#include "ForegroundMask.h"
#include <numeric>
#include <algorithm>
#include <cmath>
#include <map>

namespace {

static inline int roundDivPos(int i, int total, int parts) {
    return static_cast<int>(std::lround(static_cast<double>(i) * static_cast<double>(total) / static_cast<double>(parts)));
}

// integral01: cv::integral(binary01, integral01, CV_32S) => (H+1)x(W+1)
static inline int rectSum32S(const cv::Mat& integral01, int x, int y, int w, int h) {
    if (w <= 0 || h <= 0) return 0;
    const int x1 = x, y1 = y;
    const int x2 = x + w, y2 = y + h;
    const int A = integral01.at<int>(y1, x1);
    const int B = integral01.at<int>(y1, x2);
    const int C = integral01.at<int>(y2, x1);
    const int D = integral01.at<int>(y2, x2);
    return D - B - C + A;
}

// ----------------------------
// 原理说明（实现细节层）
// ----------------------------
// `SpriteDetector` 与 `GridSpriteDetector` 共用同一套“证据验证”：
// - Line Evidence：候选网格线应穿过背景缝隙(gutter)，在 mask 上表现为“线条带前景像素少”
// - Cell Evidence：候选网格均分后，格子占用具有规则性；并支持 sparse-grid（允许大量空格，但约束更严）
//
// 与 `GridSpriteDetector` 的主要区别：这里不会枚举所有 rows/cols，而是先估计周期(pitch)来生成少量候选：
// - 使用前景投影（每一行/列是否存在前景）形成 0/1 序列
// - 将“连续前景段”的中心点作为事件，统计相邻事件中心点的距离分布，取峰值作为 pitch 估计
// - 再围绕 (H/pitchY, W/pitchX) 形成邻域候选，并加入少量常见网格兜底
//
// 这样多数精灵表可以用很小的候选集合完成验证，因此速度更快；但当掩码质量差或排布特殊时可能漏检，
// 这也是项目里保留 `GridSpriteDetector` 作为更稳健版本的原因。

struct CellOccupancyStats {
    double nonEmptyRatio = 0.0;
    double meanPixels = 0.0;
};

static int countComponentsAboveArea(const cv::Mat& binary01, int minAreaPx) {
    if (binary01.empty()) return 0;
    cv::Mat labels, stats, centroids;
    const int n = cv::connectedComponentsWithStats(binary01, labels, stats, centroids, 8, CV_32S);
    if (n <= 1) return 0;

    std::vector<int> areas;
    areas.reserve(static_cast<size_t>(n - 1));
    for (int i = 1; i < n; ++i) areas.push_back(stats.at<int>(i, cv::CC_STAT_AREA));
    const size_t p90i = (areas.size() * 9) / 10;
    std::nth_element(areas.begin(), areas.begin() + p90i, areas.end());
    const int areaP90 = areas[p90i];
    const int areaThresh = std::max(minAreaPx, std::max(1, areaP90 / 10));

    int count = 0;
    for (int i = 1; i < n; ++i) {
        const int area = stats.at<int>(i, cv::CC_STAT_AREA);
        if (area >= areaThresh) count++;
    }
    return count;
}

static bool passCellEvidence(
    const CellOccupancyStats& occ,
    double lineErr,
    int rows,
    int cols,
    int spriteComponents
) {
    // 与 GridSpriteDetector 相同：先尝试 dense 模式；否则进入 sparse 模式并提高约束。
    if (occ.meanPixels < 10.0) return false;
    if (occ.nonEmptyRatio > 0.70) return true;

    constexpr int kMinSpriteComponents = 8;
    constexpr int kMinCells = 16;
    constexpr int kMaxCellsPerComponent = 10;
    constexpr double kMinNonEmptyRatio = 0.05;
    constexpr double kMaxLineErrSparse = 0.28;

    const int cells = rows * cols;
    if (spriteComponents < kMinSpriteComponents) return false;
    if (cells < kMinCells) return false;
    if (cells > spriteComponents * kMaxCellsPerComponent) return false;
    if (occ.nonEmptyRatio < kMinNonEmptyRatio) return false;
    if (lineErr > kMaxLineErrSparse) return false;
    return true;
}

static CellOccupancyStats computeCellOccupancyStats(const cv::Mat& integral01, int W, int H, int rows, int cols) {
    CellOccupancyStats s{};
    if (rows <= 0 || cols <= 0) return s;
    const int totalCells = rows * cols;
    const int totalPix = rectSum32S(integral01, 0, 0, W, H);
    s.meanPixels = (totalCells > 0) ? (static_cast<double>(totalPix) / static_cast<double>(totalCells)) : 0.0;

    const int pixThresh = std::max(6, static_cast<int>(std::lround(s.meanPixels * 0.20)));
    int nonEmpty = 0;
    for (int r = 0; r < rows; ++r) {
        const int y0 = roundDivPos(r, H, rows);
        const int y1 = roundDivPos(r + 1, H, rows);
        const int h = std::max(0, y1 - y0);
        for (int c = 0; c < cols; ++c) {
            const int x0 = roundDivPos(c, W, cols);
            const int x1 = roundDivPos(c + 1, W, cols);
            const int w = std::max(0, x1 - x0);
            if (w <= 0 || h <= 0) continue;
            const int pix = rectSum32S(integral01, x0, y0, w, h);
            if (pix >= pixThresh) nonEmpty++;
        }
    }
    s.nonEmptyRatio = (totalCells > 0) ? (static_cast<double>(nonEmpty) / static_cast<double>(totalCells)) : 0.0;
    return s;
}

// 检测网格分割线是否能贴近“空隙(gutter)”：
// - 使用 integral01（0/1）快速统计每条候选分割线上的前景像素数
// - 返回值越小越好
static double checkGridLinesFast(const cv::Mat& integral01, int W, int H, int rows, int cols, int radius) {
    if (rows <= 1 || cols <= 1) return 1.0;
    radius = std::clamp(radius, 0, 24);

    long long cutPixels = 0;
    long long totalCheck = 0;

    // 若大部分分割线附近找不到足够“空”的线，则认为不是规则网格
    constexpr double kMaxLineFill = 0.10; // 单条线允许的最大填充比例（0/1像素）
    int badLines = 0;
    int totalLines = 0;

    // 横线
    for (int r = 1; r < rows; ++r) {
        const int y = roundDivPos(r, H, rows);
        int minPix = W;
        for (int dy = -radius; dy <= radius; ++dy) {
            const int yy = y + dy;
            if (yy < 0 || yy >= H) continue;
            minPix = std::min(minPix, rectSum32S(integral01, 0, yy, W, 1));
        }
        cutPixels += minPix;
        totalCheck += W;
        totalLines++;
        if (minPix > static_cast<int>(std::lround(W * kMaxLineFill))) badLines++;
    }

    // 竖线
    for (int c = 1; c < cols; ++c) {
        const int x = roundDivPos(c, W, cols);
        int minPix = H;
        for (int dx = -radius; dx <= radius; ++dx) {
            const int xx = x + dx;
            if (xx < 0 || xx >= W) continue;
            minPix = std::min(minPix, rectSum32S(integral01, xx, 0, 1, H));
        }
        cutPixels += minPix;
        totalCheck += H;
        totalLines++;
        if (minPix > static_cast<int>(std::lround(H * kMaxLineFill))) badLines++;
    }

    if (totalCheck <= 0) return 1.0;
    const double baseErr = static_cast<double>(cutPixels) / static_cast<double>(totalCheck);
    const double badRatio = (totalLines > 0) ? (static_cast<double>(badLines) / static_cast<double>(totalLines)) : 0.0;
    return baseErr + 0.12 * badRatio;
}

// 辅助：基于“有无前景”的投影段距，估计平均步长（像素）。
struct StepEstimate {
    int stepPx = 0;
    int support = 0;
    int segments = 0;
};

struct ComponentSet {
    int count = 0;
    std::vector<cv::Point2d> centers;
    std::vector<int> widths;
    std::vector<int> heights;
};

static ComponentSet extractComponents(const cv::Mat& binary01, int minAreaPx) {
    ComponentSet out{};
    if (binary01.empty()) return out;
    cv::Mat labels, stats, centroids;
    const int n = cv::connectedComponentsWithStats(binary01, labels, stats, centroids, 8, CV_32S);
    if (n <= 1) return out;

    std::vector<int> areas;
    areas.reserve(static_cast<size_t>(n - 1));
    for (int i = 1; i < n; ++i) areas.push_back(stats.at<int>(i, cv::CC_STAT_AREA));
    const size_t p90i = (areas.size() * 9) / 10;
    std::nth_element(areas.begin(), areas.begin() + p90i, areas.end());
    const int areaP90 = areas[p90i];
    const int areaThresh = std::max(minAreaPx, std::max(1, areaP90 / 10));

    out.centers.reserve(static_cast<size_t>(std::max(0, n - 1)));
    out.widths.reserve(static_cast<size_t>(std::max(0, n - 1)));
    out.heights.reserve(static_cast<size_t>(std::max(0, n - 1)));
    for (int i = 1; i < n; ++i) {
        const int area = stats.at<int>(i, cv::CC_STAT_AREA);
        if (area < areaThresh) continue;
        out.count++;
        out.widths.push_back(stats.at<int>(i, cv::CC_STAT_WIDTH));
        out.heights.push_back(stats.at<int>(i, cv::CC_STAT_HEIGHT));
        out.centers.emplace_back(centroids.at<double>(i, 0), centroids.at<double>(i, 1));
    }
    return out;
}

static double medianInt(std::vector<int> v) {
    if (v.empty()) return 0.0;
    const size_t mid = v.size() / 2;
    std::nth_element(v.begin(), v.begin() + mid, v.end());
    double m = static_cast<double>(v[mid]);
    if ((v.size() & 1u) == 0u) {
        std::nth_element(v.begin(), v.begin() + (mid - 1), v.end());
        m = 0.5 * (m + static_cast<double>(v[mid - 1]));
    }
    return m;
}

static std::vector<double> cluster1D(std::vector<double> vals, double eps) {
    std::vector<double> centers;
    if (vals.empty()) return centers;
    std::sort(vals.begin(), vals.end());
    eps = std::max(0.0, eps);

    double sum = vals[0];
    int cnt = 1;
    double mean = vals[0];
    for (size_t i = 1; i < vals.size(); ++i) {
        const double v = vals[i];
        if (std::abs(v - mean) <= eps) {
            sum += v;
            cnt++;
            mean = sum / static_cast<double>(cnt);
        } else {
            centers.push_back(mean);
            sum = v;
            cnt = 1;
            mean = v;
        }
    }
    centers.push_back(mean);
    return centers;
}

static StepEstimate estimateStepFromComponentCenters(const ComponentSet& comps, int axisLenPx, bool isRow) {
    StepEstimate out{};
    if (comps.count < 2) return out;

    const double medW = medianInt(comps.widths);
    const double medH = medianInt(comps.heights);
    // Wide epsilon to avoid splitting a single animation row/col into multiple clusters due to pose variation.
    const double eps = std::max(2.0, (isRow ? (0.60 * medH) : (0.60 * medW)));

    std::vector<double> vals;
    vals.reserve(comps.centers.size());
    for (const auto& p : comps.centers) vals.push_back(isRow ? p.y : p.x);
    const std::vector<double> centers = cluster1D(std::move(vals), eps);
    out.segments = static_cast<int>(centers.size());
    if (centers.size() < 2) return out;

    std::map<int, int> distCounts;
    for (size_t i = 0; i + 1 < centers.size(); ++i) {
        const double d = centers[i + 1] - centers[i];
        if (d <= eps) continue;
        const int q = static_cast<int>(std::lround(d / 2.0)) * 2; // quantize to 2px
        if (q >= 6 && q <= axisLenPx) distCounts[q]++;
    }
    int bestStep = 0;
    int bestCount = 0;
    double bestScore = -1.0;
    for (const auto& [step, count] : distCounts) {
        const double score = static_cast<double>(count) / static_cast<double>(std::max(1, step));
        if (score > bestScore) { bestScore = score; bestStep = step; bestCount = count; }
    }
    out.stepPx = bestStep;
    out.support = bestCount;
    return out;
}

static StepEstimate findStepByGapsIntegral(const cv::Mat& integral01, int W, int H, bool isRow) {
    StepEstimate out{};
    std::vector<uint8_t> proj;
    if (isRow) {
        proj.resize(static_cast<size_t>(H), 0);
        for (int y = 0; y < H; ++y) {
            proj[static_cast<size_t>(y)] = rectSum32S(integral01, 0, y, W, 1) > 0 ? 1 : 0;
        }
    } else {
        proj.resize(static_cast<size_t>(W), 0);
        for (int x = 0; x < W; ++x) {
            proj[static_cast<size_t>(x)] = rectSum32S(integral01, x, 0, 1, H) > 0 ? 1 : 0;
        }
    }

    struct Seg { int start, end; };
    std::vector<Seg> segs;
    bool in = false;
    int start = 0;
    for (int i = 0; i < static_cast<int>(proj.size()); ++i) {
        if (proj[static_cast<size_t>(i)] && !in) { in = true; start = i; }
        else if (!proj[static_cast<size_t>(i)] && in) { in = false; segs.push_back({start, i}); }
    }
    if (in) segs.push_back({start, static_cast<int>(proj.size())});

    out.segments = static_cast<int>(segs.size());
    if (segs.size() < 2) return out;

    std::map<int, int> distCounts;
    for (size_t i = 0; i + 1 < segs.size(); ++i) {
        const int c1 = (segs[i].start + segs[i].end) / 2;
        const int c2 = (segs[i + 1].start + segs[i + 1].end) / 2;
        const int dist = c2 - c1;
        const int q = (dist + 2) / 5 * 5; // 量化到 5px
        if (q > 0) distCounts[q]++;
    }

    int bestStep = 0;
    int maxCount = 0;
    for (auto const& [step, count] : distCounts) {
        if (count > maxCount) { maxCount = count; bestStep = step; }
    }
    out.stepPx = bestStep;
    out.support = maxCount;
    return out;
}

} // namespace

SpriteSheetInfo SpriteDetector::detectFixedGrid(const cv::Mat& image, int rows, int cols, bool generateSprites) {
    SpriteSheetInfo info;
    info.isSpriteSheet = false;
    if (image.empty()) return info;
    if (rows < 2 || cols < 2) return info;

    // 统一前景掩码（复用当前项目的规则）
    cv::Mat mask = SpriteMask::makeForegroundMask(image, /*alphaThreshold=*/20, /*floodDiff=*/5);
    if (mask.empty() || cv::countNonZero(mask) == 0) return info;

    cv::Mat maskCuts;
    {
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
        cv::erode(mask, maskCuts, kernel, cv::Point(-1, -1), 1);
    }

    const int W = image.cols;
    const int H = image.rows;

    cv::Mat raw01 = (mask > 0);
    cv::Mat rawIntegral;
    cv::integral(raw01, rawIntegral, CV_32S);

    cv::Mat cuts01 = (maskCuts > 0);
    cv::Mat cutsIntegral;
    cv::integral(cuts01, cutsIntegral, CV_32S);

    const int spriteComponents = countComponentsAboveArea(cuts01, /*minAreaPx=*/32);

    const int minCellPx = 6;
    if (W / cols < minCellPx || H / rows < minCellPx) return info;

    const CellOccupancyStats occ = computeCellOccupancyStats(rawIntegral, W, H, rows, cols);
    const int radius = std::clamp(std::min(W / cols, H / rows) / 10, 2, 12);
    const double lineCuts = checkGridLinesFast(cutsIntegral, W, H, rows, cols, radius);
    const double lineRaw = checkGridLinesFast(rawIntegral, W, H, rows, cols, radius);
    const double lineErr = 0.65 * lineCuts + 0.35 * lineRaw;
    if (lineErr >= 0.35) return info;
    if (!passCellEvidence(occ, lineErr, rows, cols, spriteComponents)) return info;

    info.isSpriteSheet = true;
    info.rows = rows;
    info.cols = cols;
    info.cellWidth = static_cast<int>(std::lround(static_cast<double>(W) / cols));
    info.cellHeight = static_cast<int>(std::lround(static_cast<double>(H) / rows));

    if (generateSprites) {
        info.sprites.reserve(static_cast<size_t>(rows) * static_cast<size_t>(cols));
        for (int r = 0; r < rows; ++r) {
            const int y0 = roundDivPos(r, H, rows);
            const int y1 = roundDivPos(r + 1, H, rows);
            for (int c = 0; c < cols; ++c) {
                const int x0 = roundDivPos(c, W, cols);
                const int x1 = roundDivPos(c + 1, W, cols);
                info.sprites.push_back({x0, y0, std::max(0, x1 - x0), std::max(0, y1 - y0)});
            }
        }
    }

    return info;
}

SpriteSheetInfo SpriteDetector::detect(const cv::Mat& image, bool generateSprites) {
    SpriteSheetInfo info;
    info.isSpriteSheet = false;
    if (image.empty()) return info;

    cv::Mat mask = SpriteMask::makeForegroundMask(image, /*alphaThreshold=*/20, /*floodDiff=*/5);
    if (mask.empty() || cv::countNonZero(mask) == 0) return info;

    const int W = image.cols;
    const int H = image.rows;

    // 积分图：用于快速统计行/列/格子像素
    cv::Mat raw01 = (mask > 0);
    cv::Mat rawIntegral;
    cv::integral(raw01, rawIntegral, CV_32S);

    // 候选生成：先用投影估计步长，缩小搜索空间（相比 GridSpriteDetector 枚举更少，因此更快）
    const StepEstimate stepY_proj = findStepByGapsIntegral(rawIntegral, W, H, /*isRow=*/true);
    const StepEstimate stepX_proj = findStepByGapsIntegral(rawIntegral, W, H, /*isRow=*/false);

    const int minCellPx = 6;
    std::vector<std::pair<int, int>> candidates;

    auto addCandidate = [&](int r, int c) {
        if (r < 2 || c < 2) return;
        if (W / c < minCellPx || H / r < minCellPx) return;
        candidates.emplace_back(r, c);
    };

    if (stepY_proj.stepPx > 0 && stepX_proj.stepPx > 0) {
        const int r0 = static_cast<int>(std::lround(static_cast<double>(H) / static_cast<double>(stepY_proj.stepPx)));
        const int c0 = static_cast<int>(std::lround(static_cast<double>(W) / static_cast<double>(stepX_proj.stepPx)));
        for (int dr = -2; dr <= 2; ++dr) {
            for (int dc = -2; dc <= 2; ++dc) {
                addCandidate(r0 + dr, c0 + dc);
            }
        }
    }

    // 常见网格兜底（少量）
    for (int r = 2; r <= 12; ++r) {
        for (int c = 2; c <= 24; ++c) {
            addCandidate(r, c);
        }
    }

    std::sort(candidates.begin(), candidates.end());
    candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());

    // 线检测使用两种 mask：腐蚀版容忍边缘，原始版防止过切
    cv::Mat maskCuts;
    {
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
        cv::erode(mask, maskCuts, kernel, cv::Point(-1, -1), 1);
    }
    cv::Mat cuts01 = (maskCuts > 0);
    cv::Mat cutsIntegral;
    cv::integral(cuts01, cutsIntegral, CV_32S);

    const ComponentSet comps = extractComponents(cuts01, /*minAreaPx=*/32);
    const int spriteComponents = comps.count;
    const StepEstimate stepY_comp = estimateStepFromComponentCenters(comps, H, /*isRow=*/true);
    const StepEstimate stepX_comp = estimateStepFromComponentCenters(comps, W, /*isRow=*/false);
    const StepEstimate stepY = (stepY_comp.support >= stepY_proj.support) ? stepY_comp : stepY_proj;
    const StepEstimate stepX = (stepX_comp.support >= stepX_proj.support) ? stepX_comp : stepX_proj;

    struct CandScore {
        int r = 0;
        int c = 0;
        double lineErr = 1e9;
        double nonEmptyRatio = 0.0;
        double meanPixels = 0.0;
    };

    std::vector<CandScore> accepted;
    accepted.reserve(candidates.size());

    for (const auto& [r, c] : candidates) {
        const CellOccupancyStats occ = computeCellOccupancyStats(rawIntegral, W, H, r, c);
        const int radius = std::clamp(std::min(W / c, H / r) / 10, 2, 12);
        const double lineCuts = checkGridLinesFast(cutsIntegral, W, H, r, c, radius);
        if (lineCuts > 0.55) continue;
        const double lineRaw = checkGridLinesFast(rawIntegral, W, H, r, c, radius);
        const double lineErr = 0.65 * lineCuts + 0.35 * lineRaw;
        if (lineErr > 0.65) continue;
        if (!passCellEvidence(occ, lineErr, r, c, spriteComponents)) continue;

        accepted.push_back({r, c, lineErr, occ.nonEmptyRatio, occ.meanPixels});
    }

    if (accepted.empty()) return info;

    // 选择策略：
    // 1) 优先最小 lineErr
    // 2) 在接近最小误差的候选中，使用(可选)步长估计作为 tie-break（避免 10x10 被 5x20/20x5 混淆）
    // 3) 再优先更密的网格（避免子谐波粗网格）
    double minErr = accepted[0].lineErr;
    for (const auto& a : accepted) minErr = std::min(minErr, a.lineErr);
    const double slack = std::max(0.10, minErr * 0.50);

    auto stepReliable = [](const StepEstimate& e) {
        return (e.stepPx > 0) && (e.support >= 3) && (e.segments >= 4);
    };
    const bool useStepX = stepReliable(stepX);
    const bool useStepY = stepReliable(stepY);

    auto stepMismatch = [&](const CandScore& a) -> double {
        double m = 0.0;
        int used = 0;
        if (useStepX) {
            const double pitch = static_cast<double>(W) / static_cast<double>(a.c);
            m += std::abs(std::log(pitch / static_cast<double>(stepX.stepPx)));
            used++;
        }
        if (useStepY) {
            const double pitch = static_cast<double>(H) / static_cast<double>(a.r);
            m += std::abs(std::log(pitch / static_cast<double>(stepY.stepPx)));
            used++;
        }
        return used > 0 ? (m / used) : 0.0;
    };

    auto selectBestFrom = [&](const std::vector<CandScore>& pool) -> CandScore {
        double minErrLocal = pool[0].lineErr;
        for (const auto& a : pool) minErrLocal = std::min(minErrLocal, a.lineErr);
        const double slackLocal = std::max(0.10, minErrLocal * 0.50);

        CandScore bestLocal = pool[0];
        for (const auto& a : pool) {
            if (a.lineErr > minErrLocal + slackLocal) continue;

            if (useStepX || useStepY) {
                const double am = stepMismatch(a);
                const double bm = stepMismatch(bestLocal);
                if (std::abs(am - bm) > 1e-9) {
                    if (am < bm) bestLocal = a;
                    continue;
                }
            }

            const int aCells = a.r * a.c;
            const int bCells = bestLocal.r * bestLocal.c;
            if (aCells != bCells) { if (aCells > bCells) bestLocal = a; continue; }
            if (a.nonEmptyRatio > bestLocal.nonEmptyRatio + 1e-6) { bestLocal = a; continue; }
            if (a.meanPixels > bestLocal.meanPixels + 1e-6) { bestLocal = a; continue; }
            if (a.lineErr < bestLocal.lineErr) bestLocal = a;
        }
        return bestLocal;
    };

    std::vector<CandScore> dense;
    dense.reserve(accepted.size());
    constexpr double kDenseRatio = 0.70;
    constexpr double kMaxComponentsPerCellCoarse = 1.60;
    for (const auto& a : accepted) {
        if (a.nonEmptyRatio < kDenseRatio) continue;
        const int cells = a.r * a.c;
        if (cells <= 0) continue;
        if (spriteComponents > static_cast<int>(std::lround(static_cast<double>(cells) * kMaxComponentsPerCellCoarse))) continue;
        dense.push_back(a);
    }

    CandScore best = dense.empty() ? selectBestFrom(accepted) : selectBestFrom(dense);

    // 最终阈值
    if (best.lineErr >= 0.35) return info;

    info.isSpriteSheet = true;
    info.rows = best.r;
    info.cols = best.c;
    info.cellWidth = static_cast<int>(std::lround(static_cast<double>(W) / best.c));
    info.cellHeight = static_cast<int>(std::lround(static_cast<double>(H) / best.r));

    if (generateSprites) {
        info.sprites.reserve(static_cast<size_t>(best.r) * static_cast<size_t>(best.c));
        for (int r = 0; r < best.r; ++r) {
            const int y0 = roundDivPos(r, H, best.r);
            const int y1 = roundDivPos(r + 1, H, best.r);
            for (int c = 0; c < best.c; ++c) {
                const int x0 = roundDivPos(c, W, best.c);
                const int x1 = roundDivPos(c + 1, W, best.c);
                info.sprites.push_back({x0, y0, std::max(0, x1 - x0), std::max(0, y1 - y0)});
            }
        }
    }

    return info;
}

// 保持不变的辅助函数（旧接口保留；当前 detect() 已使用 ForegroundMask 统一实现）
cv::Mat SpriteDetector::getBinaryMask(const cv::Mat& image) {
    cv::Mat mask;
    if (image.channels() == 4) {
        std::vector<cv::Mat> channels;
        cv::split(image, channels);
        cv::threshold(channels[3], mask, 20, 255, cv::THRESH_BINARY);
        return mask;
    } 
    cv::Mat gray;
    if (image.channels() == 3) cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    else gray = image.clone();

    cv::Mat floodMask = cv::Mat::zeros(gray.rows + 2, gray.cols + 2, CV_8UC1);
    cv::Mat workingImg = image.clone();
    cv::Scalar diff(5, 5, 5); 
    if (image.channels() == 1) diff = cv::Scalar(5);
    int flags = 4 + (255 << 8) + cv::FLOODFILL_MASK_ONLY;

    std::vector<cv::Point> seeds = {
        cv::Point(0, 0), cv::Point(image.cols - 1, 0),
        cv::Point(0, image.rows - 1), cv::Point(image.cols - 1, image.rows - 1)
    };
    for(const auto& seed : seeds) {
        if (floodMask.at<uchar>(seed.y + 1, seed.x + 1) == 0) {
            cv::floodFill(workingImg, floodMask, seed, cv::Scalar(), 0, diff, diff, flags);
        }
    }
    cv::Mat roi = floodMask(cv::Rect(1, 1, image.cols, image.rows));
    cv::bitwise_not(roi, mask);
    return mask;
}

void SpriteDetector::computeProjections(const cv::Mat&, std::vector<int>&, std::vector<int>&) {}
std::optional<std::pair<int, int>> SpriteDetector::analyzeProjection(const std::vector<int>&, int) { return std::nullopt; }
std::vector<SpriteInfo> SpriteDetector::extractSpritesByContours(const cv::Mat&) { return {}; }
