#include "GridSpriteDetector.h"
#include "ForegroundMask.h"
#include <cmath>
#include <algorithm>

namespace {

// integral01: cv::integral(binary01, integral01, CV_32S) gives (rows+1)x(cols+1).
static inline int rectSum32S(const cv::Mat& integral01, int x, int y, int w, int h) {
    if (w <= 0 || h <= 0) return 0;
    const int x1 = x;
    const int y1 = y;
    const int x2 = x + w;
    const int y2 = y + h;
    const int A = integral01.at<int>(y1, x1);
    const int B = integral01.at<int>(y1, x2);
    const int C = integral01.at<int>(y2, x1);
    const int D = integral01.at<int>(y2, x2);
    return D - B - C + A;
}

static inline int roundDivPos(int i, int total, int parts) {
    // returns round(i * total / parts) for i>=0, parts>0
    return static_cast<int>(std::lround(static_cast<double>(i) * static_cast<double>(total) / static_cast<double>(parts)));
}

// ----------------------------
// 原理说明（实现细节层）
// ----------------------------
// 这份 .cpp 里的函数实现围绕一个核心想法：
//
//   “规则网格精灵表” = 同一组 rows/cols 在全图范围内均分，并且大多数分割线应穿过背景缝隙(gutter)。
//
// 因此我们的“证据”来自两方面：
// - Line Evidence：所有内部分割线应尽量“空”（前景像素少），且在小范围搜索内能找到更空的位置。
// - Cell Evidence：均分后的格子应呈现规律的占用（密集表：大多数格子非空；稀疏表：允许空格但要更强约束）。
//
// 实现上用到两个常见加速技巧：
// - 积分图 integral01：O(1) 求任意矩形区域的前景像素和，用于快速算线条带/格子像素数；
// - 剪枝：先用更“干净”的腐蚀 maskCuts 做便宜筛选，再结合 raw+cuts 得分做精筛。
//
// 注意：本工程刻意排除了 1xN / Nx1（只认二维网格），并且网格作用域是整图 (0,0,W,H)。

struct StepEstimate {
    int stepPx = 0;      // estimated pitch in pixels
    int support = 0;     // histogram peak count (higher => more reliable)
    int segments = 0;    // number of non-zero segments in projection
};

// Estimate pitch (in px) from 0/1 integral projection gaps along X or Y.
// This is used only as a tie-breaker to disambiguate factorized grids (e.g., 10x10 vs 5x20).
static StepEstimate estimateStepByGapsIntegral(const cv::Mat& integral01, int W, int H, bool isRow) {
    StepEstimate out{};
    std::vector<uint8_t> proj;
    if (isRow) {
        proj.resize(static_cast<size_t>(H), 0);
        for (int y = 0; y < H; ++y) proj[static_cast<size_t>(y)] = rectSum32S(integral01, 0, y, W, 1) > 0 ? 1 : 0;
    } else {
        proj.resize(static_cast<size_t>(W), 0);
        for (int x = 0; x < W; ++x) proj[static_cast<size_t>(x)] = rectSum32S(integral01, x, 0, 1, H) > 0 ? 1 : 0;
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
        const int q = (dist + 2) / 5 * 5; // quantize to 5px
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

struct CellOccupancyStats {
    double nonEmptyRatio = 0.0;   // fraction of cells considered non-empty
    double meanPixels = 0.0;      // mean foreground pixels per cell
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

    // Adaptive area threshold: ignore small speckles/particles by referencing large components (p90).
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

static StepEstimate estimateStepFromComponentCenters(
    const ComponentSet& comps,
    int axisLenPx,
    bool isRow
) {
    StepEstimate out{};
    if (comps.count < 2) return out;

    const double medW = medianInt(comps.widths);
    const double medH = medianInt(comps.heights);
    // Use a relatively wide clustering epsilon so pose variation within a row/col doesn't split into multiple clusters.
    // (Over-splitting here causes superharmonic grids like 9x5 to win over true 5x5.)
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

static int countComponentsAboveArea(const cv::Mat& binary01, int minAreaPx) {
    // binary01 is expected to be CV_8U with values 0/255 (or 0/1).
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
    // Cell Evidence 负责抑制误报：
    // - 如果只靠“线很空”，普通插画/单个角色有时也会碰巧通过；
    // - 规则精灵表往往在格子层面呈现更稳定的占用规律。
    //
    // 这里分两类：
    // - dense：大多数格子都应该有内容（nonEmptyRatio 高）
    // - sparse：允许空格很多，但必须引入额外的约束（组件数、线误差更低、cells 与 components 比例受控）
    // Shared floor: if the average per-cell pixels is extremely low, the grid is almost certainly too fine.
    if (occ.meanPixels < 10.0) return false;

    // Dense "rules-grid" sheets: most cells contain something.
    if (occ.nonEmptyRatio > 0.70) return true;

    // Sparse sheets (many empty cells) support:
    // - require multiple foreground components (avoid single-sprite / illustration false positives)
    // - require sufficiently "clean" grid lines
    // - cap grid size relative to components to avoid choosing superharmonic grids inside padding
    constexpr int kMinSpriteComponents = 8;
    constexpr int kMinCells = 16;              // avoid tiny coarse grids (e.g. 2x2) passing by accident
    constexpr int kMaxCellsPerComponent = 10;  // allow empties but prevent extreme over-segmentation
    constexpr double kMinNonEmptyRatio = 0.05; // allow very sparse (e.g. 1/17 columns filled)
    constexpr double kMaxLineErrSparse = 0.28; // stricter than final okLines threshold

    const int cells = rows * cols;
    if (spriteComponents < kMinSpriteComponents) return false;
    if (cells < kMinCells) return false;
    if (cells > spriteComponents * kMaxCellsPerComponent) return false;
    if (occ.nonEmptyRatio < kMinNonEmptyRatio) return false;
    if (lineErr > kMaxLineErrSparse) return false;
    return true;
}

static CellOccupancyStats computeCellOccupancyStats(
    const cv::Mat& integral01,
    const cv::Rect& bounds,
    int rows,
    int cols
) {
    CellOccupancyStats s{};
    if (rows <= 0 || cols <= 0) return s;

    const int totalCells = rows * cols;
    const int totalPix = rectSum32S(integral01, 0, 0, bounds.width, bounds.height);
    if (totalCells <= 0) return s;
    s.meanPixels = static_cast<double>(totalPix) / static_cast<double>(totalCells);

    // Adaptive threshold:
    // - allow very small sprites (dense sheets) by using a low absolute floor
    // - scale with mean pixels per cell to avoid biasing toward coarser (subharmonic) grids
    const int pixThresh = std::max(6, static_cast<int>(std::lround(s.meanPixels * 0.20)));

    int nonEmpty = 0;
    for (int r = 0; r < rows; ++r) {
        const int y0 = roundDivPos(r, bounds.height, rows);
        const int y1 = roundDivPos(r + 1, bounds.height, rows);
        const int h = std::max(0, y1 - y0);
        for (int c = 0; c < cols; ++c) {
            const int x0 = roundDivPos(c, bounds.width, cols);
            const int x1 = roundDivPos(c + 1, bounds.width, cols);
            const int w = std::max(0, x1 - x0);
            if (w <= 0 || h <= 0) continue;
            const int pix = rectSum32S(integral01, x0, y0, w, h);
            if (pix >= pixThresh) nonEmpty++;
        }
    }

    s.nonEmptyRatio = (totalCells > 0) ? (static_cast<double>(nonEmpty) / static_cast<double>(totalCells)) : 0.0;
    return s;
}

} // namespace

SpriteSheetInfo GridSpriteDetector::detectFixedGrid(const cv::Mat& image, int rows, int cols, bool generateSprites) {
    SpriteSheetInfo info;
    info.isSpriteSheet = false;
    if (image.empty()) return info;
    if (rows <= 0 || cols <= 0) return info;
    // Only accept true 2D grids (a*b). 1xN / Nx1 are treated as "not a sheet" for this project.
    if (rows < 2 || cols < 2) return info;

    cv::Mat mask = SpriteMask::makeForegroundMask(image, /*alphaThreshold=*/20, /*floodDiff=*/5);
    if (mask.empty()) return info;
    if (cv::countNonZero(mask) == 0) return info;

    // IMPORTANT: For rules-based a*b sheets, the grid is defined over the full image area.
    // Cropping to content bounds can destroy alignment when sprites have internal padding.
    const cv::Rect bounds(0, 0, image.cols, image.rows);

    // For line checks, shrink edges slightly to tolerate anti-aliased borders near grid lines.
    cv::Mat maskCuts;
    {
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
        cv::erode(mask, maskCuts, kernel, cv::Point(-1, -1), 1);
    }

    cv::Mat fill01 = (mask(bounds) > 0);
    cv::Mat fillIntegral01;
    cv::integral(fill01, fillIntegral01, CV_32S);

    cv::Mat cuts01 = (maskCuts(bounds) > 0);
    cv::Mat cutsIntegral01;
    cv::integral(cuts01, cutsIntegral01, CV_32S);

    // Estimate how many distinct sprites/parts exist (helps accept sparse sheets without raising false positives).
    const int spriteComponents = countComponentsAboveArea(cuts01, /*minAreaPx=*/32);

    const int minCellPx = 6;
    const int cellWMin = bounds.width / cols;
    const int cellHMin = bounds.height / rows;
    if (cellWMin < minCellPx || cellHMin < minCellPx) return info;

    // Wider search so boundaries can snap onto gutters even when (bounds.width % cols) or padding accumulates drift.
    const int searchRadiusPx = std::clamp(std::min(cellWMin, cellHMin) / 10, 2, 12);
    const double lineErrCuts = checkGridLines(mask, cutsIntegral01, bounds, rows, cols, searchRadiusPx);
    const double lineErrRaw = checkGridLines(mask, fillIntegral01, bounds, rows, cols, searchRadiusPx);
    const double lineErr = 0.65 * lineErrCuts + 0.35 * lineErrRaw;
    const CellOccupancyStats occ = computeCellOccupancyStats(fillIntegral01, bounds, rows, cols);

    // Hard validation (tuned for "规则网格切分" rather than generic segmentation)
    // lineErr includes an extra penalty when many boundaries have no real gutter.
    const bool okLines = (lineErr < 0.35);
    const bool okCells = passCellEvidence(occ, lineErr, rows, cols, spriteComponents);
    const bool okNotTrivial = (rows >= 2 && cols >= 2);
    if (!(okLines && okCells && okNotTrivial)) return info;

    info.isSpriteSheet = true;
    info.rows = rows;
    info.cols = cols;
    info.cellWidth = static_cast<int>(std::lround(static_cast<double>(bounds.width) / cols));
    info.cellHeight = static_cast<int>(std::lround(static_cast<double>(bounds.height) / rows));

    if (generateSprites) {
        info.sprites.reserve(static_cast<size_t>(rows) * static_cast<size_t>(cols));
        for (int r = 0; r < rows; ++r) {
            const int y0 = bounds.y + roundDivPos(r, bounds.height, rows);
            const int y1 = bounds.y + roundDivPos(r + 1, bounds.height, rows);
            for (int c = 0; c < cols; ++c) {
                const int x0 = bounds.x + roundDivPos(c, bounds.width, cols);
                const int x1 = bounds.x + roundDivPos(c + 1, bounds.width, cols);
                info.sprites.push_back({x0, y0, std::max(0, x1 - x0), std::max(0, y1 - y0)});
            }
        }
    }
    return info;
}

SpriteSheetInfo GridSpriteDetector::detect(const cv::Mat& image, bool generateSprites) {
    SpriteSheetInfo info;
    info.isSpriteSheet = false;
    if (image.empty()) return info;

    cv::Mat mask = SpriteMask::makeForegroundMask(image, /*alphaThreshold=*/20, /*floodDiff=*/5);
    if (mask.empty()) return info;
    if (cv::countNonZero(mask) == 0) return info;

    const cv::Rect bounds(0, 0, image.cols, image.rows);

    cv::Mat maskCuts;
    {
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
        cv::erode(mask, maskCuts, kernel, cv::Point(-1, -1), 1);
    }

    cv::Mat fill01 = (mask(bounds) > 0);
    cv::Mat fillIntegral01;
    cv::integral(fill01, fillIntegral01, CV_32S);

    cv::Mat cuts01 = (maskCuts(bounds) > 0);
    cv::Mat cutsIntegral01;
    cv::integral(cuts01, cutsIntegral01, CV_32S);

    // Component count is used to support sparse sheets with many empty cells while keeping false positives low.
    const ComponentSet comps = extractComponents(cuts01, /*minAreaPx=*/32);
    const int spriteComponents = comps.count;
    // Step estimates (tie-breaker only; do not hard-filter candidates).
    // Prefer component-center step (more robust to large intra-cell padding), fallback to projection-gap step.
    const StepEstimate stepX_comp = estimateStepFromComponentCenters(comps, bounds.width, /*isRow=*/false);
    const StepEstimate stepY_comp = estimateStepFromComponentCenters(comps, bounds.height, /*isRow=*/true);
    const StepEstimate stepX_proj = estimateStepByGapsIntegral(fillIntegral01, bounds.width, bounds.height, /*isRow=*/false);
    const StepEstimate stepY_proj = estimateStepByGapsIntegral(fillIntegral01, bounds.width, bounds.height, /*isRow=*/true);
    const StepEstimate stepX = (stepX_comp.support >= stepX_proj.support) ? stepX_comp : stepX_proj;
    const StepEstimate stepY = (stepY_comp.support >= stepY_proj.support) ? stepY_comp : stepY_proj;

    struct Candidate {
        int r = 0;
        int c = 0;
        double lineErr = 1.0;
        double nonEmptyRatio = 0.0;
        double meanPixels = 0.0;
    };
    Candidate best{};

    const int minCellPx = 6;
    const int maxRows = std::min(256, std::max(1, bounds.height / minCellPx));
    const int maxCols = std::min(256, std::max(1, bounds.width / minCellPx));

    std::vector<Candidate> accepted;
    accepted.reserve(static_cast<size_t>(maxRows) * static_cast<size_t>(maxCols) / 4);

    for (int r = 1; r <= maxRows; ++r) {
        for (int c = 1; c <= maxCols; ++c) {
            // Only accept true 2D grids (a*b). 1xN / Nx1 are treated as "not a sheet".
            if (r < 2 || c < 2) continue;

            // If the image is sparse, cap grid size relative to components to avoid superharmonics.
            if (spriteComponents >= 8) {
                constexpr int kMaxCellsPerComponent = 10;
                const int cells = r * c;
                if (cells > spriteComponents * kMaxCellsPerComponent) continue;
            }

            const int cellWMin = bounds.width / c;
            const int cellHMin = bounds.height / r;
            if (cellWMin < minCellPx || cellHMin < minCellPx) continue;

            const int searchRadiusPx = std::clamp(std::min(cellWMin, cellHMin) / 10, 2, 12);
            const double lineErrCuts = checkGridLines(mask, cutsIntegral01, bounds, r, c, searchRadiusPx);
            if (lineErrCuts > 0.55) continue; // prune early (cheap pass, cuts-mask)
            const double lineErrRaw = checkGridLines(mask, fillIntegral01, bounds, r, c, searchRadiusPx);
            const double lineErr = 0.65 * lineErrCuts + 0.35 * lineErrRaw;
            if (lineErr > 0.65) continue; // prune early (combined)

            const CellOccupancyStats occ = computeCellOccupancyStats(fillIntegral01, bounds, r, c);
            if (!passCellEvidence(occ, lineErr, r, c, spriteComponents)) continue;

            accepted.push_back(Candidate{r, c, lineErr, occ.nonEmptyRatio, occ.meanPixels});
        }
    }

    if (accepted.empty()) return info;

    // Selection strategy:
    // 1) Minimize lineErr (grid-line consistency)
    // 2) Among near-min lineErr solutions, use (optional) step estimates as a tie-breaker to avoid
    //    factorization/superharmonic ambiguity (e.g., 10x10 should beat 5x20 when both have similar lineErr).
    // 3) Then pick the densest grid (avoid subharmonic coarse grids like 2x2 beating 4x6)
    // 4) Tie-breakers: higher nonEmptyRatio, then higher meanPixels, then lower lineErr
    auto stepReliable = [](const StepEstimate& e) {
        return (e.stepPx > 0) && (e.support >= 3) && (e.segments >= 4);
    };
    const bool useStepX = stepReliable(stepX);
    const bool useStepY = stepReliable(stepY);

    auto stepMismatch = [&](const Candidate& a) -> double {
        double m = 0.0;
        int used = 0;
        if (useStepX) {
            const double pitch = static_cast<double>(bounds.width) / static_cast<double>(a.c);
            m += std::abs(std::log(pitch / static_cast<double>(stepX.stepPx)));
            used++;
        }
        if (useStepY) {
            const double pitch = static_cast<double>(bounds.height) / static_cast<double>(a.r);
            m += std::abs(std::log(pitch / static_cast<double>(stepY.stepPx)));
            used++;
        }
        return used > 0 ? (m / used) : 0.0;
    };

    auto selectBestFrom = [&](const std::vector<Candidate>& pool) -> Candidate {
        double minErr = pool[0].lineErr;
        for (const auto& a : pool) minErr = std::min(minErr, a.lineErr);
        const double slack = std::max(0.10, minErr * 0.50);

        Candidate bestLocal = pool[0];
        for (const auto& a : pool) {
            if (a.lineErr > minErr + slack) continue;

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
            if (std::abs(a.nonEmptyRatio - bestLocal.nonEmptyRatio) > 1e-6) {
                if (a.nonEmptyRatio > bestLocal.nonEmptyRatio) bestLocal = a;
                continue;
            }
            if (std::abs(a.meanPixels - bestLocal.meanPixels) > 1e-6) {
                if (a.meanPixels > bestLocal.meanPixels) bestLocal = a;
                continue;
            }
            if (a.lineErr < bestLocal.lineErr) bestLocal = a;
        }
        return bestLocal;
    };

    // Prefer dense candidates when they exist (prevents superharmonic sparse grids like 9x5 winning over true 5x5),
    // but reject "too coarse" dense candidates where one cell would contain multiple sprites/components.
    std::vector<Candidate> dense;
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

    best = dense.empty() ? selectBestFrom(accepted) : selectBestFrom(dense);

    if (best.r <= 0 || best.c <= 0) return info;

    info.isSpriteSheet = true;
    info.rows = best.r;
    info.cols = best.c;
    info.cellWidth = static_cast<int>(std::lround(static_cast<double>(bounds.width) / best.c));
    info.cellHeight = static_cast<int>(std::lround(static_cast<double>(bounds.height) / best.r));

    if (generateSprites) {
        info.sprites.reserve(static_cast<size_t>(best.r) * static_cast<size_t>(best.c));
        for (int r = 0; r < best.r; ++r) {
            const int y0 = bounds.y + roundDivPos(r, bounds.height, best.r);
            const int y1 = bounds.y + roundDivPos(r + 1, bounds.height, best.r);
            for (int c = 0; c < best.c; ++c) {
                const int x0 = bounds.x + roundDivPos(c, bounds.width, best.c);
                const int x1 = bounds.x + roundDivPos(c + 1, bounds.width, best.c);
                info.sprites.push_back({x0, y0, std::max(0, x1 - x0), std::max(0, y1 - y0)});
            }
        }
    }
    return info;
}

cv::Rect GridSpriteDetector::getContentBounds(const cv::Mat& mask) {
    // 寻找非零像素的包围盒
    // cv::findNonZero 很快，或者直接用 boundingRect
    std::vector<cv::Point> points;
    cv::findNonZero(mask, points);
    return cv::boundingRect(points);
}

double GridSpriteDetector::checkGridLines(const cv::Mat&, const cv::Mat& integral01, const cv::Rect& bounds, int rows, int cols, int searchRadiusPx) {
    if (rows <= 1 && cols <= 1) return 0.0;

    long long cutPixels = 0;
    int totalCheckPixels = 0;

    const int cellWMin = bounds.width / cols;
    const int cellHMin = bounds.height / rows;
    (void)cellWMin;
    (void)cellHMin;
    const int radius = std::clamp(searchRadiusPx, 0, 24);

    // If most boundaries can't find a sufficiently "empty" gutter, it's not a clean grid sheet.
    // This reduces false positives on single sprites / illustrations.
    constexpr double kMaxLineFill = 0.10; // 10% tolerance (after erosion) for dense/tight sheets
    int badLines = 0;
    int totalLines = 0;
    long long sumAbsOffset = 0;
    int maxAbsOffset = 0;

    // 检查横线
    for (int r = 1; r < rows; ++r) {
        const int yRel = roundDivPos(r, bounds.height, rows);
        const int y = yRel; // relative to bounds ROI (mask01/integral space)
        int minLinePixels = bounds.width;
        int bestDy = 0;
        for (int dy = -radius; dy <= radius; ++dy) {
            const int yy = y + dy;
            if (yy < 0 || yy >= bounds.height) continue;
            const int p = rectSum32S(integral01, 0, yy, bounds.width, 1);
            if (p < minLinePixels) {
                minLinePixels = p;
                bestDy = dy;
            }
        }
        cutPixels += minLinePixels;
        totalCheckPixels += bounds.width;

        totalLines++;
        if (minLinePixels > static_cast<int>(std::lround(bounds.width * kMaxLineFill))) badLines++;
        sumAbsOffset += std::llabs(static_cast<long long>(bestDy));
        maxAbsOffset = std::max(maxAbsOffset, std::abs(bestDy));
    }

    // 检查竖线
    for (int c = 1; c < cols; ++c) {
        const int xRel = roundDivPos(c, bounds.width, cols);
        const int x = xRel;
        int minLinePixels = bounds.height;
        int bestDx = 0;
        for (int dx = -radius; dx <= radius; ++dx) {
            const int xx = x + dx;
            if (xx < 0 || xx >= bounds.width) continue;
            const int p = rectSum32S(integral01, xx, 0, 1, bounds.height);
            if (p < minLinePixels) {
                minLinePixels = p;
                bestDx = dx;
            }
        }
        cutPixels += minLinePixels;
        totalCheckPixels += bounds.height;

        totalLines++;
        if (minLinePixels > static_cast<int>(std::lround(bounds.height * kMaxLineFill))) badLines++;
        sumAbsOffset += std::llabs(static_cast<long long>(bestDx));
        maxAbsOffset = std::max(maxAbsOffset, std::abs(bestDx));
    }

    if (totalCheckPixels == 0) return 0.0;
    const double baseErr = (double)cutPixels / totalCheckPixels;
    if (totalLines > 0) {
        const double badRatio = static_cast<double>(badLines) / static_cast<double>(totalLines);
        const double avgAbsOffset = static_cast<double>(sumAbsOffset) / static_cast<double>(totalLines);
        // Penalize when boundaries need large local shifts to find gutters => likely not a truly regular grid.
        const double offsetPenalty =
            (radius > 0) ? (0.03 * (avgAbsOffset / radius) + 0.03 * (static_cast<double>(maxAbsOffset) / radius)) : 0.0;
        return baseErr + 0.12 * badRatio + offsetPenalty;
    }
    return baseErr;
}

double GridSpriteDetector::computeFilledCellRatio(const cv::Mat& integral01, const cv::Rect& bounds, int rows, int cols, double perCellFillThreshold) {
    if (rows <= 0 || cols <= 0) return 0.0;
    const int totalCells = rows * cols;
    int filled = 0;

    for (int r = 0; r < rows; ++r) {
        const int y0 = roundDivPos(r, bounds.height, rows);
        const int y1 = roundDivPos(r + 1, bounds.height, rows);
        const int h = std::max(0, y1 - y0);
        for (int c = 0; c < cols; ++c) {
            const int x0 = roundDivPos(c, bounds.width, cols);
            const int x1 = roundDivPos(c + 1, bounds.width, cols);
            const int w = std::max(0, x1 - x0);
            const int area = w * h;
            if (area <= 0) continue;
            const int pix = rectSum32S(integral01, x0, y0, w, h);
            const double ratio = static_cast<double>(pix) / static_cast<double>(area);
            if (ratio >= perCellFillThreshold) filled++;
        }
    }
    return totalCells > 0 ? (static_cast<double>(filled) / static_cast<double>(totalCells)) : 0.0;
}

