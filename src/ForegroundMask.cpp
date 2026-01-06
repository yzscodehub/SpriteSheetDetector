#include "ForegroundMask.h"

#include <algorithm>

namespace SpriteMask {

cv::Mat makeForegroundMask(const cv::Mat& image, int alphaThreshold, int floodDiff) {
    cv::Mat mask;
    if (image.empty()) return mask;

    // Alpha-based: most robust for spritesheets.
    if (image.channels() == 4) {
        std::vector<cv::Mat> channels;
        cv::split(image, channels);
        cv::threshold(channels[3], mask, alphaThreshold, 255, cv::THRESH_BINARY);
        return mask;
    }

    // Flood-fill background from corners, then invert to get foreground.
    cv::Mat working = image.clone();
    cv::Mat floodMask = cv::Mat::zeros(image.rows + 2, image.cols + 2, CV_8UC1);

    cv::Scalar diff = (image.channels() == 1) ? cv::Scalar(floodDiff) : cv::Scalar(floodDiff, floodDiff, floodDiff);
    int flags = 4 + (255 << 8) + cv::FLOODFILL_MASK_ONLY;

    const std::vector<cv::Point> seeds = {
        {0, 0},
        {image.cols - 1, 0},
        {0, image.rows - 1},
        {image.cols - 1, image.rows - 1},
    };

    for (const auto& seed : seeds) {
        if (seed.x < 0 || seed.y < 0 || seed.x >= image.cols || seed.y >= image.rows) continue;
        if (floodMask.at<uchar>(seed.y + 1, seed.x + 1) == 0) {
            cv::floodFill(working, floodMask, seed, cv::Scalar(), nullptr, diff, diff, flags);
        }
    }

    cv::Mat roi = floodMask(cv::Rect(1, 1, image.cols, image.rows));
    cv::bitwise_not(roi, mask);
    return mask;
}

} // namespace SpriteMask



