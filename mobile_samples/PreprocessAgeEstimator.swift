import UIKit

private let inputWidth = 128
private let inputHeight = 128
private let channels = 3

private let mean: [Float] = [0.485, 0.456, 0.406]
private let std: [Float] = [0.229, 0.224, 0.225]

/// Converts a UIImage into a Float32 NCHW tensor buffer suitable for ONNX Runtime.
///
/// Steps:
/// 1. Resize to 128x128 (RGB).
/// 2. Convert to float32 in [0, 1].
/// 3. Normalize per channel: (x - mean) / std.
/// 4. Store in NCHW layout: [1, 3, 128, 128].
func preprocessToNchwFloatBuffer(image: UIImage) -> [Float]? {
    guard let cgImage = image.cgImage else { return nil }

    let colorSpace = CGColorSpaceCreateDeviceRGB()
    let bytesPerPixel = 4
    let bytesPerRow = bytesPerPixel * inputWidth
    let bitmapInfo = CGImageAlphaInfo.premultipliedLast.rawValue

    var rawData = [UInt8](repeating: 0, count: inputWidth * inputHeight * bytesPerPixel)
    guard let context = CGContext(
        data: &rawData,
        width: inputWidth,
        height: inputHeight,
        bitsPerComponent: 8,
        bytesPerRow: bytesPerRow,
        space: colorSpace,
        bitmapInfo: bitmapInfo
    ) else {
        return nil
    }

    context.interpolationQuality = .high
    context.draw(cgImage, in: CGRect(x: 0, y: 0, width: inputWidth, height: inputHeight))

    let stride = inputWidth * inputHeight
    var output = [Float](repeating: 0, count: 1 * channels * stride)

    for i in 0..<stride {
        let base = i * bytesPerPixel
        let r = Float(rawData[base]) / 255.0
        let g = Float(rawData[base + 1]) / 255.0
        let b = Float(rawData[base + 2]) / 255.0

        output[i] = (r - mean[0]) / std[0]
        output[stride + i] = (g - mean[1]) / std[1]
        output[2 * stride + i] = (b - mean[2]) / std[2]
    }

    return output
}
