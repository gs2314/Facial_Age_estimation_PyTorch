package com.example.ageestimation

import android.graphics.Bitmap
import java.nio.ByteBuffer
import java.nio.ByteOrder

private const val INPUT_WIDTH = 128
private const val INPUT_HEIGHT = 128
private const val CHANNELS = 3

private val MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
private val STD = floatArrayOf(0.229f, 0.224f, 0.225f)

/**
 * Converts a Bitmap into a Float32 NCHW tensor buffer suitable for ONNX Runtime.
 *
 * Steps:
 * 1. Resize to 128x128 (RGB).
 * 2. Convert to float32 in [0, 1].
 * 3. Normalize per channel: (x - mean) / std.
 * 4. Store in NCHW layout: [1, 3, 128, 128].
 */
fun preprocessToNchwFloatBuffer(bitmap: Bitmap): ByteBuffer {
    val resized = Bitmap.createScaledBitmap(bitmap, INPUT_WIDTH, INPUT_HEIGHT, true)
    val buffer = ByteBuffer.allocateDirect(4 * 1 * CHANNELS * INPUT_HEIGHT * INPUT_WIDTH)
    buffer.order(ByteOrder.nativeOrder())

    val pixels = IntArray(INPUT_WIDTH * INPUT_HEIGHT)
    resized.getPixels(pixels, 0, INPUT_WIDTH, 0, 0, INPUT_WIDTH, INPUT_HEIGHT)

    val stride = INPUT_HEIGHT * INPUT_WIDTH
    for (i in pixels.indices) {
        val pixel = pixels[i]
        val r = ((pixel shr 16) and 0xFF) / 255.0f
        val g = ((pixel shr 8) and 0xFF) / 255.0f
        val b = (pixel and 0xFF) / 255.0f

        val base = i
        buffer.putFloat(base * 4, (r - MEAN[0]) / STD[0])
        buffer.putFloat((stride + base) * 4, (g - MEAN[1]) / STD[1])
        buffer.putFloat((2 * stride + base) * 4, (b - MEAN[2]) / STD[2])
    }

    return buffer
}
