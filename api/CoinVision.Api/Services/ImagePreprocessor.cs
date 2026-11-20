using SkiaSharp;

namespace CoinVision.Api.Services;

public sealed record PreprocessedImage(float[] Tensor, int Width, int Height);

public interface IImagePreprocessor
{
    PreprocessedImage Process(byte[] content, int width, int height, float[] mean, float[] std);
}

/// <summary>
/// Reproduces the Python pipeline: decode as RGB, resize to 448x224 with a high-quality filter,
/// scale to [0, 1] and normalise with the ImageNet statistics, laid out as CHW.
/// </summary>
public sealed class ImagePreprocessor : IImagePreprocessor
{
    // Bilinear downscaling with mipmaps is the closest Skia analogue of torchvision's
    // Resize(antialias=True): both prefilter the source instead of point-sampling it.
    private static readonly SKSamplingOptions Sampling =
        new(SKFilterMode.Linear, SKMipmapMode.Linear);

    public PreprocessedImage Process(byte[] content, int width, int height, float[] mean, float[] std)
    {
        using var decoded = SKBitmap.Decode(content)
            ?? throw new InvalidOperationException("the image could not be decoded");
        if (decoded.Width <= 0 || decoded.Height <= 0)
        {
            throw new InvalidOperationException("the image has no pixels");
        }

        using var scaled = new SKBitmap(
            new SKImageInfo(width, height, SKColorType.Rgba8888, SKAlphaType.Unpremul));
        if (!decoded.ScalePixels(scaled, Sampling))
        {
            throw new InvalidOperationException("resize failed");
        }

        var pixels = scaled.Pixels;
        if (pixels.Length != width * height)
        {
            throw new InvalidOperationException($"unexpected pixel count {pixels.Length} for {width}x{height}");
        }

        var plane = width * height;
        var tensor = new float[3 * plane];
        for (var index = 0; index < plane; index++)
        {
            var pixel = pixels[index];
            tensor[index] = (pixel.Red / 255f - mean[0]) / std[0];
            tensor[plane + index] = (pixel.Green / 255f - mean[1]) / std[1];
            tensor[2 * plane + index] = (pixel.Blue / 255f - mean[2]) / std[2];
        }

        return new PreprocessedImage(tensor, width, height);
    }
}
