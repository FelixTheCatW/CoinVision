using System.Text.Json;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SkiaSharp;

namespace CoinVision.DateApi.Services;

/// <summary>What the exported detector says about itself.</summary>
public sealed class DetectorMeta
{
    public int CanvasWidth { get; init; } = 512;
    public int CanvasHeight { get; init; } = 320;
    public int Stride { get; init; } = 4;
    public int HeatWidth { get; init; } = 128;
    public int HeatHeight { get; init; } = 80;
    public double Expand { get; init; } = 1.3;
    public double HeatMin { get; init; } = 0.05;
    public int TopK { get; init; } = 3;
}

/// <summary>
/// Port of `src/coinvision/date_detector.py`: the year block predicted outright, as a rotated
/// rectangle, so the candidate list does not depend on the ring sweep happening to cover it.
///
/// The network reads a letterboxed 512x320 canvas and returns seven maps at stride 4 - a centre
/// heatmap, width and height as canvas fractions, the doubled angle as (cos, sin), and a sub-cell
/// offset. The offset is what makes the quad usable: without it the centre is up to one stride out,
/// which on a 1500 px photo is wider than a crop holding a 50 px year block can afford.
/// </summary>
public sealed class DateDetector : IDisposable
{
    private readonly InferenceSession _session;
    private readonly string _inputName;

    public DateDetector(string onnxPath, string metaPath)
    {
        if (!File.Exists(onnxPath))
        {
            throw new FileNotFoundException($"date detector not found: {onnxPath}", onnxPath);
        }

        var options = new Microsoft.ML.OnnxRuntime.SessionOptions
        {
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
        };
        _session = new InferenceSession(onnxPath, options);
        _inputName = _session.InputMetadata.Keys.First();

        using var document = JsonDocument.Parse(File.ReadAllText(metaPath));
        var root = document.RootElement;
        var canvas = root.GetProperty("canvas");
        var heatmap = root.GetProperty("heatmap");
        Meta = new DetectorMeta
        {
            CanvasWidth = canvas[0].GetInt32(),
            CanvasHeight = canvas[1].GetInt32(),
            Stride = root.GetProperty("stride").GetInt32(),
            HeatWidth = heatmap[0].GetInt32(),
            HeatHeight = heatmap[1].GetInt32(),
            Expand = root.TryGetProperty("expand", out var expand) ? expand.GetDouble() : 1.3,
            HeatMin = root.TryGetProperty("heat_min", out var heatMin) ? heatMin.GetDouble() : 0.05,
            TopK = root.TryGetProperty("top_k", out var topK) ? topK.GetInt32() : 3,
        };
    }

    public DetectorMeta Meta { get; }

    /// <summary>Year windows in original-image pixels, strongest first; empty when nothing fires.</summary>
    public IReadOnlyList<Window> Detect(SKBitmap source)
    {
        var meta = Meta;
        var width = source.Width;
        var height = source.Height;
        var scale = Math.Min(meta.CanvasWidth / (double)width, meta.CanvasHeight / (double)height);
        var newWidth = Math.Max(1, (int)Math.Round(width * scale));
        var newHeight = Math.Max(1, (int)Math.Round(height * scale));
        var offsetX = (meta.CanvasWidth - newWidth) / 2;
        var offsetY = (meta.CanvasHeight - newHeight) / 2;

        using var canvas = new SKBitmap(new SKImageInfo(meta.CanvasWidth, meta.CanvasHeight,
                                                        SKColorType.Rgba8888, SKAlphaType.Unpremul));
        using (var paint = new SKCanvas(canvas))
        {
            paint.Clear(SKColors.Black);
            using var image = SKImage.FromBitmap(source);
            // this SkiaSharp build has no cubic filter; linear on a 0.34x downscale is the closest
            // thing to the Python reference's LANCZOS, and the parity check compares the quads
            paint.DrawImage(image, SKRect.Create(offsetX, offsetY, newWidth, newHeight),
                            new SKSamplingOptions(SKFilterMode.Linear, SKMipmapMode.None));
        }

        var tensor = new DenseTensor<float>(
            new[] { 1, 3, meta.CanvasHeight, meta.CanvasWidth });
        var strideBytes = canvas.RowBytes;
        var pixels = canvas.GetPixelSpan();
        for (var y = 0; y < meta.CanvasHeight; y++)
        {
            for (var x = 0; x < meta.CanvasWidth; x++)
            {
                var offset = y * strideBytes + x * 4;
                tensor[0, 0, y, x] = pixels[offset] / 255f;
                tensor[0, 1, y, x] = pixels[offset + 1] / 255f;
                tensor[0, 2, y, x] = pixels[offset + 2] / 255f;
            }
        }

        using var results = _session.Run(new[] { NamedOnnxValue.CreateFromTensor(_inputName, tensor) });
        var maps = results.First().AsTensor<float>();
        return Decode(maps, scale, offsetX, offsetY);
    }

    private IReadOnlyList<Window> Decode(Tensor<float> maps, double scale, int offsetX, int offsetY)
    {
        var meta = Meta;
        var heat = new float[meta.HeatHeight * meta.HeatWidth];
        for (var row = 0; row < meta.HeatHeight; row++)
        {
            for (var col = 0; col < meta.HeatWidth; col++)
            {
                heat[row * meta.HeatWidth + col] = 1f / (1f + MathF.Exp(-maps[0, 0, row, col]));
            }
        }

        var picked = new List<(int Row, int Col)>();
        var working = (float[])heat.Clone();
        for (var pick = 0; pick < meta.TopK; pick++)
        {
            var best = -1;
            var bestValue = 0f;
            for (var index = 0; index < working.Length; index++)
            {
                if (working[index] > bestValue)
                {
                    bestValue = working[index];
                    best = index;
                }
            }

            if (best < 0 || bestValue < meta.HeatMin)
            {
                break;
            }

            var row = best / meta.HeatWidth;
            var col = best % meta.HeatWidth;
            picked.Add((row, col));
            // a year block lights a cluster of cells; keeping the neighbours would repeat the answer
            for (var r = Math.Max(0, row - 2); r <= Math.Min(meta.HeatHeight - 1, row + 2); r++)
            {
                for (var c = Math.Max(0, col - 2); c <= Math.Min(meta.HeatWidth - 1, col + 2); c++)
                {
                    working[r * meta.HeatWidth + c] = 0f;
                }
            }
        }

        var windows = new List<Window>(picked.Count);
        foreach (var (row, col) in picked)
        {
            var cx = (col + 0.5 + maps[0, 5, row, col]) * meta.Stride;
            var cy = (row + 0.5 + maps[0, 6, row, col]) * meta.Stride;
            var boxWidth = maps[0, 1, row, col] * meta.CanvasWidth;
            var boxHeight = maps[0, 2, row, col] * meta.CanvasHeight;
            if (boxWidth <= 1 || boxHeight <= 1)
            {
                continue;
            }

            var angle = 0.5 * Math.Atan2(maps[0, 4, row, col], maps[0, 3, row, col]);
            if (ToWindow(cx, cy, boxWidth * meta.Expand, boxHeight * meta.Expand, angle,
                         scale, offsetX, offsetY) is { } window)
            {
                windows.Add(window);
            }
        }

        return windows;
    }

    /// <summary>Canvas-space rotated rectangle to a Window in image pixels, with its bounding box.</summary>
    private static Window? ToWindow(double cx, double cy, double width, double height, double angle,
                                    double scale, int offsetX, int offsetY)
    {
        var ux = Math.Cos(angle);
        var uy = Math.Sin(angle);
        var vx = -uy;
        var vy = ux;
        var halfWidth = 0.5 * width;
        var halfHeight = 0.5 * height;
        var corners = new (double X, double Y)[4];
        var order = new (int Sx, int Sy)[] { (-1, -1), (1, -1), (1, 1), (-1, 1) };
        for (var index = 0; index < 4; index++)
        {
            var (sx, sy) = order[index];
            var x = cx + sx * halfWidth * ux + sy * halfHeight * vx;
            var y = cy + sx * halfWidth * uy + sy * halfHeight * vy;
            corners[index] = ((x - offsetX) / scale, (y - offsetY) / scale);
        }

        // the pipeline crops through Window, so the quad is converted back to the same parameters
        var centerX = (corners[0].X + corners[1].X + corners[2].X + corners[3].X) / 4.0;
        var centerY = (corners[0].Y + corners[1].Y + corners[2].Y + corners[3].Y) / 4.0;
        var arcSpan = Math.Sqrt(Math.Pow(corners[1].X - corners[0].X, 2) +
                                Math.Pow(corners[1].Y - corners[0].Y, 2));
        var radialSpan = Math.Sqrt(Math.Pow(corners[3].X - corners[0].X, 2) +
                                   Math.Pow(corners[3].Y - corners[0].Y, 2));
        if (arcSpan < 4 || radialSpan < 4)
        {
            return null;
        }

        var angleDeg = Math.Atan2(corners[1].Y - corners[0].Y, corners[1].X - corners[0].X) * 180.0 / Math.PI;
        var minX = corners.Min(corner => corner.X);
        var minY = corners.Min(corner => corner.Y);
        var maxX = corners.Max(corner => corner.X);
        var maxY = corners.Max(corner => corner.Y);
        return new Window(new Circle(centerX, centerY, 0), centerX, centerY, angleDeg,
                          arcSpan, radialSpan, (int)minX, (int)minY, (int)maxX, (int)maxY,
                          Source: "detector");
    }

    public void Dispose() => _session.Dispose();
}
