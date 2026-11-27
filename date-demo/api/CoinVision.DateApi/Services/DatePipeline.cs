using System.Diagnostics;
using CoinVision.DateApi.Models;
using SkiaSharp;

namespace CoinVision.DateApi.Services;

/// <summary>
/// The whole demo in five stages: decode, find coins, sweep the rim, score every window, answer.
/// The answer is only "is there a readable date" and the crop the model picked - there is no year
/// classifier here on purpose, the whole-frame one belongs to the previous experiment.
/// </summary>
public sealed class DatePipeline(Localizer localizer, WindowScorer scorer, DateDetector detector,
                                 ILogger<DatePipeline> logger)
{
    /// <summary>Read from the model's metadata: the cut belongs to the score scale that produced it.</summary>
    public double Threshold => scorer.Meta.Threshold;
    private const int PreviewMax = 1000;

    public DetectResponse Run(byte[] content, string fileName, int? forcedWindow = null)
    {
        var total = Stopwatch.StartNew();
        using var decoded = SKBitmap.Decode(content);
        var decodeMs = total.Elapsed.TotalMilliseconds;
        if (decoded is null || decoded.Width <= 0 || decoded.Height <= 0)
        {
            return new DetectResponse(false, "the image could not be decoded", fileName, content.Length,
                                      null, null, false, 0, Threshold, null, null,
                                      scorer.Width, scorer.Height, Array.Empty<CoinDto>(), 0,
                                      Array.Empty<WindowDto>(), null);
        }

        using var bitmap = EnsureRgba(decoded);
        var width = bitmap.Width;
        var height = bitmap.Height;

        var watch = Stopwatch.StartNew();
        var gray = ToGray(bitmap);
        var circles = localizer.DetectCoins(gray, width, height, maxCoins: 2);
        var detectMs = watch.Elapsed.TotalMilliseconds;

        watch.Restart();
        var windows = new List<Window>();
        foreach (var circle in circles)
        {
            windows.AddRange(localizer.RingWindows(circle, scorer.Meta.ArcDegrees,
                                                   scorer.Meta.StepDegrees, scorer.Meta.Bands));
        }

        // The ring sweep is blind, so it misses the year on about one photo in twelve - mostly on
        // photos where the detected circle is not the coin. The detector proposes the block
        // outright, and both kinds of candidate then compete in one ranking.
        var ringWindows = windows.Count;
        windows.AddRange(detector.Detect(bitmap));
        var detectorWindows = windows.Count - ringWindows;

        var sweepMs = watch.Elapsed.TotalMilliseconds;

        watch.Restart();
        var crops = new List<SKBitmap>(windows.Count);
        try
        {
            foreach (var window in windows)
            {
                crops.Add(Localizer.CropWindow(bitmap, window, scorer.Width, scorer.Height));
            }

            var cropMs = watch.Elapsed.TotalMilliseconds;

            watch.Restart();
            var scores = scorer.Score(crops);
            var scoreMs = watch.Elapsed.TotalMilliseconds;

            var best = -1;
            var bestScore = 0f;
            for (var index = 0; index < scores.Length; index++)
            {
                if (scores[index] > bestScore)
                {
                    bestScore = scores[index];
                    best = index;
                }
            }

            if (forcedWindow is >= 0 && forcedWindow < windows.Count)
            {
                best = forcedWindow.Value;
                // Score this one alone: a disagreement between the batch and the single-crop path
                // is the fastest way to tell a tensor bug from a crop bug.
                bestScore = scorer.Score(new[] { crops[best] })[0];
            }

            var top = Enumerable.Range(0, windows.Count)
                .OrderByDescending(index => scores[index])
                .Take(5)
                .Select(index => new WindowDto(
                    index,
                    windows[index].CenterX,
                    windows[index].CenterY,
                    windows[index].AngleDeg,
                    windows[index].ArcSpan,
                    windows[index].RadialSpan,
                    windows[index].X0,
                    windows[index].Y0,
                    windows[index].X1,
                    windows[index].Y1,
                    scores[index]))
                .ToList();

            var cropPng = best >= 0 ? ToBase64(crops[best]) : null;
            var previewPng = ToBase64(BuildPreview(bitmap, circles, best >= 0 ? windows[best] : null));
            var hasDate = bestScore >= Threshold;

            logger.LogInformation(
                "{File} {Width}x{Height}: {Coins} coin(s), {Windows} window(s) " +
                "({Ring} ring + {Detected} detector), best {Score:F3} from {Source} -> {HasDate}",
                fileName, width, height, circles.Count, windows.Count, ringWindows,
                detectorWindows, bestScore, best >= 0 ? windows[best].Source : "-", hasDate);

            total.Stop();
            return new DetectResponse(
                true, null, fileName, content.Length, width, height,
                hasDate, Math.Round(bestScore, 6), Threshold,
                cropPng, previewPng, scorer.Width, scorer.Height,
                circles.Select(circle => new CoinDto(Math.Round(circle.Cx, 1), Math.Round(circle.Cy, 1),
                                                     Math.Round(circle.Radius, 1))).ToList(),
                windows.Count, top,
                new TimingDto(Math.Round(decodeMs, 2), Math.Round(detectMs, 2), Math.Round(sweepMs, 2),
                              Math.Round(cropMs, 2), Math.Round(scoreMs, 2),
                              Math.Round(total.Elapsed.TotalMilliseconds, 2)));
        }
        finally
        {
            foreach (var crop in crops)
            {
                crop.Dispose();
            }
        }
    }

    private static float[] ToGray(SKBitmap bitmap)
    {
        var width = bitmap.Width;
        var height = bitmap.Height;
        var stride = bitmap.RowBytes;
        var pixels = bitmap.GetPixelSpan();
        var gray = new float[width * height];
        for (var y = 0; y < height; y++)
        {
            for (var x = 0; x < width; x++)
            {
                var offset = y * stride + x * 4;
                var luma = (pixels[offset] * 299 + pixels[offset + 1] * 587 + pixels[offset + 2] * 114) / 1000;
                gray[y * width + x] = luma / 255f;
            }
        }

        return gray;
    }

    private static SKBitmap EnsureRgba(SKBitmap source)
    {
        if (source.ColorType == SKColorType.Rgba8888 && source.AlphaType == SKAlphaType.Unpremul)
        {
            return source.Copy();
        }

        var copy = new SKBitmap(new SKImageInfo(source.Width, source.Height, SKColorType.Rgba8888,
                                                SKAlphaType.Unpremul));
        using var canvas = new SKCanvas(copy);
        canvas.Clear(SKColors.Black);
        canvas.DrawBitmap(source, 0, 0);
        return copy;
    }

    private static SKBitmap BuildPreview(SKBitmap source, IReadOnlyList<Circle> circles, Window? chosen)
    {
        var scale = Math.Min(1.0, PreviewMax / (double)Math.Max(source.Width, source.Height));
        var width = Math.Max((int)(source.Width * scale), 8);
        var height = Math.Max((int)(source.Height * scale), 8);
        var preview = new SKBitmap(new SKImageInfo(width, height, SKColorType.Rgba8888,
                                                   SKAlphaType.Unpremul));
        using (var canvas = new SKCanvas(preview))
        {
            canvas.Clear(SKColors.Black);
            using var image = SKImage.FromBitmap(source);
            canvas.DrawImage(image, SKRect.Create(width, height),
                             new SKSamplingOptions(SKFilterMode.Linear, SKMipmapMode.Linear));

            using var circlePaint = new SKPaint
            {
                Style = SKPaintStyle.Stroke,
                StrokeWidth = 2,
                Color = new SKColor(0, 160, 255),
                IsAntialias = true,
            };
            foreach (var circle in circles)
            {
                canvas.DrawCircle((float)(circle.Cx * scale), (float)(circle.Cy * scale),
                                  (float)(circle.Radius * scale), circlePaint);
            }

            if (chosen is not null)
            {
                using var windowPaint = new SKPaint
                {
                    Style = SKPaintStyle.Stroke,
                    StrokeWidth = 3,
                    Color = new SKColor(255, 70, 220),
                    IsAntialias = true,
                };
                using var path = new SKPath();
                var corners = WindowCorners(chosen);
                path.MoveTo((float)(corners[0].X * scale), (float)(corners[0].Y * scale));
                for (var index = 1; index < corners.Count; index++)
                {
                    path.LineTo((float)(corners[index].X * scale), (float)(corners[index].Y * scale));
                }

                path.Close();
                canvas.DrawPath(path, windowPaint);
            }
        }

        return preview;
    }

    /// <summary>The four corners of the window's rotated rectangle, not its bounding box.</summary>
    public static IReadOnlyList<(double X, double Y)> WindowCorners(Window window)
    {
        var alpha = -window.AngleDeg * Math.PI / 180.0;
        var cosA = Math.Cos(alpha);
        var sinA = Math.Sin(alpha);
        var ux = cosA;
        var uy = -sinA;
        var vx = sinA;
        var vy = cosA;
        var halfArc = window.ArcSpan / 2.0;
        var halfRadial = window.RadialSpan / 2.0;
        return new[]
        {
            (window.CenterX - halfArc * ux - halfRadial * vx, window.CenterY - halfArc * uy - halfRadial * vy),
            (window.CenterX + halfArc * ux - halfRadial * vx, window.CenterY + halfArc * uy - halfRadial * vy),
            (window.CenterX + halfArc * ux + halfRadial * vx, window.CenterY + halfArc * uy + halfRadial * vy),
            (window.CenterX - halfArc * ux + halfRadial * vx, window.CenterY - halfArc * uy + halfRadial * vy),
        };
    }

    private static string ToBase64(SKBitmap bitmap)
    {
        using var image = SKImage.FromBitmap(bitmap);
        using var data = image.Encode(SKEncodedImageFormat.Png, 100);
        return Convert.ToBase64String(data.ToArray());
    }
}
