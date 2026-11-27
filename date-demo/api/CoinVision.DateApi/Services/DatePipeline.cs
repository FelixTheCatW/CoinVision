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

            var topIndices = Enumerable.Range(0, windows.Count)
                .OrderByDescending(index => scores[index])
                .Take(5)
                .ToList();
            var top = topIndices
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
                    scores[index],
                    windows[index].Source))
                .ToList();

            var cropPng = best >= 0 ? ToBase64(crops[best]) : null;
            var previewPng = ToBase64(BuildPreview(bitmap, circles, windows, scores, best, topIndices));
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

    /// <summary>
    /// What the pipeline saw, drawn over the photo: the coins, the rim sweep as a score-coloured
    /// grid, the detector's quads, and the five windows the ranker liked most. Without this the
    /// answer is a number and a crop; with it the sweep and the ranking are visible.
    /// </summary>
    private static SKBitmap BuildPreview(SKBitmap source, IReadOnlyList<Circle> circles,
                                         IReadOnlyList<Window> windows, float[] scores,
                                         int best, IReadOnlyList<int> topIndices)
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

            using var typeface = SKTypeface.FromFamilyName("Segoe UI") ?? SKTypeface.Default;
            using var small = new SKFont(typeface, 13);
            using var text = new SKPaint { Color = SKColors.White, IsAntialias = true };

            // The rim sweep, one dot per ring window: red where the ranker thought it saw the year.
            // Seeing the dots is the point - the sweep is blind, and the ridge of red is it guessing.
            using var dot = new SKPaint { IsAntialias = true };
            for (var index = 0; index < windows.Count; index++)
            {
                if (windows[index].Source != "ring")
                {
                    continue;
                }

                dot.Color = ScoreColor(scores[index]);
                canvas.DrawCircle((float)(windows[index].CenterX * scale),
                                  (float)(windows[index].CenterY * scale),
                                  ScoreRadius(scores[index]), dot);
            }

            // The detector's quads: few, tight, and the only candidates that are not a sector of a
            // ring. Amber and dashed so they read as a different kind of proposal.
            using var detectorPaint = new SKPaint
            {
                Style = SKPaintStyle.Stroke,
                StrokeWidth = 2,
                Color = new SKColor(255, 185, 0, 235),
                IsAntialias = true,
                PathEffect = SKPathEffect.CreateDash(new[] { 9f, 6f }, 0),
            };
            foreach (var window in windows)
            {
                if (window.Source != "detector")
                {
                    continue;
                }

                DrawQuad(canvas, window, scale, detectorPaint);
            }

            // The top five: the winner thick and solid, the rest thin and dashed, and a numbered
            // badge on each so the list in the corner can be matched to a place on the coin.
            var placed = new List<SKPoint>();
            for (var rank = 0; rank < topIndices.Count; rank++)
            {
                var index = topIndices[rank];
                var isBest = index == best;
                using var paint = new SKPaint
                {
                    Style = SKPaintStyle.Stroke,
                    StrokeWidth = isBest ? 4.5f : 1.8f,
                    Color = isBest ? new SKColor(255, 70, 220) : new SKColor(255, 70, 220, 165),
                    IsAntialias = true,
                    PathEffect = isBest ? null : SKPathEffect.CreateDash(new[] { 7f, 5f }, 0),
                };
                DrawQuad(canvas, windows[index], scale, paint);

                var window = windows[index];
                // The badge goes on the quad's own corner, not its centre: five windows around one
                // date would otherwise stack their badges on the very digits they are pointing at.
                // The first corner inside the frame wins, so no badge ends up off-canvas.
                var cx = 0f;
                var cy = 0f;
                foreach (var corner in WindowCorners(window))
                {
                    var px = (float)(corner.X * scale);
                    var py = (float)(corner.Y * scale);
                    if (px >= 12 && py >= 12 && px <= width - 12 && py <= height - 12)
                    {
                        cx = px;
                        cy = py;
                        break;
                    }
                }

                if (cx == 0f && cy == 0f)
                {
                    cx = Math.Clamp((float)(window.CenterX * scale), 12f, width - 12f);
                    cy = Math.Clamp((float)(window.CenterY * scale), 12f, height - 12f);
                }

                // Three detector quads over one date put their badges in the same pixel; push the
                // later ones down so all five numbers stay readable.
                while (placed.Any(point => Math.Abs(point.X - cx) < 15 && Math.Abs(point.Y - cy) < 15)
                       && cy > 18f)
                {
                    cy -= 18f;
                }

                cy = Math.Clamp(cy, 14f, height - 14f);

                placed.Add(new SKPoint(cx, cy));
                using var badge = new SKPaint
                {
                    IsAntialias = true,
                    Color = isBest ? new SKColor(255, 70, 220) : new SKColor(40, 0, 38, 225),
                };
                using var badgeEdge = new SKPaint
                {
                    Style = SKPaintStyle.Stroke,
                    StrokeWidth = 1.5f,
                    Color = new SKColor(255, 70, 220),
                    IsAntialias = true,
                };
                var radius = isBest ? 11f : 9f;
                canvas.DrawCircle(cx, cy, radius, badge);
                canvas.DrawCircle(cx, cy, radius, badgeEdge);
                using var number = new SKPaint { Color = SKColors.White, IsAntialias = true };
                var digits = (rank + 1).ToString();
                canvas.DrawText(digits, cx - small.MeasureText(digits) / 2f, cy + 5f, small, number);
            }

            // The coins themselves, drawn last so the outline stays on top.
            using var circlePaint = new SKPaint
            {
                Style = SKPaintStyle.Stroke,
                StrokeWidth = 3,
                Color = new SKColor(0, 200, 255),
                IsAntialias = true,
            };
            foreach (var circle in circles)
            {
                canvas.DrawCircle((float)(circle.Cx * scale), (float)(circle.Cy * scale),
                                  (float)(circle.Radius * scale), circlePaint);
            }

            DrawLegend(canvas, width, height, small, text);
            DrawTopList(canvas, width, windows, scores, topIndices, best, small);
        }

        return preview;
    }

    /// <summary>
    /// Score as a colour, in steps rather than a ramp: the ranker's scores are bimodal (34 of 63
    /// windows under 0.1, 8 over 0.99), so a continuous scale paints the whole rim one colour.
    /// </summary>
    private static SKColor ScoreColor(float score) => score switch
    {
        >= 0.99f => new SKColor(255, 45, 45, 235),
        >= 0.50f => new SKColor(255, 140, 30, 225),
        >= 0.10f => new SKColor(240, 215, 60, 220),
        >= 0.01f => new SKColor(90, 150, 255, 210),
        _ => new SKColor(150, 160, 190, 120),
    };

    private static float ScoreRadius(float score) => score switch
    {
        >= 0.99f => 5.5f,
        >= 0.50f => 4.5f,
        >= 0.10f => 3.5f,
        _ => 2.5f,
    };

    /// <summary>
    /// The five windows the ranker liked, as a list in the corner: the numbers are badges on the
    /// coin, and putting their scores here keeps the labels from stacking on top of each other when
    /// neighbouring windows sit two millimetres apart.
    /// </summary>
    private static void DrawTopList(SKCanvas canvas, int width, IReadOnlyList<Window> windows,
                                    float[] scores, IReadOnlyList<int> topIndices, int best,
                                    SKFont font)
    {
        var rows = topIndices.Count;
        var boxWidth = 196f;
        var boxHeight = 17f * rows + 26f;
        var left = width - boxWidth - 10f;
        var top = 10f;
        using var background = new SKPaint { Color = new SKColor(0, 0, 0, 185), IsAntialias = true };
        canvas.DrawRoundRect(new SKRect(left, top, left + boxWidth, top + boxHeight), 6, 6, background);

        using var header = new SKPaint { Color = new SKColor(255, 170, 235), IsAntialias = true };
        canvas.DrawText("что выбрал ранкер", left + 9f, top + 16f, font, header);

        for (var rank = 0; rank < rows; rank++)
        {
            var index = topIndices[rank];
            var y = top + 32f + rank * 17f;
            var isBest = index == best;
            using var chip = new SKPaint
            {
                Style = SKPaintStyle.Fill,
                Color = isBest ? new SKColor(255, 70, 220) : new SKColor(255, 70, 220, 90),
                IsAntialias = true,
            };
            canvas.DrawCircle(left + 14f, y - 4f, isBest ? 5.5f : 4.5f, chip);

            using var row = new SKPaint { Color = isBest ? SKColors.White : new SKColor(220, 215, 230), IsAntialias = true };
            var origin = windows[index].Source == "detector" ? "дет" : "кол";
            var size = $"{windows[index].ArcSpan:F0}x{windows[index].RadialSpan:F0}";
            canvas.DrawText($"{rank + 1}  {scores[index]:F3}  {origin}  {size}",
                            left + 24f, y, font, row);
        }
    }

    /// <summary>A caption on a dark pill, because white text on a bright coin disappears.</summary>
    private static void DrawLabel(SKCanvas canvas, string value, float x, float y, SKFont font,
                                  bool strong, SKColor? tint = null)
    {
        var width = font.MeasureText(value);
        using var background = new SKPaint
        {
            Color = strong ? new SKColor(60, 0, 55, 225) : new SKColor(0, 0, 0, 175),
            IsAntialias = true,
        };
        using var stroke = new SKPaint
        {
            Style = SKPaintStyle.Stroke,
            StrokeWidth = strong ? 1.5f : 1f,
            Color = tint ?? new SKColor(255, 70, 220, strong ? (byte)235 : (byte)140),
            IsAntialias = true,
        };
        var rect = new SKRect(x - 4, y - 13, x + width + 4, y + 4);
        canvas.DrawRoundRect(rect, 4, 4, background);
        canvas.DrawRoundRect(rect, 4, 4, stroke);

        using var text = new SKPaint { Color = tint ?? SKColors.White, IsAntialias = true };
        canvas.DrawText(value, x, y, font, text);
    }

    private static void DrawQuad(SKCanvas canvas, Window window, double scale, SKPaint paint)
    {
        using var path = new SKPath();
        var corners = WindowCorners(window);
        path.MoveTo((float)(corners[0].X * scale), (float)(corners[0].Y * scale));
        for (var index = 1; index < corners.Count; index++)
        {
            path.LineTo((float)(corners[index].X * scale), (float)(corners[index].Y * scale));
        }

        path.Close();
        canvas.DrawPath(path, paint);
    }

    private static void DrawLegend(SKCanvas canvas, int width, int height, SKFont font, SKPaint text)
    {
        string[] rows =
        {
            "круг монеты",
            "окна: красное = скор выше",
            "квад детектора",
            "топ-5, толстый = выбор",
        };
        var boxWidth = 210f;
        var boxHeight = 16f * rows.Length + 14f;
        var left = 10f;
        var top = height - boxHeight - 10f;
        using var background = new SKPaint
        {
            Color = new SKColor(0, 0, 0, 165),
            IsAntialias = true,
        };
        canvas.DrawRoundRect(new SKRect(left, top, left + boxWidth, top + boxHeight), 6, 6, background);

        using var swatch = new SKPaint { IsAntialias = true, StrokeWidth = 3 };
        for (var row = 0; row < rows.Length; row++)
        {
            var y = top + 16f + row * 16f;
            switch (row)
            {
                case 0:
                    swatch.Style = SKPaintStyle.Stroke;
                    swatch.Color = new SKColor(0, 170, 255);
                    canvas.DrawCircle(left + 12f, y - 4f, 5f, swatch);
                    break;
                case 1:
                    swatch.Style = SKPaintStyle.Fill;
                    swatch.Color = ScoreColor(0.995f);
                    canvas.DrawCircle(left + 8f, y - 4f, 4f, swatch);
                    swatch.Color = ScoreColor(0.2f);
                    canvas.DrawCircle(left + 17f, y - 4f, 4f, swatch);
                    swatch.Color = ScoreColor(0.005f);
                    canvas.DrawCircle(left + 26f, y - 4f, 3f, swatch);
                    break;
                case 2:
                    swatch.Style = SKPaintStyle.Stroke;
                    swatch.Color = new SKColor(255, 185, 0);
                    canvas.DrawLine(left + 5f, y - 4f, left + 21f, y - 4f, swatch);
                    break;
                default:
                    swatch.Style = SKPaintStyle.Stroke;
                    swatch.Color = new SKColor(255, 70, 220);
                    canvas.DrawRect(new SKRect(left + 5f, y - 9f, left + 21f, y + 1f), swatch);
                    break;
            }

            text.Color = new SKColor(235, 235, 245);
            canvas.DrawText(rows[row], left + 28f, y, font, text);
        }
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
