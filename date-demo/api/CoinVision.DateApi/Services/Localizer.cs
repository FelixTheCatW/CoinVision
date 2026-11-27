using SkiaSharp;

namespace CoinVision.DateApi.Services;

/// <summary>A coin as the detector sees it, in original-image pixels.</summary>
public sealed record Circle(double Cx, double Cy, double Radius);

/// <summary>
/// One arc window: the rotated rectangle `crop_upright` will unroll. `Box` is only its bounding
/// rectangle, for drawing.
/// </summary>
public sealed record Window(
    Circle Circle,
    double CenterX,
    double CenterY,
    double AngleDeg,
    double ArcSpan,
    double RadialSpan,
    int X0,
    int Y0,
    int X1,
    int Y1,
    string Source = "ring");

/// <summary>
/// Port of `src/coinvision/localize.py`: gradient-vote circle centres, a rim radius per centre, the
/// arc sweep and the upright crop. Same constants and the same order of operations, so the windows
/// and crops match the Python reference the model was trained against.
/// </summary>
public sealed class Localizer
{
    private const int SearchLongest = 800;
    private const double EdgePercentile = 96.0;
    private const int AngleSamples = 360;
    private const int RadiusSamples = 220;

    // A coin shows several concentric rings of gradient - a shield inside, the rim outside - and the
    // rim is the one farthest from the centre. Every peak reaching this share of the strongest peak
    // is a candidate and the outermost of them becomes the radius.
    private const double RimPeakShare = 0.6;

    // Two centres closer than this share of the larger radius are the same coin, not two coins.
    private const double ConcentricShare = 0.3;

    // The accumulator also fires on captions, watermarks and slab edges. A real rim is visible at
    // every angle; a candidate whose per-angle gradient peak collapses on part of the arc is not a
    // coin, and the next peak is tried instead. If that leaves a slot empty, the best rejected peak
    // is used - a coin cut by the frame keeps a rim on part of the arc only.
    private const double RimMinScore = 0.12;
    private const int MaxPicks = 8;

    public IReadOnlyList<Circle> DetectCoins(float[] gray, int width, int height, int maxCoins)
    {
        var scale = SearchLongest / (double)Math.Max(width, height);
        float[] small;
        int smallWidth, smallHeight;
        if (scale >= 1.0)
        {
            small = gray;
            smallWidth = width;
            smallHeight = height;
            scale = 1.0;
        }
        else
        {
            smallWidth = Math.Max((int)(width * scale), 8);
            smallHeight = Math.Max((int)(height * scale), 8);
            small = ImageOps.ResizeBilinear(gray, width, height, smallWidth, smallHeight);
        }

        var shortest = Math.Min(smallWidth, smallHeight);
        var smooth = ImageOps.Gaussian(small, smallWidth, smallHeight, 1.2);
        var (gradY, gradX) = ImageOps.Gradient(smooth, smallWidth, smallHeight);
        var magnitude = new float[small.Length];
        for (var index = 0; index < magnitude.Length; index++)
        {
            magnitude[index] = (float)Math.Sqrt(gradX[index] * gradX[index] + gradY[index] * gradY[index]);
        }

        var threshold = ImageOps.Percentile(magnitude, EdgePercentile);
        var edgeX = new List<int>();
        var edgeY = new List<int>();
        var weight = new List<float>();
        for (var index = 0; index < magnitude.Length; index++)
        {
            if (magnitude[index] >= threshold)
            {
                edgeX.Add(index % smallWidth);
                edgeY.Add(index / smallWidth);
                weight.Add(magnitude[index]);
            }
        }

        if (edgeX.Count < 50)
        {
            return Array.Empty<Circle>();
        }

        var accumulator = new float[small.Length];
        var radii = ImageOps.Linspace(0.18 * shortest, 0.58 * shortest, 20);
        foreach (var radiusValue in radii)
        {
            var radius = Math.Max(radiusValue, 1.0);
            foreach (var sign in new[] { 1.0, -1.0 })
            {
                for (var edge = 0; edge < edgeX.Count; edge++)
                {
                    var direction = Math.Atan2(gradY[edgeY[edge] * smallWidth + edgeX[edge]],
                                               gradX[edgeY[edge] * smallWidth + edgeX[edge]]);
                    var voteX = (int)Math.Round(edgeX[edge] + sign * radius * Math.Cos(direction),
                                                MidpointRounding.ToEven);
                    var voteY = (int)Math.Round(edgeY[edge] + sign * radius * Math.Sin(direction),
                                                MidpointRounding.ToEven);
                    if (voteX < 0 || voteX >= smallWidth || voteY < 0 || voteY >= smallHeight)
                    {
                        continue;
                    }

                    accumulator[voteY * smallWidth + voteX] += weight[edge] / (float)radius;
                }
            }
        }

        accumulator = ImageOps.Gaussian(accumulator, smallWidth, smallHeight, 3.0);
        var circles = new List<Circle>(maxCoins);
        var rejected = new List<(double Score, Circle Circle)>();
        var working = (float[])accumulator.Clone();
        for (var pick = 0; pick < MaxPicks && circles.Count < maxCoins; pick++)
        {
            var flat = 0;
            var best = 0f;
            for (var index = 0; index < working.Length; index++)
            {
                if (working[index] > best)
                {
                    best = working[index];
                    flat = index;
                }
            }

            if (best <= 0f)
            {
                break;
            }

            var centerY = flat / smallWidth;
            var centerX = flat % smallWidth;
            var radius = RimRadius(magnitude, smallWidth, smallHeight, centerX, centerY, shortest);
            if (radius is not null)
            {
                var circle = new Circle(centerX / scale, centerY / scale, radius.Value / scale);
                var score = RimScore(magnitude, smallWidth, smallHeight, centerX, centerY, radius.Value);
                if (score >= RimMinScore)
                {
                    circles.Add(circle);
                }
                else
                {
                    rejected.Add((score, circle));
                }
            }

            var radiusPixels = Math.Max(radius ?? 0.2 * shortest, 0.2 * shortest);
            var limit = radiusPixels * 1.1;
            for (var y = 0; y < smallHeight; y++)
            {
                var dy = y - centerY;
                for (var x = 0; x < smallWidth; x++)
                {
                    var dx = x - centerX;
                    if (dx * dx + dy * dy <= limit * limit)
                    {
                        working[y * smallWidth + x] = 0f;
                    }
                }
            }
        }

        // A coin cut by the frame leaves a rim on part of the arc only, so the filter can reject
        // every peak; then the best rejected one beats nothing. A photo where one circle did pass
        // keeps only that one - filling the second slot is how a caption becomes a coin.
        if (circles.Count == 0 && rejected.Count > 0)
        {
            rejected.Sort((left, right) => right.Score.CompareTo(left.Score));
            circles.Add(rejected[0].Circle);
        }

        return MergeConcentric(circles);
    }

    /// <summary>One circle per coin: two peaks on the same coin keep the circle farther out.</summary>
    private static IReadOnlyList<Circle> MergeConcentric(List<Circle> circles)
    {
        var merged = new List<Circle>(circles.Count);
        foreach (var circle in circles)
        {
            var absorbed = false;
            for (var index = 0; index < merged.Count; index++)
            {
                var kept = merged[index];
                var gapX = circle.Cx - kept.Cx;
                var gapY = circle.Cy - kept.Cy;
                var gap = Math.Sqrt(gapX * gapX + gapY * gapY);
                if (gap > ConcentricShare * Math.Max(circle.Radius, kept.Radius))
                {
                    continue;
                }

                if (circle.Radius > kept.Radius)
                {
                    merged[index] = circle;
                }

                absorbed = true;
                break;
            }

            if (!absorbed)
            {
                merged.Add(circle);
            }
        }

        return merged;
    }

    /// <summary>How much of the circle sits on an edge: 1 all the way round, 0 nothing.</summary>
    private static double RimScore(float[] magnitude, int width, int height, double centerX,
                                   double centerY, double radius)
    {
        const int samples = 180;
        const int band = 15;
        var angles = ImageOps.Linspace(0.0, 2.0 * Math.PI, samples, endpoint: false);
        var radii = ImageOps.Linspace(0.9 * radius, 1.1 * radius, band);
        var heights = new double[samples];
        for (var index = 0; index < samples; index++)
        {
            var best = 0.0;
            for (var step = 0; step < band; step++)
            {
                var value = ImageOps.SampleBilinear(magnitude, width, height,
                    centerX + radii[step] * Math.Cos(angles[index]),
                    centerY + radii[step] * Math.Sin(angles[index]));
                if (value > best)
                {
                    best = value;
                }
            }

            heights[index] = best;
        }

        var median = Median(heights);
        if (median <= 1e-6)
        {
            return 0.0;
        }

        Array.Sort(heights);
        var index10 = Math.Clamp((int)(0.10 * heights.Length), 0, heights.Length - 1);
        return heights[index10] / median;
    }

    /// <summary>Read the rim radius off the radial median profile of the gradient magnitude.</summary>
    private static double? RimRadius(float[] magnitude, int width, int height, int centerX,
                                     int centerY, int shortest)
    {
        var angles = ImageOps.Linspace(0.0, 2.0 * Math.PI, AngleSamples, endpoint: false);
        var probe = ImageOps.Linspace(0.18 * shortest, 0.62 * shortest, RadiusSamples);
        var profile = new float[probe.Length];
        var samples = new double[angles.Length];
        for (var index = 0; index < probe.Length; index++)
        {
            for (var angle = 0; angle < angles.Length; angle++)
            {
                samples[angle] = ImageOps.SampleBilinear(
                    magnitude, width, height,
                    centerX + probe[index] * Math.Cos(angles[angle]),
                    centerY + probe[index] * Math.Sin(angles[angle]));
            }

            profile[index] = (float)Median(samples);
        }

        var smooth = ImageOps.Gaussian(profile, profile.Length, 1, 2.0);
        var strongest = 0f;
        foreach (var value in smooth)
        {
            if (value > strongest)
            {
                strongest = value;
            }
        }

        if (strongest <= 1e-6f)
        {
            return null;
        }

        var tallest = 0;
        for (var index = 1; index < smooth.Length; index++)
        {
            if (smooth[index] > smooth[tallest])
            {
                tallest = index;
            }
        }

        // Local maxima, then the outermost one that is comparable in height with the strongest.
        var outermost = -1;
        for (var index = 1; index < smooth.Length - 1; index++)
        {
            if (smooth[index] >= smooth[index - 1] && smooth[index] >= smooth[index + 1]
                && smooth[index] >= RimPeakShare * strongest)
            {
                outermost = index;
            }
        }

        return probe[outermost >= 0 ? outermost : tallest];
    }

    private static double Median(double[] values)
    {
        var sorted = (double[])values.Clone();
        Array.Sort(sorted);
        var middle = sorted.Length / 2;
        return sorted.Length % 2 == 0 ? 0.5 * (sorted[middle - 1] + sorted[middle]) : sorted[middle];
    }

    /// <summary>The arc sweep: `count` positions a full turn, one window per radial band.</summary>
    public IReadOnlyList<Window> RingWindows(Circle circle, double arcDegrees, double stepDegrees,
                                             IReadOnlyList<(double Inner, double Outer)> bands)
    {
        var windows = new List<Window>();
        var arc = arcDegrees * Math.PI / 180.0;
        var step = stepDegrees * Math.PI / 180.0;
        var count = (int)Math.Round(2.0 * Math.PI / step, MidpointRounding.ToEven);
        foreach (var (innerFraction, outerFraction) in bands)
        {
            var inner = circle.Radius * innerFraction;
            var outer = circle.Radius * outerFraction;
            var radiusMid = 0.5 * (inner + outer);
            for (var index = 0; index < count; index++)
            {
                var mid = index * step;
                var half = 0.5 * arc;
                var xs = new List<double>(4);
                var ys = new List<double>(4);
                foreach (var sign in new[] { -1.0, 1.0 })
                {
                    foreach (var radius in new[] { inner, outer })
                    {
                        xs.Add(circle.Cx + radius * Math.Cos(mid + sign * half));
                        ys.Add(circle.Cy + radius * Math.Sin(mid + sign * half));
                    }
                }

                windows.Add(new Window(
                    circle,
                    circle.Cx + radiusMid * Math.Cos(mid),
                    circle.Cy + radiusMid * Math.Sin(mid),
                    WrapDegrees(-(mid * 180.0 / Math.PI + 90.0)),
                    arc * radiusMid,
                    outer - inner,
                    (int)xs.Min(),
                    (int)ys.Min(),
                    (int)xs.Max(),
                    (int)ys.Max()));
            }
        }

        return windows;
    }

    /// <summary>
    /// `crop_upright`: the window unrolled into a `width x height` upright rectangle. The affine is
    /// the same one the Python side builds, only inverted and handed to Skia, whose sampler is far
    /// faster than a per-pixel loop and cubic like PIL's BICUBIC.
    /// </summary>
    public static SKBitmap CropWindow(SKBitmap source, Window window, int width, int height,
                                      double margin = 0.0)
    {
        var arc = window.ArcSpan * (1.0 + margin);
        var radial = window.RadialSpan * (1.0 + margin);
        var alpha = -window.AngleDeg * Math.PI / 180.0;
        var cosA = Math.Cos(alpha);
        var sinA = Math.Sin(alpha);
        var scaleX = arc / width;
        var scaleY = radial / height;
        var a = cosA * scaleX;
        var b = sinA * scaleY;
        var d = -sinA * scaleX;
        var e = cosA * scaleY;
        var c = window.CenterX - a * (width / 2.0) - b * (height / 2.0);
        var f = window.CenterY - d * (width / 2.0) - e * (height / 2.0);

        var determinant = a * e - b * d;
        var inverse = new SKMatrix
        {
            ScaleX = (float)(e / determinant),
            SkewX = (float)(-b / determinant),
            TransX = (float)((-e * c + b * f) / determinant),
            SkewY = (float)(-d / determinant),
            ScaleY = (float)(a / determinant),
            TransY = (float)((d * c - a * f) / determinant),
            Persp0 = 0,
            Persp1 = 0,
            Persp2 = 1,
        };

        var info = new SKImageInfo(width, height, SKColorType.Rgba8888, SKAlphaType.Unpremul);
        var crop = new SKBitmap(info);
        using (var canvas = new SKCanvas(crop))
        {
            canvas.Clear(SKColors.Black);
            canvas.Concat(ref inverse);
            canvas.DrawBitmap(source, 0, 0, new SKSamplingOptions(SKCubicResampler.CatmullRom));
        }

        return crop;
    }

    private static double WrapDegrees(double value)
    {
        var wrapped = value % 360.0;
        if (wrapped > 180.0)
        {
            wrapped -= 360.0;
        }
        else if (wrapped < -180.0)
        {
            wrapped += 360.0;
        }

        return wrapped;
    }

    private static byte ToByte(double value)
    {
        if (value < 0)
        {
            return 0;
        }

        return value > 255 ? (byte)255 : (byte)Math.Round(value);
    }
}
