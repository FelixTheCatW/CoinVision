namespace CoinVision.DateApi.Services;

/// <summary>
/// The numpy/scipy primitives `coinvision.localize` leans on, ported one to one: PIL's luminance,
/// its bilinear resize, scipy's reflect-mode gaussian, numpy's percentile with linear interpolation
/// and numpy's gradient (central differences inside, one-sided at the border).
/// Everything is float32 like the Python side; the point is that the C# circles and crops match.
/// </summary>
public static class ImageOps
{
    /// <summary>PIL "L": ITU-R 601-2 luma on 8-bit channels, then scaled to [0, 1].</summary>
    public static float[] ToGray(byte[] rgba, int width, int height)
    {
        var gray = new float[width * height];
        for (var index = 0; index < gray.Length; index++)
        {
            var offset = index * 4;
            var luma = (rgba[offset] * 299 + rgba[offset + 1] * 587 + rgba[offset + 2] * 114) / 1000;
            gray[index] = luma / 255f;
        }

        return gray;
    }

    /// <summary>PIL bilinear resize: destination centres map through (i + 0.5) * scale - 0.5.</summary>
    public static float[] ResizeBilinear(float[] source, int width, int height, int targetWidth,
                                         int targetHeight)
    {
        var target = new float[targetWidth * targetHeight];
        var scaleX = (double)width / targetWidth;
        var scaleY = (double)height / targetHeight;
        for (var y = 0; y < targetHeight; y++)
        {
            var sourceY = (y + 0.5) * scaleY - 0.5;
            var y0 = (int)Math.Floor(sourceY);
            var fy = sourceY - y0;
            for (var x = 0; x < targetWidth; x++)
            {
                var sourceX = (x + 0.5) * scaleX - 0.5;
                var x0 = (int)Math.Floor(sourceX);
                var fx = sourceX - x0;
                var top = Clamp(y0, height - 1) * width;
                var bottom = Clamp(y0 + 1, height - 1) * width;
                var a = Lerp(source[top + Clamp(x0, width - 1)], source[top + Clamp(x0 + 1, width - 1)], fx);
                var b = Lerp(source[bottom + Clamp(x0, width - 1)],
                             source[bottom + Clamp(x0 + 1, width - 1)], fx);
                target[y * targetWidth + x] = (float)Lerp(a, b, fy);
            }
        }

        return target;
    }

    /// <summary>Separable gaussian, scipy defaults: truncate 4 sigma, reflect border.</summary>
    public static float[] Gaussian(float[] source, int width, int height, double sigma)
    {
        if (sigma <= 0)
        {
            return (float[])source.Clone();
        }

        var radius = (int)(4.0 * sigma + 0.5);
        var kernel = new double[2 * radius + 1];
        var total = 0.0;
        for (var index = -radius; index <= radius; index++)
        {
            var weight = Math.Exp(-0.5 * (index / sigma) * (index / sigma));
            kernel[index + radius] = weight;
            total += weight;
        }

        for (var index = 0; index < kernel.Length; index++)
        {
            kernel[index] /= total;
        }

        var horizontal = new float[source.Length];
        for (var y = 0; y < height; y++)
        {
            for (var x = 0; x < width; x++)
            {
                var sum = 0.0;
                for (var k = -radius; k <= radius; k++)
                {
                    sum += kernel[k + radius] * source[y * width + Reflect(x + k, width)];
                }

                horizontal[y * width + x] = (float)sum;
            }
        }

        var result = new float[source.Length];
        for (var y = 0; y < height; y++)
        {
            for (var x = 0; x < width; x++)
            {
                var sum = 0.0;
                for (var k = -radius; k <= radius; k++)
                {
                    sum += kernel[k + radius] * horizontal[Reflect(y + k, height) * width + x];
                }

                result[y * width + x] = (float)sum;
            }
        }

        return result;
    }

    /// <summary>numpy.gradient on both axes: central differences inside, one-sided at the border.</summary>
    public static (float[] Gy, float[] Gx) Gradient(float[] source, int width, int height)
    {
        var gy = new float[source.Length];
        var gx = new float[source.Length];
        for (var y = 0; y < height; y++)
        {
            for (var x = 0; x < width; x++)
            {
                var index = y * width + x;
                var left = x == 0 ? index : index - 1;
                var right = x == width - 1 ? index : index + 1;
                gx[index] = (float)((source[right] - source[left]) / 2.0);

                var up = y == 0 ? index : index - width;
                var down = y == height - 1 ? index : index + width;
                gy[index] = (float)((source[down] - source[up]) / 2.0);
            }
        }

        return (gy, gx);
    }

    /// <summary>numpy.percentile with linear interpolation.</summary>
    public static double Percentile(float[] values, double percentile)
    {
        var sorted = (float[])values.Clone();
        Array.Sort(sorted);
        var position = (sorted.Length - 1) * percentile / 100.0;
        var lower = (int)Math.Floor(position);
        var upper = Math.Min(lower + 1, sorted.Length - 1);
        var fraction = position - lower;
        return sorted[lower] + (sorted[upper] - sorted[lower]) * fraction;
    }

    /// <summary>scipy.ndimage.map_coordinates(order=1, mode="constant"): zero outside the array.</summary>
    public static double SampleBilinear(float[] source, int width, int height, double x, double y)
    {
        if (x < -0.5 || y < -0.5 || x > width - 0.5 || y > height - 0.5)
        {
            return 0.0;
        }

        var x0 = (int)Math.Floor(x);
        var y0 = (int)Math.Floor(y);
        var fx = x - x0;
        var fy = y - y0;
        var a = Pixel(source, width, height, x0, y0);
        var b = Pixel(source, width, height, x0 + 1, y0);
        var c = Pixel(source, width, height, x0, y0 + 1);
        var d = Pixel(source, width, height, x0 + 1, y0 + 1);
        return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy;
    }

    /// <summary>numpy.linspace with the endpoint included unless asked otherwise.</summary>
    public static double[] Linspace(double start, double stop, int count, bool endpoint = true)
    {
        var values = new double[count];
        var divisor = endpoint ? Math.Max(count - 1, 1) : count;
        var step = (stop - start) / divisor;
        for (var index = 0; index < count; index++)
        {
            values[index] = start + index * step;
        }

        return values;
    }

    /// <summary>Keys cubic with a = -0.5, the filter PIL calls BICUBIC; outside the array is fill 0.</summary>
    public static double Cubic(float[] source, int width, int height, double x, double y)
    {
        var x0 = (int)Math.Floor(x);
        var y0 = (int)Math.Floor(y);
        var sum = 0.0;
        for (var j = -1; j <= 2; j++)
        {
            var wy = Keys(y - (y0 + j));
            if (wy == 0.0)
            {
                continue;
            }

            for (var i = -1; i <= 2; i++)
            {
                var wx = Keys(x - (x0 + i));
                if (wx == 0.0)
                {
                    continue;
                }

                sum += wx * wy * Pixel(source, width, height, x0 + i, y0 + j);
            }
        }

        return sum;
    }

    private static double Keys(double distance)
    {
        var x = Math.Abs(distance);
        if (x <= 1.0)
        {
            return (1.5 * x - 2.5) * x * x + 1.0;
        }

        if (x < 2.0)
        {
            return ((-0.5 * x + 2.5) * x - 4.0) * x + 2.0;
        }

        return 0.0;
    }

    private static double Pixel(float[] source, int width, int height, int x, int y)
    {
        if (x < 0 || y < 0 || x >= width || y >= height)
        {
            return 0.0;
        }

        return source[y * width + x];
    }

    private static int Reflect(int index, int size)
    {
        if (size == 1)
        {
            return 0;
        }

        while (index < 0 || index >= size)
        {
            index = index < 0 ? -index - 1 : 2 * size - index - 1;
        }

        return index;
    }

    private static int Clamp(int value, int limit)
    {
        if (value < 0)
        {
            return 0;
        }

        return value > limit ? limit : value;
    }

    private static double Lerp(double a, double b, double t) => a + (b - a) * t;
}
