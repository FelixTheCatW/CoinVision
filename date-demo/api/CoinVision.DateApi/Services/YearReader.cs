using System.Text.Json;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SkiaSharp;

namespace CoinVision.DateApi.Services;

/// <summary>What the reading model tells us about itself.</summary>
public sealed class ReaderMeta
{
    public string InputName { get; init; } = "images";
    public int Width { get; init; } = 448;
    public int Height { get; init; } = 224;
    public int Digits { get; init; } = 4;
}

/// <summary>The year read off one crop, with the confidence of the whole four-digit answer.</summary>
public sealed record Reading(string Year, double Confidence, IReadOnlyList<double> Digits);

/// <summary>
/// Reads the four-digit year from the crop the localiser cut. Five fold models vote by averaging
/// their per-digit probabilities, and the confidence of the answer is the product of the winning
/// digits' probabilities - a four-digit answer is only as sure as its weakest digit.
///
/// The graph takes plain RGB in [0, 1]: the ImageNet mean and std are baked into it, which is how the
/// window scorer already works and why a caller cannot silently skip the normalisation.
/// </summary>
public sealed class YearReader : IDisposable
{
    private readonly InferenceSession[] _sessions;
    private readonly string _inputName;
    private readonly int _width;
    private readonly int _height;

    public YearReader(IEnumerable<string> onnxPaths, string metaPath)
    {
        var paths = onnxPaths.ToArray();
        if (paths.Length == 0 || paths.Any(path => !File.Exists(path)))
        {
            throw new FileNotFoundException($"reading model not found: {string.Join(", ", paths)}");
        }

        var options = new Microsoft.ML.OnnxRuntime.SessionOptions
        {
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
        };
        _sessions = paths.Select(path => new InferenceSession(path, options)).ToArray();

        using var document = JsonDocument.Parse(File.ReadAllText(metaPath));
        var input = document.RootElement.GetProperty("input");
        Meta = new ReaderMeta
        {
            InputName = input.GetProperty("name").GetString() ?? "images",
            Width = input.GetProperty("width").GetInt32(),
            Height = input.GetProperty("height").GetInt32(),
            Digits = document.RootElement.GetProperty("digits").GetInt32(),
        };
        _inputName = Meta.InputName;
        _width = Meta.Width;
        _height = Meta.Height;
    }

    public ReaderMeta Meta { get; }

    public int Width => _width;
    public int Height => _height;

    /// <summary>Read one crop; it must already be <see cref="Width"/> x <see cref="Height"/> RGBA.</summary>
    public Reading Read(SKBitmap crop)
    {
        var tensor = new DenseTensor<float>(new[] { 1, 3, _height, _width });
        var stride = crop.RowBytes;
        var pixels = crop.GetPixelSpan();
        for (var y = 0; y < _height; y++)
        {
            for (var x = 0; x < _width; x++)
            {
                var offset = y * stride + x * 4;
                tensor[0, 0, y, x] = pixels[offset] / 255f;
                tensor[0, 1, y, x] = pixels[offset + 1] / 255f;
                tensor[0, 2, y, x] = pixels[offset + 2] / 255f;
            }
        }

        var average = new double[Meta.Digits, 10];
        foreach (var session in _sessions)
        {
            using var results = session.Run(new[] { NamedOnnxValue.CreateFromTensor(_inputName, tensor) });
            var logits = results.First().AsTensor<float>();
            for (var digit = 0; digit < Meta.Digits; digit++)
            {
                var maximum = float.MinValue;
                for (var classIndex = 0; classIndex < 10; classIndex++)
                {
                    maximum = MathF.Max(maximum, logits[0, digit, classIndex]);
                }

                var total = 0.0;
                var exponentials = new double[10];
                for (var classIndex = 0; classIndex < 10; classIndex++)
                {
                    exponentials[classIndex] = Math.Exp(logits[0, digit, classIndex] - maximum);
                    total += exponentials[classIndex];
                }

                for (var classIndex = 0; classIndex < 10; classIndex++)
                {
                    average[digit, classIndex] += exponentials[classIndex] / total / _sessions.Length;
                }
            }
        }

        var digits = new char[Meta.Digits];
        var confidence = 1.0;
        var perDigit = new double[Meta.Digits];
        for (var digit = 0; digit < Meta.Digits; digit++)
        {
            var best = 0;
            for (var classIndex = 1; classIndex < 10; classIndex++)
            {
                if (average[digit, classIndex] > average[digit, best])
                {
                    best = classIndex;
                }
            }

            digits[digit] = (char)('0' + best);
            perDigit[digit] = average[digit, best];
            confidence *= average[digit, best];
        }

        return new Reading(new string(digits), confidence, perDigit);
    }

    public void Dispose()
    {
        foreach (var session in _sessions)
        {
            session.Dispose();
        }
    }
}
