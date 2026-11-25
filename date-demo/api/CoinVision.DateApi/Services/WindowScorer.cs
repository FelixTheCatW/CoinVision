using System.Text.Json;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SkiaSharp;

namespace CoinVision.DateApi.Services;

/// <summary>What the ONNX graph tells us about itself.</summary>
public sealed class ScorerMeta
{
    public string InputName { get; init; } = "windows";
    public int Width { get; init; } = 160;
    public int Height { get; init; } = 80;
    public double ArcDegrees { get; init; } = 110;
    public double StepDegrees { get; init; } = 12;
    public IReadOnlyList<(double Inner, double Outer)> Bands { get; init; } = new[] { (0.32, 1.02) };
    public double Threshold { get; init; } = 0.5;
}

/// <summary>
/// Runs the window scorer: a 3x80x160 crop in, one logit out - "does this window hold the whole
/// year". The graph has the (x - 0.5) shift baked in, so the input is plain RGB in [0, 1].
/// </summary>
public sealed class WindowScorer : IDisposable
{
    private readonly InferenceSession _session;
    private readonly string _inputName;
    private readonly int _width;
    private readonly int _height;

    public WindowScorer(string onnxPath, string metaPath)
    {
        if (!File.Exists(onnxPath))
        {
            throw new FileNotFoundException($"window scorer not found: {onnxPath}", onnxPath);
        }

        var options = new Microsoft.ML.OnnxRuntime.SessionOptions
        {
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
        };
        _session = new InferenceSession(onnxPath, options);
        _inputName = _session.InputMetadata.Keys.First();

        var json = File.ReadAllText(metaPath);
        using var document = JsonDocument.Parse(json);
        var root = document.RootElement;
        var input = root.GetProperty("input");
        var sweep = root.GetProperty("sweep");
        var bands = sweep.GetProperty("bands").EnumerateArray()
            .Select(band => (band[0].GetDouble(), band[1].GetDouble()))
            .ToArray();

        Meta = new ScorerMeta
        {
            InputName = input.GetProperty("name").GetString() ?? "windows",
            Width = input.GetProperty("width").GetInt32(),
            Height = input.GetProperty("height").GetInt32(),
            ArcDegrees = sweep.GetProperty("arc_degrees").GetDouble(),
            StepDegrees = sweep.GetProperty("step_degrees").GetDouble(),
            Bands = bands,
            Threshold = root.TryGetProperty("threshold", out var threshold) ? threshold.GetDouble() : 0.5,
        };
        _width = Meta.Width;
        _height = Meta.Height;
    }

    public ScorerMeta Meta { get; }

    public int Width => _width;
    public int Height => _height;

    /// <summary>Probability that the window contains the whole year, in input order.</summary>
    public float[] Score(IReadOnlyList<SKBitmap> crops)
    {
        if (crops.Count == 0)
        {
            return Array.Empty<float>();
        }

        var plane = _width * _height;
        var tensor = new DenseTensor<float>(new[] { crops.Count, 3, _height, _width });
        for (var index = 0; index < crops.Count; index++)
        {
            var crop = crops[index];
            var stride = crop.RowBytes;
            var pixels = crop.GetPixelSpan();
            for (var y = 0; y < _height; y++)
            {
                for (var x = 0; x < _width; x++)
                {
                    var offset = y * stride + x * 4;
                    tensor[index, 0, y, x] = pixels[offset] / 255f;
                    tensor[index, 1, y, x] = pixels[offset + 1] / 255f;
                    tensor[index, 2, y, x] = pixels[offset + 2] / 255f;
                }
            }
        }

        using var results = _session.Run(new[] { NamedOnnxValue.CreateFromTensor(_inputName, tensor) });
        var logits = results.First().AsTensor<float>();
        var probabilities = new float[crops.Count];
        for (var index = 0; index < crops.Count; index++)
        {
            probabilities[index] = 1f / (1f + MathF.Exp(-logits[index]));
        }

        return probabilities;
    }

    public void Dispose() => _session.Dispose();
}
