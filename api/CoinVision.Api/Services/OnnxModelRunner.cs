using CoinVision.Api.Models;
using Microsoft.Extensions.Options;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace CoinVision.Api.Services;

/// <summary>Settings that point the API at the exported ONNX files.</summary>
public sealed class ModelOptions
{
    public const string Section = "Model";

    /// <summary>Directory holding model.json and the .onnx files produced by `coinvision export`.</summary>
    public string Directory { get; set; } = "../../onnx";

    /// <summary>Upper bound on how many graphs are averaged; they are taken in file order.</summary>
    public int MaxModels { get; set; } = 8;
}

public sealed record ModelOutput(float[] Probabilities);

/// <summary>
/// Loads the exported graph(s) once and runs them. Several checkpoints are averaged, which is what
/// the cross-validation measured as the multi-seed bag (small but consistent help on rare classes).
/// </summary>
public sealed class OnnxModelRunner : IDisposable
{
    private readonly IReadOnlyList<InferenceSession> _sessions;
    private readonly ILogger<OnnxModelRunner> _logger;

    public OnnxModelRunner(IOptions<ModelOptions> options, IHostEnvironment environment, ILogger<OnnxModelRunner> logger)
    {
        _logger = logger;
        var directory = ResolveDirectory(options.Value.Directory, environment);
        var metadataPath = Path.Combine(directory, "model.json");
        if (!File.Exists(metadataPath))
        {
            throw new FileNotFoundException(
                $"model.json not found in {directory}. Run `uv run coinvision export --checkpoint ... --out-dir {directory}` first.");
        }

        var metadata = System.Text.Json.JsonSerializer.Deserialize<OnnxMetadata>(
            File.ReadAllText(metadataPath)) ?? throw new InvalidOperationException("model.json is empty");

        InputWidth = metadata.Input.Width;
        InputHeight = metadata.Input.Height;
        Digits = metadata.Digits;
        Classes = metadata.Classes;
        Mean = metadata.Input.Mean;
        Std = metadata.Input.Std;

        var chosen = metadata.Models.Take(Math.Max(1, options.Value.MaxModels)).ToList();
        var sessions = new List<InferenceSession>(chosen.Count);
        foreach (var entry in chosen)
        {
            var path = Path.Combine(directory, entry.File);
            if (!File.Exists(path))
            {
                throw new FileNotFoundException($"ONNX file missing: {path}");
            }

            sessions.Add(new InferenceSession(path, SessionOptions()));
        }

        _sessions = sessions;
        Files = chosen.Select(entry => entry.File).ToList();
        Sources = chosen.Select(entry => entry.Source).ToList();
        MaxAbsDiff = chosen.Select(entry => entry.MaxAbsDiff).Where(value => value.HasValue).DefaultIfEmpty(null).Max();
        ModelDirectory = directory;
        _logger.LogInformation(
            "Loaded {Count} ONNX model(s) from {Directory}, input {Width}x{Height}, digits {Digits}, classes {Classes}",
            Files.Count, directory, InputWidth, InputHeight, Digits, Classes);
    }

    public int InputWidth { get; }
    public int InputHeight { get; }
    public int Digits { get; }
    public int Classes { get; }
    public float[] Mean { get; }
    public float[] Std { get; }
    public IReadOnlyList<string> Files { get; }
    public IReadOnlyList<string> Sources { get; }
    public double? MaxAbsDiff { get; }
    public string ModelDirectory { get; }

    /// <summary>Mean softmax over all loaded graphs for a whole batch; layout [batch, digits, classes].</summary>
    public float[] RunBatch(float[] input, int batch, CancellationToken cancellationToken)
    {
        var dimensions = new[] { batch, 3, InputHeight, InputWidth };
        var tensor = new DenseTensor<float>(input, dimensions);
        var inputName = _sessions[0].InputNames[0];

        var accumulated = new float[batch * Digits * Classes];
        foreach (var session in _sessions)
        {
            cancellationToken.ThrowIfCancellationRequested();
            using var results = session.Run(
                new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(inputName, tensor) });
            var logits = results[0].AsTensor<float>();
            var values = logits.ToArray();
            for (var sample = 0; sample < batch; sample++)
            {
                for (var digit = 0; digit < Digits; digit++)
                {
                    var source = sample * Digits * Classes + digit * Classes;
                    var target = sample * Digits * Classes + digit * Classes;
                    SoftmaxInto(values, source, accumulated, target);
                }
            }

        }

        for (var index = 0; index < accumulated.Length; index++)
        {
            accumulated[index] /= _sessions.Count;
        }

        return accumulated;
    }

    /// <summary>Adds the softmax of logits[source..source+classes] into target[..].</summary>
    private void SoftmaxInto(float[] logits, int source, float[] target, int targetOffset)
    {
        var maximum = float.MinValue;
        for (var index = 0; index < Classes; index++)
        {
            if (logits[source + index] > maximum)
            {
                maximum = logits[source + index];
            }
        }

        var sum = 0.0;
        var exponentials = new double[Classes];
        for (var index = 0; index < Classes; index++)
        {
            exponentials[index] = Math.Exp(logits[source + index] - maximum);
            sum += exponentials[index];
        }

        for (var index = 0; index < Classes; index++)
        {
            target[targetOffset + index] += (float)(exponentials[index] / sum);
        }
    }

    public ModelInfo Describe() => new(
        Files.Count,
        Files,
        InputWidth,
        InputHeight,
        Digits,
        Classes,
        Sources,
        MaxAbsDiff);

    private static Microsoft.ML.OnnxRuntime.SessionOptions SessionOptions() => new()
    {
        GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
        IntraOpNumThreads = Math.Max(1, Environment.ProcessorCount / 2),
        InterOpNumThreads = 1,
    };

    private static string ResolveDirectory(string configured, IHostEnvironment environment)
    {
        if (Path.IsPathRooted(configured))
        {
            return configured;
        }

        var fromContentRoot = Path.GetFullPath(Path.Combine(environment.ContentRootPath, configured));
        return Directory.Exists(fromContentRoot)
            ? fromContentRoot
            : Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, configured));
    }

    public void Dispose()
    {
        foreach (var session in _sessions)
        {
            session.Dispose();
        }
    }
}
