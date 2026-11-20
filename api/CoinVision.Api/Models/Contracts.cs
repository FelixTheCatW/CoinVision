using System.Text.Json.Serialization;

namespace CoinVision.Api.Models;

/// <summary>Confidence of a single digit position.</summary>
public sealed record DigitPrediction(int Position, int Digit, double Confidence);

/// <summary>Lower-ranked year for the same prefix, so a technician can see what else was close.</summary>
public sealed record YearAlternative(string Year, double Probability);

/// <summary>Where the time of one image went.</summary>
public sealed record TimingBreakdown(double DecodeMs, double PreprocessMs, double InferenceMs, double TotalMs);

/// <summary>Result for one uploaded image. Always present, even when <see cref="Ok"/> is false.</summary>
public sealed record ImagePrediction(
    string FileName,
    string? ContentType,
    long SizeBytes,
    int? OriginalWidth,
    int? OriginalHeight,
    bool Ok,
    string? Error,
    string? Year,
    double? Confidence,
    IReadOnlyList<DigitPrediction> Digits,
    IReadOnlyList<YearAlternative> Alternatives,
    TimingBreakdown? TimingMs,
    string? Sha256);

/// <summary>Distribution of one timing stage over the batch.</summary>
public sealed record DistributionStats(double Mean, double Median, double P95, double Max, double Total);

/// <summary>Whole-batch statistics: timings, confidence spread and the year histogram.</summary>
public sealed record SummaryStats(
    int Count,
    int Failed,
    double WallMs,
    DistributionStats DecodeMs,
    DistributionStats PreprocessMs,
    DistributionStats InferenceMs,
    DistributionStats TotalMs,
    double? ConfidenceMean,
    double? ConfidenceMin,
    IReadOnlyDictionary<string, int> ConfidenceHistogram,
    IReadOnlyDictionary<string, int> Years);

/// <summary>The exported models this API is serving.</summary>
public sealed record ModelInfo(
    int Count,
    IReadOnlyList<string> Files,
    int InputWidth,
    int InputHeight,
    int Digits,
    int Classes,
    IReadOnlyList<string> Sources,
    double? MaxAbsDiff);

/// <summary>Response of POST /api/predict.</summary>
public sealed record BatchResponse(
    string RequestId,
    DateTimeOffset ReceivedAt,
    ModelInfo Model,
    IReadOnlyList<ImagePrediction> Items,
    SummaryStats Summary,
    IReadOnlyList<string> Warnings);

/// <summary>Response of GET /api/info, the machine-readable contract summary.</summary>
public sealed record ApiInfo(
    string Name,
    string Version,
    ModelInfo Model,
    IReadOnlyList<string> Endpoints,
    int MaxFilesPerRequest,
    long MaxFileBytes,
    IReadOnlyList<string> AcceptedContentTypes,
    string ConfidenceDefinition);

/// <summary>Shape of the ONNX metadata sidecar written by `coinvision export`.</summary>
internal sealed class OnnxMetadata
{
    [JsonPropertyName("input")]
    public OnnxInput Input { get; set; } = new();

    [JsonPropertyName("digits")]
    public int Digits { get; set; }

    [JsonPropertyName("classes")]
    public int Classes { get; set; }

    [JsonPropertyName("models")]
    public List<OnnxModelEntry> Models { get; set; } = [];
}

internal sealed class OnnxInput
{
    [JsonPropertyName("width")]
    public int Width { get; set; }

    [JsonPropertyName("height")]
    public int Height { get; set; }

    [JsonPropertyName("mean")]
    public float[] Mean { get; set; } = [];

    [JsonPropertyName("std")]
    public float[] Std { get; set; } = [];
}

internal sealed class OnnxModelEntry
{
    [JsonPropertyName("file")]
    public string File { get; set; } = string.Empty;

    [JsonPropertyName("source")]
    public string Source { get; set; } = string.Empty;

    [JsonPropertyName("max_abs_diff")]
    public double? MaxAbsDiff { get; set; }
}
