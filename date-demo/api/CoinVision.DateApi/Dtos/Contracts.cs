namespace CoinVision.DateApi.Models;

/// <summary>One coin the detector found, in original-image pixels.</summary>
public sealed record CoinDto(double Cx, double Cy, double Radius);

/// <summary>One scored window, so the page can show what the model considered.</summary>
public sealed record WindowDto(
    int Index,
    double CenterX,
    double CenterY,
    double AngleDeg,
    double ArcSpan,
    double RadialSpan,
    int X0,
    int Y0,
    int X1,
    int Y1,
    double Score,
    string Source = "ring");

/// <summary>Where the time went, in milliseconds.</summary>
public sealed record TimingDto(
    double Decode,
    double Detect,
    double Sweep,
    double Crop,
    double Score,
    double Read,
    double Total);

/// <summary>The answer for one uploaded image.</summary>
public sealed record DetectResponse(
    bool Ok,
    string? Error,
    string FileName,
    long SizeBytes,
    int? Width,
    int? Height,
    bool HasDate,
    double BestScore,
    double Threshold,
    string? CropPng,
    string? PreviewPng,
    int CropWidth,
    int CropHeight,
    IReadOnlyList<CoinDto> Coins,
    int WindowCount,
    IReadOnlyList<WindowDto> TopWindows,
    TimingDto? TimingMs,
    string? Year = null,
    double YearConfidence = 0,
    string? ReaderCropPng = null,
    IReadOnlyList<double>? YearDigits = null,
    bool ReaderFromDetector = false);

/// <summary>What the service is running, for the header of the page.</summary>
public sealed record InfoResponse(
    string Model,
    int CropWidth,
    int CropHeight,
    double ArcDegrees,
    double StepDegrees,
    IReadOnlyList<double[]> Bands,
    double Threshold,
    IReadOnlyList<string> Stages);
