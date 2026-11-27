using CoinVision.DateApi.Models;
using CoinVision.DateApi.Services;

var builder = WebApplication.CreateBuilder(args);

var baseDir = AppContext.BaseDirectory;
var modelDir = Path.Combine(baseDir, "models");
var onnxPath = Path.Combine(modelDir, "window_scorer.onnx");
var metaPath = Path.Combine(modelDir, "window_scorer.json");
var detectorPath = Path.Combine(modelDir, "date_detector.onnx");
var detectorMetaPath = Path.Combine(modelDir, "date_detector.json");
var readerMetaPath = Path.Combine(modelDir, "reading_reader.json");
var readerPaths = Enumerable.Range(0, 5)
    .Select(fold => Path.Combine(modelDir, $"reading_reader_{fold}.onnx"))
    .ToArray();

builder.Services.AddSingleton(_ => new WindowScorer(onnxPath, metaPath));
builder.Services.AddSingleton(_ => new DateDetector(detectorPath, detectorMetaPath));
builder.Services.AddSingleton(_ => new YearReader(readerPaths, readerMetaPath));
builder.Services.AddSingleton<Localizer>();
builder.Services.AddSingleton<DatePipeline>();
builder.Services.Configure<Microsoft.AspNetCore.Http.Features.FormOptions>(options =>
{
    options.MultipartBodyLengthLimit = 25 * 1024 * 1024;
});

var app = builder.Build();

app.UseDefaultFiles();
app.UseStaticFiles();

app.MapGet("/api/health", () => Results.Ok(new { status = "ok", time = DateTime.UtcNow }));

app.MapGet("/api/info", (WindowScorer scorer) => Results.Ok(new InfoResponse(
    "one ranker over two sources: 110 deg arc windows (bands 0.30-1.10 + 0.60-1.25) and the detector's year quads; "
        + "1637 photos, 5-fold recall@1 0.971+-0.017 inside the window cache, 0.927 end-to-end, threshold from metadata; "
        + "a reader trained on the detector's crops then reads the digits (0.987 with no oracle)",
    scorer.Width,
    scorer.Height,
    scorer.Meta.ArcDegrees,
    scorer.Meta.StepDegrees,
    scorer.Meta.Bands.Select(band => new[] { band.Inner, band.Outer }).ToList(),
    scorer.Meta.Threshold,
    new[] { "decode", "detect coins", "sweep windows", "score windows", "crop", "read year" })));

app.MapPost("/api/detect", async (HttpRequest request, DatePipeline pipeline, ILogger<Program> log,
                                  int? window) =>
{
    if (!request.HasFormContentType)
    {
        return Results.BadRequest(new { error = "multipart/form-data expected" });
    }

    var form = await request.ReadFormAsync();
    var file = form.Files.FirstOrDefault();
    if (file is null || file.Length == 0)
    {
        return Results.BadRequest(new { error = "no file uploaded" });
    }

    if (file.Length > 25 * 1024 * 1024)
    {
        return Results.BadRequest(new { error = "file is larger than 25 MB" });
    }

    using var stream = new MemoryStream();
    await file.CopyToAsync(stream);
    var content = stream.ToArray();

    try
    {
        var result = await Task.Run(() => pipeline.Run(content, file.FileName, window));
        return Results.Ok(result);
    }
    catch (Exception exception)
    {
        log.LogError(exception, "detect failed for {File}", file.FileName);
        return Results.Problem(detail: exception.Message, statusCode: 500);
    }
}).DisableAntiforgery();

app.Run();
