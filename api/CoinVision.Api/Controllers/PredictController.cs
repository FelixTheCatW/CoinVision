using System.Diagnostics;
using CoinVision.Api.Models;
using CoinVision.Api.Services;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Options;

namespace CoinVision.Api.Controllers;

public sealed class UploadOptions
{
    public const string Section = "Uploads";

    public int MaxFilesPerRequest { get; set; } = 50;
    public long MaxFileBytes { get; set; } = 25 * 1024 * 1024;
}

/// <summary>Recognition endpoints: multipart batch, raw single file and the machine-readable info.</summary>
[ApiController]
[Route("api")]
[Produces("application/json")]
public sealed class PredictController(
    PredictionService predictionService,
    OnnxModelRunner runner,
    IOptions<UploadOptions> uploadOptions,
    ILogger<PredictController> logger) : ControllerBase
{
    private static readonly DateTimeOffset StartedAt = DateTimeOffset.UtcNow;

    private readonly UploadOptions _uploads = uploadOptions.Value;

    /// <summary>Recognise every image in a multipart request. Field names are ignored.</summary>
    [HttpPost("predict")]
    [RequestSizeLimit(512 * 1024 * 1024)]
    [ProducesResponseType(typeof(BatchResponse), StatusCodes.Status200OK)]
    [ProducesResponseType(typeof(ProblemDetails), StatusCodes.Status400BadRequest)]
    public async Task<ActionResult<BatchResponse>> Predict(CancellationToken cancellationToken)
    {
        var requestId = Activity.Current?.Id ?? HttpContext.TraceIdentifier;
        var wall = Stopwatch.StartNew();
        var files = Request.HasFormContentType ? Request.Form.Files : new FormFileCollection();

        if (files.Count == 0)
        {
            logger.LogWarning("Request {RequestId} carried no files", requestId);
            return Problem(
                title: "No files",
                detail: "Send one or more images as multipart/form-data fields; the field name is ignored.",
                statusCode: StatusCodes.Status400BadRequest);
        }

        if (files.Count > _uploads.MaxFilesPerRequest)
        {
            logger.LogWarning(
                "Request {RequestId} carried {Count} files, limit is {Limit}",
                requestId, files.Count, _uploads.MaxFilesPerRequest);
            return Problem(
                title: "Too many files",
                detail: $"At most {_uploads.MaxFilesPerRequest} images per request.",
                statusCode: StatusCodes.Status400BadRequest);
        }

        logger.LogInformation(
            "Request {RequestId}: {Count} file(s), {Bytes} bytes",
            requestId, files.Count, files.Sum(file => file.Length));

        var warnings = new List<string>();
        var slots = new ImagePrediction?[files.Count];
        var accepted = new List<(int Index, IFormFile File)>();

        for (var index = 0; index < files.Count; index++)
        {
            var file = files[index];
            var name = Path.GetFileName(file.FileName);
            if (file.Length == 0)
            {
                slots[index] = Rejected(name, file.ContentType, 0, "empty upload");
                continue;
            }

            if (file.Length > _uploads.MaxFileBytes)
            {
                var message = $"file is larger than {_uploads.MaxFileBytes} bytes";
                warnings.Add($"{name}: {message}");
                slots[index] = Rejected(name, file.ContentType, file.Length, message);
                continue;
            }

            accepted.Add((index, file));
        }

        var predictions = await predictionService.PredictBatchAsync(
            accepted.Select(entry => entry.File).ToList(), cancellationToken);
        for (var offset = 0; offset < accepted.Count; offset++)
        {
            slots[accepted[offset].Index] = predictions[offset];
        }

        var items = slots.Select(item => item!).ToList();
        wall.Stop();
        var summary = BatchStatistics.Summarise(items, wall.Elapsed.TotalMilliseconds);

        logger.LogInformation(
            "Request {RequestId} done in {WallMs:F1} ms: {Ok} recognised, {Failed} failed, mean confidence {Confidence:F3}",
            requestId, summary.WallMs, summary.Count - summary.Failed, summary.Failed, summary.ConfidenceMean ?? 0);

        return Ok(new BatchResponse(requestId, DateTimeOffset.UtcNow, runner.Describe(), items, summary, warnings));
    }

    /// <summary>Recognise a single image sent as the raw request body (any content type).</summary>
    [HttpPost("predict/single")]
    [RequestSizeLimit(512 * 1024 * 1024)]
    [ProducesResponseType(typeof(ImagePrediction), StatusCodes.Status200OK)]
    [ProducesResponseType(typeof(ProblemDetails), StatusCodes.Status400BadRequest)]
    public async Task<ActionResult<ImagePrediction>> PredictSingle(CancellationToken cancellationToken)
    {
        using var buffer = new MemoryStream();
        await Request.Body.CopyToAsync(buffer, cancellationToken);
        var content = buffer.ToArray();
        if (content.Length == 0)
        {
            return Problem(
                title: "Empty body",
                detail: "Send the image bytes as the request body.",
                statusCode: StatusCodes.Status400BadRequest);
        }

        var fileName = Request.Headers["X-Filename"].FirstOrDefault() ?? "upload.bin";
        var formFile = new FormFile(new MemoryStream(content), 0, content.Length, "file", fileName)
        {
            Headers = new HeaderDictionary(),
            ContentType = Request.ContentType ?? "application/octet-stream",
        };

        var predictions = await predictionService.PredictBatchAsync([formFile], cancellationToken);
        return Ok(predictions[0]);
    }

    /// <summary>Everything a script needs: model identity, limits and the response contract.</summary>
    [HttpGet("info")]
    [ProducesResponseType(typeof(ApiInfo), StatusCodes.Status200OK)]
    public ActionResult<ApiInfo> Info()
    {
        var version = typeof(PredictController).Assembly.GetName().Version?.ToString() ?? "0.0.0";
        return Ok(new ApiInfo(
            "coinvision-api",
            version,
            runner.Describe(),
            [
                "POST /api/predict        multipart/form-data, one or more files, any field name",
                "POST /api/predict/single raw body, optional X-Filename header",
                "GET  /api/info           this document",
                "GET  /health             liveness plus model summary",
                "GET  /scalar/v1          interactive OpenAPI reference",
                "GET  /openapi/v1.json    OpenAPI document",
            ],
            _uploads.MaxFilesPerRequest,
            _uploads.MaxFileBytes,
            ["image/jpeg", "image/png", "image/webp", "image/bmp", "image/tiff"],
            "confidence = product of the per-digit probabilities, i.e. the probability of the whole year; digits[3].confidence isolates the last digit"));
    }

    [HttpGet("/health")]
    [ProducesResponseType(StatusCodes.Status200OK)]
    public ActionResult<object> Health() => Ok(new
    {
        status = "ok",
        models = runner.Files.Count,
        input = $"{runner.InputWidth}x{runner.InputHeight}",
        uptimeSeconds = Math.Round((DateTimeOffset.UtcNow - StartedAt).TotalSeconds, 1),
    });

    private static ImagePrediction Rejected(string fileName, string? contentType, long size, string error) =>
        new(fileName, contentType, size, null, null, false, error, null, null, [], [], null, null);
}
