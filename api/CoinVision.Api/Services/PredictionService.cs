using System.Diagnostics;
using System.Security.Cryptography;
using CoinVision.Api.Models;
using SkiaSharp;

namespace CoinVision.Api.Services;

/// <summary>
/// Decodes and preprocesses the uploads, runs them through the graphs as one batch and turns the
/// logits into per-image results. Decode and preprocess are timed per image, the inference is timed
/// for the batch and its share is attributed to every image.
/// </summary>
public sealed class PredictionService(
    OnnxModelRunner runner,
    IImagePreprocessor preprocessor,
    ILogger<PredictionService> logger)
{
    private const int AlternativeCount = 3;

    private sealed record Prepared(
        int Index,
        string FileName,
        string? ContentType,
        long SizeBytes,
        int Width,
        int Height,
        string Sha256,
        float[] Tensor,
        double DecodeMs,
        double PreprocessMs);

    public async Task<IReadOnlyList<ImagePrediction>> PredictBatchAsync(
        IReadOnlyList<IFormFile> files,
        CancellationToken cancellationToken)
    {
        var results = new ImagePrediction?[files.Count];
        var prepared = new List<Prepared>(files.Count);

        for (var index = 0; index < files.Count; index++)
        {
            var (failure, ready) = await PrepareAsync(index, files[index], cancellationToken);
            if (failure is not null)
            {
                results[index] = failure;
            }
            else if (ready is not null)
            {
                prepared.Add(ready);
            }
        }

        if (prepared.Count == 0)
        {
            return results.Select(item => item!).ToList();
        }

        var sampleSize = 3 * runner.InputHeight * runner.InputWidth;
        var batch = new float[prepared.Count * sampleSize];
        for (var offset = 0; offset < prepared.Count; offset++)
        {
            Array.Copy(prepared[offset].Tensor, 0, batch, offset * sampleSize, sampleSize);
        }

        var inferenceWatch = Stopwatch.StartNew();
        var probabilities = runner.RunBatch(batch, prepared.Count, cancellationToken);
        inferenceWatch.Stop();
        var shareMs = inferenceWatch.Elapsed.TotalMilliseconds / prepared.Count;
        var stride = runner.Digits * runner.Classes;

        for (var offset = 0; offset < prepared.Count; offset++)
        {
            var entry = prepared[offset];
            var slice = new float[stride];
            Array.Copy(probabilities, offset * stride, slice, 0, stride);

            var (year, confidence, digits, alternatives) = Interpret(slice);
            var timing = new TimingBreakdown(
                Math.Round(entry.DecodeMs, 2),
                Math.Round(entry.PreprocessMs, 2),
                Math.Round(shareMs, 2),
                Math.Round(entry.DecodeMs + entry.PreprocessMs + shareMs, 2));

            logger.LogInformation(
                "Predicted {File} ({Bytes} bytes, {Width}x{Height}) -> {Year} at {Confidence:P1} in {TotalMs:F1} ms (decode {DecodeMs:F1}, preprocess {PreprocessMs:F1}, inference share {InferenceMs:F1})",
                entry.FileName, entry.SizeBytes, entry.Width, entry.Height, year, confidence,
                timing.TotalMs, timing.DecodeMs, timing.PreprocessMs, timing.InferenceMs);

            results[entry.Index] = new ImagePrediction(
                entry.FileName, entry.ContentType, entry.SizeBytes, entry.Width, entry.Height,
                true, null, year, Math.Round(confidence, 6), digits, alternatives, timing, entry.Sha256);
        }

        return results.Select(item => item!).ToList();
    }

    private async Task<(ImagePrediction? Failure, Prepared? Ready)> PrepareAsync(
        int index,
        IFormFile file,
        CancellationToken cancellationToken)
    {
        var fileName = Path.GetFileName(file.FileName);
        byte[] content;
        try
        {
            using var buffer = new MemoryStream(capacity: (int)Math.Min(file.Length, int.MaxValue));
            await file.CopyToAsync(buffer, cancellationToken);
            content = buffer.ToArray();
        }
        catch (Exception exception)
        {
            logger.LogWarning(exception, "Could not read {File}", fileName);
            return (Failure(fileName, file.ContentType, file.Length, "could not read the upload"), null);
        }

        var sha256 = Convert.ToHexString(SHA256.HashData(content)).ToLowerInvariant();

        int width;
        int height;
        var decodeWatch = Stopwatch.StartNew();
        try
        {
            using var stream = new SKMemoryStream(content);
            using var codec = SKCodec.Create(stream) ?? throw new InvalidOperationException("unknown image format");
            width = codec.Info.Width;
            height = codec.Info.Height;
        }
        catch (Exception exception)
        {
            logger.LogWarning(exception, "Could not decode {File} ({Bytes} bytes)", fileName, content.Length);
            var decodeMs = Math.Round(decodeWatch.Elapsed.TotalMilliseconds, 2);
            return (Failure(fileName, file.ContentType, content.Length, "not a decodable image") with
            {
                Sha256 = sha256,
                TimingMs = new TimingBreakdown(decodeMs, 0, 0, decodeMs),
            }, null);
        }

        decodeWatch.Stop();

        var preprocessWatch = Stopwatch.StartNew();
        PreprocessedImage pixels;
        try
        {
            pixels = preprocessor.Process(content, runner.InputWidth, runner.InputHeight, runner.Mean, runner.Std);
        }
        catch (Exception exception)
        {
            logger.LogWarning(exception, "Could not preprocess {File}", fileName);
            var decodeMs = Math.Round(decodeWatch.Elapsed.TotalMilliseconds, 2);
            var preprocessMs = Math.Round(preprocessWatch.Elapsed.TotalMilliseconds, 2);
            return (Failure(fileName, file.ContentType, content.Length, "could not preprocess the image") with
            {
                Sha256 = sha256,
                TimingMs = new TimingBreakdown(decodeMs, preprocessMs, 0, Math.Round(decodeMs + preprocessMs, 2)),
            }, null);
        }

        preprocessWatch.Stop();

        return (null, new Prepared(
            index, fileName, file.ContentType, content.Length, width, height, sha256, pixels.Tensor,
            decodeWatch.Elapsed.TotalMilliseconds, preprocessWatch.Elapsed.TotalMilliseconds));
    }

    private (string Year, double Confidence, IReadOnlyList<DigitPrediction> Digits, IReadOnlyList<YearAlternative> Alternatives) Interpret(float[] probabilities)
    {
        var digits = new DigitPrediction[runner.Digits];
        var chosen = new int[runner.Digits];
        var confidence = 1.0;
        for (var position = 0; position < runner.Digits; position++)
        {
            var offset = position * runner.Classes;
            var best = 0;
            for (var candidate = 1; candidate < runner.Classes; candidate++)
            {
                if (probabilities[offset + candidate] > probabilities[offset + best])
                {
                    best = candidate;
                }
            }

            chosen[position] = best;
            var digitConfidence = probabilities[offset + best];
            confidence *= digitConfidence;
            digits[position] = new DigitPrediction(position, best, Math.Round(digitConfidence, 6));
        }

        // The prefix digits are constant for this dataset, so runner-up years differ in the last
        // digit: the alternatives show what the model considered instead of the winner.
        var prefix = string.Concat(chosen.Take(runner.Digits - 1));
        var lastOffset = (runner.Digits - 1) * runner.Classes;
        var alternatives = Enumerable.Range(0, runner.Classes)
            .Select(digit => new YearAlternative(prefix + digit, Math.Round(probabilities[lastOffset + digit], 6)))
            .Where(item => item.Year != prefix + chosen[^1])
            .OrderByDescending(item => item.Probability)
            .Take(AlternativeCount)
            .ToList();

        return (string.Concat(chosen), confidence, digits, alternatives);
    }

    private static ImagePrediction Failure(string fileName, string? contentType, long size, string error) =>
        new(fileName, contentType, size, null, null, false, error, null, null, [], [], null, null);
}
