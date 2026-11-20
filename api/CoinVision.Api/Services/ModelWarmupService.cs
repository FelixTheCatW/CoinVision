using System.Diagnostics;
using CoinVision.Api.Services;

namespace CoinVision.Api;

/// <summary>Runs one dummy inference at startup so the first real request does not pay the setup.</summary>
public sealed class ModelWarmupService(OnnxModelRunner runner, ILogger<ModelWarmupService> logger) : IHostedService
{
    public Task StartAsync(CancellationToken cancellationToken)
    {
        var watch = Stopwatch.StartNew();
        var zeros = new float[3 * runner.InputHeight * runner.InputWidth];
        var probabilities = runner.RunBatch(zeros, 1, cancellationToken);
        logger.LogInformation(
            "Warm-up done in {Milliseconds:F0} ms over {Models} graph(s) ({Digits}x{Classes} outputs, checksum {Checksum:F4})",
            watch.Elapsed.TotalMilliseconds,
            runner.Files.Count,
            runner.Digits,
            runner.Classes,
            probabilities.Sum());
        return Task.CompletedTask;
    }

    public Task StopAsync(CancellationToken cancellationToken) => Task.CompletedTask;
}
