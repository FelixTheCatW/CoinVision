namespace CoinVision.Api.Services;

using CoinVision.Api.Models;

/// <summary>Turns per-image timings and confidences into the batch summary.</summary>
public static class BatchStatistics
{
    private static readonly (string Label, double Lower, double Upper)[] ConfidenceBuckets =
    [
        ("<0.50", 0.0, 0.50),
        ("0.50-0.70", 0.50, 0.70),
        ("0.70-0.90", 0.70, 0.90),
        ("0.90-0.99", 0.90, 0.99),
        ("0.99-1.00", 0.99, 1.0000001),
    ];

    public static SummaryStats Summarise(IReadOnlyList<ImagePrediction> items, double wallMs)
    {
        var predictions = items.Where(item => item.Ok && item.TimingMs is not null).ToList();

        return new SummaryStats(
            items.Count,
            items.Count(item => !item.Ok),
            Math.Round(wallMs, 2),
            Distribution(predictions.Select(item => item.TimingMs!.DecodeMs)),
            Distribution(predictions.Select(item => item.TimingMs!.PreprocessMs)),
            Distribution(predictions.Select(item => item.TimingMs!.InferenceMs)),
            Distribution(predictions.Select(item => item.TimingMs!.TotalMs)),
            predictions.Count == 0 ? null : Math.Round(predictions.Average(item => item.Confidence ?? 0), 6),
            predictions.Count == 0 ? null : Math.Round(predictions.Min(item => item.Confidence ?? 0), 6),
            Histogram(predictions),
            predictions
                .Where(item => item.Year is not null)
                .GroupBy(item => item.Year!)
                .OrderByDescending(group => group.Count())
                .ToDictionary(group => group.Key, group => group.Count()));
    }

    private static DistributionStats Distribution(IEnumerable<double> values)
    {
        var sorted = values.OrderBy(value => value).ToArray();
        if (sorted.Length == 0)
        {
            return new DistributionStats(0, 0, 0, 0, 0);
        }

        var median = sorted.Length % 2 == 1
            ? sorted[sorted.Length / 2]
            : (sorted[sorted.Length / 2 - 1] + sorted[sorted.Length / 2]) / 2;
        var p95Index = Math.Min(sorted.Length - 1, (int)Math.Ceiling(0.95 * sorted.Length) - 1);

        return new DistributionStats(
            Math.Round(sorted.Average(), 2),
            Math.Round(median, 2),
            Math.Round(sorted[p95Index], 2),
            Math.Round(sorted[^1], 2),
            Math.Round(sorted.Sum(), 2));
    }

    private static Dictionary<string, int> Histogram(IReadOnlyList<ImagePrediction> predictions)
    {
        var histogram = ConfidenceBuckets.ToDictionary(bucket => bucket.Label, _ => 0);
        foreach (var prediction in predictions)
        {
            var confidence = prediction.Confidence ?? 0;
            var bucket = ConfidenceBuckets.FirstOrDefault(entry => confidence >= entry.Lower && confidence < entry.Upper);
            var label = bucket.Label ?? ConfidenceBuckets[^1].Label;
            histogram[label] = histogram.GetValueOrDefault(label) + 1;
        }

        return histogram;
    }
}
