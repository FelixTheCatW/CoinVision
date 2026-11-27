using System.IO;
using System.Runtime.InteropServices;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using SkiaSharp;

namespace ImgExplorer;

/// <summary>
/// Декодирование изображений через SkiaSharp: поддерживаются png, jpg, webp, gif
/// (у gif берётся первый кадр) и любые другие форматы, которые понимает Skia.
/// </summary>
public static class ImageLoader
{
    public const string FileFilter =
        "Изображения (*.png;*.jpg;*.jpeg;*.webp;*.gif)|*.png;*.jpg;*.jpeg;*.webp;*.gif|" +
        "PNG (*.png)|*.png|" +
        "JPEG (*.jpg;*.jpeg)|*.jpg;*.jpeg|" +
        "WebP (*.webp)|*.webp|" +
        "GIF (*.gif)|*.gif|" +
        "Все файлы (*.*)|*.*";

    /// <summary>Загружает файл и возвращает bitmap с плотностью 96 dpi (1 пиксель изображения = 1 DIP).</summary>
    public static BitmapSource Load(string path)
    {
        using var codec = SKCodec.Create(path)
            ?? throw new InvalidDataException($"Файл «{Path.GetFileName(path)}» не является изображением поддерживаемого формата.");

        var info = new SKImageInfo(codec.Info.Width, codec.Info.Height, SKColorType.Bgra8888, SKAlphaType.Premul);
        using var decoded = new SKBitmap(info);

        var result = codec.GetPixels(info, decoded.GetPixels());
        if (result is not (SKCodecResult.Success or SKCodecResult.IncompleteInput))
        {
            throw new InvalidDataException($"Не удалось декодировать изображение ({result}).");
        }

        using var bitmap = ApplyOrigin(decoded, codec.EncodedOrigin);

        var stride = bitmap.Width * 4;
        var source = BitmapSource.Create(
            bitmap.Width,
            bitmap.Height,
            96,
            96,
            PixelFormats.Pbgra32,
            null,
            bitmap.Bytes,
            stride);

        source.Freeze();
        return source;
    }

    /// <summary>
    /// Разворачивает изображение согласно EXIF-ориентации и приводит его к формату BGRA.
    /// </summary>
    private static SKBitmap ApplyOrigin(SKBitmap source, SKEncodedOrigin origin)
    {
        var swapAxes = origin is SKEncodedOrigin.LeftTop or SKEncodedOrigin.RightTop
            or SKEncodedOrigin.RightBottom or SKEncodedOrigin.LeftBottom;

        var width = swapAxes ? source.Height : source.Width;
        var height = swapAxes ? source.Width : source.Height;

        var target = new SKBitmap(new SKImageInfo(width, height, SKColorType.Bgra8888, SKAlphaType.Premul));

        var sourcePixels = MemoryMarshal.Cast<byte, uint>(source.GetPixelSpan());
        var targetPixels = MemoryMarshal.Cast<byte, uint>(target.GetPixelSpan());

        var sourceStride = source.RowBytes / 4;
        var targetStride = target.RowBytes / 4;
        int sourceWidth = source.Width, sourceHeight = source.Height;

        for (var y = 0; y < height; y++)
        {
            for (var x = 0; x < width; x++)
            {
                var (sx, sy) = origin switch
                {
                    SKEncodedOrigin.TopRight => (sourceWidth - 1 - x, y),
                    SKEncodedOrigin.BottomRight => (sourceWidth - 1 - x, sourceHeight - 1 - y),
                    SKEncodedOrigin.BottomLeft => (x, sourceHeight - 1 - y),
                    SKEncodedOrigin.LeftTop => (y, x),
                    SKEncodedOrigin.RightTop => (y, sourceHeight - 1 - x),
                    SKEncodedOrigin.RightBottom => (sourceWidth - 1 - y, sourceHeight - 1 - x),
                    SKEncodedOrigin.LeftBottom => (sourceWidth - 1 - y, x),
                    _ => (x, y),
                };

                targetPixels[(y * targetStride) + x] = sourcePixels[(sy * sourceStride) + sx];
            }
        }

        return target;
    }
}
