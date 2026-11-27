using System.Globalization;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Shapes;
using Microsoft.Win32;
using Path = System.IO.Path;

namespace ImgExplorer;

/// <summary>
/// Просмотр изображения в масштабе 1:1 с выделением области по четырём точкам.
/// </summary>
public partial class MainWindow : Window
{
    private const int MaxPoints = 4;
    private const double MarkerArm = 6;

    private static readonly Brush[] PointBrushes =
    {
        CreateBrush(Color.FromRgb(0x35, 0xD0, 0x7F)),
        CreateBrush(Color.FromRgb(0xFF, 0xA0, 0x00)),
        CreateBrush(Color.FromRgb(0x4F, 0xA3, 0xFF)),
        CreateBrush(Color.FromRgb(0xD4, 0x6B, 0xFF)),
    };

    private static readonly Brush SelectionStrokeBrush = CreateBrush(Color.FromRgb(0x00, 0xE5, 0xFF));
    private static readonly Brush SelectionFillBrush = CreateBrush(Color.FromArgb(0x33, 0x00, 0xE5, 0xFF));
    private static readonly Brush PreviewStrokeBrush = CreateBrush(Color.FromArgb(0xB0, 0x00, 0xE5, 0xFF));

    /// <summary>Точки области в порядке нажатия Q (1-4).</summary>
    private readonly List<(int X, int Y)> _points = new();

    private BitmapSource? _image;
    private (int X, int Y)? _hoverPoint;

    private string _cursorText = "—";
    private string? _message;

    public MainWindow()
    {
        InitializeComponent();
        UpdateStatus();
    }

    // ---------- открытие изображения ----------

    private void OnOpenClick(object sender, RoutedEventArgs e) => OpenImageDialog();

    private void OpenImageDialog()
    {
        var dialog = new OpenFileDialog
        {
            Title = "Открыть изображение",
            Filter = ImageLoader.FileFilter,
            CheckFileExists = true,
        };

        if (dialog.ShowDialog(this) == true)
        {
            LoadImage(dialog.FileName);
        }
    }

    private void LoadImage(string path)
    {
        BitmapSource image;
        try
        {
            image = ImageLoader.Load(path);
        }
        catch (Exception ex)
        {
            MessageBox.Show(this, ex.Message, "Не удалось открыть изображение",
                MessageBoxButton.OK, MessageBoxImage.Warning);
            return;
        }

        _image = image;
        ImageView.Source = image;
        Placeholder.Visibility = Visibility.Collapsed;

        var fileName = Path.GetFileName(path);
        FileNameText.Text = $"{fileName} — {image.PixelWidth}×{image.PixelHeight} px";
        Title = $"{fileName} ({image.PixelWidth}×{image.PixelHeight}) — img-explorer";

        ClearSelection(message: null);
        _hoverPoint = null;
        _cursorText = "—";
        _message = "Изображение открыто в масштабе 1:1. Q — четыре точки области, W — другое изображение.";
        Scroll.ScrollToHome();
        UpdateStatus();
    }

    // ---------- клавиатура ----------

    private void OnPreviewKeyDown(object sender, KeyEventArgs e)
    {
        switch (e.Key)
        {
            case Key.W:
                e.Handled = true;
                OpenImageDialog();
                return;
            case Key.Q:
                e.Handled = true;
                RememberPoint();
                return;
        }

        if (e.Key == Key.Escape)
        {
            e.Handled = true;
            ClearSelection();
        }
    }

    // ---------- точки области ----------

    private void RememberPoint()
    {
        if (_image is null)
        {
            _message = "Сначала откройте изображение (W).";
            UpdateStatus();
            return;
        }

        // Четыре точки уже заданы — Q начинает новую область.
        if (_points.Count >= MaxPoints)
        {
            _points.Clear();
        }

        if (!TryGetPixelUnderCursor(out var point))
        {
            _message = "Курсор вне изображения — наведите указатель на картинку и повторите.";
            UpdateStatus();
            return;
        }

        ApplyPoint(point);
    }

    /// <summary>Добавляет точку области; на четвёртой точке копирует координаты в буфер обмена.</summary>
    private void ApplyPoint((int X, int Y) point)
    {
        _points.Add(point);

        if (_points.Count == MaxPoints)
        {
            var text = FormatPoints(_points);
            _message = CopyToClipboard(text)
                ? $"Скопировано: {text}"
                : $"Область задана, но скопировать в буфер обмена не удалось: {text}";
        }
        else
        {
            _message = $"Точка {_points.Count} = {point.X}, {point.Y}. Дальше Q.";
        }

        RedrawOverlay();
        UpdateStatus();
    }

    private void ClearSelection(string? message = "Выделение снято.")
    {
        _points.Clear();
        _message = _image is null ? null : message;
        RedrawOverlay();
        UpdateStatus();
    }

    private bool TryGetPixelUnderCursor(out (int X, int Y) point)
    {
        point = default;
        if (_image is null)
        {
            return false;
        }

        var position = Mouse.GetPosition(ImageView);

        // Позиция ровно на правой/нижней границе относится к последнему пикселю.
        if (position.X < -0.5 || position.Y < -0.5 ||
            position.X > _image.PixelWidth || position.Y > _image.PixelHeight)
        {
            return false;
        }

        var x = Math.Clamp((int)Math.Floor(position.X), 0, _image.PixelWidth - 1);
        var y = Math.Clamp((int)Math.Floor(position.Y), 0, _image.PixelHeight - 1);
        point = (x, y);
        return true;
    }

    private static bool CopyToClipboard(string text)
    {
        try
        {
            Clipboard.SetText(text);
            return true;
        }
        catch (Exception)
        {
            return false;
        }
    }

    // ---------- мышь ----------

    private void OnImageHostMouseMove(object sender, MouseEventArgs e)
    {
        if (_image is null)
        {
            return;
        }

        var previewVisible = _points.Count is > 0 and < MaxPoints;

        if (TryGetPixelUnderCursor(out var point))
        {
            _hoverPoint = point;
            _cursorText = string.Create(CultureInfo.InvariantCulture, $"{point.X}, {point.Y}");
        }
        else
        {
            _hoverPoint = null;
            _cursorText = "—";
        }

        if (previewVisible)
        {
            RedrawOverlay();
        }

        UpdateStatus();
    }

    private void OnImageHostMouseLeave(object sender, MouseEventArgs e)
    {
        _hoverPoint = null;
        _cursorText = "—";

        if (_points.Count is > 0 and < MaxPoints)
        {
            RedrawOverlay();
        }

        UpdateStatus();
    }

    // ---------- отрисовка маски ----------

    private void RedrawOverlay()
    {
        Overlay.Children.Clear();

        if (_image is null)
        {
            return;
        }

        if (_points.Count >= 3)
        {
            var polygon = new Polygon
            {
                Stroke = SelectionStrokeBrush,
                StrokeThickness = 1,
                Fill = _points.Count == MaxPoints ? SelectionFillBrush : null,
                IsHitTestVisible = false,
            };

            foreach (var point in _points)
            {
                polygon.Points.Add(new Point(point.X + 0.5, point.Y + 0.5));
            }

            Overlay.Children.Add(polygon);
        }
        else if (_points.Count == 2)
        {
            Overlay.Children.Add(CreateLine(_points[0], _points[1], SelectionStrokeBrush, dashed: false));
        }

        // Пока область не замкнута — пунктир от последней точки к курсору
        // (и замыкающая сторона, когда точек уже три).
        if (_points.Count is > 0 and < MaxPoints && _hoverPoint is { } hover)
        {
            Overlay.Children.Add(CreateLine(_points[^1], hover, PreviewStrokeBrush, dashed: true));

            if (_points.Count == 3)
            {
                Overlay.Children.Add(CreateLine(hover, _points[0], PreviewStrokeBrush, dashed: true));
            }
        }

        for (var i = 0; i < _points.Count; i++)
        {
            AddMarker(_points[i], PointBrushes[i], i + 1);
        }
    }

    private static Line CreateLine((int X, int Y) from, (int X, int Y) to, Brush stroke, bool dashed) => new()
    {
        X1 = from.X + 0.5,
        Y1 = from.Y + 0.5,
        X2 = to.X + 0.5,
        Y2 = to.Y + 0.5,
        Stroke = stroke,
        StrokeThickness = 1,
        StrokeDashArray = dashed ? new DoubleCollection([4.0, 3.0]) : null,
        IsHitTestVisible = false,
    };

    private void AddMarker((int X, int Y) point, Brush brush, int number)
    {
        var cx = point.X + 0.5;
        var cy = point.Y + 0.5;

        Overlay.Children.Add(new Line
        {
            X1 = cx - MarkerArm,
            Y1 = cy,
            X2 = cx + MarkerArm,
            Y2 = cy,
            Stroke = brush,
            StrokeThickness = 1,
        });

        Overlay.Children.Add(new Line
        {
            X1 = cx,
            Y1 = cy - MarkerArm,
            X2 = cx,
            Y2 = cy + MarkerArm,
            Stroke = brush,
            StrokeThickness = 1,
        });

        var ring = new Ellipse
        {
            Width = 5,
            Height = 5,
            Stroke = brush,
            StrokeThickness = 1,
            IsHitTestVisible = false,
        };

        Canvas.SetLeft(ring, cx - 2.5);
        Canvas.SetTop(ring, cy - 2.5);
        Overlay.Children.Add(ring);

        var label = new TextBlock
        {
            Text = number.ToString(CultureInfo.InvariantCulture),
            Foreground = brush,
            FontSize = 11,
            FontWeight = FontWeights.SemiBold,
            IsHitTestVisible = false,
        };

        Canvas.SetLeft(label, cx + MarkerArm + 1);
        Canvas.SetTop(label, cy - MarkerArm - 12);
        Overlay.Children.Add(label);
    }

    // ---------- статус ----------

    private void UpdateStatus()
    {
        var text = $"Курсор: {_cursorText}";

        for (var i = 0; i < MaxPoints; i++)
        {
            text += $"    {i + 1}: " + (i < _points.Count ? FormatPoint(_points[i]) : "—");
        }

        if (!string.IsNullOrEmpty(_message))
        {
            text += $"    |    {_message}";
        }

        StatusText.Text = text;
    }

    private static string FormatPoint((int X, int Y) point) =>
        string.Create(CultureInfo.InvariantCulture, $"{point.X}, {point.Y}");

    /// <summary>Координаты всех точек через запятую без пробелов: x1,y1,x2,y2,...</summary>
    private static string FormatPoints(IEnumerable<(int X, int Y)> points) =>
        string.Join(",", points.Select(point =>
            string.Create(CultureInfo.InvariantCulture, $"{point.X},{point.Y}")));

    private static Brush CreateBrush(Color color)
    {
        var brush = new SolidColorBrush(color);
        brush.Freeze();
        return brush;
    }
}
